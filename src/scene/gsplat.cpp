/* SPDX-FileCopyrightText: 2026 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "scene/gsplat.h"
#include "scene/gsplat_receiver.h"

#include "scene/bake.h"
#include "scene/integrator.h"
#include "scene/shader_nodes.h"
#include "util/algorithm.h"
#include "util/log.h"
#include "util/math_gsplat.h"
#include "util/math_spherical_harmonics.h"
#include "util/tbb.h"
#include "util/time.h"

#include <tbb/parallel_sort.h>

#include <limits>

CCL_NAMESPACE_BEGIN

void Scene::device_update_gsplat_raster()
{
  KernelGSplatRaster &raster = dscene.data.gsplat_raster;
  raster = {};
  auto clear = [&]() {
    dscene.gsplat_raster_splats.free();
    dscene.gsplat_raster_tiles.free();
    dscene.gsplat_raster_indices.free();
    dscene.gsplat_shader_shadow_transmission.free();
  };
  const KernelCamera &cam = dscene.data.cam;
  /* The independent layer has a single pinhole projection. Keep general
   * transport for cameras and scene features requiring per-path evaluation. */
  if (!integrator->get_use_gsplat_raster() || device->info.type != DEVICE_OPTIX ||
      cam.type != CAMERA_PERSPECTIVE || cam.aperturesize != 0.0f || cam.num_motion_steps != 0 ||
      cam.have_perspective_motion || cam.interocular_offset != 0.0f || cam.width <= 0 ||
      cam.height <= 0 || !dscene.data.film.is_rec709 || dscene.data.bvh.have_volumes ||
      has_shadow_catcher() || bake_manager->get_baking())
  {
    clear();
    return;
  }

  /* A single mesh depth cannot represent a transparent surface stack. */
  for (const Shader *shader : shaders) {
    if (shader->reference_count() && shader->has_surface_transparent) {
      clear();
      return;
    }
  }
  raster.receive_shadows = integrator->get_gsplat_receive_shadows();
  raster.shadow_strength = integrator->get_gsplat_shadow_strength();
  raster.contact_distance = integrator->get_gsplat_contact_distance();
  for (const Object *object : objects) {
    if (!(object->get_geometry()->primitive_type() & PRIMITIVE_GSPLAT) &&
        (object->visibility_for_tracing() & PATH_RAY_VISIBILITY_SHADOW))
    {
      raster.have_shadow_geometry = true;
      break;
    }
  }
  /* Static single-closure material data. The packed w channel carries artistic
   * blocker opacity; RGB retains the legacy transmission data. Unsupported
   * graphs use full opacity instead of guessing a transmission fraction. */
  auto *shadow_transmission = dscene.gsplat_shader_shadow_transmission.alloc(
      dscene.shaders.size());
  std::fill_n(shadow_transmission, dscene.shaders.size(), make_float4(0, 0, 0, 1));
  int glass_shaders = 0;
  for (Shader *shader : shaders) {
    if (shader->id < dscene.shaders.size() && shader->is_default_gsplat) {
      /* Mark only captured-radiance shaders for secondary receiver shading. */
      shadow_transmission[shader->id].x = -1;
      continue;
    }
    if (!shader->graph || shader->id >= dscene.shaders.size()) {
      continue;
    }
    ShaderOutput *surface = shader->graph->output()->input("Surface")->link;
    if (!surface) {
      continue;
    }
    ShaderNode *node = surface->parent;
    float3 color = one_float3();
    float transmission = 0, ior = 1.5f;
    bool thin = false;
    if (node->type == PrincipledBsdfNode::get_node_type()) {
      auto *bsdf = static_cast<PrincipledBsdfNode *>(node);
      if (bsdf->input("Base Color")->link || bsdf->input("Transmission Weight")->link ||
          bsdf->input("Metallic")->link || bsdf->input("IOR")->link)
      {
        continue;
      }
      color = bsdf->get_base_color();
      transmission = bsdf->get_transmission_weight() * (1 - bsdf->get_metallic());
      ior = bsdf->get_ior();
      thin = bsdf->is_thin_wall();
    }
    else if (node->type == GlassBsdfNode::get_node_type() ||
             node->type == RefractionBsdfNode::get_node_type())
    {
      if (node->input("Color")->link || node->input("IOR")->link) {
        continue;
      }
      color = static_cast<BsdfNode *>(node)->get_color();
      transmission = 1;
      ior = node->type == GlassBsdfNode::get_node_type() ?
                static_cast<GlassBsdfNode *>(node)->get_IOR() :
                static_cast<RefractionBsdfNode *>(node)->get_IOR();
    }
    shadow_transmission[shader->id] = gsplat_material_shadow_transmission(
        color, transmission, ior, thin);
    /* The receiver overlay uses an artistic opacity rather than physical
     * transmission. Keep the packed buffer layout for other device kernels. */
    shadow_transmission[shader->id].w = gsplat_material_shadow_opacity(
        transmission, integrator->get_gsplat_glass_shadow_opacity());
    glass_shaders += transmission > 0;
  }
  dscene.gsplat_shader_shadow_transmission.copy_to_device();
  LOG_DEBUG << "GSplat transmissive shadow shaders: " << glass_shaders;
  const double start_time = time_dt();
  const int width = int(cam.width), height = int(cam.height);
  const int tiles_x = divide_up(width, GSPLAT_RASTER_TILE_SIZE);
  const int tiles_y = divide_up(height, GSPLAT_RASTER_TILE_SIZE);
  struct Projected {
    KernelGSplat splat;
    int4 tiles;
    size_t order;
  };
  vector<Projected> projected;
  const float3 p = transform_perspective(&cam.rastertocamera, zero_float3());
  const float3 px = transform_perspective(&cam.rastertocamera, make_float3(1, 0, 0));
  const float3 py = transform_perspective(&cam.rastertocamera, make_float3(0, 1, 0));
  const float2 focal = make_float2(1.0f / (px.x / px.z - p.x / p.z),
                                   1.0f / (py.y / py.z - p.y / p.z));
  const float dilation = 0.3f * sqr(cam.differential_scale);
  const float3 eye = make_float3(
      cam.cameratoworld.x.w, cam.cameratoworld.y.w, cam.cameratoworld.z.w);

  size_t next_order = 0;
  std::vector<float4> receiver_samples;
  for (const Object *object : objects) {
    if (!gsplat_raster_object_supported(object) ||
        !(object->visibility_for_tracing() & PATH_RAY_VISIBILITY_CAMERA))
    {
      continue;
    }
    const PointCloud *cloud = static_cast<const PointCloud *>(object->get_geometry());
    const Transform model_view = cam.worldtocamera * object->get_tfm();
    const Transform inverse_object = transform_inverse(object->get_tfm());
    const float3 local_eye = transform_point(&inverse_object, eye);
    const float3x3 view_axes = make_float3x3(
        make_float3(model_view.x), make_float3(model_view.y), make_float3(model_view.z));
    const auto *positions = cloud->get_position();
    const auto *scales = cloud->attributes.find(ATTR_STD_GSPLAT_SCALE)->data<packed_float3>();
    const auto *rotations = cloud->attributes.find(ATTR_STD_GSPLAT_ROTATION)->data<Quaternion>();
    const auto *colors = cloud->attributes.find(ATTR_STD_GSPLAT_RADIANCE_BASE)->data<float4>();
    if (raster.receive_shadows && raster.have_shadow_geometry && raster.shadow_strength > 0) {
      /* Sample before any camera/frustum culling. Moving or zooming the camera
       * must not change the surface model used to anchor shadow receivers. */
      const size_t stride = std::max(size_t(1), divide_up(cloud->num_points(), size_t(65536)));
      for (size_t i = 0; i < cloud->num_points(); i += stride) {
        const float3 world = transform_point(&object->get_tfm(), positions[i]);
        if (isfinite_safe(world) && colors[i].w >= 0.05f) {
          receiver_samples.push_back(make_float4(world, colors[i].w));
        }
      }
    }
    const Attribute *sh = cloud->attributes.find(
        ATTR_STD_GSPLAT_RADIANCE_SPHERICAL_HARMONICS_REST);
    /* Worker completion order must not affect coplanar compositing. Reserve
     * keys for every source particle, including particles culled by projection,
     * so keys remain unique across objects. */
    const size_t object_order = next_order;
    next_order += cloud->num_points();
    enumerable_thread_specific<vector<Projected>> local_projected;
    parallel_for(
        blocked_range<size_t>(0, cloud->num_points(), 4096),
        [&](const blocked_range<size_t> &range) {
          vector<Projected> &output = local_projected.local();
          for (size_t i = range.begin(); i < range.end(); ++i) {
            const float3 center = transform_point(&model_view, positions[i]);
            if (!(center.z > cam.nearclip && center.z < cam.nearclip + cam.cliplength) ||
                !(colors[i].w >= GSPLAT_RASTER_ALPHA_MIN))
            {
              continue;
            }
            const float3 world = transform_point(&object->get_tfm(), positions[i]);
            const float3 screen = transform_perspective(&cam.worldtoraster, world);
            /* Match the reference raster's 20% dilated center-frustum culling.
             * A near-camera floater with an off-screen center can otherwise project
             * to an enormous ellipse covering the entire image. */
            if (!isfinite_safe(screen) || screen.x < -0.1f * width || screen.x > 1.1f * width ||
                screen.y < -0.1f * height || screen.y > 1.1f * height)
            {
              continue;
            }
            const float3x3 axes = view_axes * quaternion_to_rotation(rotations[i]) *
                                  float3x3_scale(scales[i]);
            const float3 cov = gsplat_project_covariance(center, axes, focal, dilation);
            const float det = cov.x * cov.z - sqr(cov.y);
            if (!(det > 0 && isfinite_safe(cov))) {
              continue;
            }
            const float2 radius = make_float2(sqrtf(GSPLAT_RASTER_EXTENT * cov.x),
                                              sqrtf(GSPLAT_RASTER_EXTENT * cov.z));
            if (!isfinite_safe(screen) || (!isfinite_safe(radius.x) || !isfinite_safe(radius.y)) ||
                screen.x + radius.x < 0 || screen.y + radius.y < 0 ||
                screen.x - radius.x >= width || screen.y - radius.y >= height)
            {
              continue;
            }
            Projected item;
            const float2 depth_gradient = gsplat_project_depth_gradient(center, axes, focal, cov);
            item.splat.center = make_float4(screen.x, screen.y, center.z, colors[i].w);
            item.splat.conic = make_float4(
                cov.z / det, -cov.y / det, cov.x / det, depth_gradient.x);
            const float3 direction = safe_normalize(float3(positions[i]) - local_eye);
            float3 color = spherical_harmonics_evaluate_band_L0(direction, make_float3(colors[i]));
            if (sh) {
              color += spherical_harmonics_evaluate_rest(
                  sh->data<PackedSphericalHarmonicsRest>()[i], direction);
            }
            item.splat.color = make_float4(max(color + 0.5f, zero_float3()), depth_gradient.y);
            const float3 scale = fabs(float3(scales[i]));
            const float3x3 rotation = quaternion_to_rotation(rotations[i]);
            const float3 local_normal = rotation * ((scale.x <= scale.y && scale.x <= scale.z) ?
                                                        make_float3(1, 0, 0) :
                                                    (scale.y <= scale.z) ? make_float3(0, 1, 0) :
                                                                           make_float3(0, 0, 1));
            float3 normal = safe_normalize(
                transform_direction_transposed(&inverse_object, local_normal));
            if (dot(normal, eye - world) < 0) {
              normal = -normal;
            }
            if (reduce_min(scale) > 0.85f * reduce_max(scale)) {
              normal = zero_float3();
            }
            item.splat.receiver_plane = make_float4(normal, integrator->get_gsplat_receiver_bias());
            item.tiles = make_int4(
                int(clamp(floorf((screen.x - radius.x) / GSPLAT_RASTER_TILE_SIZE),
                          0.0f,
                          float(tiles_x - 1))),
                int(clamp(floorf((screen.y - radius.y) / GSPLAT_RASTER_TILE_SIZE),
                          0.0f,
                          float(tiles_y - 1))),
                int(clamp(floorf((screen.x + radius.x) / GSPLAT_RASTER_TILE_SIZE),
                          0.0f,
                          float(tiles_x - 1))),
                int(clamp(floorf((screen.y + radius.y) / GSPLAT_RASTER_TILE_SIZE),
                          0.0f,
                          float(tiles_y - 1))));
            item.order = object_order + i;
            output.push_back(item);
          }
        });
    size_t projected_count = projected.size();
    for (const vector<Projected> &output : local_projected) {
      projected_count += output.size();
    }
    projected.reserve(projected_count);
    for (const vector<Projected> &output : local_projected) {
      projected.insert(projected.end(), output.begin(), output.end());
    }
  }
  if (projected.empty()) {
    /* Enable even for an off-screen cloud so camera rays still skip its BVH. */
    clear();
    raster.enabled = true;
    return;
  }
  const double project_time = time_dt();
  tbb::parallel_sort(
      projected.begin(), projected.end(), [](const Projected &a, const Projected &b) {
        return a.splat.center.z < b.splat.center.z ||
               (a.splat.center.z == b.splat.center.z && a.order < b.order);
      });

  const double sort_time = time_dt();
  vector<uint2> tiles(size_t(tiles_x) * tiles_y, make_uint2(0, 0));
  for (const Projected &item : projected) {
    for (int y = item.tiles.y; y <= item.tiles.w; ++y) {
      for (int x = item.tiles.x; x <= item.tiles.z; ++x) {
        ++tiles[size_t(y) * tiles_x + x].y;
      }
    }
  }
  size_t count = 0;
  for (uint2 &tile : tiles) {
    tile.x = uint(count);
    count += tile.y;
    tile.y = tile.x;
  }
  if (count > std::numeric_limits<uint>::max() ||
      projected.size() > size_t(std::numeric_limits<int>::max()))
  {
    clear();
    return;
  }
  /* Cache a continuous receiver guide independently of mesh depth. A guide is
   * fitted from nearby Gaussian centers in the same visible depth layer, not
   * from their often unstable shortest-axis normals. Per-pixel work is then
   * just four guide reads rather than a fit on every path sample. */
  vector<float4> receivers;
  const double receiver_start = time_dt();
  if (raster.receive_shadows && raster.have_shadow_geometry && raster.shadow_strength > 0) {
    const int2 grid = gsplat_shadow_receiver_grid(width, height);
    const float2 step = make_float2(float(width) / grid.x, float(height) / grid.y);
    vector<vector<uint>> center_bins(size_t(tiles_x) * tiles_y);
    for (uint i = 0; i < projected.size(); ++i) {
      const float4 c = projected[i].splat.center;
      if (c.x >= 0 && c.y >= 0 && c.x < width && c.y < height) {
        center_bins[size_t(int(c.y) / GSPLAT_RASTER_TILE_SIZE) * tiles_x +
                    int(c.x) / GSPLAT_RASTER_TILE_SIZE]
            .push_back(i);
      }
    }
    receivers.resize(size_t(grid.x + 1) * (grid.y + 1), zero_float4());
    parallel_for(
        blocked_range<size_t>(0, receivers.size(), 32), [&](const blocked_range<size_t> &range) {
          vector<uint> candidates;
          vector<std::pair<float, float>> depths;
          vector<float2> layers;
          vector<float4> points;
          vector<float> weights;
          for (size_t index = range.begin(); index < range.end(); ++index) {
            const int gx = int(index % (grid.x + 1)), gy = int(index / (grid.x + 1));
            const float2 pixel = make_float2(gx * step.x, gy * step.y);
            candidates.clear();
            for (int ty = max(0, int(floorf((pixel.y - 2 * step.y) / GSPLAT_RASTER_TILE_SIZE)));
                 ty <=
                 min(tiles_y - 1, int(floorf((pixel.y + 2 * step.y) / GSPLAT_RASTER_TILE_SIZE)));
                 ++ty)
            {
              for (int tx = max(0, int(floorf((pixel.x - 2 * step.x) / GSPLAT_RASTER_TILE_SIZE)));
                   tx <=
                   min(tiles_x - 1, int(floorf((pixel.x + 2 * step.x) / GSPLAT_RASTER_TILE_SIZE)));
                   ++tx)
              {
                for (uint i : center_bins[size_t(ty) * tiles_x + tx]) {
                  const float4 c = projected[i].splat.center;
                  if (sqr((c.x - pixel.x) / step.x) + sqr((c.y - pixel.y) / step.y) <= 4) {
                    candidates.push_back(i);
                  }
                }
              }
            }
            layers.clear();
            for (uint i : candidates) {
              const KernelGSplat &s = projected[i].splat;
              const float2 local = make_float2((s.center.x - pixel.x) / step.x,
                                               (s.center.y - pixel.y) / step.y);
              const float radius2 = len_squared(local);
              const float weight = s.center.w * expf(-0.5f * radius2) *
                                   (1 - smoothstep(2.25f, 4.0f, radius2));
              if (weight > 1e-5f) {
                layers.push_back(make_float2(1 / s.center.z, weight));
              }
            }
            std::sort(layers.begin(), layers.end(), [](const float2 a, const float2 b) {
              return a.x < b.x;
            });
            const float anchor = GSplatShadowReceiverFit::front_layer(layers.data(),
                                                                      layers.size());
            if (!(anchor > 0)) {
              continue;
            }
            points.clear();
            for (uint i : candidates) {
              const KernelGSplat &s = projected[i].splat;
              const float2 offset = make_float2(s.center.x, s.center.y) - pixel;
              const float layer = 1 - smoothstep(
                                          0.025f, 0.075f, fabsf(1 / s.center.z - anchor) / anchor);
              const float2 local = make_float2(offset.x / step.x, offset.y / step.y);
              const float radius2 = len_squared(local);
              const float weight = s.center.w * expf(-0.5f * radius2) * layer *
                                   (1 - smoothstep(2.25f, 4.0f, radius2));
              if (weight > 1e-5f) {
                points.push_back(make_float4(local.x, local.y, 1 / s.center.z - anchor, weight));
              }
            }
            weights.resize(points.size());
            for (size_t i = 0; i < points.size(); ++i) {
              weights[i] = points[i].w;
            }
            float4 plane = zero_float4();
            for (int iteration = 0; iteration < 3; ++iteration) {
              GSplatShadowReceiverFit fit;
              for (size_t i = 0; i < points.size(); ++i) {
                fit.add(make_float2(points[i].x, points[i].y), points[i].z, weights[i]);
              }
              plane = fit.solve(anchor);
              if (!(plane.x > 0)) {
                break;
              }
              depths.clear();
              float weight_sum = 0;
              for (const float4 &p : points) {
                const float residual = fabsf(anchor + p.z - plane.x - plane.y * p.x -
                                             plane.z * p.y);
                depths.emplace_back(residual, p.w);
                weight_sum += p.w;
              }
              std::sort(depths.begin(), depths.end());
              float sum = 0, mad = 0;
              for (const auto &d : depths) {
                sum += d.second;
                mad = d.first;
                if (sum >= 0.5f * weight_sum) {
                  break;
                }
              }
              const float scale = fmaxf(3 * 1.4826f * mad, anchor * 0.001f);
              for (size_t i = 0; i < points.size(); ++i) {
                const float4 &p = points[i];
                const float residual = anchor + p.z - plane.x - plane.y * p.x - plane.z * p.y;
                weights[i] = p.w / (1 + sqr(residual / scale));
              }
            }
            if (plane.x > 0 && fabsf(plane.x - anchor) < 0.075f * anchor) {
              receivers[index] = make_float4(plane.x, plane.y / step.x, plane.z / step.y, plane.w);
            }
          }
        });
    /* Regularize the receiver geometry, not the shadow image. Overlapping
     * affine patches suppress depth jitter without blurring a mesh silhouette.
     * Large depth steps are excluded, and unsupported patches are left alone. */
    for (int pass = 0; pass < 2; ++pass) {
      const vector<float4> source = receivers;
      parallel_for(
          blocked_range<size_t>(0, receivers.size(), 64), [&](const blocked_range<size_t> &range) {
            for (size_t index = range.begin(); index < range.end(); ++index) {
              const float4 center = source[index];
              if (!(center.x > 0)) {
                continue;
              }
              const int gx = int(index % (grid.x + 1)), gy = int(index / (grid.x + 1));
              GSplatShadowReceiverFit fit;
              for (int y = max(0, gy - 3); y <= min(grid.y, gy + 3); ++y) {
                for (int x = max(0, gx - 3); x <= min(grid.x, gx + 3); ++x) {
                  const float4 neighbor = source[size_t(y) * (grid.x + 1) + x];
                  if (!(neighbor.x > 0 && neighbor.w > 0)) {
                    continue;
                  }
                  const float2 offset = make_float2(x - gx, y - gy);
                  const float layer = 1 - smoothstep(0.1f, 0.2f, fabsf(neighbor.x / center.x - 1));
                  const float weight = neighbor.w * layer * expf(-len_squared(offset) / 4.5f);
                  fit.add(offset, neighbor.x - center.x, weight);
                }
              }
              const float4 plane = fit.solve(center.x);
              if (plane.w > 0.5f && fabsf(plane.x / center.x - 1) < 0.05f) {
                receivers[index] = make_float4(
                    plane.x, plane.y / step.x, plane.z / step.y, plane.w);
              }
            }
          });
    }
    /* Repair isolated back-layer islands only when a consistent front plane
     * surrounds the query in all four quadrants. This keeps a sparse tabletop
     * patch from falling through to the floor without extrapolating its plane
     * across a silhouette or a depth edge. */
    {
      const vector<float4> source = receivers;
      parallel_for(blocked_range<size_t>(0, receivers.size(), 64),
                   [&](const blocked_range<size_t> &range) {
        vector<float2> layers;
        for (size_t index = range.begin(); index < range.end(); ++index) {
          const int gx = int(index % (grid.x + 1)), gy = int(index / (grid.x + 1));
          layers.clear();
          for (int y = max(0, gy - 3); y <= min(grid.y, gy + 3); ++y) {
            for (int x = max(0, gx - 3); x <= min(grid.x, gx + 3); ++x) {
              const float4 n = source[size_t(y) * (grid.x + 1) + x];
              const float2 offset = make_float2(x - gx, y - gy);
              const float z = n.x - n.y * offset.x * step.x - n.z * offset.y * step.y;
              if (z > 0 && n.w > 0.5f) {
                layers.push_back(make_float2(z, n.w * expf(-len_squared(offset) / 4.5f)));
              }
            }
          }
          std::sort(layers.begin(), layers.end(), [](const float2 a, const float2 b) {
            return a.x < b.x;
          });
          const float anchor = GSplatShadowReceiverFit::front_layer(layers.data(), layers.size());
          const float4 center = source[index];
          if (!(anchor > 0 && (center.x <= 0 || anchor > center.x * 1.075f))) {
            continue;
          }
          GSplatShadowReceiverFit fit;
          float quadrant[4] = {};
          float total = 0;
          for (int y = max(0, gy - 3); y <= min(grid.y, gy + 3); ++y) {
            for (int x = max(0, gx - 3); x <= min(grid.x, gx + 3); ++x) {
              const float4 n = source[size_t(y) * (grid.x + 1) + x];
              const float2 offset = make_float2(x - gx, y - gy);
              const float z = n.x - n.y * offset.x * step.x - n.z * offset.y * step.y;
              if (!(n.w > 0.5f && fabsf(z / anchor - 1) < 0.025f)) {
                continue;
              }
              const float w = n.w * expf(-len_squared(offset) / 4.5f);
              fit.add(offset, n.x - anchor, w);
              total += w;
              if (x != gx && y != gy) {
                quadrant[(x > gx ? 1 : 0) + (y > gy ? 2 : 0)] += w;
              }
            }
          }
          const float4 plane = fit.solve(anchor);
          if (plane.w > 0.75f && quadrant[0] > 0.04f * total &&
              quadrant[1] > 0.04f * total && quadrant[2] > 0.04f * total &&
              quadrant[3] > 0.04f * total)
          {
            receivers[index] = make_float4(plane.x, plane.y / step.x, plane.z / step.y, plane.w);
          }
        }
      });
    }
    /* The image guide selects visible layers, but no longer defines their
     * geometry by itself. Anchor supported patches to fixed world-space
     * neighborhoods, then analytically reproject those planes. Curves, edges,
     * sparse areas, and incompatible layers keep the conservative fallback. */
    const GSplatWorldReceiver world_receiver(std::move(receiver_samples));
    vector<float4> anchors(receivers.size(), zero_float4());
    parallel_for(blocked_range<size_t>(0, receivers.size(), 32),
                 [&](const blocked_range<size_t> &range) {
      for (size_t index = range.begin(); index < range.end(); ++index) {
        const float4 old = receivers[index];
        if (!(old.x > 0 && old.w > 0)) {
          continue;
        }
        const float2 pixel = make_float2((index % (grid.x + 1)) * step.x,
                                         (index / (grid.x + 1)) * step.y);
        const float3 p = transform_perspective(&cam.rastertocamera,
                                               make_float3(pixel.x, pixel.y, 0));
        const float3 q = p / p.z;
        const float3 px = transform_perspective(&cam.rastertocamera,
                                                make_float3(pixel.x + 1, pixel.y, 0));
        const float3 py = transform_perspective(&cam.rastertocamera,
                                                make_float3(pixel.x, pixel.y + 1, 0));
        const float3 dx = px / px.z - q, dy = py / py.z - q;
        const float3 seed = transform_point(&cam.cameratoworld, q / old.x);
        float confidence;
        const float4 plane = world_receiver.plane(seed, confidence);
        if (!(confidence > 0.5f)) {
          continue;
        }
        const float3 normal = transform_direction_transposed(&cam.cameratoworld,
                                                             make_float3(plane));
        const float denominator = plane.w - dot(make_float3(plane), eye);
        if (fabsf(denominator) < 1e-8f) {
          continue;
        }
        const float inverse = dot(normal, q) / denominator;
        if (inverse > 0) {
          const float agreement = fabsf(dot(
              normal, gsplat_receiver_normal(q, dx, dy, make_float3(old))));
          const float weight = smoothstep(0.5f, 0.95f, confidence) *
                               smoothstep(0.85f, 0.98f, agreement) *
                               (1 - smoothstep(0.01f, 0.05f, fabsf(inverse / old.x - 1)));
          anchors[index] = make_float4(inverse,
                                       dot(normal, dx) / denominator,
                                       dot(normal, dy) / denominator,
                                       weight);
        }
      }
    });
    /* An isolated accepted anchor must never replace a whole guide cell.
     * Spread compatible plane constraints with a compact, smoothly vanishing
     * support and retain the original guide as a regularizer. Blend geometry,
     * not rendered shadow colors, without crossing depth discontinuities. */
    const vector<float4> original = receivers;
    parallel_for(blocked_range<size_t>(0, receivers.size(), 32),
                 [&](const blocked_range<size_t> &range) {
      for (size_t index = range.begin(); index < range.end(); ++index) {
        const float4 old = original[index];
        if (!(old.x > 0 && old.w > 0)) {
          continue;
        }
        const int gx = int(index % (grid.x + 1)), gy = int(index / (grid.x + 1));
        float3 sum = make_float3(old);
        float total = 1;
        for (int y = max(0, gy - 4); y <= min(grid.y, gy + 4); ++y) {
          for (int x = max(0, gx - 4); x <= min(grid.x, gx + 4); ++x) {
            const size_t neighbor = size_t(y) * (grid.x + 1) + x;
            const float4 anchor = anchors[neighbor], source = original[neighbor];
            const float2 offset = make_float2((gx - x) * step.x, (gy - y) * step.y);
            const float distance2 = float(sqr(gx - x) + sqr(gy - y));
            const float inverse = anchor.x + anchor.y * offset.x + anchor.z * offset.y;
            const float source_inverse = source.x + source.y * offset.x + source.z * offset.y;
            if (!(anchor.w > 0 && inverse > 0 && source_inverse > 0)) {
              continue;
            }
            const float layer = 1 - smoothstep(0.01f, 0.04f, fabsf(source_inverse / old.x - 1));
            const float weight = anchor.w * layer * expf(-distance2 / 8) *
                                 (1 - smoothstep(9.0f, 16.0f, distance2));
            sum += weight * make_float3(inverse, anchor.y, anchor.z);
            total += weight;
          }
        }
        receivers[index] = make_float4(sum / total, old.w);
      }
    });
    raster.receiver_offset = int(projected.size());
    LOG_DEBUG << "GSplat receiver guide: " << receivers.size() << " records, "
              << 1000 * (time_dt() - receiver_start) << " ms";
  }
  auto *splats = dscene.gsplat_raster_splats.alloc(projected.size() + receivers.size());
  auto *indices = dscene.gsplat_raster_indices.alloc(count);
  for (size_t i = 0; i < projected.size(); ++i) {
    const Projected &item = projected[i];
    splats[i] = item.splat;
    for (int y = item.tiles.y; y <= item.tiles.w; ++y) {
      for (int x = item.tiles.x; x <= item.tiles.z; ++x) {
        indices[tiles[size_t(y) * tiles_x + x].y++] = uint(i);
      }
    }
  }
  for (size_t i = 0; i < receivers.size(); ++i) {
    splats[projected.size() + i] = {};
    splats[projected.size() + i].center = receivers[i];
  }
  auto *device_tiles = dscene.gsplat_raster_tiles.alloc(tiles.size());
  std::copy(tiles.begin(), tiles.end(), device_tiles);
  const double bin_time = time_dt();
  dscene.gsplat_raster_splats.copy_to_device();
  dscene.gsplat_raster_tiles.copy_to_device();
  dscene.gsplat_raster_indices.copy_to_device();
  raster.enabled = true;
  raster.tiles_x = tiles_x;
  raster.tiles_y = tiles_y;
  LOG_DEBUG << "GSplat raster: " << projected.size() << " splats, " << count << " tile entries; "
            << "project " << 1000 * (project_time - start_time) << " ms, sort "
            << 1000 * (sort_time - project_time) << " ms, bin " << 1000 * (bin_time - sort_time)
            << " ms, upload " << 1000 * (time_dt() - bin_time) << " ms";
}

CCL_NAMESPACE_END
