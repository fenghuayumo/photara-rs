/* SPDX-FileCopyrightText: 2026 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include <gtest/gtest.h>
#include <vector>

#include "kernel/film/read.h"
#include "scene/gsplat_receiver.h"
#include "util/math_gsplat.h"

CCL_NAMESPACE_BEGIN

TEST(GSplatRaster, WorldReceiverFixedSurface)
{
  std::vector<float4> points;
  const float3 offset = make_float3(100, -40, 7);
  const float3 expected = safe_normalize(make_float3(-0.3f, 0.2f, 1));
  for (int y = -12; y <= 12; ++y) {
    for (int x = -12; x <= 12; ++x) {
      const float u = x * 0.03f, v = y * 0.03f;
      points.push_back(make_float4(offset + make_float3(u, v, 0.3f * u - 0.2f * v), 1));
    }
  }
  const GSplatWorldReceiver receiver(points);
  /* Queries from different camera footprints must recover the same world
   * plane, including far from the origin and across spatial cell boundaries. */
  for (const float3 q : {make_float3(-0.1f, 0.05f, -0.04f),
                          zero_float3(), make_float3(0.12f, -0.09f, 0.054f)})
  {
    float confidence;
    const float4 plane = receiver.plane(offset + q, confidence);
    EXPECT_GT(confidence, 0.99f);
    EXPECT_GT(fabsf(dot(make_float3(plane), expected)), 0.99999f);
    EXPECT_NEAR(dot(make_float3(plane), offset + q), plane.w, 2e-5f);
  }
  float confidence;
  receiver.plane(offset + make_float3(0, 0, 3), confidence);
  EXPECT_FLOAT_EQ(confidence, 0);
}

TEST(GSplatRaster, WorldReceiverRejectsUnsupportedGeometry)
{
  float confidence;
  GSplatWorldReceiver({make_float4(0, 0, 0, 1)}).plane(zero_float3(), confidence);
  EXPECT_FLOAT_EQ(confidence, 0);
  std::vector<float4> line, layers;
  for (int i = -40; i <= 40; ++i) {
    line.push_back(make_float4(i * 0.01f, 0, 0, 1));
  }
  GSplatWorldReceiver(line).plane(zero_float3(), confidence);
  EXPECT_FLOAT_EQ(confidence, 0);
  for (int y = -8; y <= 8; ++y) {
    for (int x = -8; x <= 8; ++x) {
      layers.push_back(make_float4(x * 0.03f, y * 0.03f, -0.02f, 1));
      layers.push_back(make_float4(x * 0.03f, y * 0.03f, 0.02f, 1));
    }
  }
  GSplatWorldReceiver(layers).plane(zero_float3(), confidence);
  /* Mixed layers must stay below the start of the host's confidence ramp. */
  EXPECT_LT(confidence, 0.5f);
}

TEST(GSplatRaster, WorldReceiverNoisySurface)
{
  std::vector<float4> points;
  for (int y = -16; y <= 16; ++y) {
    for (int x = -16; x <= 16; ++x) {
      const float noise = 0.001f * sinf(x * 2.1f + y * 0.7f);
      points.push_back(make_float4(x * 0.02f, y * 0.02f, noise, 1));
    }
  }
  const GSplatWorldReceiver receiver(points);
  for (float x : {-0.12f, 0.0f, 0.12f}) {
    float confidence;
    const float4 plane = receiver.plane(make_float3(x, 0, 0.001f), confidence);
    EXPECT_GT(confidence, 0.9f);
    EXPECT_GT(fabsf(plane.z), 0.999f);
    EXPECT_LT(fabsf(plane.w), 0.0005f);
  }
}

TEST(GSplatRaster, MeshOcclusionUsesLocalEllipseDepth)
{
  const float4 center = make_float4(50, 50, 2, 0.8f);
  const float4 conic = make_float4(1, 0, 1, 0.04f);
  const float4 color = make_float4(1, 0, 0, -0.01f);
  EXPECT_NEAR(gsplat_projected_depth(center, conic, color, make_float3(60, 55, 0)),
              2.35f, 1e-6f);
  /* The center is in front of a mesh at z=2.1, but its right-hand ellipse
   * fragment is behind it. The opposite fragment remains visible. */
  EXPECT_GT(gsplat_projected_depth(center, conic, color, make_float3(60, 50, 0)), 2.1f);
  EXPECT_LT(gsplat_projected_depth(center, conic, color, make_float3(40, 50, 0)), 2.1f);
}

TEST(GSplatRaster, ReceiverPlaneFit)
{
  GSplatShadowReceiverFit fit;
  for (int y = -3; y <= 3; ++y) {
    for (int x = -3; x <= 3; ++x) {
      fit.add(make_float2(x, y), 0.03f + 0.1f * x - 0.05f * y, 1);
    }
  }
  const float4 plane = fit.solve(2);
  EXPECT_NEAR(plane.x, 2.03f, 1e-5f);
  EXPECT_NEAR(plane.y, 0.1f, 1e-5f);
  EXPECT_NEAR(plane.z, -0.05f, 1e-5f);
  EXPECT_GT(plane.w, 0.99f);
}

TEST(GSplatRaster, ReceiverFrontLayerRejectsBrightFloater)
{
  std::vector<float2> depths;
  for (int i = 0; i < 100; ++i) {
    depths.push_back(make_float2(1, 1));
  }
  for (int i = 0; i < 20; ++i) {
    depths.push_back(make_float2(2, 1));
  }
  depths.push_back(make_float2(4, 100));
  EXPECT_NEAR(GSplatShadowReceiverFit::front_layer(depths.data(), depths.size()), 2, 1e-6f);
  EXPECT_FLOAT_EQ(GSplatShadowReceiverFit::front_layer(depths.data(), 7), 0);
}

TEST(GSplatRaster, ReceiverRejectsSparseAndMultipleLayers)
{
  GSplatShadowReceiverFit sparse, line, layers;
  sparse.add(make_float2(0, 0), 0, 1);
  EXPECT_FLOAT_EQ(sparse.solve(2).w, 0);
  for (int x = -10; x <= 10; ++x) {
    line.add(make_float2(x, 0), 0.1f * x, 1);
  }
  EXPECT_FLOAT_EQ(line.solve(2).w, 0);
  for (int y = -3; y <= 3; ++y) {
    for (int x = -3; x <= 3; ++x) {
      layers.add(make_float2(x, y), -0.2f, 1);
      layers.add(make_float2(x, y), 0.2f, 1);
    }
  }
  EXPECT_FLOAT_EQ(layers.solve(2).w, 0);
}

TEST(GSplatRaster, ReceiverResolutionIndependentGrid)
{
  const int2 low = gsplat_shadow_receiver_grid(960, 640);
  const int2 high = gsplat_shadow_receiver_grid(1920, 1280);
  EXPECT_EQ(low.x, high.x);
  EXPECT_EQ(low.y, high.y);
  EXPECT_EQ(gsplat_shadow_receiver_grid(65, 65).x, 5);
}

TEST(GSplatRaster, ReceiverNormalFromPerspectivePlane)
{
  /* z = 2 + 0.3*x - 0.2*y, with an off-axis, asymmetric projection.
   * The recovered normal must be independent of pixel position and scale. */
  const float3 expected = safe_normalize(make_float3(0.3f, -0.2f, -1));
  for (float depth_scale : {0.01f, 1.0f, 100.0f}) {
    for (float x : {-0.8f, 0.0f, 1.2f}) {
      const float3 q = make_float3(x, 0.4f, 1);
      const float3 inverse = make_float3(
          (1 - 0.3f * q.x + 0.2f * q.y) / 2, -0.3f / (2 * 700), 0.2f / (2 * 450));
      const float3 normal = gsplat_receiver_normal(
          q, make_float3(1.0f / 700, 0, 0), make_float3(0, 1.0f / 450, 0), inverse * depth_scale);
      EXPECT_NEAR(normal.x, expected.x, 1e-6f);
      EXPECT_NEAR(normal.y, expected.y, 1e-6f);
      EXPECT_NEAR(normal.z, expected.z, 1e-6f);
    }
  }
}

TEST(GSplatRaster, LinearShadowStrength)
{
  const float3 source = make_float3(0.2f, 0.6f, 0.9f);
  const float3 linear = color_srgb_to_linear_v3(source);
  for (float strength : {0.0f, 0.25f, 0.5f, 0.8f, 1.0f}) {
    const float3 result = color_srgb_to_linear_v3(
        gsplat_shadow_color(source, make_float3(1 - strength)));
    EXPECT_NEAR(result.x, linear.x * (1 - strength), 1e-6f);
    EXPECT_NEAR(result.y, linear.y * (1 - strength), 1e-6f);
    EXPECT_NEAR(result.z, linear.z * (1 - strength), 1e-6f);
  }
  const float3 unchanged = gsplat_shadow_color(source, one_float3());
  EXPECT_FLOAT_EQ(unchanged.x, source.x);
  EXPECT_FLOAT_EQ(unchanged.y, source.y);
  EXPECT_FLOAT_EQ(unchanged.z, source.z);
}

TEST(GSplatRaster, ConditionalShadowDepth)
{
  const float3 center = make_float3(0, 0, 2);
  const float2 focal = make_float2(100, 100);
  const float3x3 axes = make_float3x3(
      make_float3(0.2f, 0, 0), make_float3(0, 0.3f, 0), make_float3(0.1f, 0, 0));
  const float3 covariance = gsplat_project_covariance(center, axes, focal, 0.3f);
  const float2 gradient = gsplat_project_depth_gradient(center, axes, focal, covariance);
  EXPECT_NEAR(gradient.x, 1.0f / 100.3f, 1e-7f);
  EXPECT_FLOAT_EQ(gradient.y, 0);

  const float3x3 collapsed = float3x3_scale(zero_float3());
  const float2 flat = gsplat_project_depth_gradient(
      center, collapsed, focal, gsplat_project_covariance(center, collapsed, focal, 0.3f));
  EXPECT_FLOAT_EQ(flat.x, 0);
  EXPECT_FLOAT_EQ(flat.y, 0);
}

TEST(GSplatRaster, ThinProjectedGaussian)
{
  const float3 covariance = gsplat_project_covariance(make_float3(0, 0, 2),
                                                      float3x3_scale(make_float3(0.02f, 0.04f, 0)),
                                                      make_float2(100, 100),
                                                      0.3f);
  EXPECT_NEAR(covariance.x, 1.3f, 1e-6f);
  EXPECT_NEAR(covariance.y, 0.0f, 1e-6f);
  EXPECT_NEAR(covariance.z, 4.3f, 1e-6f);
  EXPECT_NEAR(gsplat_projected_alpha(make_float3(1 / covariance.x, 0, 1 / covariance.z),
                                     make_float2(sqrtf(covariance.x), 0),
                                     0.7f),
              0.7f * expf(-0.5f),
              1e-6f);
}

TEST(GSplatRaster, SupportAndAlphaCutoff)
{
  EXPECT_FLOAT_EQ(gsplat_projected_alpha(make_float3(1, 0, 1), make_float2(0, 0), 1), 0.99f);
  EXPECT_FLOAT_EQ(gsplat_projected_alpha(make_float3(1, 0, 1), make_float2(3, 0), 1), 0);
  EXPECT_FLOAT_EQ(gsplat_projected_alpha(make_float3(1, 0, 1), make_float2(0, 0), 0.003f), 0);
}

TEST(GSplatRaster, MaterialShadowOpacity)
{
  EXPECT_FLOAT_EQ(gsplat_material_shadow_opacity(0, 0.5f), 1);
  EXPECT_FLOAT_EQ(gsplat_material_shadow_opacity(1, 0.5f), 0.5f);
  EXPECT_FLOAT_EQ(gsplat_material_shadow_opacity(0.5f, 0.5f), 0.75f);
  EXPECT_FLOAT_EQ(gsplat_material_shadow_opacity(1, 0), 0);
  EXPECT_FLOAT_EQ(gsplat_material_shadow_opacity(1, 1), 1);
}

TEST(GSplatRaster, MaterialShadowTransmission)
{
  const float4 opaque = gsplat_material_shadow_transmission(one_float3(), 0, 1.5f, false);
  EXPECT_FLOAT_EQ(opaque.x, 0);
  EXPECT_FLOAT_EQ(opaque.w, 0);

  const float4 clear = gsplat_material_shadow_transmission(one_float3(), 1, 1.5f, false);
  EXPECT_NEAR(clear.x, 0.96f, 1e-6f);
  EXPECT_NEAR(clear.x * clear.x, 0.9216f, 1e-6f);
  EXPECT_FLOAT_EQ(clear.w, 1);

  const float3 tint = make_float3(0.25f, 0.7f, 1);
  const float4 shell = gsplat_material_shadow_transmission(tint, 0.5f, 1, false);
  EXPECT_NEAR(shell.x * shell.x, 0.125f, 1e-6f);
  EXPECT_NEAR(shell.y * shell.y, 0.35f, 1e-6f);
  EXPECT_NEAR(shell.z * shell.z, 0.5f, 1e-6f);

  const float4 thin = gsplat_material_shadow_transmission(tint, 0.5f, 1, true);
  EXPECT_NEAR(thin.x, 0.125f, 1e-6f);
  EXPECT_NEAR(thin.y, 0.35f, 1e-6f);
  EXPECT_NEAR(thin.z, 0.5f, 1e-6f);
}

TEST(GSplatRaster, SecondaryRayPlanarLimit)
{
  for (float thickness : {0.0f, 1e-18f, 0.001f}) {
    float t = 0;
    const float alpha = gsplat_ray_alpha(make_float3(0.1f, 0, 1),
                                         make_float3(0, 0, -1),
                                         zero_float3(),
                                         make_float3(0.1f, 0.2f, thickness),
                                         identity_quaternion(),
                                         0.7f,
                                         t);
    EXPECT_NEAR(t, 1, 1e-6f);
    EXPECT_NEAR(alpha, 0.7f * expf(-0.5f), 1e-6f);
  }
}

static KernelFilmConvert raster_film(const int samples)
{
  KernelFilmConvert film = {};
  film.pass_offset = 0;
  film.pass_gsplat = 4;
  film.pass_sample_count = PASS_UNUSED;
  film.num_components = 4;
  film.scale = film.scale_exposure = film.gsplat_scale = 1.0f / samples;
  film.exposure = 1;
  return film;
}

TEST(GSplatRaster, TrainingColorCompositing)
{
  KernelFilmConvert film = raster_film(4);
  /* Opaque black mesh/world, 50% red layer. Decode AFTER blending in the
   * training color space: linear output must be 0.214, rather than 0.5. */
  const float buffer[] = {0, 0, 0, 0, 2, 0, 0, 2};
  float pixel[4];
  film_get_pass_pixel_combined(&film, buffer, pixel);
  EXPECT_NEAR(pixel[0], 0.21404114f, 1e-6f);
  EXPECT_FLOAT_EQ(pixel[1], 0);
  EXPECT_FLOAT_EQ(pixel[2], 0);
  EXPECT_FLOAT_EQ(pixel[3], 1);
}

TEST(GSplatRaster, TransparentFilmPremultiplication)
{
  KernelFilmConvert film = raster_film(4);
  const float buffer[] = {0, 0, 0, 4, 2, 0, 0, 2};
  float pixel[4];
  film_get_pass_pixel_combined(&film, buffer, pixel);
  EXPECT_NEAR(pixel[0], 0.5f, 1e-6f);
  EXPECT_NEAR(pixel[3], 0.5f, 1e-6f);
}

TEST(GSplatRaster, AdaptiveAndDenoisedLayerNormalization)
{
  KernelFilmConvert film = raster_film(32);
  film.pass_sample_count = 8;
  film.pass_use_filter = true;
  film.scale = film.scale_exposure = 1;
  float buffer[] = {0, 0, 0, 0, 2, 0, 0, 2, 0};
  const uint samples = 4;
  memcpy(buffer + 8, &samples, sizeof(samples));
  float pixel[4];
  film_get_pass_pixel_combined(&film, buffer, pixel);
  EXPECT_NEAR(pixel[0], 0.21404114f, 1e-6f);
  film.is_denoised = true;
  film_get_pass_pixel_combined(&film, buffer, pixel);
  EXPECT_NEAR(pixel[0], 0.21404114f, 1e-6f);
}

CCL_NAMESPACE_END
