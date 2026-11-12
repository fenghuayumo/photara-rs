/* SPDX-License-Identifier: Apache-2.0 */

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <stdexcept>
#include <string>
#include <vector>

#include "photara/photara.h"
#include "session/display_driver.h"

namespace py = pybind11;

namespace {

using FloatArray = py::array_t<float, py::array::c_style | py::array::forcecast>;
using IndexArray = py::array_t<int, py::array::c_style | py::array::forcecast>;

FloatArray require_float_array(const py::handle &object, const char *name)
{
  FloatArray array = FloatArray::ensure(object);
  if (!array) {
    throw std::runtime_error(std::string(name) + " must be a numeric array");
  }
  return array;
}

FloatArray require_vector3(const py::handle &object, const char *name)
{
  FloatArray array = require_float_array(object, name);
  if (array.ndim() != 1 || array.shape(0) != 3) {
    throw std::runtime_error(std::string(name) + " must have shape (3,)");
  }
  return array;
}

FloatArray require_matrix(const py::handle &object, const char *name)
{
  FloatArray array = require_float_array(object, name);
  if (array.ndim() != 2 || array.shape(0) != 4 || array.shape(1) != 4) {
    throw std::runtime_error(std::string(name) + " must have shape (4, 4)");
  }
  return array;
}

PhotaraDevice parse_device(const std::string &device)
{
  std::string upper;
  upper.reserve(device.size());
  for (char c : device) {
    upper.push_back(char(std::toupper(static_cast<unsigned char>(c))));
  }

  if (upper == "AUTO" || upper.empty()) {
    return PHOTARA_DEVICE_AUTO;
  }
  if (upper == "OPTIX") {
    return PHOTARA_DEVICE_OPTIX;
  }
  throw std::runtime_error("Unknown device '" + device + "' (expected AUTO or OPTIX)");
}

class Context {
 public:
  Context(const int width,
          const int height,
          const int samples,
          const int seed,
          const int max_bounces,
          const bool use_denoise,
          const bool use_adaptive_sampling,
          const std::string &device,
          const std::string &resource_dir,
          bool viewport)
  {
    PhotaraOptions options = {};
    options.width = width;
    options.height = height;
    options.samples = samples;
    options.seed = seed;
    options.max_bounces = max_bounces;
    options.use_denoise = use_denoise ? 1 : 0;
    options.use_adaptive_sampling = use_adaptive_sampling ? 1 : 0;
    options.device = parse_device(device);
    options.resource_dir = resource_dir.c_str();

    context_ = viewport ? photara_create_viewport(&options) : photara_create(&options);
    if (context_ == nullptr) {
      throw std::runtime_error(photara_last_error(nullptr));
    }
  }

  ~Context()
  {
    if (context_ != nullptr) {
      photara_destroy(context_);
      context_ = nullptr;
    }
  }

  Context(const Context &) = delete;
  Context &operator=(const Context &) = delete;

  int add_material(const FloatArray &base_color,
                   const float metallic,
                   const float roughness,
                   const float ior,
                   const float transmission,
                   const FloatArray &emission_color,
                   const float emission_strength)
  {
    const PhotaraMaterialDesc desc = make_material_desc(base_color,
                                                        metallic,
                                                        roughness,
                                                        ior,
                                                        transmission,
                                                        emission_color,
                                                        emission_strength);

    const int result = photara_add_material(context_, &desc);
    if (result < 0) {
      throw_last_error();
    }
    return result;
  }

  void update_material(const int material_index,
                       const FloatArray &base_color,
                       const float metallic,
                       const float roughness,
                       const float ior,
                       const float transmission,
                       const FloatArray &emission_color,
                       const float emission_strength)
  {
    const PhotaraMaterialDesc desc = make_material_desc(base_color,
                                                        metallic,
                                                        roughness,
                                                        ior,
                                                        transmission,
                                                        emission_color,
                                                        emission_strength);

    if (photara_update_material(context_, material_index, &desc) != 0) {
      throw_last_error();
    }
  }

  void set_material_base_color_texture(const int material_index,
                                       const std::string &filename,
                                       const std::string &colorspace,
                                       const int interpolation,
                                       const int extension)
  {
    PhotaraTextureDesc desc = {};
    desc.filename = filename.c_str();
    desc.colorspace = colorspace.empty() ? nullptr : colorspace.c_str();
    desc.interpolation = interpolation;
    desc.extension = extension;
    if (photara_set_material_base_color_texture(context_, material_index, &desc) != 0) {
      throw_last_error();
    }
  }

  void clear_material_base_color_texture(const int material_index)
  {
    if (photara_clear_material_base_color_texture(context_, material_index) != 0) {
      throw_last_error();
    }
  }

  void set_background_texture(const std::string &filename,
                              const std::string &colorspace,
                              const int interpolation,
                              const int extension)
  {
    PhotaraTextureDesc desc = {};
    desc.filename = filename.c_str();
    desc.colorspace = colorspace.empty() ? nullptr : colorspace.c_str();
    desc.interpolation = interpolation;
    desc.extension = extension;
    if (photara_set_background_texture(context_, &desc) != 0) {
      throw_last_error();
    }
  }

  void clear_background_texture()
  {
    if (photara_clear_background_texture(context_) != 0) {
      throw_last_error();
    }
  }

  int add_mesh(const FloatArray &positions,
               const IndexArray &indices,
               const py::object &normals,
               const py::object &uvs,
               const py::object &matrix,
               const int material_index,
               const int object_id)
  {
    if (positions.ndim() != 2 || positions.shape(1) != 3) {
      throw std::runtime_error("vertices must have shape (N, 3)");
    }
    if (indices.ndim() != 2 || indices.shape(1) != 3) {
      throw std::runtime_error("indices must have shape (T, 3)");
    }
    if (positions.shape(0) <= 0 || indices.shape(0) <= 0) {
      throw std::runtime_error("Mesh must contain vertices and triangles");
    }

    const int num_vertices = int(positions.shape(0));
    const int num_triangles = int(indices.shape(0));

    const float *normals_ptr = nullptr;
    FloatArray normals_array;
    if (!normals.is_none()) {
      normals_array = require_float_array(normals, "normals");
      if (normals_array.ndim() != 2 || normals_array.shape(0) != num_vertices ||
          normals_array.shape(1) != 3)
      {
        throw std::runtime_error("normals must have shape (N, 3) matching vertices");
      }
      normals_ptr = normals_array.data();
    }

    const float *uvs_ptr = nullptr;
    FloatArray uvs_array;
    if (!uvs.is_none()) {
      uvs_array = require_float_array(uvs, "uvs");
      if (uvs_array.ndim() != 2 || uvs_array.shape(0) != num_vertices ||
          uvs_array.shape(1) != 2)
      {
        throw std::runtime_error("uvs must have shape (N, 2) matching vertices");
      }
      uvs_ptr = uvs_array.data();
    }

    const float *matrix_ptr = nullptr;
    FloatArray matrix_array;
    if (!matrix.is_none()) {
      matrix_array = require_matrix(matrix, "matrix");
      matrix_ptr = matrix_array.data();
    }

    PhotaraMeshDesc desc = {};
    desc.positions = positions.data();
    desc.normals = normals_ptr;
    desc.uvs = uvs_ptr;
    desc.indices = indices.data();
    desc.num_vertices = num_vertices;
    desc.num_triangles = num_triangles;
    desc.matrix = matrix_ptr;
    desc.material_index = material_index;
    desc.object_id = object_id;

    const int result = photara_add_mesh(context_, &desc);
    if (result < 0) {
      throw_last_error();
    }
    return result;
  }

  int add_instance(const int source_object, const py::object &matrix, const int object_id)
  {
    const float *matrix_ptr = nullptr;
    FloatArray matrix_array;
    if (!matrix.is_none()) {
      matrix_array = require_matrix(matrix, "matrix");
      matrix_ptr = matrix_array.data();
    }

    const int result = photara_add_instance(context_, source_object, matrix_ptr, object_id);
    if (result < 0) {
      throw_last_error();
    }
    return result;
  }

  void update_mesh(const int object,
                   const FloatArray &positions,
                   const py::object &normals,
                   const py::object &uvs)
  {
    if (positions.ndim() != 2 || positions.shape(1) != 3 || positions.shape(0) <= 0) {
      throw std::runtime_error("positions must have shape (N, 3) with N > 0");
    }
    const int num_vertices = int(positions.shape(0));

    const float *normals_ptr = nullptr;
    FloatArray normals_array;
    if (!normals.is_none()) {
      normals_array = require_float_array(normals, "normals");
      if (normals_array.ndim() != 2 || normals_array.shape(0) != num_vertices ||
          normals_array.shape(1) != 3)
      {
        throw std::runtime_error("normals must have shape (N, 3) matching vertices");
      }
      normals_ptr = normals_array.data();
    }

    const float *uvs_ptr = nullptr;
    FloatArray uvs_array;
    if (!uvs.is_none()) {
      uvs_array = require_float_array(uvs, "uvs");
      if (uvs_array.ndim() != 2 || uvs_array.shape(0) != num_vertices ||
          uvs_array.shape(1) != 2)
      {
        throw std::runtime_error("uvs must have shape (N, 2) matching vertices");
      }
      uvs_ptr = uvs_array.data();
    }

    PhotaraMeshUpdateDesc desc = {};
    desc.positions = positions.data();
    desc.normals = normals_ptr;
    desc.uvs = uvs_ptr;
    desc.num_vertices = num_vertices;

    if (photara_update_mesh(context_, object, &desc) != 0) {
      throw_last_error();
    }
  }

  void replace_mesh(const int object,
                    const FloatArray &positions,
                    const IndexArray &indices,
                    const py::object &normals,
                    const py::object &uvs,
                    const int material_index)
  {
    if (positions.ndim() != 2 || positions.shape(1) != 3 || positions.shape(0) <= 0) {
      throw std::runtime_error("positions must have shape (N, 3) with N > 0");
    }
    if (indices.ndim() != 2 || indices.shape(1) != 3 || indices.shape(0) <= 0) {
      throw std::runtime_error("indices must have shape (T, 3) with T > 0");
    }
    const int num_vertices = int(positions.shape(0));

    const float *normals_ptr = nullptr;
    FloatArray normals_array;
    if (!normals.is_none()) {
      normals_array = require_float_array(normals, "normals");
      if (normals_array.ndim() != 2 || normals_array.shape(0) != num_vertices ||
          normals_array.shape(1) != 3)
      {
        throw std::runtime_error("normals must have shape (N, 3) matching vertices");
      }
      normals_ptr = normals_array.data();
    }

    const float *uvs_ptr = nullptr;
    FloatArray uvs_array;
    if (!uvs.is_none()) {
      uvs_array = require_float_array(uvs, "uvs");
      if (uvs_array.ndim() != 2 || uvs_array.shape(0) != num_vertices ||
          uvs_array.shape(1) != 2)
      {
        throw std::runtime_error("uvs must have shape (N, 2) matching vertices");
      }
      uvs_ptr = uvs_array.data();
    }

    PhotaraMeshReplaceDesc desc = {};
    desc.positions = positions.data();
    desc.normals = normals_ptr;
    desc.uvs = uvs_ptr;
    desc.indices = indices.data();
    desc.num_vertices = num_vertices;
    desc.num_triangles = int(indices.shape(0));
    desc.material_index = material_index;

    if (photara_replace_mesh(context_, object, &desc) != 0) {
      throw_last_error();
    }
  }

  void set_object_transform(const int object, const FloatArray &matrix)
  {
    const FloatArray matrix_array = require_matrix(matrix, "matrix");
    if (photara_set_object_transform(context_, object, matrix_array.data()) != 0) {
      throw_last_error();
    }
  }

  void set_object_visible(const int object, const bool visible)
  {
    if (photara_set_object_visible(context_, object, visible ? 1 : 0) != 0) {
      throw_last_error();
    }
  }

  void set_object_id(const int object, const int object_id)
  {
    if (photara_set_object_id(context_, object, object_id) != 0) {
      throw_last_error();
    }
  }

  void set_object_material(const int object, const int material_index)
  {
    if (photara_set_object_material(context_, object, material_index) != 0) {
      throw_last_error();
    }
  }

  void remove_object(const int object)
  {
    if (photara_remove_object(context_, object) != 0) {
      throw_last_error();
    }
  }

  void clear_objects()
  {
    if (photara_clear_objects(context_) != 0) {
      throw_last_error();
    }
  }

  void set_camera(const FloatArray &position,
                  const FloatArray &target,
                  const py::object &up,
                  const float fov_degrees,
                  const float near_clip,
                  const float far_clip)
  {
    check_vector3(position, "position");
    check_vector3(target, "target");
    PhotaraCameraDesc desc = {};
    copy_vector3(position, desc.position);
    copy_vector3(target, desc.target);
    if (!up.is_none()) {
      const FloatArray up_array = require_vector3(up, "up");
      copy_vector3(up_array, desc.up);
    }
    desc.fov_degrees = fov_degrees;
    desc.near_clip = near_clip;
    desc.far_clip = far_clip;

    if (photara_set_camera(context_, &desc) != 0) {
      throw_last_error();
    }
  }

  void set_camera_matrix(const FloatArray &matrix,
                         const float fov_degrees,
                         const float near_clip,
                         const float far_clip)
  {
    const FloatArray matrix_array = require_matrix(matrix, "matrix");
    if (photara_set_camera_matrix(
            context_, matrix_array.data(), fov_degrees, near_clip, far_clip) != 0)
    {
      throw_last_error();
    }
  }

  void queue_camera_matrix(const unsigned long long job,
                           const FloatArray &matrix,
                           const float fov_degrees,
                           const float near_clip,
                           const float far_clip)
  {
    const FloatArray matrix_array = require_matrix(matrix, "matrix");
    if (photara_queue_camera_matrix(
            context_, job, matrix_array.data(), fov_degrees, near_clip, far_clip) != 0)
    {
      throw_last_error();
    }
  }

  int add_area_light(const FloatArray &position,
                     const FloatArray &target,
                     const py::object &up,
                     const float size_u,
                     const float size_v,
                     const FloatArray &color,
                     const float strength)
  {
    const PhotaraAreaLightDesc desc = make_area_light_desc(
        position, target, up, size_u, size_v, color, strength);
    const int result = photara_add_area_light(context_, &desc);
    if (result < 0) {
      throw_last_error();
    }
    return result;
  }

  void update_area_light(const int light,
                         const FloatArray &position,
                         const FloatArray &target,
                         const py::object &up,
                         const float size_u,
                         const float size_v,
                         const FloatArray &color,
                         const float strength)
  {
    const PhotaraAreaLightDesc desc = make_area_light_desc(
        position, target, up, size_u, size_v, color, strength);
    if (photara_update_area_light(context_, light, &desc) != 0) {
      throw_last_error();
    }
  }

  int add_sun_light(const FloatArray &direction,
                    const FloatArray &color,
                    const float strength,
                    const float angle_degrees)
  {
    const PhotaraSunLightDesc desc = make_sun_light_desc(
        direction, color, strength, angle_degrees);
    const int result = photara_add_sun_light(context_, &desc);
    if (result < 0) {
      throw_last_error();
    }
    return result;
  }

  void update_sun_light(const int light,
                        const FloatArray &direction,
                        const FloatArray &color,
                        const float strength,
                        const float angle_degrees)
  {
    const PhotaraSunLightDesc desc = make_sun_light_desc(
        direction, color, strength, angle_degrees);
    if (photara_update_sun_light(context_, light, &desc) != 0) {
      throw_last_error();
    }
  }

  int add_point_light(const FloatArray &position,
                      const FloatArray &color,
                      const float strength,
                      const float radius)
  {
    const PhotaraPointLightDesc desc = make_point_light_desc(
        position, color, strength, radius);
    const int result = photara_add_point_light(context_, &desc);
    if (result < 0) {
      throw_last_error();
    }
    return result;
  }

  void update_point_light(const int light,
                          const FloatArray &position,
                          const FloatArray &color,
                          const float strength,
                          const float radius)
  {
    const PhotaraPointLightDesc desc = make_point_light_desc(
        position, color, strength, radius);
    if (photara_update_point_light(context_, light, &desc) != 0) {
      throw_last_error();
    }
  }

  int add_spot_light(const FloatArray &position,
                     const FloatArray &direction,
                     const py::object &up,
                     const FloatArray &color,
                     const float strength,
                     const float radius,
                     const float angle_degrees,
                     const float smooth)
  {
    const PhotaraSpotLightDesc desc = make_spot_light_desc(
        position, direction, up, color, strength, radius, angle_degrees, smooth);
    const int result = photara_add_spot_light(context_, &desc);
    if (result < 0) {
      throw_last_error();
    }
    return result;
  }

  void update_spot_light(const int light,
                         const FloatArray &position,
                         const FloatArray &direction,
                         const py::object &up,
                         const FloatArray &color,
                         const float strength,
                         const float radius,
                         const float angle_degrees,
                         const float smooth)
  {
    const PhotaraSpotLightDesc desc = make_spot_light_desc(
        position, direction, up, color, strength, radius, angle_degrees, smooth);
    if (photara_update_spot_light(context_, light, &desc) != 0) {
      throw_last_error();
    }
  }

  void set_light_transform(const int light, const FloatArray &matrix)
  {
    const FloatArray matrix_array = require_matrix(matrix, "matrix");
    if (photara_set_light_transform(context_, light, matrix_array.data()) != 0) {
      throw_last_error();
    }
  }

  void set_light_visible(const int light, const bool visible)
  {
    if (photara_set_light_visible(context_, light, visible ? 1 : 0) != 0) {
      throw_last_error();
    }
  }

  void remove_light(const int light)
  {
    if (photara_remove_light(context_, light) != 0) {
      throw_last_error();
    }
  }

  void clear_lights()
  {
    if (photara_clear_lights(context_) != 0) {
      throw_last_error();
    }
  }

  void clear_scene()
  {
    if (photara_clear_scene(context_) != 0) {
      throw_last_error();
    }
  }

  void set_background(const FloatArray &color, const float strength)
  {
    check_vector3(color, "color");
    const float rgb[3] = {float(color.data()[0]),
                          float(color.data()[1]),
                          float(color.data()[2])};
    if (photara_set_background(context_, rgb, strength) != 0) {
      throw_last_error();
    }
  }

  void set_seed(const int seed)
  {
    if (photara_set_seed(context_, seed) != 0) {
      throw_last_error();
    }
  }

  void set_samples(const int samples)
  {
    if (photara_set_samples(context_, samples) != 0) {
      throw_last_error();
    }
  }

  py::dict neeat_stats()
  {
    int lights = 0;
    float max_pdf = 0;
    if (photara_get_neeat_stats(context_, &lights, &max_pdf) != 0) { throw_last_error(); }
    py::dict result;
    result["active_lights"] = lights;
    result["max_pdf"] = max_pdf;
    int tiles = 0;
    float local_max_pdf = 0;
    if (photara_get_neeat_local_stats(context_, &tiles, &local_max_pdf) != 0) {
      throw_last_error();
    }
    result["local_tiles"] = tiles;
    result["local_max_pdf"] = local_max_pdf;
    return result;
  }

  void configure_texture_lod(const int method)
  {
    if (photara_set_texture_lod(context_, static_cast<PhotaraTextureLOD>(method)) != 0) {
      throw_last_error();
    }
  }

  int texture_lod()
  {
    PhotaraTextureLOD method;
    if (photara_get_texture_lod(context_, &method) != 0) { throw_last_error(); }
    return int(method);
  }

  void configure_neeat(const bool enabled)
  {
    if (photara_set_neeat(context_, enabled) != 0) { throw_last_error(); }
  }

  void set_render_mode(const int mode, const int samples_per_frame)
  {
    if (photara_set_render_mode(context_, static_cast<PhotaraRenderMode>(mode),
                               samples_per_frame) != 0) {
      throw_last_error();
    }
  }

  unsigned long long render_begin()
  {
    unsigned long long job = 0;
    if (photara_render_begin(context_, &job) != 0) {
      throw_last_error();
    }
    return job;
  }

  py::dict render_poll(const unsigned long long job)
  {
    PhotaraRenderProgress progress = {};
    if (photara_render_poll(context_, job, &progress) != 0) {
      throw_last_error();
    }
    py::dict result;
    result["state"] = progress.state;
    result["progress"] = progress.progress;
    result["current_sample"] = progress.current_sample;
    result["num_samples"] = progress.num_samples;
    result["status"] = progress.status ? progress.status : "";
    result["substatus"] = progress.substatus ? progress.substatus : "";
    return result;
  }

  void render_request_cancel(const unsigned long long job)
  {
    if (photara_render_request_cancel(context_, job) != 0) {
      throw_last_error();
    }
  }

  void render_cancel(const unsigned long long job)
  {
    if (photara_render_cancel(context_, job) != 0) {
      throw_last_error();
    }
  }

  void render_finish(const unsigned long long job)
  {
    if (photara_render_finish(context_, job) != 0) {
      throw_last_error();
    }
  }

  void render()
  {
    if (photara_render(context_) != 0) {
      throw_last_error();
    }
  }

  void attach_display(py::capsule capsule)
  {
    const char *abi = "photara.DisplayDriver.v1.4d9862575ebf7cca";
    if (!PyCapsule_IsValid(capsule.ptr(), abi)) {
      throw std::runtime_error("Blender/Photara display driver ABI mismatch");
    }
    void *pointer = PyCapsule_GetPointer(capsule.ptr(), abi);
    PyCapsule_SetName(capsule.ptr(), "photara.DisplayDriver.consumed");
    if (photara_attach_display(context_, pointer) != 0) { throw_last_error(); }
  }

  void draw_display()
  {
    if (photara_draw_display(context_) != 0) { throw_last_error(); }
  }

  py::object preview(unsigned long long revision)
  {
    int w = width(), h = height();
    py::array_t<float> rgba(std::vector<py::ssize_t>{h, w, 4});
    py::array_t<float> depth(std::vector<py::ssize_t>{h, w});
    int result = photara_copy_preview(context_, rgba.mutable_data(), depth.mutable_data(),
                                      size_t(w) * h, &revision);
    if (result < 0) { throw_last_error(); }
    if (result == 1) { return py::none(); }
    return py::make_tuple(revision, rgba, depth);
  }

  void resize(int width, int height)
  {
    if (photara_resize(context_, width, height) != 0) {
      throw_last_error();
    }
  }

  void configure_film(float exposure, bool transparent, bool glass, float roughness,
                      int filter_type, float filter_width)
  {
    if (photara_configure_film(context_, exposure, transparent, glass, roughness,
                               filter_type, filter_width) != 0) {
      throw_last_error();
    }
  }

  int add_gaussian_splats(const FloatArray &positions, const FloatArray &scales,
                         const FloatArray &rotations, const FloatArray &colors,
                         const FloatArray &sh, const FloatArray &matrix, int object_id)
  {
    const auto n=positions.ndim()==2 ? positions.shape(0) : 0;
    if (positions.ndim()!=2 || positions.shape(1)!=3 || scales.ndim()!=2 ||
        scales.shape(0)!=n || scales.shape(1)!=3 || rotations.ndim()!=2 ||
        rotations.shape(0)!=n || rotations.shape(1)!=4 || colors.ndim()!=2 ||
        colors.shape(0)!=n || colors.shape(1)!=4 || sh.ndim()!=2 || sh.shape(0)!=n ||
        sh.shape(1)!=45 || matrix.size()!=16) throw std::runtime_error("Invalid Gaussian array shapes");
    const int result=photara_add_gaussian_splats(context_,int(n),positions.data(),scales.data(),
        rotations.data(),colors.data(),sh.data(),matrix.data(),object_id);
    if(result<0) throw_last_error();
    return result;
  }

  void configure_sampling(float threshold, int min_samples, double time_limit, int start_sample)
  {
    if (photara_configure_sampling(context_, threshold, min_samples, time_limit, start_sample) != 0) {
      throw_last_error();
    }
  }

  void configure_dlss(float factor)
  {
    if (photara_configure_dlss(context_, factor) != 0) { throw_last_error(); }
  }

  void configure_denoiser(int type, int passes, int prefilter, int quality, bool use_gpu)
  {
    if (photara_configure_denoiser(context_, type, passes, prefilter, quality, use_gpu) != 0) {
      throw_last_error();
    }
  }

  py::array get_pass(const std::string &name)
  {
    const int channels = photara_pass_channels(context_, name.c_str());
    if (channels < 0) {
      throw std::runtime_error("Unknown render pass '" + name + "'");
    }
    const int width = photara_render_width(context_);
    const int height = photara_render_height(context_);
    if (width <= 0 || height <= 0) {
      throw std::runtime_error("Render resolution is not available");
    }

    py::array_t<float> result(std::vector<py::ssize_t>{height, width, channels});
    if (photara_copy_pass(context_, name.c_str(), result.mutable_data(), size_t(result.size())) !=
        0)
    {
      throw_last_error();
    }
    return result;
  }

  int width() const
  {
    return photara_render_width(context_);
  }

  int height() const
  {
    return photara_render_height(context_);
  }

  std::string device_name() const
  {
    return photara_device_name(context_);
  }

  std::string last_error() const
  {
    return photara_last_error(context_);
  }

 private:
  static void check_vector3(const FloatArray &array, const char *name)
  {
    if (array.ndim() != 1 || array.shape(0) != 3) {
      throw std::runtime_error(std::string(name) + " must have shape (3,)");
    }
  }

  static void copy_vector3(const FloatArray &array, float dst[3])
  {
    const float *data = array.data();
    dst[0] = data[0];
    dst[1] = data[1];
    dst[2] = data[2];
  }

  static PhotaraMaterialDesc make_material_desc(const FloatArray &base_color,
                                                const float metallic,
                                                const float roughness,
                                                const float ior,
                                                const float transmission,
                                                const FloatArray &emission_color,
                                                const float emission_strength)
  {
    check_vector3(base_color, "base_color");
    check_vector3(emission_color, "emission_color");

    PhotaraMaterialDesc desc = {};
    copy_vector3(base_color, desc.base_color);
    desc.metallic = metallic;
    desc.roughness = roughness;
    desc.ior = ior;
    desc.transmission = transmission;
    copy_vector3(emission_color, desc.emission_color);
    desc.emission_strength = emission_strength;
    return desc;
  }

  static PhotaraAreaLightDesc make_area_light_desc(const FloatArray &position,
                                                   const FloatArray &target,
                                                   const py::object &up,
                                                   const float size_u,
                                                   const float size_v,
                                                   const FloatArray &color,
                                                   const float strength)
  {
    check_vector3(position, "position");
    check_vector3(target, "target");
    check_vector3(color, "color");

    PhotaraAreaLightDesc desc = {};
    copy_vector3(position, desc.position);
    copy_vector3(target, desc.target);
    if (!up.is_none()) {
      const FloatArray up_array = require_vector3(up, "up");
      copy_vector3(up_array, desc.up);
    }
    desc.size_u = size_u;
    desc.size_v = size_v;
    copy_vector3(color, desc.color);
    desc.strength = strength;
    return desc;
  }

  static PhotaraSunLightDesc make_sun_light_desc(const FloatArray &direction,
                                                 const FloatArray &color,
                                                 const float strength,
                                                 const float angle_degrees)
  {
    check_vector3(direction, "direction");
    check_vector3(color, "color");

    PhotaraSunLightDesc desc = {};
    copy_vector3(direction, desc.direction);
    copy_vector3(color, desc.color);
    desc.strength = strength;
    desc.angle_degrees = angle_degrees;
    return desc;
  }

  static PhotaraPointLightDesc make_point_light_desc(const FloatArray &position,
                                                     const FloatArray &color,
                                                     const float strength,
                                                     const float radius)
  {
    check_vector3(position, "position");
    check_vector3(color, "color");

    PhotaraPointLightDesc desc = {};
    copy_vector3(position, desc.position);
    copy_vector3(color, desc.color);
    desc.strength = strength;
    desc.radius = radius;
    return desc;
  }

  static PhotaraSpotLightDesc make_spot_light_desc(const FloatArray &position,
                                                   const FloatArray &direction,
                                                   const py::object &up,
                                                   const FloatArray &color,
                                                   const float strength,
                                                   const float radius,
                                                   const float angle_degrees,
                                                   const float smooth)
  {
    check_vector3(position, "position");
    check_vector3(direction, "direction");
    check_vector3(color, "color");

    PhotaraSpotLightDesc desc = {};
    copy_vector3(position, desc.position);
    copy_vector3(direction, desc.direction);
    copy_vector3(color, desc.color);
    if (!up.is_none()) {
      const FloatArray up_array = require_vector3(up, "up");
      copy_vector3(up_array, desc.up);
    }
    desc.strength = strength;
    desc.radius = radius;
    desc.angle_degrees = angle_degrees;
    desc.smooth = smooth;
    return desc;
  }

  [[noreturn]] void throw_last_error() const
  {
    throw std::runtime_error(photara_last_error(context_));
  }

  PhotaraContext *context_ = nullptr;
};

} /* namespace */

static void fill_display_device(void *pointer, int type, const uint8_t *uuid, size_t size)
{
  auto *device = static_cast<ccl::GraphicsInteropDevice *>(pointer);
  device->type = static_cast<ccl::GraphicsInteropDevice::Type>(type);
  device->uuid.assign(uuid, uuid + size);
}

PYBIND11_MODULE(_photara, module)
{
  module.def("auto_denoiser", &photara_auto_denoiser);
  module.def("display_device_allocator", []() {
    return py::capsule(reinterpret_cast<void *>(&fill_display_device),
                       "photara.DisplayDeviceAllocator.v1");
  });
  module.doc() = "Photara renderer native bindings";

  py::class_<Context>(module, "Context")
      .def(py::init<int,
                    int,
                    int,
                    int,
                    int,
                    bool,
                    bool,
                    const std::string &,
                    const std::string &, bool>(),
           py::arg("width"),
           py::arg("height"),
           py::arg("samples"),
           py::arg("seed"),
           py::arg("max_bounces"),
           py::arg("use_denoise"),
           py::arg("use_adaptive_sampling"),
           py::arg("device"),
           py::arg("resource_dir"), py::arg("viewport") = false)
      .def("add_material",
           &Context::add_material,
           py::arg("base_color"),
           py::arg("metallic"),
           py::arg("roughness"),
           py::arg("ior"),
           py::arg("transmission"),
           py::arg("emission_color"),
           py::arg("emission_strength"))
      .def("update_material",
           &Context::update_material,
           py::arg("material_index"),
           py::arg("base_color"),
           py::arg("metallic"),
           py::arg("roughness"),
           py::arg("ior"),
           py::arg("transmission"),
           py::arg("emission_color"),
           py::arg("emission_strength"))
      .def("set_material_base_color_texture",
           &Context::set_material_base_color_texture,
           py::arg("material_index"),
           py::arg("filename"),
           py::arg("colorspace") = "",
           py::arg("interpolation") = 0,
           py::arg("extension") = 0)
      .def("clear_material_base_color_texture",
           &Context::clear_material_base_color_texture,
           py::arg("material_index"))
      .def("add_mesh",
           &Context::add_mesh,
           py::arg("positions"),
           py::arg("indices"),
           py::arg("normals") = py::none(),
           py::arg("uvs") = py::none(),
           py::arg("matrix") = py::none(),
           py::arg("material_index") = -1,
           py::arg("object_id") = 0)
      .def("add_instance",
           &Context::add_instance,
           py::arg("source_object"),
           py::arg("matrix") = py::none(),
           py::arg("object_id") = 0)
      .def("update_mesh",
           &Context::update_mesh,
           py::arg("object"),
           py::arg("positions"),
           py::arg("normals") = py::none(),
           py::arg("uvs") = py::none())
      .def("replace_mesh",
           &Context::replace_mesh,
           py::arg("object"),
           py::arg("positions"),
           py::arg("indices"),
           py::arg("normals") = py::none(),
           py::arg("uvs") = py::none(),
           py::arg("material_index") = -1)
      .def("set_object_transform",
           &Context::set_object_transform,
           py::arg("object"),
           py::arg("matrix"))
      .def("set_object_visible",
           &Context::set_object_visible,
           py::arg("object"),
           py::arg("visible"))
      .def("set_object_id",
           &Context::set_object_id,
           py::arg("object"),
           py::arg("object_id"))
      .def("set_object_material",
           &Context::set_object_material,
           py::arg("object"),
           py::arg("material_index"))
      .def("remove_object", &Context::remove_object, py::arg("object"))
      .def("clear_objects", &Context::clear_objects)
      .def("set_camera",
           &Context::set_camera,
           py::arg("position"),
           py::arg("target"),
           py::arg("up") = py::none(),
           py::arg("fov_degrees") = 45.0f,
           py::arg("near_clip") = 0.001f,
           py::arg("far_clip") = 1000.0f)
      .def("set_camera_matrix",
           &Context::set_camera_matrix,
           py::arg("matrix"),
           py::arg("fov_degrees") = 45.0f,
           py::arg("near_clip") = 0.001f,
           py::arg("far_clip") = 1000.0f)
      .def("queue_camera_matrix",
           &Context::queue_camera_matrix,
           py::arg("job"),
           py::arg("matrix"),
           py::arg("fov_degrees"),
           py::arg("near_clip") = 0.001f,
           py::arg("far_clip") = 1000.0f)
      .def("add_area_light",
           &Context::add_area_light,
           py::arg("position"),
           py::arg("target"),
           py::arg("up") = py::none(),
           py::arg("size_u") = 1.0f,
           py::arg("size_v") = 1.0f,
           py::arg("color"),
           py::arg("strength") = 10.0f)
      .def("update_area_light",
           &Context::update_area_light,
           py::arg("light"),
           py::arg("position"),
           py::arg("target"),
           py::arg("up") = py::none(),
           py::arg("size_u") = 1.0f,
           py::arg("size_v") = 1.0f,
           py::arg("color"),
           py::arg("strength") = 10.0f)
      .def("add_sun_light",
           &Context::add_sun_light,
           py::arg("direction"),
           py::arg("color"),
           py::arg("strength") = 1.0f,
           py::arg("angle_degrees") = 0.0f)
      .def("update_sun_light",
           &Context::update_sun_light,
           py::arg("light"),
           py::arg("direction"),
           py::arg("color"),
           py::arg("strength") = 1.0f,
           py::arg("angle_degrees") = 0.0f)
      .def("add_point_light",
           &Context::add_point_light,
           py::arg("position"),
           py::arg("color"),
           py::arg("strength") = 100.0f,
           py::arg("radius") = 0.0f)
      .def("update_point_light",
           &Context::update_point_light,
           py::arg("light"),
           py::arg("position"),
           py::arg("color"),
           py::arg("strength") = 100.0f,
           py::arg("radius") = 0.0f)
      .def("add_spot_light",
           &Context::add_spot_light,
           py::arg("position"),
           py::arg("direction"),
           py::arg("up") = py::none(),
           py::arg("color"),
           py::arg("strength") = 100.0f,
           py::arg("radius") = 0.0f,
           py::arg("angle_degrees") = 45.0f,
           py::arg("smooth") = 0.5f)
      .def("update_spot_light",
           &Context::update_spot_light,
           py::arg("light"),
           py::arg("position"),
           py::arg("direction"),
           py::arg("up") = py::none(),
           py::arg("color"),
           py::arg("strength") = 100.0f,
           py::arg("radius") = 0.0f,
           py::arg("angle_degrees") = 45.0f,
           py::arg("smooth") = 0.5f)
      .def("set_light_transform",
           &Context::set_light_transform,
           py::arg("light"),
           py::arg("matrix"))
      .def("set_light_visible",
           &Context::set_light_visible,
           py::arg("light"),
           py::arg("visible"))
      .def("remove_light", &Context::remove_light, py::arg("light"))
      .def("clear_lights", &Context::clear_lights)
      .def("clear_scene", &Context::clear_scene)
      .def("set_background",
           &Context::set_background,
           py::arg("color"),
           py::arg("strength") = 1.0f)
      .def("set_background_texture",
           &Context::set_background_texture,
           py::arg("filename"),
           py::arg("colorspace") = "",
           py::arg("interpolation") = 0,
           py::arg("extension") = 0)
      .def("clear_background_texture", &Context::clear_background_texture)
      .def("set_seed", &Context::set_seed, py::arg("seed"))
      .def("set_samples", &Context::set_samples, py::arg("samples"))
      .def("configure_texture_lod", &Context::configure_texture_lod, py::arg("method"))
      .def("texture_lod", &Context::texture_lod)
      .def("neeat_stats", &Context::neeat_stats)
      .def("configure_neeat", &Context::configure_neeat, py::arg("enabled"))
      .def("set_render_mode", &Context::set_render_mode,
           py::arg("mode"), py::arg("samples_per_frame"))
      .def("render_begin", &Context::render_begin)
      .def("render_poll", &Context::render_poll, py::arg("job"))
      .def("render_request_cancel", &Context::render_request_cancel)
      .def("render_cancel",
           &Context::render_cancel,
           py::arg("job"),
           py::call_guard<py::gil_scoped_release>())
      .def("render_finish", &Context::render_finish, py::arg("job"))
      .def("render", &Context::render, py::call_guard<py::gil_scoped_release>())
      .def("attach_display", &Context::attach_display)
      .def("draw_display", &Context::draw_display)
      .def("preview", &Context::preview)
      .def("resize", &Context::resize)
      .def("configure_film", &Context::configure_film)
      .def("add_gaussian_splats", &Context::add_gaussian_splats)
      .def("configure_sampling", &Context::configure_sampling)
      .def("configure_dlss", &Context::configure_dlss)
      .def("configure_denoiser", &Context::configure_denoiser)
      .def("get_pass", &Context::get_pass, py::arg("name"))
      .def_property_readonly("width", &Context::width)
      .def_property_readonly("height", &Context::height)
      .def_property_readonly("device_name", &Context::device_name)
      .def_property_readonly("last_error", &Context::last_error);
}
