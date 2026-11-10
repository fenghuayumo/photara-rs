/* SPDX-License-Identifier: Apache-2.0 */

#include "photara/photara.h"

#ifdef _WIN32
#  include <windows.h>
#  include <dbghelp.h>
#  include <cstdio>
#endif

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include "device/denoise.h"
#include "device/device.h"
#include "kernel/types.h"
#include "scene/attribute.h"
#include "scene/background.h"
#include "scene/camera.h"
#include "scene/film.h"
#include "scene/integrator.h"
#include "scene/light.h"
#include "scene/mesh.h"
#include "scene/object.h"
#include "scene/pass.h"
#include "scene/scene.h"
#include "scene/shader.h"
#include "scene/shader_graph.h"
#include "scene/shader_nodes.h"
#include "session/buffers.h"
#include "session/output_driver.h"
#include "session/session.h"
#include "util/array.h"
#include "util/log.h"
#include "util/math.h"
#include "util/path.h"
#include "util/system.h"
#include "util/transform.h"
#include "util/types.h"
#include "util/types_normal.h"
#include "util/unique_ptr.h"
#include "util/vector.h"

using namespace ccl;

namespace {

constexpr float kPi = 3.14159265358979323846f;

struct PassSpec {
  const char *name;
  PassType type;
  int channels;
};

const PassSpec kPasses[] = {
    {"combined", PASS_COMBINED, 4},
    {"depth", PASS_DEPTH, 1},
    {"normal", PASS_NORMAL, 3},
    {"albedo", PASS_DIFFUSE_COLOR, 3},
    {"object_id", PASS_OBJECT_ID, 1},
};

std::mutex g_create_error_mutex;
std::string g_create_error;

#ifdef _WIN32
LONG WINAPI photara_crash_handler(EXCEPTION_POINTERS *info)
{
  if (info->ExceptionRecord->ExceptionCode != EXCEPTION_ACCESS_VIOLATION) {
    return EXCEPTION_CONTINUE_SEARCH;
  }

  static std::mutex print_mutex;
  const std::lock_guard<std::mutex> lock(print_mutex);

  std::fprintf(stderr,
               "\n[photara] ACCESS VIOLATION at %p (access address %p)\n",
               info->ExceptionRecord->ExceptionAddress,
               reinterpret_cast<void *>(info->ExceptionRecord->ExceptionInformation[1]));

  void *frames[64];
  const USHORT count = CaptureStackBackTrace(0, 64, frames, nullptr);

  static bool symbols_initialized = false;
  static bool symbols_ok = false;
  HANDLE process = GetCurrentProcess();
  if (!symbols_initialized) {
    symbols_initialized = true;
    SymSetOptions(SYMOPT_UNDNAME | SYMOPT_DEFERRED_LOADS | SYMOPT_LOAD_LINES);
    symbols_ok = SymInitialize(process, nullptr, TRUE) != FALSE;
  }

  for (USHORT index = 0; index < count; index++) {
    const DWORD64 address = reinterpret_cast<DWORD64>(frames[index]);

    char storage[sizeof(SYMBOL_INFO) + 256] = {};
    SYMBOL_INFO *symbol = reinterpret_cast<SYMBOL_INFO *>(storage);
    symbol->SizeOfStruct = sizeof(SYMBOL_INFO);
    symbol->MaxNameLen = 255;

    DWORD64 displacement = 0;
    if (symbols_ok && SymFromAddr(process, address, &displacement, symbol)) {
      IMAGEHLP_LINE64 line = {};
      line.SizeOfStruct = sizeof(line);
      DWORD line_displacement = 0;
      if (SymGetLineFromAddr64(process, address, &line_displacement, &line)) {
        std::fprintf(stderr,
                     "  [%02u] %s+0x%llx (%s:%lu)\n",
                     unsigned(index),
                     symbol->Name,
                     static_cast<unsigned long long>(displacement),
                     line.FileName,
                     unsigned long(line.LineNumber));
      }
      else {
        std::fprintf(stderr,
                     "  [%02u] %s+0x%llx\n",
                     unsigned(index),
                     symbol->Name,
                     static_cast<unsigned long long>(displacement));
      }
    }
    else {
      HMODULE module = nullptr;
      char module_name[MAX_PATH] = {};
      if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                                 GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                             reinterpret_cast<LPCSTR>(frames[index]),
                             &module))
      {
        GetModuleFileNameA(module, module_name, MAX_PATH);
      }
      std::fprintf(stderr,
                   "  [%02u] %p %s\n",
                   unsigned(index),
                   frames[index],
                   module_name);
    }
  }
  std::fflush(stderr);
  return EXCEPTION_CONTINUE_SEARCH;
}
#endif

void set_create_error(const std::string &message)
{
  const std::lock_guard<std::mutex> lock(g_create_error_mutex);
  g_create_error = message;
}

float3 float3_from_ptr(const float *v)
{
  return make_float3(v[0], v[1], v[2]);
}

float3 safe_normalize(const float3 &v, const char *what)
{
  const float length_squared = dot(v, v);
  if (!(length_squared > 1e-20f)) {
    throw std::runtime_error(std::string("Cannot normalize zero-length ") + what);
  }
  return v * (1.0f / std::sqrt(length_squared));
}

/* Build a Cycles transform whose local axes are (right, up, forward), with
 * `origin` as translation. Cycles stores the rows of the column-vector affine
 * matrix, so the basis vectors are placed in the columns here. */
Transform basis_transform(const float3 &right,
                          const float3 &up,
                          const float3 &forward,
                          const float3 &origin)
{
  Transform tfm;
  tfm.x = make_float4(right.x, up.x, forward.x, origin.x);
  tfm.y = make_float4(right.y, up.y, forward.y, origin.y);
  tfm.z = make_float4(right.z, up.z, forward.z, origin.z);
  return tfm;
}

/* Input is a row-major 4x4 affine matrix with the translation in the last
 * column. Cycles stores the first three rows directly. */
Transform transform_from_matrix(const float *m)
{
  Transform tfm;
  tfm.x = make_float4(m[0], m[1], m[2], m[3]);
  tfm.y = make_float4(m[4], m[5], m[6], m[7]);
  tfm.z = make_float4(m[8], m[9], m[10], m[11]);
  return tfm;
}

void make_orthonormal_basis(const float3 &forward_in,
                            const float *up_hint,
                            float3 &right,
                            float3 &up,
                            float3 &forward)
{
  forward = safe_normalize(forward_in, "view/direction vector");

  float3 hint = up_hint ? float3_from_ptr(up_hint) : make_float3(0.0f, 1.0f, 0.0f);
  if (!(dot(hint, hint) > 1e-20f)) {
    hint = make_float3(0.0f, 1.0f, 0.0f);
  }
  hint = normalize(hint);

  if (std::fabs(dot(hint, forward)) > 0.9999f) {
    hint = (std::fabs(forward.z) < 0.9f) ? make_float3(0.0f, 0.0f, 1.0f) :
                                           make_float3(1.0f, 0.0f, 0.0f);
  }

  right = normalize(cross(hint, forward));
  up = cross(forward, right);
}

Shader *make_principled_shader(Scene *scene, const PhotaraMaterialDesc &desc)
{
  unique_ptr<ShaderGraph> graph = make_unique<ShaderGraph>();
  PrincipledBsdfNode *bsdf = graph->create_node<PrincipledBsdfNode>();

  bsdf->set_base_color(float3_from_ptr(desc.base_color));
  bsdf->set_metallic(desc.metallic);
  bsdf->set_roughness(desc.roughness);
  bsdf->set_ior(desc.ior);
  bsdf->set_transmission_weight(desc.transmission);
  bsdf->set_emission_color(float3_from_ptr(desc.emission_color));
  bsdf->set_emission_strength(desc.emission_strength);

  graph->connect(bsdf->output("BSDF"), graph->output()->input("Surface"));

  Shader *shader = scene->create_node<Shader>();
  shader->set_graph(std::move(graph));
  shader->tag_update(scene);
  return shader;
}

Shader *make_emission_shader(Scene *scene, const float3 &color, const float strength)
{
  unique_ptr<ShaderGraph> graph = make_unique<ShaderGraph>();
  EmissionNode *emission = graph->create_node<EmissionNode>();
  emission->set_color(color);
  emission->set_strength(strength);
  graph->connect(emission->output("Emission"), graph->output()->input("Surface"));

  Shader *shader = scene->create_node<Shader>();
  shader->set_graph(std::move(graph));
  shader->tag_update(scene);
  return shader;
}

BackgroundNode *configure_background_shader(Scene *scene,
                                            const float3 &color,
                                            const float strength)
{
  unique_ptr<ShaderGraph> graph = make_unique<ShaderGraph>();
  BackgroundNode *background = graph->create_node<BackgroundNode>();
  background->set_color(color);
  background->set_strength(strength);
  graph->connect(background->output("Background"), graph->output()->input("Surface"));

  /* The default background shader is referenced by the scene and is what the
   * standalone XML loader populates. Replacing its graph keeps the lifetime
   * and shader-manager ownership identical to Cycles. */
  Shader *shader = scene->default_background;
  shader->set_graph(std::move(graph));
  shader->tag_update(scene);
  return background;
}

void assign_single_shader(Light *light, Shader *shader)
{
  array<Node *> used_shaders;
  used_shaders.push_back_slow(shader);
  light->set_used_shaders(used_shaders);
}

bool find_device(const PhotaraDevice requested, DeviceInfo &device, std::string &error)
{
  if (requested != PHOTARA_DEVICE_AUTO && requested != PHOTARA_DEVICE_OPTIX) {
    error = "Only OptiX is supported";
    return false;
  }

  for (const DeviceInfo &info : Device::available_devices(DEVICE_MASK_OPTIX)) {
    if (info.meets_driver_requirement) {
      device = info;
      return true;
    }
  }

  error = "OptiX device is not available (requires an NVIDIA GPU, a current driver and a build with OptiX support)";
  return false;
}

/* Copies the completed full-frame passes out of the renderer. */
class CaptureDriver : public OutputDriver {
 public:
  struct PassBuffer {
    int channels = 0;
    std::vector<float> pixels;
  };

  explicit CaptureDriver(std::vector<PassSpec> passes) : pass_specs_(std::move(passes)) {}

  void clear()
  {
    const std::lock_guard<std::mutex> lock(mutex_);
    width_ = 0;
    height_ = 0;
    passes_.clear();
  }

  bool has_data() const
  {
    const std::lock_guard<std::mutex> lock(mutex_);
    return width_ > 0 && height_ > 0 && !passes_.empty();
  }

  int width() const
  {
    const std::lock_guard<std::mutex> lock(mutex_);
    return width_;
  }

  int height() const
  {
    const std::lock_guard<std::mutex> lock(mutex_);
    return height_;
  }

  bool copy_pass(const char *name, float *dst, const size_t dst_count)
  {
    const std::lock_guard<std::mutex> lock(mutex_);
    if (width_ <= 0 || height_ <= 0) {
      return false;
    }
    const auto it = passes_.find(name);
    if (it == passes_.end()) {
      return false;
    }
    const size_t count = size_t(width_) * size_t(height_) * size_t(it->second.channels);
    if (dst_count < count) {
      return false;
    }
    std::memcpy(dst, it->second.pixels.data(), count * sizeof(float));
    return true;
  }

  void write_render_tile(const Tile &tile) override
  {
    if (tile.size.x != tile.full_size.x || tile.size.y != tile.full_size.y) {
      return;
    }

    const std::lock_guard<std::mutex> lock(mutex_);
    width_ = tile.size.x;
    height_ = tile.size.y;
    passes_.clear();

    for (const PassSpec &spec : pass_specs_) {
      std::vector<float> pixels(size_t(width_) * size_t(height_) * size_t(spec.channels));
      if (tile.get_pass_pixels(spec.name, spec.channels, pixels.data())) {
        passes_.emplace(spec.name, PassBuffer{spec.channels, std::move(pixels)});
      }
    }
  }

 private:
  mutable std::mutex mutex_;
  int width_ = 0;
  int height_ = 0;
  std::vector<PassSpec> pass_specs_;
  std::map<std::string, PassBuffer> passes_;
};

} /* namespace */

struct PhotaraContext {
  std::mutex mutex;
  std::string last_error;
  std::string device_name;
  int width = 0;
  int height = 0;
  bool rendered = false;

  unique_ptr<Session> session;
  Scene *scene = nullptr;
  SessionParams session_params;
  BufferParams buffer_params;
  CaptureDriver *capture = nullptr;
  BackgroundNode *background_node = nullptr;
  std::vector<Shader *> materials;

  ~PhotaraContext()
  {
    session.reset();
  }
};

namespace {

void set_error(PhotaraContext *ctx, const std::string &message)
{
  ctx->last_error = message;
}

template<typename Function>
int guarded(PhotaraContext *ctx, Function &&function)
{
  if (ctx == nullptr) {
    set_create_error("NULL Photara context");
    return -1;
  }
  const std::lock_guard<std::mutex> lock(ctx->mutex);
  try {
    function();
    return 0;
  }
  catch (const std::exception &exception) {
    set_error(ctx, exception.what());
    return -1;
  }
  catch (...) {
    set_error(ctx, "Unknown error");
    return -1;
  }
}

} /* namespace */

extern "C" {

PhotaraContext *photara_create(const PhotaraOptions *options)
{
  if (options == nullptr) {
    set_create_error("NULL options passed to photara_create");
    return nullptr;
  }

  try {
    if (options->width <= 0 || options->height <= 0) {
      throw std::runtime_error("Render width and height must be positive");
    }
    if (options->width > 32768 || options->height > 32768) {
      throw std::runtime_error("Render resolution is too large");
    }

    static std::once_flag init_flag;
    std::call_once(init_flag, []() {
      log_init(nullptr);
      system_max_open_files_ensure();
#ifdef _WIN32
      if (std::getenv("PHOTARA_CRASH_HANDLER") != nullptr) {
        AddVectoredExceptionHandler(1, photara_crash_handler);
      }
#endif
    });

    path_init(options->resource_dir ? options->resource_dir : "");

    if (const char *level = std::getenv("PHOTARA_LOG_LEVEL")) {
      if (std::strcmp(level, "trace") == 0) {
        LOG_LEVEL = LOG_LEVEL_TRACE;
      }
      else if (std::strcmp(level, "debug") == 0) {
        LOG_LEVEL = LOG_LEVEL_DEBUG;
      }
      else if (std::strcmp(level, "info") == 0) {
        LOG_LEVEL = LOG_LEVEL_INFO;
      }
    }

    DeviceInfo device;
    std::string device_error;
    if (!find_device(options->device, device, device_error)) {
      throw std::runtime_error(device_error);
    }

    auto ctx = std::make_unique<PhotaraContext>();
    ctx->width = options->width;
    ctx->height = options->height;
    ctx->device_name = device.description;

    SessionParams &params = ctx->session_params;
    params.device = device;
    params.denoise_device = device;
    params.headless = true;
    params.background = true;
    params.samples = std::max(options->samples > 0 ? options->samples : 64, 1);
    params.use_auto_tile = false;
    params.tile_size = 0;
    params.use_resolution_divider = false;
    params.pixel_size = 1;
    params.threads = 0;
    params.shadingsystem = SHADINGSYSTEM_SVM;

    SceneParams scene_params;
    scene_params.shadingsystem = SHADINGSYSTEM_SVM;
    scene_params.bvh_type = BVH_TYPE_STATIC;

    ctx->session = make_unique<Session>(params, scene_params);
    if (ctx->session->progress.get_error()) {
      throw std::runtime_error(ctx->session->progress.get_error_message());
    }

    ctx->scene = ctx->session->scene.get();
    Scene *scene = ctx->scene;

    /* Render configuration. */
    Integrator *integrator = scene->integrator;
    const int max_bounces = std::clamp(options->max_bounces > 0 ? options->max_bounces : 8, 1, 64);
    integrator->set_seed(std::max(options->seed, 0));
    integrator->set_max_bounce(max_bounces);
    integrator->set_max_diffuse_bounce(max_bounces);
    integrator->set_max_glossy_bounce(max_bounces);
    integrator->set_max_transmission_bounce(max_bounces);
    integrator->set_use_adaptive_sampling(options->use_adaptive_sampling != 0);

    if (options->use_denoise) {
      const DenoiserType denoiser = (device.denoisers & DENOISER_OPTIX) ? DENOISER_OPTIX :
                                                                          DENOISER_OPENIMAGEDENOISE;
      integrator->set_use_denoise(true);
      integrator->set_denoiser_type(denoiser);
      integrator->set_denoiser_passes(DENOISER_PASS_ALBEDO | DENOISER_PASS_NORMAL);
      integrator->set_denoiser_prefilter(DENOISER_PREFILTER_ACCURATE);
      integrator->set_denoise_use_gpu(true);
    }
    else {
      integrator->set_use_denoise(false);
    }

    scene->film->set_display_pass(PASS_COMBINED);

    /* Named passes requested by the Python wrapper and C API users. The
     * environment filter is a development aid; all passes are created by
     * default. */
    std::vector<PassSpec> active_passes;
    {
      const char *pass_filter = std::getenv("PHOTARA_PASSES");
      for (const PassSpec &spec : kPasses) {
        if (pass_filter != nullptr && std::strstr(pass_filter, spec.name) == nullptr) {
          continue;
        }
        active_passes.push_back(spec);
        Pass *pass = scene->create_node<Pass>();
        pass->set_name(ustring(spec.name));
        pass->set_type(spec.type);
      }
    }

    /* Default camera looking down +Z from (0, 0, -5). */
    {
      PhotaraCameraDesc camera = {};
      camera.position[0] = 0.0f;
      camera.position[1] = 0.0f;
      camera.position[2] = -5.0f;
      camera.target[0] = 0.0f;
      camera.target[1] = 0.0f;
      camera.target[2] = 0.0f;
      camera.up[0] = 0.0f;
      camera.up[1] = 1.0f;
      camera.up[2] = 0.0f;
      camera.fov_degrees = 45.0f;
      camera.near_clip = 0.001f;
      camera.far_clip = 1000.0f;

      Camera *cam = scene->camera;
      float3 right, up, forward;
      make_orthonormal_basis(float3_from_ptr(camera.target) - float3_from_ptr(camera.position),
                             camera.up,
                             right,
                             up,
                             forward);

      cam->set_camera_type(CAMERA_PERSPECTIVE);
      cam->set_fov(camera.fov_degrees * kPi / 180.0f);
      cam->set_nearclip(camera.near_clip);
      cam->set_farclip(camera.far_clip);
      cam->set_full_width(ctx->width);
      cam->set_full_height(ctx->height);
      cam->set_matrix(basis_transform(right, up, forward, float3_from_ptr(camera.position)));
      cam->need_flags_update = true;
      cam->compute_auto_viewplane();
      cam->update(scene);
    }

    /* Neutral default world, so an empty scene is not pitch black. */
    {
      /* Cycles uses scene->default_background as the fallback shader when the
       * Background node has no explicit shader. Configure that shader and leave
       * the socket empty, matching the XML loader. */
      ctx->background_node = configure_background_shader(
          scene, make_float3(0.05f, 0.05f, 0.05f), 1.0f);
      scene->background->set_use_shader(true);
    }

    ctx->buffer_params.width = ctx->width;
    ctx->buffer_params.height = ctx->height;
    ctx->buffer_params.full_width = ctx->width;
    ctx->buffer_params.full_height = ctx->height;

    {
      auto capture = make_unique<CaptureDriver>(active_passes);
      ctx->capture = capture.get();
      ctx->session->set_output_driver(std::move(capture));
    }

    set_create_error("");
    return ctx.release();
  }
  catch (const std::exception &exception) {
    set_create_error(exception.what());
    return nullptr;
  }
  catch (...) {
    set_create_error("Unknown error while creating the Photara context");
    return nullptr;
  }
}

void photara_destroy(PhotaraContext *ctx)
{
  delete ctx;
}

int photara_add_material(PhotaraContext *ctx, const PhotaraMaterialDesc *desc)
{
  if (ctx == nullptr) {
    return -1;
  }
  const std::lock_guard<std::mutex> lock(ctx->mutex);
  try {
    if (desc == nullptr) {
      throw std::runtime_error("NULL material description");
    }
    Shader *shader = make_principled_shader(ctx->scene, *desc);
    ctx->materials.push_back(shader);
    return int(ctx->materials.size()) - 1;
  }
  catch (const std::exception &exception) {
    set_error(ctx, exception.what());
    return -1;
  }
}

int photara_add_mesh(PhotaraContext *ctx, const PhotaraMeshDesc *desc)
{
  return guarded(ctx, [&]() {
    if (desc == nullptr) {
      throw std::runtime_error("NULL mesh description");
    }
    if (desc->positions == nullptr || desc->indices == nullptr) {
      throw std::runtime_error("Mesh positions and indices are required");
    }
    if (desc->num_vertices <= 0 || desc->num_triangles <= 0) {
      throw std::runtime_error("Mesh must have at least one vertex and one triangle");
    }

    for (int i = 0; i < desc->num_triangles * 3; i++) {
      const int index = desc->indices[i];
      if (index < 0 || index >= desc->num_vertices) {
        throw std::runtime_error("Mesh triangle index is out of range");
      }
    }

    Shader *shader = ctx->scene->default_surface;
    if (desc->material_index >= 0) {
      if (size_t(desc->material_index) >= ctx->materials.size()) {
        throw std::runtime_error("Mesh material index is out of range");
      }
      shader = ctx->materials[size_t(desc->material_index)];
    }

    Mesh *mesh = ctx->scene->create_node<Mesh>();
    mesh->resize_mesh(desc->num_vertices, desc->num_triangles);

    packed_float3 *positions = mesh->get_position_for_write();
    for (int i = 0; i < desc->num_vertices; i++) {
      positions[i] = packed_float3(make_float3(desc->positions[i * 3 + 0],
                                               desc->positions[i * 3 + 1],
                                               desc->positions[i * 3 + 2]));
    }

    std::memcpy(mesh->get_triangles().data(),
                desc->indices,
                sizeof(int) * size_t(desc->num_triangles) * 3);

    bool *smooth = mesh->get_smooth().data();
    int *shaders = mesh->get_shader().data();
    std::fill(smooth, smooth + desc->num_triangles, true);
    std::fill(shaders, shaders + desc->num_triangles, 0);

    array<Node *> used_shaders;
    used_shaders.push_back_slow(shader);
    mesh->set_used_shaders(used_shaders);

    if (desc->normals != nullptr) {
      Attribute *attribute = mesh->attributes.add(ATTR_STD_VERTEX_NORMAL);
      packed_normal *normals = attribute->data_for_write<packed_normal>();
      for (int i = 0; i < desc->num_vertices; i++) {
        normals[i] = packed_normal(float3_from_ptr(desc->normals + i * 3));
      }
    }

    if (desc->uvs != nullptr) {
      Attribute *attribute = mesh->attributes.add(ATTR_STD_UV);
      float2 *uvs = attribute->data_for_write<float2>();
      for (int triangle = 0; triangle < desc->num_triangles; triangle++) {
        for (int corner = 0; corner < 3; corner++) {
          const int vertex = desc->indices[triangle * 3 + corner];
          uvs[triangle * 3 + corner] = make_float2(desc->uvs[vertex * 2 + 0],
                                                   desc->uvs[vertex * 2 + 1]);
        }
      }
    }

    mesh->tag_position_modified();
    mesh->tag_triangles_modified();
    mesh->tag_shader_modified();
    mesh->tag_smooth_modified();

    Object *object = ctx->scene->create_node<Object>();
    object->set_geometry(mesh);
    object->set_tfm(desc->matrix ? transform_from_matrix(desc->matrix) : transform_identity());
    object->set_pass_id(desc->object_id);
  });
}

int photara_set_camera(PhotaraContext *ctx, const PhotaraCameraDesc *desc)
{
  return guarded(ctx, [&]() {
    if (desc == nullptr) {
      throw std::runtime_error("NULL camera description");
    }

    Camera *camera = ctx->scene->camera;
    float3 right, up, forward;
    make_orthonormal_basis(float3_from_ptr(desc->target) - float3_from_ptr(desc->position),
                           desc->up,
                           right,
                           up,
                           forward);

    camera->set_camera_type(CAMERA_PERSPECTIVE);
    camera->set_fov((desc->fov_degrees > 0.0f ? desc->fov_degrees : 45.0f) * kPi / 180.0f);
    if (desc->near_clip > 0.0f) {
      camera->set_nearclip(desc->near_clip);
    }
    if (desc->far_clip > 0.0f) {
      camera->set_farclip(desc->far_clip);
    }
    camera->set_full_width(ctx->width);
    camera->set_full_height(ctx->height);
    camera->set_matrix(basis_transform(right, up, forward, float3_from_ptr(desc->position)));
    camera->need_flags_update = true;
    camera->compute_auto_viewplane();
    camera->update(ctx->scene);
  });
}

int photara_add_area_light(PhotaraContext *ctx, const PhotaraAreaLightDesc *desc)
{
  return guarded(ctx, [&]() {
    if (desc == nullptr) {
      throw std::runtime_error("NULL area light description");
    }
    if (!(desc->size_u > 0.0f) || !(desc->size_v > 0.0f)) {
      throw std::runtime_error("Area light dimensions must be positive");
    }

    float3 right, up, forward;
    /* Local +Z points from the target towards the light, so emission (-Z)
     * points at the target. */
    make_orthonormal_basis(float3_from_ptr(desc->position) - float3_from_ptr(desc->target),
                           desc->up,
                           right,
                           up,
                           forward);

    AreaLight *light = ctx->scene->create_node<AreaLight>();
    light->set_sizeu(desc->size_u);
    light->set_sizev(desc->size_v);
    light->set_use_mis(true);
    assign_single_shader(
        light, make_emission_shader(ctx->scene, float3_from_ptr(desc->color), desc->strength));

    Object *object = ctx->scene->create_node<Object>();
    object->set_geometry(light);
    object->set_tfm(basis_transform(right, up, forward, float3_from_ptr(desc->position)));
    object->set_visibility(PATH_RAY_VISIBILITY_ALL & ~PATH_RAY_VISIBILITY_CAMERA);
  });
}

int photara_add_sun_light(PhotaraContext *ctx, const PhotaraSunLightDesc *desc)
{
  return guarded(ctx, [&]() {
    if (desc == nullptr) {
      throw std::runtime_error("NULL sun light description");
    }

    /* Cycles' sun direction is -local Z. */
    float3 right, up, forward;
    make_orthonormal_basis(-float3_from_ptr(desc->direction), nullptr, right, up, forward);

    SunLight *light = ctx->scene->create_node<SunLight>();
    light->set_angle(std::max(desc->angle_degrees, 0.0f) * kPi / 180.0f);
    light->set_use_mis(true);
    assign_single_shader(
        light, make_emission_shader(ctx->scene, float3_from_ptr(desc->color), desc->strength));

    Object *object = ctx->scene->create_node<Object>();
    object->set_geometry(light);
    object->set_tfm(basis_transform(
        right, up, forward, make_float3(0.0f, 0.0f, 0.0f)));
    object->set_visibility(PATH_RAY_VISIBILITY_ALL & ~PATH_RAY_VISIBILITY_CAMERA);
  });
}

int photara_set_background(PhotaraContext *ctx, const float color[3], const float strength)
{
  return guarded(ctx, [&]() {
    if (color == nullptr) {
      throw std::runtime_error("NULL background color");
    }
    if (ctx->background_node == nullptr) {
      ctx->background_node = configure_background_shader(
          ctx->scene, float3_from_ptr(color), strength);
      ctx->scene->background->set_use_shader(true);
      return;
    }
    /* Update the existing node instead of replacing the shader graph. Replacing
     * the graph of the referenced default background has proven to leave the
     * scene in an invalid state before the first render. */
    ctx->background_node->set_color(float3_from_ptr(color));
    ctx->background_node->set_strength(strength);
    ctx->scene->default_background->tag_update(ctx->scene);
  });
}

int photara_render(PhotaraContext *ctx)
{
  return guarded(ctx, [&]() {
    ctx->rendered = false;
    if (ctx->capture != nullptr) {
      ctx->capture->clear();
    }

    ctx->session->reset(ctx->session_params, ctx->buffer_params);
    ctx->session->start();
    ctx->session->wait();

    if (ctx->session->progress.get_error()) {
      throw std::runtime_error(ctx->session->progress.get_error_message());
    }
    if (ctx->session->device->have_error()) {
      throw std::runtime_error(ctx->session->device->error_message());
    }
    if (ctx->capture != nullptr && !ctx->capture->has_data()) {
      throw std::runtime_error("Renderer finished without producing a render result");
    }
    ctx->rendered = true;
  });
}

int photara_render_width(const PhotaraContext *ctx)
{
  if (ctx == nullptr) {
    return 0;
  }
  if (ctx->rendered && ctx->capture != nullptr) {
    return ctx->capture->width();
  }
  return ctx->width;
}

int photara_render_height(const PhotaraContext *ctx)
{
  if (ctx == nullptr) {
    return 0;
  }
  if (ctx->rendered && ctx->capture != nullptr) {
    return ctx->capture->height();
  }
  return ctx->height;
}

int photara_pass_channels(const PhotaraContext *ctx, const char *name)
{
  (void)ctx;
  if (name == nullptr) {
    return -1;
  }
  for (const PassSpec &spec : kPasses) {
    if (std::strcmp(spec.name, name) == 0) {
      return spec.channels;
    }
  }
  return -1;
}

int photara_copy_pass(PhotaraContext *ctx, const char *name, float *dst, const size_t dst_count)
{
  return guarded(ctx, [&]() {
    if (name == nullptr || dst == nullptr) {
      throw std::runtime_error("NULL pass name or destination buffer");
    }
    if (!ctx->rendered) {
      throw std::runtime_error("Render the scene before reading passes");
    }
    if (ctx->capture == nullptr) {
      throw std::runtime_error("Render passes are unavailable (no output driver was created)");
    }
    if (!ctx->capture->copy_pass(name, dst, dst_count)) {
      throw std::runtime_error(std::string("Render pass '") + name +
                               "' is unavailable or the destination buffer is too small");
    }
  });
}

const char *photara_device_name(const PhotaraContext *ctx)
{
  return ctx ? ctx->device_name.c_str() : "";
}

const char *photara_last_error(const PhotaraContext *ctx)
{
  if (ctx != nullptr) {
    return ctx->last_error.c_str();
  }
  const std::lock_guard<std::mutex> lock(g_create_error_mutex);
  return g_create_error.c_str();
}

} /* extern "C" */
