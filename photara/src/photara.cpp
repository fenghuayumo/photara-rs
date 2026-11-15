/* SPDX-License-Identifier: Apache-2.0 */

#include "photara/photara.h"
#include "realtime/frame_scheduler.h"
#include "realtime/temporal_history.h"

#ifdef _WIN32
#  include <windows.h>
#  include <dbghelp.h>
#  include <cstdio>
#endif

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "device/denoise.h"
#include "device/device.h"
#include "integrator/denoiser.h"
#include "kernel/types.h"
#include "scene/attribute.h"
#include "scene/background.h"
#include "scene/camera.h"
#include "scene/film.h"
#include "scene/integrator.h"
#include "scene/light.h"
#include "scene/mesh.h"
#include "scene/pointcloud.h"
#include "util/math_spherical_harmonics.h"
#include "scene/object.h"
#include "scene/pass.h"
#include "scene/scene.h"
#ifdef WITH_OPTIX
#  include "device/optix/device_impl.h"
#  include "bvh/optix.h"
#endif
#include "scene/shader.h"
#include "scene/shader_graph.h"
#include "scene/shader_nodes.h"
#include "session/buffers.h"
#include "session/output_driver.h"
#include "session/display_driver.h"
#include "session/session.h"
#include "util/array.h"
#include "util/colorspace.h"
#include "util/log.h"
#include "util/hash.h"
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
    {"stable_radiance", PASS_STABLE_RADIANCE, 3},
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

ustring parse_texture_colorspace(const char *value)
{
  if (value == nullptr || value[0] == '\0') {
    return u_colorspace_auto;
  }

  std::string name(value);
  for (char &c : name) {
    c = char(std::tolower(static_cast<unsigned char>(c)));
  }

  if (name == "auto") {
    return u_colorspace_auto;
  }
  if (name == "srgb" || name == "__builtin_srgb") {
    return u_colorspace_srgb;
  }
  if (name == "scene_linear" || name == "linear") {
    return u_colorspace_scene_linear;
  }
  if (name == "data") {
    return u_colorspace_data;
  }
  throw std::runtime_error(
      std::string("Unknown texture colorspace '") + value +
      "' (expected auto, sRGB, scene_linear or data)");
}

InterpolationType parse_texture_interpolation(const int value)
{
  if (value < INTERPOLATION_LINEAR || value >= INTERPOLATION_NUM_TYPES) {
    throw std::runtime_error("Texture interpolation is out of range");
  }
  return static_cast<InterpolationType>(value);
}

ExtensionType parse_texture_extension(const int value)
{
  if (value < EXTENSION_REPEAT || value >= EXTENSION_NUM_TYPES) {
    throw std::runtime_error("Texture extension is out of range");
  }
  return static_cast<ExtensionType>(value);
}

void configure_image_texture(ImageTextureNode *node, const PhotaraTextureDesc &desc)
{
  if (desc.filename == nullptr || desc.filename[0] == '\0') {
    throw std::runtime_error("Texture filename is required");
  }
  const ustring colorspace = parse_texture_colorspace(desc.colorspace);
  const InterpolationType interpolation = parse_texture_interpolation(desc.interpolation);
  const ExtensionType extension = parse_texture_extension(desc.extension);

  node->handle.clear();
  node->set_filename(ustring(desc.filename));
  node->set_colorspace(colorspace);
  node->set_interpolation(interpolation);
  node->set_extension(extension);
  node->set_projection(NODE_IMAGE_PROJ_FLAT);
  node->set_animated(false);
}

void configure_environment_texture(EnvironmentTextureNode *node,
                                   const PhotaraTextureDesc &desc)
{
  if (desc.filename == nullptr || desc.filename[0] == '\0') {
    throw std::runtime_error("Environment texture filename is required");
  }
  const ustring colorspace = parse_texture_colorspace(desc.colorspace);
  const InterpolationType interpolation = parse_texture_interpolation(desc.interpolation);

  node->handle.clear();
  node->set_filename(ustring(desc.filename));
  node->set_colorspace(colorspace);
  node->set_interpolation(interpolation);
  node->set_projection(NODE_ENVIRONMENT_EQUIRECTANGULAR);
  node->set_animated(false);
}

void apply_material_desc(PrincipledBsdfNode *bsdf,
                         MixColorNode *mix,
                         const PhotaraMaterialDesc &desc)
{
  const float3 base_color = float3_from_ptr(desc.base_color);
  bsdf->set_base_color(base_color);
  bsdf->set_metallic(desc.metallic);
  bsdf->set_roughness(desc.roughness);
  bsdf->set_ior(desc.ior);
  bsdf->set_transmission_weight(desc.transmission);
  bsdf->set_emission_color(float3_from_ptr(desc.emission_color));
  bsdf->set_emission_strength(desc.emission_strength);
  if (mix != nullptr) {
    mix->set_a(base_color);
  }
}

unique_ptr<ShaderGraph> make_principled_graph(const PhotaraMaterialDesc &desc,
                                              const PhotaraTextureDesc *texture,
                                              PrincipledBsdfNode **bsdf_out,
                                              MixColorNode **mix_out,
                                              ImageTextureNode **texture_out,
                                              const float alpha_cutoff = -1.0f)
{
  unique_ptr<ShaderGraph> graph = make_unique<ShaderGraph>();
  PrincipledBsdfNode *bsdf = graph->create_node<PrincipledBsdfNode>();
  MixColorNode *mix = graph->create_node<MixColorNode>();
  mix->set_blend_type(NODE_MIX_MUL);
  mix->set_fac(1.0f);
  mix->set_b(make_float3(1.0f, 1.0f, 1.0f));
  apply_material_desc(bsdf, mix, desc);

  graph->connect(mix->output("Result"), bsdf->input("Base Color"));
  graph->connect(bsdf->output("BSDF"), graph->output()->input("Surface"));

  ImageTextureNode *texture_node = nullptr;
  if (texture != nullptr) {
    texture_node = graph->create_node<ImageTextureNode>();
    configure_image_texture(texture_node, *texture);
    graph->connect(texture_node->output("Color"), mix->input("B"));
    if (alpha_cutoff >= 0) {
      auto *test = graph->create_node<MathNode>();
      test->set_math_type(NODE_MATH_GREATER_THAN);
      test->set_value2(alpha_cutoff);
      auto *hole = graph->create_node<TransparentBsdfNode>();
      hole->set_color(one_float3());
      auto *mask = graph->create_node<MixClosureNode>();
      graph->disconnect(graph->output()->input("Surface"));
      graph->connect(texture_node->output("Alpha"), test->input("Value1"));
      graph->connect(test->output("Value"), mask->input("Fac"));
      graph->connect(hole->output("BSDF"), mask->input("Closure1"));
      graph->connect(bsdf->output("BSDF"), mask->input("Closure2"));
      graph->connect(mask->output("Closure"), graph->output()->input("Surface"));
    }
  }

  if (bsdf_out != nullptr) {
    *bsdf_out = bsdf;
  }
  if (mix_out != nullptr) {
    *mix_out = mix;
  }
  if (texture_out != nullptr) {
    *texture_out = texture_node;
  }
  return graph;
}

Shader *make_principled_shader(Scene *scene,
                               const PhotaraMaterialDesc &desc,
                               PrincipledBsdfNode **bsdf_out,
                               MixColorNode **mix_out)
{
  unique_ptr<ShaderGraph> graph = make_principled_graph(
      desc, nullptr, bsdf_out, mix_out, nullptr);

  Shader *shader = scene->create_node<Shader>();
  shader->set_graph(std::move(graph));
  shader->tag_update(scene);
  return shader;
}

struct EmissionShader {
  Shader *shader = nullptr;
  EmissionNode *node = nullptr;
};

EmissionShader make_emission_shader(Scene *scene, const float3 &color, const float strength)
{
  unique_ptr<ShaderGraph> graph = make_unique<ShaderGraph>();
  EmissionNode *emission = graph->create_node<EmissionNode>();
  emission->set_color(color);
  emission->set_strength(strength);
  graph->connect(emission->output("Emission"), graph->output()->input("Surface"));

  Shader *shader = scene->create_node<Shader>();
  shader->set_graph(std::move(graph));
  shader->tag_update(scene);
  return {shader, emission};
}

struct BackgroundShader {
  BackgroundNode *background = nullptr;
  EnvironmentTextureNode *texture = nullptr;
};

BackgroundShader configure_background_shader(Scene *scene,
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
  return {background, nullptr};
}

BackgroundShader configure_background_environment(Scene *scene,
                                                  const PhotaraTextureDesc &desc,
                                                  const float3 &color,
                                                  const float strength)
{
  unique_ptr<ShaderGraph> graph = make_unique<ShaderGraph>();
  BackgroundNode *background = graph->create_node<BackgroundNode>();
  EnvironmentTextureNode *texture = graph->create_node<EnvironmentTextureNode>();

  background->set_color(color);
  background->set_strength(strength);
  configure_environment_texture(texture, desc);
  graph->connect(texture->output("Color"), background->input("Color"));
  graph->connect(background->output("Background"), graph->output()->input("Surface"));

  Shader *shader = scene->default_background;
  shader->set_graph(std::move(graph));
  shader->tag_update(scene);
  return {background, texture};
}

void assign_single_shader(Light *light, Shader *shader)
{
  array<Node *> used_shaders;
  used_shaders.push_back_slow(shader);
  light->set_used_shaders(used_shaders);
}

bool find_device(const PhotaraDevice requested, DeviceInfo &device, std::string &error)
{
  uint mask;
  switch (requested) {
    case PHOTARA_DEVICE_AUTO:
      mask = DEVICE_MASK_OPTIX | DEVICE_MASK_METAL;
      break;
    case PHOTARA_DEVICE_OPTIX:
      mask = DEVICE_MASK_OPTIX;
      break;
    case PHOTARA_DEVICE_METAL:
      mask = DEVICE_MASK_METAL;
      break;
    default:
      error = "Unsupported render device (expected AUTO, OPTIX or METAL)";
      return false;
  }

  for (const DeviceInfo &info : Device::available_devices(mask)) {
    if (info.meets_driver_requirement) {
      device = info;
      return true;
    }
  }

  error = requested == PHOTARA_DEVICE_METAL ?
              "Metal device is not available (requires a supported Apple GPU and a Metal-enabled macOS build)" :
          requested == PHOTARA_DEVICE_OPTIX ?
              "OptiX device is not available (requires an NVIDIA GPU, a current driver and a build with OptiX support)" :
              "No supported render device is available (OptiX or Metal)";
  return false;
}

/* Prefer GPU OIDN, then OptiX, then the CPU OIDN fallback on Metal. */
DenoiserType automatic_denoiser(const DeviceInfo &device)
{
  if (device.denoisers & DENOISER_OPENIMAGEDENOISE) {
    return DENOISER_OPENIMAGEDENOISE;
  }
  if (device.denoisers & DENOISER_OPTIX) {
    return DENOISER_OPTIX;
  }
#ifdef WITH_OPENIMAGEDENOISE
  return DENOISER_OPENIMAGEDENOISE;
#else
  return DENOISER_NONE;
#endif
}

/* Copies the completed full-frame passes out of the renderer. */
class CaptureDriver : public OutputDriver {
 public:
  bool gpu_display = false;

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
    ++revision_;
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

  bool update_render_tile(const Tile &tile) override
  {
    write_render_tile(tile);
    return has_data();
  }

  int preview(float *rgba, float *depth, size_t pixels, unsigned long long *revision)
  {
    const std::lock_guard<std::mutex> lock(mutex_);
    if (revision_ == *revision || width_ <= 0 || pixels != size_t(width_) * height_) {
      return 1;
    }
    const auto color = passes_.find("combined");
    const auto z = passes_.find("depth");
    if (color == passes_.end() || z == passes_.end()) { return 1; }
    std::memcpy(rgba, color->second.pixels.data(), pixels * 4 * sizeof(float));
    std::memcpy(depth, z->second.pixels.data(), pixels * sizeof(float));
    *revision = revision_;
    return 0;
  }

  void write_render_tile(const Tile &tile) override
  {
    if (gpu_display) { return; }
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
    ++revision_;
  }

 private:
  mutable std::mutex mutex_;
  unsigned long long revision_ = 0;
  int width_ = 0;
  int height_ = 0;
  std::vector<PassSpec> pass_specs_;
  std::map<std::string, PassBuffer> passes_;
};

} /* namespace */

struct PhotaraContext {
  struct MaterialRecord {
    Shader *shader = nullptr;
    PrincipledBsdfNode *bsdf = nullptr;
    MixColorNode *mix = nullptr;
    ImageTextureNode *texture = nullptr;
    PhotaraMaterialDesc material_desc = {};
    float alpha_cutoff = -1.0f;
  };

  enum LightType {
    LIGHT_TYPE_AREA,
    LIGHT_TYPE_SUN,
    LIGHT_TYPE_POINT,
    LIGHT_TYPE_SPOT,
  };

  struct ObjectRecord {
    Object *object = nullptr;
    Mesh *mesh = nullptr;
    bool active = false;
  };

  struct LightRecord {
    Light *light = nullptr;
    Object *object = nullptr;
    Shader *shader = nullptr;
    EmissionNode *emission = nullptr;
    int type = LIGHT_TYPE_AREA;
    bool active = false;
  };

  struct RenderJobRecord {
    unsigned long long id = 0;
    int state = PHOTARA_RENDER_DONE;
    bool active = false;
    bool cancel_requested = false;
    std::string status;
    std::string substatus;
  };

  std::mutex mutex;
  std::string last_error;
  std::string device_name;
  int width = 0;
  int height = 0;
  bool rendered = false;
  bool gpu_display = false;
  PhotaraRenderMode render_mode = PHOTARA_RENDER_REFERENCE;
  photara::realtime::FrameScheduler realtime;
  int job_samples = 0;
  SessionParams active_frame_params;
  PhotaraRenderMode last_frame_mode = PHOTARA_RENDER_REFERENCE;
  photara::realtime::TemporalHistory history;
  int history_seed = 0;
  std::shared_ptr<DenoiseTemporalHistory> independent_dlss_history;

  unique_ptr<Session> session;
  Scene *scene = nullptr;
  SessionParams session_params;
  BufferParams buffer_params;
  CaptureDriver *capture = nullptr;
  BackgroundNode *background_node = nullptr;
  EnvironmentTextureNode *background_texture = nullptr;
  bool background_texture_active = false;
  float3 background_color = make_float3(0.05f, 0.05f, 0.05f);
  float background_strength = 1.0f;

  std::vector<MaterialRecord> materials;
  std::vector<ObjectRecord> objects;
  std::vector<int> free_object_slots;
  std::map<Geometry *, int> mesh_refs;
  std::vector<LightRecord> lights;
  std::vector<int> free_light_slots;

  unsigned long long next_job_id = 1;
  RenderJobRecord render_job;

  ~PhotaraContext()
  {
    if (session != nullptr && render_job.active &&
        render_job.state == PHOTARA_RENDER_RUNNING)
    {
      session->cancel(true);
    }
    if (session) session->wait();
    history.free_gpu();
    session.reset();
  }
};

namespace {

void set_error(PhotaraContext *ctx, const std::string &message)
{
  ctx->last_error = message;
}

void ensure_render_idle(PhotaraContext *ctx)
{
  if (ctx->render_job.active && ctx->render_job.state == PHOTARA_RENDER_RUNNING) {
    throw std::runtime_error("Cannot modify the scene while a render is running");
  }
}

template<typename Function>
int guarded(PhotaraContext *ctx, Function &&function, const bool preserve_history = false)
{
  if (ctx == nullptr) {
    set_create_error("NULL Photara context");
    return -1;
  }
  const std::lock_guard<std::mutex> lock(ctx->mutex);
  try {
    ensure_render_idle(ctx);
    function();
    if (!preserve_history) ctx->history.clear();
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

template<typename Function>
int guarded_read(PhotaraContext *ctx, Function &&function)
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

template<typename Function>
int guarded_index(PhotaraContext *ctx, Function &&function)
{
  if (ctx == nullptr) {
    set_create_error("NULL Photara context");
    return -1;
  }
  const std::lock_guard<std::mutex> lock(ctx->mutex);
  try {
    ensure_render_idle(ctx);
    const int result = function();
    if (result >= 0) ctx->history.clear();
    return result >= 0 ? result : -1;
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

PhotaraContext::MaterialRecord &require_material(PhotaraContext *ctx, const int index)
{
  if (index < 0 || size_t(index) >= ctx->materials.size()) {
    throw std::runtime_error("Material index is out of range");
  }
  return ctx->materials[size_t(index)];
}

Shader *resolve_material_shader(PhotaraContext *ctx, const int index)
{
  if (index < 0) {
    return ctx->scene->default_surface;
  }
  return require_material(ctx, index).shader;
}

PhotaraContext::ObjectRecord &require_object(PhotaraContext *ctx, const int handle)
{
  if (handle < 0 || size_t(handle) >= ctx->objects.size() ||
      !ctx->objects[size_t(handle)].active)
  {
    throw std::runtime_error("Object handle is invalid or was removed");
  }
  return ctx->objects[size_t(handle)];
}

PhotaraContext::LightRecord &require_light(PhotaraContext *ctx, const int handle)
{
  if (handle < 0 || size_t(handle) >= ctx->lights.size() ||
      !ctx->lights[size_t(handle)].active)
  {
    throw std::runtime_error("Light handle is invalid or was removed");
  }
  return ctx->lights[size_t(handle)];
}

int register_object(PhotaraContext *ctx, Object *object, Mesh *mesh)
{
  PhotaraContext::ObjectRecord record;
  record.object = object;
  record.mesh = mesh;
  record.active = true;

  int handle;
  if (!ctx->free_object_slots.empty()) {
    handle = ctx->free_object_slots.back();
    ctx->free_object_slots.pop_back();
    ctx->objects[size_t(handle)] = record;
  }
  else {
    handle = int(ctx->objects.size());
    ctx->objects.push_back(record);
  }
  return handle;
}

int register_light(PhotaraContext *ctx,
                   Light *light,
                   Object *object,
                   Shader *shader,
                   EmissionNode *emission,
                   const int type)
{
  PhotaraContext::LightRecord record;
  record.light = light;
  record.object = object;
  record.shader = shader;
  record.emission = emission;
  record.type = type;
  record.active = true;

  int handle;
  if (!ctx->free_light_slots.empty()) {
    handle = ctx->free_light_slots.back();
    ctx->free_light_slots.pop_back();
    ctx->lights[size_t(handle)] = record;
  }
  else {
    handle = int(ctx->lights.size());
    ctx->lights.push_back(record);
  }
  return handle;
}

void clear_objects_impl(PhotaraContext *ctx)
{
  std::vector<Geometry *> meshes;
  meshes.reserve(ctx->mesh_refs.size());
  for (const auto &entry : ctx->mesh_refs) {
    meshes.push_back(entry.first);
  }

  for (PhotaraContext::ObjectRecord &record : ctx->objects) {
    if (record.active) {
      ctx->scene->delete_node(record.object);
      record = PhotaraContext::ObjectRecord{};
    }
  }
  for (Geometry *mesh : meshes) {
    ctx->scene->delete_node(mesh);
  }

  ctx->objects.clear();
  ctx->free_object_slots.clear();
  ctx->mesh_refs.clear();
}

void clear_lights_impl(PhotaraContext *ctx)
{
  for (PhotaraContext::LightRecord &record : ctx->lights) {
    if (record.active) {
      ctx->scene->delete_node(record.object);
      ctx->scene->delete_node(record.light);
      record = PhotaraContext::LightRecord{};
    }
  }
  ctx->lights.clear();
  ctx->free_light_slots.clear();
}

void apply_area_light_desc(Scene *scene,
                           AreaLight *light,
                           Object *object,
                           EmissionNode *emission,
                           Shader *shader,
                           const PhotaraAreaLightDesc &desc)
{
  if (!(desc.size_u > 0.0f) || !(desc.size_v > 0.0f)) {
    throw std::runtime_error("Area light dimensions must be positive");
  }

  float3 right, up, forward;
  /* Local +Z points from the target towards the light, so emission (-Z)
   * points at the target. */
  make_orthonormal_basis(float3_from_ptr(desc.position) - float3_from_ptr(desc.target),
                         desc.up,
                         right,
                         up,
                         forward);

  light->set_sizeu(desc.size_u);
  light->set_sizev(desc.size_v);
  light->tag_update(scene);
  emission->set_color(float3_from_ptr(desc.color));
  emission->set_strength(desc.strength);
  shader->tag_update(scene);

  object->set_tfm(basis_transform(right, up, forward, float3_from_ptr(desc.position)));
  object->tag_update(scene);
}

void apply_sun_light_desc(Scene *scene,
                          SunLight *light,
                          Object *object,
                          EmissionNode *emission,
                          Shader *shader,
                          const PhotaraSunLightDesc &desc)
{
  /* Cycles' sun direction is -local Z. */
  float3 right, up, forward;
  make_orthonormal_basis(-float3_from_ptr(desc.direction), nullptr, right, up, forward);

  light->set_angle(std::max(desc.angle_degrees, 0.0f) * kPi / 180.0f);
  light->tag_update(scene);
  emission->set_color(float3_from_ptr(desc.color));
  emission->set_strength(desc.strength);
  shader->tag_update(scene);

  object->set_tfm(basis_transform(right, up, forward, make_float3(0.0f, 0.0f, 0.0f)));
  object->tag_update(scene);
}

void validate_point_light_desc(const PhotaraPointLightDesc &desc)
{
  for (int i = 0; i < 3; i++) {
    if (!std::isfinite(desc.position[i])) {
      throw std::runtime_error("Point light position must be finite");
    }
    if (!std::isfinite(desc.color[i]) || desc.color[i] < 0.0f) {
      throw std::runtime_error("Point light color must be finite and non-negative");
    }
  }
  if (!std::isfinite(desc.strength) || desc.strength < 0.0f) {
    throw std::runtime_error("Point light strength must be finite and non-negative");
  }
  if (!std::isfinite(desc.radius) || desc.radius < 0.0f) {
    throw std::runtime_error("Point light radius must be finite and non-negative");
  }
}

void validate_spot_light_desc(const PhotaraSpotLightDesc &desc)
{
  for (int i = 0; i < 3; i++) {
    if (!std::isfinite(desc.position[i]) || !std::isfinite(desc.direction[i]) ||
        !std::isfinite(desc.up[i]))
    {
      throw std::runtime_error("Spot light position/direction/up must be finite");
    }
    if (!std::isfinite(desc.color[i]) || desc.color[i] < 0.0f) {
      throw std::runtime_error("Spot light color must be finite and non-negative");
    }
  }
  if (!(dot(float3_from_ptr(desc.direction), float3_from_ptr(desc.direction)) > 1e-20f)) {
    throw std::runtime_error("Spot light direction must be non-zero");
  }
  if (!std::isfinite(desc.strength) || desc.strength < 0.0f) {
    throw std::runtime_error("Spot light strength must be finite and non-negative");
  }
  if (!std::isfinite(desc.radius) || desc.radius < 0.0f) {
    throw std::runtime_error("Spot light radius must be finite and non-negative");
  }
  if (!std::isfinite(desc.angle_degrees) || desc.angle_degrees <= 0.0f ||
      desc.angle_degrees >= 180.0f)
  {
    throw std::runtime_error("Spot light angle must be in (0, 180) degrees");
  }
  if (!std::isfinite(desc.smooth)) {
    throw std::runtime_error("Spot light smooth must be finite");
  }
}

void apply_point_light_desc(Scene *scene,
                            PointLight *light,
                            Object *object,
                            EmissionNode *emission,
                            Shader *shader,
                            const PhotaraPointLightDesc &desc)
{
  light->set_radius(desc.radius);
  light->set_is_sphere(desc.radius > 0.0f);
  light->set_use_mis(desc.radius > 0.0f);
  light->tag_update(scene);
  emission->set_color(float3_from_ptr(desc.color));
  emission->set_strength(desc.strength);
  shader->tag_update(scene);

  object->set_tfm(basis_transform(make_float3(1.0f, 0.0f, 0.0f),
                                  make_float3(0.0f, 1.0f, 0.0f),
                                  make_float3(0.0f, 0.0f, 1.0f),
                                  float3_from_ptr(desc.position)));
  object->tag_update(scene);
}

void apply_spot_light_desc(Scene *scene,
                           SpotLight *light,
                           Object *object,
                           EmissionNode *emission,
                           Shader *shader,
                           const PhotaraSpotLightDesc &desc)
{
  float3 right, up, forward;
  make_orthonormal_basis(-float3_from_ptr(desc.direction), desc.up, right, up, forward);

  light->set_radius(desc.radius);
  light->set_is_sphere(desc.radius > 0.0f);
  light->set_use_mis(desc.radius > 0.0f);
  light->set_angle(desc.angle_degrees * kPi / 180.0f);
  light->set_smooth(std::clamp(desc.smooth, 1e-4f, 1.0f));
  light->tag_update(scene);
  emission->set_color(float3_from_ptr(desc.color));
  emission->set_strength(desc.strength);
  shader->tag_update(scene);

  object->set_tfm(basis_transform(right, up, forward, float3_from_ptr(desc.position)));
  object->tag_update(scene);
}

void write_mesh_normals(Mesh *mesh, const float *normals)
{
  Attribute *attribute = mesh->attributes.find(ATTR_STD_VERTEX_NORMAL);
  if (attribute == nullptr) {
    attribute = mesh->attributes.add(ATTR_STD_VERTEX_NORMAL);
  }
  packed_normal *dst = attribute->data_for_write<packed_normal>();
  const size_t num_vertices = mesh->num_verts();
  for (size_t i = 0; i < num_vertices; i++) {
    dst[i] = packed_normal(float3_from_ptr(normals + i * 3));
  }
  attribute->modified = true;
}

void write_mesh_uvs(Mesh *mesh, const float *uvs)
{
  Attribute *attribute = mesh->attributes.find(ATTR_STD_UV);
  if (attribute == nullptr) {
    attribute = mesh->attributes.add(ATTR_STD_UV);
  }
  float2 *dst = attribute->data_for_write<float2>();
  const int *indices = mesh->get_triangles().data();
  const size_t num_triangles = mesh->num_triangles();
  for (size_t triangle = 0; triangle < num_triangles; triangle++) {
    for (int corner = 0; corner < 3; corner++) {
      const int vertex = indices[triangle * 3 + corner];
      dst[triangle * 3 + corner] = make_float2(uvs[vertex * 2 + 0], uvs[vertex * 2 + 1]);
    }
  }
  attribute->modified = true;
}

} /* namespace */

extern "C" {

static PhotaraContext *photara_create_impl(const PhotaraOptions *options, bool viewport)
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
    params.headless = !viewport;
    params.preview_update_interval = viewport ? (1.0 / 60.0) : 0.0;
    params.background = !viewport;
    params.stop_when_done = true;
    params.use_neeat = true;
    params.samples = std::max(options->samples > 0 ? options->samples : 64, 1);
    params.use_auto_tile = false;
    params.tile_size = 0;
    params.use_resolution_divider = false;
    params.pixel_size = 1;
    params.threads = 0;
    params.shadingsystem = SHADINGSYSTEM_SVM;

    SceneParams scene_params;
    scene_params.shadingsystem = SHADINGSYSTEM_SVM;
    scene_params.auto_texture_cache = true;
    scene_params.texture_cache_path = "photara_tx";
    /* Dynamic BVH keeps object transforms separate from geometry, so instances
     * can be added and animated after the first render. Static BVH bakes
     * single-user transforms into the mesh and breaks later instancing. */
    scene_params.bvh_type = BVH_TYPE_DYNAMIC;

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
    ctx->history_seed = std::max(options->seed, 0);
    integrator->set_max_bounce(max_bounces);
    integrator->set_max_diffuse_bounce(max_bounces);
    integrator->set_max_glossy_bounce(max_bounces);
    integrator->set_max_transmission_bounce(max_bounces);
    integrator->set_use_adaptive_sampling(options->use_adaptive_sampling != 0);

    if (options->use_denoise) {
      const DenoiserType denoiser = automatic_denoiser(device);
      if (denoiser == DENOISER_NONE) {
        throw std::runtime_error("Denoising is unavailable in this build; disable denoising or enable OpenImageDenoise");
      }
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
        if (pass_filter == nullptr && spec.type == PASS_STABLE_RADIANCE) continue;
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
      /* Leave camera dirty until Scene::update uploads it to the device. */
    }

    /* Neutral default world, so an empty scene is not pitch black. */
    {
      /* Cycles uses scene->default_background as the fallback shader when the
       * Background node has no explicit shader. Configure that shader and leave
       * the socket empty, matching the XML loader. */
      const float3 background_color = make_float3(0.05f, 0.05f, 0.05f);
      ctx->background_color = background_color;
      ctx->background_strength = 1.0f;
      const BackgroundShader background = configure_background_shader(
          scene, background_color, 1.0f);
      ctx->background_node = background.background;
      ctx->background_texture = background.texture;
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

PhotaraContext *photara_create(const PhotaraOptions *options)
{
  return photara_create_impl(options, false);
}
PhotaraContext *photara_create_viewport(const PhotaraOptions *options)
{
  return photara_create_impl(options, true);
}

void photara_destroy(PhotaraContext *ctx)
{
  delete ctx;
}

int photara_add_material(PhotaraContext *ctx, const PhotaraMaterialDesc *desc)
{
  return guarded_index(ctx, [&]() -> int {
    if (desc == nullptr) {
      throw std::runtime_error("NULL material description");
    }

    PrincipledBsdfNode *bsdf = nullptr;
    MixColorNode *mix = nullptr;
    Shader *shader = make_principled_shader(ctx->scene, *desc, &bsdf, &mix);

    PhotaraContext::MaterialRecord record;
    record.shader = shader;
    record.bsdf = bsdf;
    record.mix = mix;
    record.material_desc = *desc;
    ctx->materials.push_back(record);
    return int(ctx->materials.size()) - 1;
  });
}

int photara_update_material(PhotaraContext *ctx,
                            const int material_index,
                            const PhotaraMaterialDesc *desc)
{
  return guarded(ctx, [&]() {
    if (desc == nullptr) {
      throw std::runtime_error("NULL material description");
    }
    PhotaraContext::MaterialRecord &record = require_material(ctx, material_index);
    if (record.bsdf == nullptr) {
      throw std::runtime_error("Material is not editable");
    }
    apply_material_desc(record.bsdf, record.mix, *desc);
    record.material_desc = *desc;
    if (record.alpha_cutoff >= 0 && record.texture) {
      const string filename = record.texture->get_filename().string();
      const string colorspace = record.texture->get_colorspace().string();
      PhotaraTextureDesc texture = {filename.c_str(), colorspace.c_str(),
                                   int(record.texture->get_interpolation()),
                                   int(record.texture->get_extension())};
      auto graph = make_principled_graph(*desc, &texture, &record.bsdf, &record.mix,
                                        &record.texture, record.alpha_cutoff);
      record.shader->set_graph(std::move(graph));
    }
    record.shader->tag_update(ctx->scene);
  });
}

int photara_set_material_base_color_texture(PhotaraContext *ctx,
                                            const int material_index,
                                            const PhotaraTextureDesc *desc)
{
  return guarded(ctx, [&]() {
    if (desc == nullptr) {
      throw std::runtime_error("NULL texture description");
    }
    PhotaraContext::MaterialRecord &record = require_material(ctx, material_index);
    if (record.shader == nullptr || record.mix == nullptr) {
      throw std::runtime_error("Material is not editable");
    }

    /* A finalized shader graph cannot be modified, so rebuild it with the
     * texture node before assigning it to the existing Shader. */
    PrincipledBsdfNode *bsdf = nullptr;
    MixColorNode *mix = nullptr;
    ImageTextureNode *texture = nullptr;
    unique_ptr<ShaderGraph> graph = make_principled_graph(
        record.material_desc, desc, &bsdf, &mix, &texture, record.alpha_cutoff);
    record.shader->set_graph(std::move(graph));
    record.bsdf = bsdf;
    record.mix = mix;
    record.texture = texture;

    record.shader->tag_update(ctx->scene);
    ctx->scene->image_manager->tag_update();
  });
}

int photara_set_material_alpha_clip(PhotaraContext *ctx, const int material, const float cutoff)
{
  return guarded(ctx, [&]() {
    if (!std::isfinite(cutoff) || cutoff > 1) throw std::runtime_error("Invalid alpha cutoff");
    auto &record = require_material(ctx, material);
    if (!record.texture && cutoff >= 0) throw std::runtime_error("Alpha clipping requires a base-color texture");
    if (!record.texture) { record.alpha_cutoff = -1; return; }
    const string filename = record.texture->get_filename().string();
    const string colorspace = record.texture->get_colorspace().string();
    PhotaraTextureDesc texture = {filename.c_str(), colorspace.c_str(),
                                 int(record.texture->get_interpolation()),
                                 int(record.texture->get_extension())};
    auto graph = make_principled_graph(record.material_desc, &texture,
                                      &record.bsdf, &record.mix, &record.texture, cutoff);
    record.shader->set_graph(std::move(graph));
    record.alpha_cutoff = cutoff;
    record.shader->tag_update(ctx->scene);
    ctx->scene->image_manager->tag_update();
  });
}

int photara_get_omm_stats(PhotaraContext *ctx, int *supported, int *enabled,
                         int *meshes, unsigned long long *transparent_states)
{
  return guarded(ctx, [&]() {
    if (!supported || !enabled || !meshes || !transparent_states)
      throw std::runtime_error("NULL OMM diagnostics output");
    *supported = *enabled = *meshes = 0; *transparent_states = 0;
#ifdef WITH_OPTIX
    auto *device = ctx->scene->device;
    if (device->info.type == DEVICE_OPTIX) {
      auto *optix = static_cast<OptiXDevice *>(device);
      *supported = optix->omm_supported; *enabled = optix->omm_enabled;
      for (Geometry *geom : ctx->scene->geometry) {
        if (geom->bvh && geom->bvh->params.bvh_layout == BVH_LAYOUT_OPTIX) {
          auto *bvh = static_cast<BVHOptiX *>(geom->bvh.get());
          if (bvh->omm_array_data) { ++*meshes; *transparent_states += bvh->omm_transparent; }
        }
      }
    }
#endif
  }, true);
}

int photara_clear_material_base_color_texture(PhotaraContext *ctx,
                                              const int material_index)
{
  return guarded(ctx, [&]() {
    PhotaraContext::MaterialRecord &record = require_material(ctx, material_index);
    if (record.shader == nullptr || record.mix == nullptr) {
      throw std::runtime_error("Material is not editable");
    }
    if (record.texture == nullptr) {
      return;
    }

    PrincipledBsdfNode *bsdf = nullptr;
    MixColorNode *mix = nullptr;
    unique_ptr<ShaderGraph> graph = make_principled_graph(
        record.material_desc, nullptr, &bsdf, &mix, nullptr);
    record.shader->set_graph(std::move(graph));
    record.bsdf = bsdf;
    record.mix = mix;
    record.texture = nullptr;
    record.alpha_cutoff = -1;

    record.shader->tag_update(ctx->scene);
    ctx->scene->image_manager->tag_update();
  });
}

int photara_add_gaussian_splats(PhotaraContext *ctx, int count,
    const float *positions, const float *scales, const float *rotations,
    const float *colors_opacity, const float *sh_rest, const float *matrix, int object_id)
{
  return guarded_index(ctx, [&]() -> int {
    if (count <= 0 || !positions || !scales || !rotations || !colors_opacity) {
      throw std::runtime_error("Gaussian arrays must be nonempty");
    }
    auto validate = [](const float *data, size_t n) {
      for (size_t i = 0; i < n; ++i) {
        if (!std::isfinite(data[i])) { throw std::runtime_error("Nonfinite Gaussian data"); }
      }
    };
    validate(positions, size_t(count)*3); validate(scales, size_t(count)*3);
    validate(rotations, size_t(count)*4); validate(colors_opacity, size_t(count)*4);
    if (sh_rest) { validate(sh_rest, size_t(count)*45); }
    for (int i=0;i<count;++i) {
      for (int j=0;j<3;++j) if (scales[i*3+j]<0) throw std::runtime_error("Negative Gaussian scale");
      const float *q=rotations+i*4;
      if (!(q[0]*q[0]+q[1]*q[1]+q[2]*q[2]+q[3]*q[3]>0)) throw std::runtime_error("Zero quaternion");
      if (colors_opacity[i*4+3]<0 || colors_opacity[i*4+3]>1) throw std::runtime_error("Invalid opacity");
    }
    PointCloud *cloud=ctx->scene->create_node<PointCloud>();
    cloud->resize(count); cloud->set_render_as(PointCloud::RENDER_AS_GSPLATS);
    cloud->create_missing_gsplat_attributes();
    auto *p=cloud->get_position_for_write();
    auto *scale=cloud->attributes.find(ATTR_STD_GSPLAT_SCALE)->data_for_write<packed_float3>();
    auto *rotation=cloud->attributes.find(ATTR_STD_GSPLAT_ROTATION)->data_for_write<Quaternion>();
    auto *color=cloud->attributes.find(ATTR_STD_GSPLAT_RADIANCE_BASE)->data_for_write<float4>();
    for (int i=0;i<count;++i) {
      p[i]=make_float3(positions[i*3],positions[i*3+1],positions[i*3+2]);
      scale[i]=make_float3(scales[i*3],scales[i*3+1],scales[i*3+2]);
      const float *q=rotations+i*4;
      const float length=std::sqrt(q[0]*q[0]+q[1]*q[1]+q[2]*q[2]+q[3]*q[3]);
      rotation[i]=make_quaternion(q[0]/length,q[1]/length,q[2]/length,q[3]/length);
      const float *c=colors_opacity+i*4; color[i]=make_float4((c[0]-.5f)/0.28209479177387814f,(c[1]-.5f)/0.28209479177387814f,(c[2]-.5f)/0.28209479177387814f,c[3]);
    }
    if (sh_rest) {
      auto *sh=cloud->attributes.add(ATTR_STD_GSPLAT_RADIANCE_SPHERICAL_HARMONICS_REST)
          ->data_for_write<PackedSphericalHarmonicsRest>();
      for (int i=0;i<count;++i) {
        spherical_harmonics_rest_fill_zero(sh[i]);
        for(int k=0;k<15;++k) {
          const float *c=sh_rest+size_t(i)*45+k*3;
          spherical_harmonics_rest_set_coefficient(sh[i],k,make_float3(c[0],c[1],c[2]));
        }
      }
    }
    array<Node *> shaders; shaders.resize(1); shaders[0]=ctx->scene->default_gsplat;
    cloud->set_used_shaders(shaders); cloud->update_gsplat_radii(); cloud->tag_update(ctx->scene,true);
    Object *object=ctx->scene->create_node<Object>(); object->set_geometry(cloud);
    object->set_tfm(matrix ? transform_from_matrix(matrix) : transform_identity());
    object->set_pass_id(object_id); object->tag_update(ctx->scene);
    const int handle=register_object(ctx,object,nullptr); ctx->mesh_refs[cloud]=1;
    return handle;
  });
}

int photara_add_mesh(PhotaraContext *ctx, const PhotaraMeshDesc *desc)
{
  return guarded_index(ctx, [&]() -> int {
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

    Shader *shader = resolve_material_shader(ctx, desc->material_index);

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
    object->tag_update(ctx->scene);

    const int handle = register_object(ctx, object, mesh);
    ctx->mesh_refs[mesh] = 1;
    return handle;
  });
}

int photara_add_instance(PhotaraContext *ctx,
                         const int source_object,
                         const float *matrix,
                         const int object_id)
{
  return guarded_index(ctx, [&]() -> int {
    const PhotaraContext::ObjectRecord &source = require_object(ctx, source_object);

    Geometry *geometry = source.object->get_geometry();
    Mesh *source_mesh = source.mesh;
    Object *object = ctx->scene->create_node<Object>();
    object->set_geometry(geometry);
    object->set_tfm(matrix ? transform_from_matrix(matrix) : transform_identity());
    object->set_pass_id(object_id);
    object->tag_update(ctx->scene);

    const int handle = register_object(ctx, object, source_mesh);
    ctx->mesh_refs[geometry] += 1;
    return handle;
  });
}

int photara_set_object_transform(PhotaraContext *ctx, const int object, const float matrix[16])
{
  return guarded(ctx, [&]() {
    if (matrix == nullptr) {
      throw std::runtime_error("NULL object transform");
    }
    PhotaraContext::ObjectRecord &record = require_object(ctx, object);
    record.object->set_tfm(transform_from_matrix(matrix));
    record.object->tag_update(ctx->scene);
  });
}

int photara_set_object_visible(PhotaraContext *ctx, const int object, const int visible)
{
  return guarded(ctx, [&]() {
    PhotaraContext::ObjectRecord &record = require_object(ctx, object);
    record.object->set_visibility(visible ? PATH_RAY_VISIBILITY_ALL : 0);
    record.object->tag_update(ctx->scene);
  });
}

int photara_set_object_id(PhotaraContext *ctx, const int object, const int object_id)
{
  return guarded(ctx, [&]() {
    PhotaraContext::ObjectRecord &record = require_object(ctx, object);
    record.object->set_pass_id(object_id);
    record.object->tag_update(ctx->scene);
  });
}

int photara_set_object_material(PhotaraContext *ctx, const int object, const int material_index)
{
  return guarded(ctx, [&]() {
    PhotaraContext::ObjectRecord &record = require_object(ctx, object);
    Shader *shader = resolve_material_shader(ctx, material_index);

    array<Node *> used_shaders;
    used_shaders.push_back_slow(shader);
    if (!record.mesh) { throw std::runtime_error("Gaussian clouds use captured radiance, not mesh materials"); }
    record.mesh->set_used_shaders(used_shaders);
    record.mesh->tag_shader_modified();
    record.mesh->tag_update(ctx->scene, false);
  });
}

int photara_update_mesh(PhotaraContext *ctx,
                        const int object,
                        const PhotaraMeshUpdateDesc *desc)
{
  return guarded(ctx, [&]() {
    if (desc == nullptr) {
      throw std::runtime_error("NULL mesh update description");
    }
    if (desc->positions == nullptr) {
      throw std::runtime_error("Mesh positions are required");
    }
    if (desc->num_vertices <= 0) {
      throw std::runtime_error("Mesh must have at least one vertex");
    }

    PhotaraContext::ObjectRecord &record = require_object(ctx, object);
    Mesh *mesh = record.mesh;
    if (mesh == nullptr) {
      throw std::runtime_error("Object does not reference a mesh");
    }
    if (size_t(desc->num_vertices) != mesh->num_verts()) {
      throw std::runtime_error(
          "Mesh vertex count does not match; use photara_replace_mesh for topology changes");
    }

    packed_float3 *positions = mesh->get_position_for_write();
    for (int i = 0; i < desc->num_vertices; i++) {
      positions[i] = packed_float3(make_float3(desc->positions[i * 3 + 0],
                                               desc->positions[i * 3 + 1],
                                               desc->positions[i * 3 + 2]));
    }
    mesh->tag_position_modified();

    if (desc->normals != nullptr) {
      write_mesh_normals(mesh, desc->normals);
    }
    if (desc->uvs != nullptr) {
      write_mesh_uvs(mesh, desc->uvs);
    }

    mesh->tag_update(ctx->scene, false);
  });
}

int photara_replace_mesh(PhotaraContext *ctx,
                         const int object,
                         const PhotaraMeshReplaceDesc *desc)
{
  return guarded(ctx, [&]() {
    if (desc == nullptr) {
      throw std::runtime_error("NULL mesh replace description");
    }
    if (desc->positions == nullptr || desc->indices == nullptr) {
      throw std::runtime_error("Mesh positions and indices are required");
    }
    if (desc->num_vertices <= 0 || desc->num_triangles <= 0) {
      throw std::runtime_error("Mesh must have at least one vertex and one triangle");
    }
    for (int i = 0; i < desc->num_triangles * 3; i++) {
      if (desc->indices[i] < 0 || desc->indices[i] >= desc->num_vertices) {
        throw std::runtime_error("Mesh triangle index is out of range");
      }
    }

    PhotaraContext::ObjectRecord &record = require_object(ctx, object);
    Mesh *mesh = record.mesh;
    if (mesh == nullptr) {
      throw std::runtime_error("Object does not reference a mesh");
    }

    Shader *shader = nullptr;
    if (desc->material_index >= 0) {
      shader = resolve_material_shader(ctx, desc->material_index);
    }

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

    if (shader != nullptr) {
      array<Node *> used_shaders;
      used_shaders.push_back_slow(shader);
      mesh->set_used_shaders(used_shaders);
    }

    if (mesh->attributes.find(ATTR_STD_VERTEX_NORMAL) != nullptr) {
      mesh->attributes.remove(ATTR_STD_VERTEX_NORMAL);
    }
    if (desc->normals != nullptr) {
      write_mesh_normals(mesh, desc->normals);
    }

    if (mesh->attributes.find(ATTR_STD_UV) != nullptr) {
      mesh->attributes.remove(ATTR_STD_UV);
    }
    if (desc->uvs != nullptr) {
      write_mesh_uvs(mesh, desc->uvs);
    }

    mesh->tag_position_modified();
    mesh->tag_triangles_modified();
    mesh->tag_shader_modified();
    mesh->tag_smooth_modified();
    mesh->tag_update(ctx->scene, true);
    record.object->tag_update(ctx->scene);
  });
}

int photara_remove_object(PhotaraContext *ctx, const int object)
{
  return guarded(ctx, [&]() {
    PhotaraContext::ObjectRecord &record = require_object(ctx, object);
    Geometry *mesh = record.object->get_geometry();

    ctx->scene->delete_node(record.object);
    record = PhotaraContext::ObjectRecord{};
    ctx->free_object_slots.push_back(object);

    const auto mesh_it = ctx->mesh_refs.find(mesh);
    if (mesh_it != ctx->mesh_refs.end()) {
      if (--mesh_it->second <= 0) {
        ctx->scene->delete_node(mesh);
        ctx->mesh_refs.erase(mesh_it);
      }
    }
  });
}

int photara_clear_objects(PhotaraContext *ctx)
{
  return guarded(ctx, [&]() { clear_objects_impl(ctx); });
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
    /* Camera::device_update computes matrices and clears the dirty flag. */
  }, true);
}

int photara_set_camera_matrix(PhotaraContext *ctx,
                              const float matrix[16],
                              const float fov_degrees,
                              const float near_clip,
                              const float far_clip)
{
  return guarded(ctx, [&]() {
    if (matrix == nullptr) {
      throw std::runtime_error("NULL camera matrix");
    }

    Camera *camera = ctx->scene->camera;
    camera->set_camera_type(CAMERA_PERSPECTIVE);
    if (fov_degrees > 0.0f) {
      camera->set_fov(fov_degrees * kPi / 180.0f);
    }
    if (near_clip > 0.0f) {
      camera->set_nearclip(near_clip);
    }
    if (far_clip > 0.0f) {
      camera->set_farclip(far_clip);
    }
    camera->set_full_width(ctx->width);
    camera->set_full_height(ctx->height);
    camera->set_matrix(transform_from_matrix(matrix));
    camera->need_flags_update = true;
    camera->compute_auto_viewplane();
    /* Camera::device_update computes matrices and clears the dirty flag. */
  }, true);
}

int photara_queue_camera_matrix(PhotaraContext *ctx,
                                const unsigned long long job,
                                const float matrix[16],
                                const float fov_degrees,
                                const float near_clip,
                                const float far_clip)
{
  return guarded_read(ctx, [&]() {
    if (ctx->history.limit) {
      throw std::runtime_error("With history_samples, update the camera between completed renders");
    }
    auto &record = ctx->render_job;
    if (ctx->session_params.headless || !record.active || record.id != job ||
        record.cancel_requested || record.state != PHOTARA_RENDER_RUNNING)
    {
      throw std::runtime_error("Camera updates require a valid viewport render job");
    }
    if (!matrix || !std::isfinite(fov_degrees) || fov_degrees <= 0.0f ||
        fov_degrees >= 180.0f || !std::isfinite(near_clip) || near_clip <= 0.0f ||
        !std::isfinite(far_clip) || far_clip <= near_clip)
    {
      throw std::runtime_error("Invalid queued camera parameters");
    }
    for (int i = 0; i < 16; ++i) {
      if (!std::isfinite(matrix[i])) {
        throw std::runtime_error("Camera matrix must be finite");
      }
    }
    Camera *camera = ctx->scene->camera;
    const Transform transform = transform_from_matrix(matrix);
    const int width = ctx->width, height = ctx->height;
    ctx->session->queue_scene_update(
        [camera, transform, fov_degrees, near_clip, far_clip, width, height]() {
          camera->set_camera_type(CAMERA_PERSPECTIVE);
          camera->set_fov(fov_degrees * kPi / 180.0f);
          camera->set_nearclip(near_clip);
          camera->set_farclip(far_clip);
          camera->set_full_width(width);
          camera->set_full_height(height);
          camera->set_matrix(transform);
          camera->need_flags_update = true;
          camera->compute_auto_viewplane();
        },
        ctx->active_frame_params,
        ctx->buffer_params);
    record.state = PHOTARA_RENDER_RUNNING;
    record.status = "Rendering";
    ctx->rendered = false;
  });
}

int photara_add_area_light(PhotaraContext *ctx, const PhotaraAreaLightDesc *desc)
{
  return guarded_index(ctx, [&]() -> int {
    if (desc == nullptr) {
      throw std::runtime_error("NULL area light description");
    }
    if (!(desc->size_u > 0.0f) || !(desc->size_v > 0.0f)) {
      throw std::runtime_error("Area light dimensions must be positive");
    }
    const float3 delta = float3_from_ptr(desc->position) - float3_from_ptr(desc->target);
    if (!(dot(delta, delta) > 1e-20f)) {
      throw std::runtime_error("Area light position and target must differ");
    }

    EmissionShader emission = make_emission_shader(
        ctx->scene, float3_from_ptr(desc->color), desc->strength);

    AreaLight *light = ctx->scene->create_node<AreaLight>();
    light->set_use_mis(true);
    assign_single_shader(light, emission.shader);

    Object *object = ctx->scene->create_node<Object>();
    object->set_geometry(light);
    object->set_visibility(PATH_RAY_VISIBILITY_ALL & ~PATH_RAY_VISIBILITY_CAMERA);

    apply_area_light_desc(
        ctx->scene, light, object, emission.node, emission.shader, *desc);

    return register_light(
        ctx, light, object, emission.shader, emission.node, PhotaraContext::LIGHT_TYPE_AREA);
  });
}

int photara_update_area_light(PhotaraContext *ctx,
                              const int light_handle,
                              const PhotaraAreaLightDesc *desc)
{
  return guarded(ctx, [&]() {
    if (desc == nullptr) {
      throw std::runtime_error("NULL area light description");
    }
    PhotaraContext::LightRecord &record = require_light(ctx, light_handle);
    if (record.type != PhotaraContext::LIGHT_TYPE_AREA) {
      throw std::runtime_error("Light handle is not an area light");
    }
    apply_area_light_desc(ctx->scene,
                          static_cast<AreaLight *>(record.light),
                          record.object,
                          record.emission,
                          record.shader,
                          *desc);
  });
}

int photara_add_sun_light(PhotaraContext *ctx, const PhotaraSunLightDesc *desc)
{
  return guarded_index(ctx, [&]() -> int {
    if (desc == nullptr) {
      throw std::runtime_error("NULL sun light description");
    }
    const float3 direction = float3_from_ptr(desc->direction);
    if (!(dot(direction, direction) > 1e-20f)) {
      throw std::runtime_error("Sun light direction must be non-zero");
    }

    EmissionShader emission = make_emission_shader(
        ctx->scene, float3_from_ptr(desc->color), desc->strength);

    SunLight *light = ctx->scene->create_node<SunLight>();
    light->set_use_mis(true);
    assign_single_shader(light, emission.shader);

    Object *object = ctx->scene->create_node<Object>();
    object->set_geometry(light);
    object->set_visibility(PATH_RAY_VISIBILITY_ALL & ~PATH_RAY_VISIBILITY_CAMERA);

    apply_sun_light_desc(ctx->scene, light, object, emission.node, emission.shader, *desc);

    return register_light(
        ctx, light, object, emission.shader, emission.node, PhotaraContext::LIGHT_TYPE_SUN);
  });
}

int photara_update_sun_light(PhotaraContext *ctx,
                             const int light_handle,
                             const PhotaraSunLightDesc *desc)
{
  return guarded(ctx, [&]() {
    if (desc == nullptr) {
      throw std::runtime_error("NULL sun light description");
    }
    PhotaraContext::LightRecord &record = require_light(ctx, light_handle);
    if (record.type != PhotaraContext::LIGHT_TYPE_SUN) {
      throw std::runtime_error("Light handle is not a sun light");
    }
    apply_sun_light_desc(ctx->scene,
                         static_cast<SunLight *>(record.light),
                         record.object,
                         record.emission,
                         record.shader,
                         *desc);
  });
}

int photara_add_point_light(PhotaraContext *ctx, const PhotaraPointLightDesc *desc)
{
  return guarded_index(ctx, [&]() -> int {
    if (desc == nullptr) {
      throw std::runtime_error("NULL point light description");
    }
    validate_point_light_desc(*desc);

    EmissionShader emission = make_emission_shader(
        ctx->scene, float3_from_ptr(desc->color), desc->strength);

    PointLight *light = ctx->scene->create_node<PointLight>();
    assign_single_shader(light, emission.shader);

    Object *object = ctx->scene->create_node<Object>();
    object->set_geometry(light);
    object->set_visibility(PATH_RAY_VISIBILITY_ALL & ~PATH_RAY_VISIBILITY_CAMERA);

    apply_point_light_desc(ctx->scene, light, object, emission.node, emission.shader, *desc);

    return register_light(
        ctx, light, object, emission.shader, emission.node, PhotaraContext::LIGHT_TYPE_POINT);
  });
}

int photara_update_point_light(PhotaraContext *ctx,
                               const int light_handle,
                               const PhotaraPointLightDesc *desc)
{
  return guarded(ctx, [&]() {
    if (desc == nullptr) {
      throw std::runtime_error("NULL point light description");
    }
    validate_point_light_desc(*desc);
    PhotaraContext::LightRecord &record = require_light(ctx, light_handle);
    if (record.type != PhotaraContext::LIGHT_TYPE_POINT) {
      throw std::runtime_error("Light handle is not a point light");
    }
    apply_point_light_desc(ctx->scene,
                           static_cast<PointLight *>(record.light),
                           record.object,
                           record.emission,
                           record.shader,
                           *desc);
  });
}

int photara_add_spot_light(PhotaraContext *ctx, const PhotaraSpotLightDesc *desc)
{
  return guarded_index(ctx, [&]() -> int {
    if (desc == nullptr) {
      throw std::runtime_error("NULL spot light description");
    }
    validate_spot_light_desc(*desc);

    EmissionShader emission = make_emission_shader(
        ctx->scene, float3_from_ptr(desc->color), desc->strength);

    SpotLight *light = ctx->scene->create_node<SpotLight>();
    assign_single_shader(light, emission.shader);

    Object *object = ctx->scene->create_node<Object>();
    object->set_geometry(light);
    object->set_visibility(PATH_RAY_VISIBILITY_ALL & ~PATH_RAY_VISIBILITY_CAMERA);

    apply_spot_light_desc(ctx->scene, light, object, emission.node, emission.shader, *desc);

    return register_light(
        ctx, light, object, emission.shader, emission.node, PhotaraContext::LIGHT_TYPE_SPOT);
  });
}

int photara_update_spot_light(PhotaraContext *ctx,
                              const int light_handle,
                              const PhotaraSpotLightDesc *desc)
{
  return guarded(ctx, [&]() {
    if (desc == nullptr) {
      throw std::runtime_error("NULL spot light description");
    }
    validate_spot_light_desc(*desc);
    PhotaraContext::LightRecord &record = require_light(ctx, light_handle);
    if (record.type != PhotaraContext::LIGHT_TYPE_SPOT) {
      throw std::runtime_error("Light handle is not a spot light");
    }
    apply_spot_light_desc(ctx->scene,
                          static_cast<SpotLight *>(record.light),
                          record.object,
                          record.emission,
                          record.shader,
                          *desc);
  });
}

int photara_set_light_transform(PhotaraContext *ctx,
                                const int light_handle,
                                const float matrix[16])
{
  return guarded(ctx, [&]() {
    if (matrix == nullptr) {
      throw std::runtime_error("NULL light transform");
    }
    PhotaraContext::LightRecord &record = require_light(ctx, light_handle);
    record.object->set_tfm(transform_from_matrix(matrix));
    record.object->tag_update(ctx->scene);
  });
}

int photara_set_light_visible(PhotaraContext *ctx,
                              const int light_handle,
                              const int visible)
{
  return guarded(ctx, [&]() {
    PhotaraContext::LightRecord &record = require_light(ctx, light_handle);
    record.object->set_visibility(
        visible ? (PATH_RAY_VISIBILITY_ALL & ~PATH_RAY_VISIBILITY_CAMERA) : 0);
    record.object->tag_update(ctx->scene);
  });
}

int photara_remove_light(PhotaraContext *ctx, const int light_handle)
{
  return guarded(ctx, [&]() {
    PhotaraContext::LightRecord &record = require_light(ctx, light_handle);
    ctx->scene->delete_node(record.object);
    ctx->scene->delete_node(record.light);
    record = PhotaraContext::LightRecord{};
    ctx->free_light_slots.push_back(light_handle);
  });
}

int photara_clear_lights(PhotaraContext *ctx)
{
  return guarded(ctx, [&]() { clear_lights_impl(ctx); });
}

int photara_set_background(PhotaraContext *ctx, const float color[3], const float strength)
{
  return guarded(ctx, [&]() {
    if (color == nullptr) {
      throw std::runtime_error("NULL background color");
    }
    const float3 background_color = float3_from_ptr(color);
    ctx->background_color = background_color;
    ctx->background_strength = strength;

    if (ctx->background_node == nullptr) {
      const BackgroundShader background = configure_background_shader(
          ctx->scene, background_color, strength);
      ctx->background_node = background.background;
      ctx->background_texture = background.texture;
      ctx->scene->background->set_use_shader(true);
    }

    if (ctx->background_texture_active) {
      /* The Background Color input is linked to the environment; only the
       * strength is meaningful until the texture is cleared. */
      ctx->background_node->set_strength(strength);
    }
    else {
      ctx->background_node->set_color(background_color);
      ctx->background_node->set_strength(strength);
    }
    ctx->scene->default_background->tag_update(ctx->scene);
  });
}

int photara_set_background_texture(PhotaraContext *ctx, const PhotaraTextureDesc *desc)
{
  return guarded(ctx, [&]() {
    if (desc == nullptr) {
      throw std::runtime_error("NULL texture description");
    }

    const BackgroundShader background = configure_background_environment(
        ctx->scene, *desc, ctx->background_color, ctx->background_strength);
    ctx->background_node = background.background;
    ctx->background_texture = background.texture;
    ctx->background_texture_active = true;
    ctx->scene->background->set_use_shader(true);

    ctx->scene->default_background->tag_update(ctx->scene);
    ctx->scene->image_manager->tag_update();
  });
}

int photara_clear_background_texture(PhotaraContext *ctx)
{
  return guarded(ctx, [&]() {
    if (!ctx->background_texture_active) {
      return;
    }

    const BackgroundShader background = configure_background_shader(
        ctx->scene, ctx->background_color, ctx->background_strength);
    ctx->background_node = background.background;
    ctx->background_texture = nullptr;
    ctx->background_texture_active = false;
    ctx->scene->background->set_use_shader(true);

    ctx->scene->default_background->tag_update(ctx->scene);
    ctx->scene->image_manager->tag_update();
  });
}

int photara_clear_scene(PhotaraContext *ctx)
{
  return guarded(ctx, [&]() {
    clear_objects_impl(ctx);
    clear_lights_impl(ctx);
  });
}

int photara_set_seed(PhotaraContext *ctx, const int seed)
{
  return guarded(ctx, [&]() {
    if (ctx->history_seed != std::max(seed, 0)) ctx->history.clear();
    ctx->history_seed = std::max(seed, 0);
    ctx->scene->integrator->set_seed(std::max(seed, 0));
    ctx->session_params.reset_neeat = true;
  }, true);
}

int photara_set_samples(PhotaraContext *ctx, const int samples)
{
  return guarded(ctx, [&]() {
    if (samples <= 0) {
      throw std::runtime_error("Samples must be positive");
    }
    ctx->session_params.samples = samples;
  }, true);
}

int photara_get_neeat_stats(PhotaraContext *ctx, int *lights, float *max_pdf)
{
  return guarded(ctx, [&]() {
    if (!lights || !max_pdf) throw std::runtime_error("NULL NEE-AT stats output");
    *lights = 0;
    *max_pdf = 0;
    auto &dscene = ctx->scene->dscene;
    const int clusters = dscene.data.integrator.neeat_cluster_count;
    if (clusters) dscene.neeat_distribution.copy_from_device();
    *lights = dscene.data.integrator.neeat_count;
    const int stride = dscene.data.integrator.neeat_cluster_size;
    for (int i = 0; i < *lights; ++i) {
      *max_pdf = std::max(*max_pdf, dscene.neeat_distribution[i / stride].y *
                                      dscene.neeat_lights[i].pdf);
    }
  }, true);
}

int photara_get_neeat_local_stats(PhotaraContext *ctx, int *tiles, float *max_pdf)
{
  return guarded(ctx, [&]() {
    if (!tiles || !max_pdf) throw std::runtime_error("NULL NEE-AT stats output");
    auto &dscene = ctx->scene->dscene;
    const int count = dscene.data.integrator.neeat_cluster_count;
    if (count) dscene.neeat_distribution.copy_from_device();
    *tiles = count ? dscene.data.integrator.neeat_local_count : 0;
    *max_pdf = 0;
    std::vector<float> max_leaf_pdf(count, 0.0f);
    for (int i = 0; i < dscene.data.integrator.neeat_count; ++i) {
      const int cluster = i / dscene.data.integrator.neeat_cluster_size;
      max_leaf_pdf[cluster] = std::max(max_leaf_pdf[cluster], dscene.neeat_lights[i].pdf);
    }
    for (int table = 1; table <= *tiles; ++table) {
      for (int i = 0; i < count; ++i) {
        *max_pdf = std::max(*max_pdf, dscene.neeat_distribution[table * (count + 1) + i].y *
                                        max_leaf_pdf[i]);
      }
    }
  }, true);
}

int photara_set_texture_lod(PhotaraContext *ctx, const PhotaraTextureLOD method)
{
  return guarded(ctx, [&]() {
    if (method != PHOTARA_TEXTURE_LOD_RAY_CONES &&
        method != PHOTARA_TEXTURE_LOD_RAY_DIFFERENTIALS) {
      throw std::runtime_error("Invalid texture LOD method");
    }
    ctx->scene->integrator->set_texture_lod_method(static_cast<TextureLODMethod>(method));
    ctx->scene->integrator->tag_update(ctx->scene, Integrator::UPDATE_ALL);
    ctx->session_params.reset_neeat = true;
  });
}

int photara_get_texture_lod(PhotaraContext *ctx, PhotaraTextureLOD *method)
{
  return guarded(ctx, [&]() {
    if (!method) throw std::runtime_error("NULL texture LOD output");
    *method = static_cast<PhotaraTextureLOD>(ctx->scene->integrator->get_texture_lod_method());
  }, true);
}

int photara_set_nee_candidates(PhotaraContext *ctx, const int count)
{
  return guarded(ctx, [&]() {
    if (count < 1 || count > 16) throw std::runtime_error("NEE candidates must be in [1, 16]");
    ctx->session_params.nee_candidates = count;
  });
}

int photara_get_neeat_update_counts(PhotaraContext *ctx, uint64_t *gpu, uint64_t *cpu)
{
  return guarded(ctx, [&]() {
    if (!gpu || !cpu) throw std::runtime_error("NULL NEE-AT update counts output");
    *gpu = ctx->scene->light_manager->neeat_gpu_updates;
    *cpu = ctx->scene->light_manager->neeat_cpu_updates;
  }, true);
}

int photara_set_neeat_tree_fraction(PhotaraContext *ctx, const float fraction)
{
  return guarded(ctx, [&]() {
    if (!std::isfinite(fraction) || fraction < 0.0f || fraction > 1.0f)
      throw std::runtime_error("NEE-AT tree fraction must be finite and in [0,1]");
    ctx->session_params.neeat_tree_fraction = fraction;
    ctx->session_params.reset_neeat = true;
  });
}

int photara_set_neeat(PhotaraContext *ctx, const int enabled)
{
  return guarded(ctx, [&]() {
    ctx->session_params.use_neeat = enabled != 0;
    ctx->session_params.reset_neeat = true;
  });
}

int photara_set_render_mode(PhotaraContext *ctx,
                            const PhotaraRenderMode mode,
                            const int samples_per_frame)
{
  return guarded(ctx, [&]() {
    if (mode != PHOTARA_RENDER_REFERENCE && mode != PHOTARA_RENDER_REALTIME) {
      throw std::runtime_error("Unknown render mode");
    }
    ctx->realtime.configure(samples_per_frame);
    ctx->render_mode = mode;
    ctx->session_params.reset_neeat = true;
  });
}

int photara_set_history_samples(PhotaraContext *ctx, const int history_samples)
{
  return guarded(ctx, [&]() {
    if (history_samples < 0 || history_samples > 64) {
      throw std::runtime_error("history_samples must be an integer in [0, 64]");
    }
    ctx->history.free_gpu();
    ctx->history.limit = history_samples;
    ctx->history.frame_index = 0;
    ctx->scene->integrator->set_seed(ctx->history_seed);
    if (history_samples) {
      for (const PassType type : {PASS_POSITION, PASS_NORMAL, PASS_ROUGHNESS, PASS_OBJECT_ID}) {
        bool found = false;
        for (const auto &pass : ctx->scene->passes) found |= pass->get_type() == type;
        if (!found) {
          Pass *pass = ctx->scene->create_node<Pass>();
          pass->set_type(type);
        }
      }
      ctx->session->set_frame_reconstruction([ctx](RenderBuffers *buffers, const int samples) {
        ctx->history.reconstruct(buffers, samples, *ctx->scene->camera,
                                 ctx->scene->dscene.data.integrator.pixel_jitter);
      });
    }
    else {
      ctx->session->set_frame_reconstruction({});
      ctx->history.free_gpu();
      ctx->session->set_denoise_temporal_history(nullptr);
      ctx->independent_dlss_history.reset();
    }
  });
}

int photara_get_history_backend(PhotaraContext *ctx, int *gpu, uint64_t *updates,
                                uint64_t *pixel_readback_bytes)
{
  return guarded(ctx, [&]() {
    if (!gpu || !updates || !pixel_readback_bytes) throw std::runtime_error("NULL history backend output");
    /* Preserve the diagnostic ABI: reconstruction is GPU-only, with no pixel readback. */
    *gpu = 1;
    *updates = ctx->history.gpu_updates;
    *pixel_readback_bytes = 0;
  }, true);
}

int photara_reset_history(PhotaraContext *ctx)
{
  return guarded(ctx, [&]() { ctx->history.frame_index = 0; });
}

int photara_get_history_stats(PhotaraContext *ctx, int *stored_frames,
                             uint64_t *reused_pixels, uint64_t *max_history_frames,
                             double *mean_history_frames, uint64_t *memory_bytes, int *sample_seed)
{
  return guarded(ctx, [&]() {
    if (!stored_frames || !reused_pixels || !max_history_frames ||
        !mean_history_frames || !memory_bytes || !sample_seed)
      throw std::runtime_error("NULL history stats output");
    ctx->history.sync_stats();
    *stored_frames = ctx->history.stored_frames();
    *reused_pixels = ctx->history.reused_pixels;
    *max_history_frames = ctx->history.reused_frames;
    *mean_history_frames = ctx->history.mean_history_frames;
    *memory_bytes = ctx->history.memory_bytes();
    *sample_seed = ctx->scene->integrator->get_seed();
  }, true);
}

int photara_render_begin(PhotaraContext *ctx, unsigned long long *job)
{
  if (ctx == nullptr) {
    set_create_error("NULL Photara context");
    return -1;
  }
  if (job == nullptr) {
    set_error(ctx, "NULL render job pointer");
    return -1;
  }

  const std::lock_guard<std::mutex> lock(ctx->mutex);
  try {
    if (ctx->render_job.active && ctx->render_job.state == PHOTARA_RENDER_RUNNING) {
      throw std::runtime_error("A render is already running");
    }

    if (ctx->history.limit && !ctx->rendered) ctx->history.clear();
    ctx->last_error.clear();
    ctx->rendered = false;
    if (ctx->capture != nullptr) {
      ctx->capture->clear();
    }

    /* A previous cancelled job leaves Progress::cancel set; clear it before
     * starting the next render. No render is running at this point. */
    ctx->session->progress.reset();
    SessionParams frame_params = ctx->session_params;
    ctx->job_samples = frame_params.samples;
    if (ctx->render_mode == PHOTARA_RENDER_REALTIME) {
      // Keep the sequence length stable across frames. Only the subset moves.
      frame_params.samples = photara::realtime::FrameScheduler::sample_period;
      frame_params.realtime_frame = true;
      frame_params.use_sample_subset = true;
      frame_params.sample_subset_offset = ctx->realtime.offset();
      frame_params.sample_subset_length = ctx->realtime.samples();
      frame_params.time_limit = 0.0;
      ctx->job_samples = ctx->realtime.samples();
    }
    if (ctx->history.limit) {
      if (!ctx->scene->geometry.empty()) {
        for (const auto &geometry : ctx->scene->geometry) {
          if (geometry->geometry_type != Geometry::MESH && !geometry->is_light()) {
            throw std::runtime_error("history_samples currently supports triangle meshes only");
          }
        }
      }
      if (ctx->scene->integrator->get_denoiser_upscale_factor() != 1.0f) {
        throw std::runtime_error("history_samples currently requires native-resolution denoising");
      }
      const bool plane_history = ctx->job_samples == 1 &&
          ctx->scene->integrator->get_use_denoise() &&
          ctx->scene->integrator->get_denoiser_type() == DENOISER_DLSS;
      if (plane_history) {
        ctx->scene->integrator->set_denoiser_passes(
            DENOISER_PASS_ALBEDO | DENOISER_PASS_NORMAL | DENOISER_PASS_SPECULAR_ALBEDO |
            DENOISER_PASS_ROUGHNESS | DENOISER_PASS_DEPTH | DENOISER_PASS_BACKWARD_MOTION |
            DENOISER_PASS_SPECULAR_MOTION);
      }
      for (const auto &material : ctx->materials) {
        if (material.alpha_cutoff >= 0 || (material.material_desc.transmission > 0 && !plane_history)) {
          throw std::runtime_error("Glass history requires GPU, DLSS, and 1 sample per frame; alpha clipping is unsupported");
        }
      }
      // Local sample zero supplies unfiltered first-hit guides on every frame.
      // Scramble the seed instead of skipping this first local sample with a subset.
      frame_params.samples = ctx->job_samples;
      frame_params.use_sample_subset = false;
      const uint index = uint(ctx->history.frame_index++);
      ctx->scene->integrator->set_seed(index == 0 ? ctx->history_seed :
          int(hash_uint2(uint(ctx->history_seed), index) & 0x7fffffffu));
      ctx->scene->integrator->set_aa_samples(frame_params.samples);
      // The explicit ring owns all cross-frame color influence. Do not also
      // accumulate unbounded opaque DLSS history behind history_samples.
      if (ctx->scene->integrator->get_denoiser_type() == DENOISER_DLSS) {
        ctx->independent_dlss_history = std::make_shared<DenoiseTemporalHistory>();
        ctx->session->set_denoise_temporal_history(ctx->independent_dlss_history);
      }
      else {
        ctx->session->set_denoise_temporal_history(nullptr);
      }
    }
    // Session::update_scene only increases AA samples unless reset_samples is
    // set. Explicitly restore the sequence length when leaving RealTime.
    if (ctx->render_mode == PHOTARA_RENDER_REALTIME ||
        ctx->last_frame_mode == PHOTARA_RENDER_REALTIME) {
      ctx->scene->integrator->set_aa_samples(frame_params.samples);
    }
    ctx->active_frame_params = frame_params;
    ctx->session->reset(frame_params, ctx->buffer_params);
    ctx->session->start();
    ctx->last_frame_mode = ctx->render_mode;
    ctx->session_params.reset_neeat = false;
    if (ctx->render_mode == PHOTARA_RENDER_REALTIME) {
      ctx->realtime.advance();
    }

    const unsigned long long id = ctx->next_job_id++;
    ctx->render_job = PhotaraContext::RenderJobRecord{};
    ctx->render_job.id = id;
    ctx->render_job.state = PHOTARA_RENDER_RUNNING;
    ctx->render_job.active = true;
    ctx->render_job.status = "Rendering";
    *job = id;
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

int photara_render_poll(PhotaraContext *ctx,
                        const unsigned long long job,
                        PhotaraRenderProgress *progress_out)
{
  if (ctx == nullptr) {
    set_create_error("NULL Photara context");
    return -1;
  }
  if (progress_out == nullptr) {
    set_error(ctx, "NULL render progress pointer");
    return -1;
  }

  const std::lock_guard<std::mutex> lock(ctx->mutex);
  try {
    if (!ctx->render_job.active || ctx->render_job.id != job) {
      throw std::runtime_error("Render job is invalid or was superseded");
    }

    PhotaraContext::RenderJobRecord &record = ctx->render_job;
    Progress &progress = ctx->session->progress;

    if (record.state == PHOTARA_RENDER_RUNNING) {
      // Preserve the upload safety rule without waiting on the UI thread.
      if (record.cancel_requested && progress.get_current_sample() > 0) {
        ctx->session->request_cancel();
      }
      if (!ctx->session->is_rendering()) {
        ctx->session->wait();

        if (progress.get_error()) {
          record.state = PHOTARA_RENDER_FAILED;
          record.status = progress.get_error_message();
          record.substatus.clear();
          set_error(ctx, record.status);
          ctx->rendered = false;
        }
        else if (ctx->session->device->have_error()) {
          record.state = PHOTARA_RENDER_FAILED;
          record.status = ctx->session->device->error_message();
          record.substatus.clear();
          set_error(ctx, record.status);
          ctx->rendered = false;
        }
        else if (progress.get_cancel()) {
          record.state = PHOTARA_RENDER_CANCELLED;
          const std::string message = progress.get_cancel_message();
          record.status = message.empty() ? "Cancelled" : message;
          record.substatus.clear();
          ctx->rendered = false;
        }
        else if ((ctx->gpu_display || (ctx->capture != nullptr && ctx->capture->has_data()))) {
          record.state = PHOTARA_RENDER_DONE;
          record.status = "Finished";
          record.substatus.clear();
          ctx->rendered = true;
        }
        else {
          record.state = PHOTARA_RENDER_FAILED;
          record.status = "Renderer finished without producing a render result";
          record.substatus.clear();
          set_error(ctx, record.status);
          ctx->rendered = false;
        }
      }
      else {
        progress.get_status(record.status, record.substatus);
      }
    }

    progress_out->state = record.state;
    progress_out->progress = (record.state == PHOTARA_RENDER_DONE) ?
                                 1.0f :
                                 progress.get_progress();
    progress_out->current_sample = ctx->session->get_current_sample();
    /* A queued view has not rendered any samples yet. Do not advertise the
     * previous camera's sample count while the worker is applying its reset. */
    if (progress_out->current_sample == 0) {
      progress_out->progress = 0.0f;
      progress_out->current_sample = 0;
    }
    progress_out->num_samples = ctx->job_samples > 0 ? ctx->job_samples :
                                                        ctx->session_params.samples;
    progress_out->status = record.status.c_str();
    progress_out->substatus = record.substatus.c_str();
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

int photara_render_request_cancel(PhotaraContext *ctx, const unsigned long long job)
{
  return guarded_read(ctx, [&]() {
    if (!ctx->render_job.active || ctx->render_job.id != job) {
      throw std::runtime_error("Render job is invalid or was superseded");
    }
    ctx->render_job.cancel_requested = true;
    /* Once upload has completed, signal cancellation immediately. Waiting for
     * the next Python poll adds another UI frame to every navigation update.
     * Keep the deferred poll path while the first sample is still uploading. */
    if (ctx->session->progress.get_current_sample() > 0) {
      ctx->session->request_cancel();
    }
  });
}

int photara_render_cancel(PhotaraContext *ctx, const unsigned long long job)
{
  if (ctx == nullptr) {
    set_create_error("NULL Photara context");
    return -1;
  }

  const std::lock_guard<std::mutex> lock(ctx->mutex);
  try {
    if (!ctx->render_job.active || ctx->render_job.id != job) {
      throw std::runtime_error("Render job is invalid or was superseded");
    }

    PhotaraContext::RenderJobRecord &record = ctx->render_job;
    if (record.state == PHOTARA_RENDER_CANCELLED) {
      return 0;
    }
    if (record.state != PHOTARA_RENDER_RUNNING) {
      throw std::runtime_error("Render job is not running");
    }

    Progress &progress = ctx->session->progress;

    /* Cancelling while the scene is still being uploaded can invalidate CUDA
     * module handles. Wait until the render loop has produced its first sample
     * (or finished) before requesting cancellation. */
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (ctx->session->is_rendering() && progress.get_current_sample() == 0 &&
           std::chrono::steady_clock::now() < deadline)
    {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    if (!ctx->session->is_rendering()) {
      if (progress.get_error()) {
        record.state = PHOTARA_RENDER_FAILED;
        record.status = progress.get_error_message();
        set_error(ctx, record.status);
        ctx->rendered = false;
      }
      else if ((ctx->gpu_display || (ctx->capture != nullptr && ctx->capture->has_data()))) {
        record.state = PHOTARA_RENDER_DONE;
        record.status = "Finished";
        ctx->rendered = true;
      }
      else {
        record.state = PHOTARA_RENDER_CANCELLED;
        const std::string message = progress.get_cancel_message();
        record.status = message.empty() ? "Cancelled" : message;
        ctx->rendered = false;
      }
    }
    else {
      /* Use the non-quick path: force-cancelling device work while kernels are
       * still being loaded can invalidate CUDA module handles. The progress
       * cancel flag stops the path tracer at its next safe polling point. */
      ctx->session->cancel(false);
      if (progress.get_error()) {
        record.state = PHOTARA_RENDER_FAILED;
        record.status = progress.get_error_message();
        set_error(ctx, record.status);
      }
      else {
        record.state = PHOTARA_RENDER_CANCELLED;
        const std::string message = progress.get_cancel_message();
        record.status = message.empty() ? "Cancelled" : message;
      }
      ctx->rendered = false;
    }
    record.substatus.clear();
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

int photara_render_finish(PhotaraContext *ctx, const unsigned long long job)
{
  if (ctx == nullptr) {
    set_create_error("NULL Photara context");
    return -1;
  }

  const std::lock_guard<std::mutex> lock(ctx->mutex);
  try {
    if (!ctx->render_job.active || ctx->render_job.id != job) {
      throw std::runtime_error("Render job is invalid or was superseded");
    }
    if (ctx->render_job.state == PHOTARA_RENDER_DONE) {
      return 0;
    }
    if (ctx->render_job.state == PHOTARA_RENDER_RUNNING) {
      throw std::runtime_error("Render job is still running");
    }

    const std::string message = ctx->render_job.status.empty() ?
                                    "Render was cancelled or failed" :
                                    ctx->render_job.status;
    set_error(ctx, message);
    return -1;
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

int photara_render(PhotaraContext *ctx)
{
  unsigned long long job = 0;
  if (photara_render_begin(ctx, &job) != 0) {
    return -1;
  }

  PhotaraRenderProgress progress = {};
  for (;;) {
    if (photara_render_poll(ctx, job, &progress) != 0) {
      return -1;
    }
    if (progress.state != PHOTARA_RENDER_RUNNING) {
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }

  return photara_render_finish(ctx, job);
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

int photara_attach_display(PhotaraContext *ctx, void *pointer)
{
  // Ownership is transferred even when validation fails.
  unique_ptr<DisplayDriver> driver(static_cast<DisplayDriver *>(pointer));
  return guarded(ctx, [&]() {
    if (!driver) { throw std::runtime_error("Missing display driver"); }
    ctx->session->set_display_driver(std::move(driver));
    ctx->gpu_display = true;
    ctx->capture->gpu_display = true;
    ctx->capture->clear();
  });
}

int photara_draw_display(PhotaraContext *ctx)
{
  // Drawing may overlap render work. DisplayDriver handles graphics locks.
  if (!ctx || !ctx->gpu_display) { return -1; }
  ctx->session->draw();
  return 0;
}

int photara_copy_preview(PhotaraContext *ctx, float *rgba, float *depth,
                          size_t pixels, unsigned long long *revision)
{
  int result = 1;
  int error = guarded_read(ctx, [&]() {
    if (!rgba || !depth || !revision) { throw std::runtime_error("NULL preview buffer"); }
    if (ctx->capture) { result = ctx->capture->preview(rgba, depth, pixels, revision); }
  });
  return error ? -1 : result;
}

int photara_resize(PhotaraContext *ctx, int width, int height)
{
  return guarded(ctx, [&]() {
    if (width <= 0 || height <= 0) {
      throw std::runtime_error("Render resolution must be positive");
    }
    ctx->width = ctx->buffer_params.width = ctx->buffer_params.full_width = width;
    ctx->height = ctx->buffer_params.height = ctx->buffer_params.full_height = height;
    ctx->rendered = false;
    if (ctx->capture) {
      ctx->capture->clear();
    }
    Camera *camera = ctx->scene->camera;
    camera->set_full_width(width);
    camera->set_full_height(height);
    camera->compute_auto_viewplane();
    /* Camera::device_update computes matrices and clears the dirty flag. */
  });
}

int photara_configure_film(PhotaraContext *ctx, float exposure, int transparent,
                           int transparent_glass, float roughness_threshold,
                           int filter_type, float filter_width)
{
  return guarded(ctx, [&]() {
    if (!std::isfinite(exposure) || exposure < 0 || filter_type < 0 || filter_type > 2 ||
        !std::isfinite(filter_width) || filter_width <= 0 ||
        !std::isfinite(roughness_threshold) || roughness_threshold < 0 || roughness_threshold > 1) {
      throw std::runtime_error("Invalid film configuration");
    }
    ctx->scene->film->set_exposure(exposure);
    ctx->scene->film->set_filter_type(static_cast<FilterType>(filter_type));
    ctx->scene->film->set_filter_width(filter_width);
    ctx->scene->film->tag_modified();
    ctx->scene->background->set_transparent(transparent != 0);
    ctx->scene->background->set_transparent_glass(transparent && transparent_glass);
    ctx->scene->background->set_transparent_roughness_threshold(roughness_threshold);
    ctx->scene->background->tag_update(ctx->scene);
  });
}

int photara_auto_denoiser(void)
{
  DeviceInfo device;
  std::string error;
  if (!find_device(PHOTARA_DEVICE_AUTO, device, error)) {
    return 0;
  }
  return automatic_denoiser(device);
}

int photara_configure_denoiser(PhotaraContext *ctx, int type, int passes,
                               int prefilter, int quality, int use_gpu)
{
  return guarded(ctx, [&]() {
    if (type != 0 && type != DENOISER_OPTIX && type != DENOISER_OPENIMAGEDENOISE && type != DENOISER_DLSS) {
      throw std::runtime_error("Unsupported denoiser type");
    }
    if (passes != 0 && passes != 1 && passes != 5) {
      throw std::runtime_error("Unsupported denoiser input passes");
    }
    if (prefilter < 1 || prefilter > 3 || quality < 1 || quality > 3) {
      throw std::runtime_error("Invalid denoiser prefilter or quality");
    }
    if (type == DENOISER_DLSS && !(ctx->session_params.device.denoisers & DENOISER_DLSS)) {
      throw std::runtime_error("DLSS Ray Reconstruction is unavailable in this build or on this device");
    }
    if (type == DENOISER_OPTIX && !(ctx->session_params.device.denoisers & DENOISER_OPTIX)) {
      throw std::runtime_error("OptiX denoising is unavailable on this device");
    }
#ifndef WITH_OPENIMAGEDENOISE
    if (type == DENOISER_OPENIMAGEDENOISE) {
      throw std::runtime_error("This build does not include OpenImageDenoise");
    }
#endif
    Integrator *integrator = ctx->scene->integrator;
    if (type != 0) {
      integrator->set_denoiser_type(static_cast<DenoiserType>(type));
    }
    integrator->set_denoiser_passes(passes);
    integrator->set_denoiser_prefilter(static_cast<DenoiserPrefilter>(prefilter));
    integrator->set_denoiser_quality(static_cast<DenoiserQuality>(quality));
    integrator->set_denoise_use_gpu(use_gpu != 0);
    integrator->tag_update(ctx->scene, Integrator::UPDATE_ALL);
  });
}

int photara_configure_sampling(PhotaraContext *ctx, float threshold, int min_samples,
                               double time_limit, int denoise_start_sample)
{
  return guarded(ctx, [&]() {
    if (!std::isfinite(threshold) || threshold < 0.0f || threshold > 1.0f ||
        min_samples < 0 || !std::isfinite(time_limit) || time_limit < 0.0 ||
        denoise_start_sample < 0) {
      throw std::runtime_error("Invalid sampling settings");
    }
    Integrator *integrator = ctx->scene->integrator;
    integrator->set_adaptive_threshold(threshold);
    integrator->set_adaptive_min_samples(min_samples);
    integrator->set_denoise_start_sample(denoise_start_sample);
    ctx->session_params.time_limit = time_limit;
    ctx->session->set_time_limit(time_limit);
    integrator->tag_update(ctx->scene, Integrator::UPDATE_ALL);
  });
}

int photara_configure_dlss(PhotaraContext *ctx, float factor)
{
  return guarded(ctx, [&]() {
    if (!(factor >= 1.0f && factor <= 3.0f)) {
      throw std::runtime_error("DLSS upscale factor must be between 1 and 3");
    }
    Integrator *integrator = ctx->scene->integrator;
    if (factor != 1.0f && integrator->get_denoiser_type() != DENOISER_DLSS) {
      throw std::runtime_error("Upscaling requires the DLSS denoiser");
    }
    integrator->set_denoiser_upscale_factor(factor);
    integrator->tag_update(ctx->scene, Integrator::UPDATE_ALL);
  });
}

int photara_copy_pass(PhotaraContext *ctx, const char *name, float *dst, const size_t dst_count)
{
  return guarded_read(ctx, [&]() {
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
