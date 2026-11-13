/* SPDX-License-Identifier: Apache-2.0 */
/* Windows viewer example: OptiX frame lifetime + Vulkan presentation. */
#define NOMINMAX
// clang-format off
#include <windows.h>
#include <windowsx.h>
#include <mmsystem.h>
// clang-format on

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <thread>
#include <vector>

#include "app/photara_xml.h"
#include "app/viewer_presenter.h"
#include "device/device.h"
#include "realtime/frame_scheduler.h"
#include "scene/camera.h"
#include "scene/film.h"
#include "scene/integrator.h"
#include "scene/pass.h"
#include "scene/scene.h"
#include "session/display_driver.h"
#include "session/session.h"
#include "util/log.h"
#include "util/path.h"
#include "util/transform.h"

using namespace ccl;

namespace {
struct TimerResolution {
  bool active = timeBeginPeriod(1) == TIMERR_NOERROR;
  ~TimerResolution()
  {
    if (active) {
      timeEndPeriod(1);
    }
  }
};

/* One persistent waiter signals actual Session completion, avoiding both a
 * per-frame thread allocation and periodic UI polling. Session outlives it. */
class RenderCompletion {
  Session &session_;
  HANDLE event_ = nullptr;
  std::mutex mutex_;
  std::condition_variable changed_;
  uint64_t generation_ = 0;
  bool stopping_ = false;
  std::thread thread_;

 public:
  explicit RenderCompletion(Session &session) : session_(session)
  {
    event_ = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (!event_) {
      throw std::runtime_error("Cannot create render completion event");
    }
    try {
      thread_ = std::thread([this] {
        std::unique_lock<std::mutex> lock(mutex_);
        uint64_t observed = 0;
        while (true) {
          changed_.wait(lock, [&] { return stopping_ || observed != generation_; });
          if (stopping_) {
            return;
          }
          observed = generation_;
          lock.unlock();
          session_.wait();
          SetEvent(event_);
          lock.lock();
        }
      });
    }
    catch (...) {
      CloseHandle(event_);
      throw;
    }
  }
  ~RenderCompletion()
  {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stopping_ = true;
    }
    changed_.notify_one();
    if (thread_.joinable()) {
      thread_.join();
    }
    CloseHandle(event_);
  }
  HANDLE event() const
  {
    return event_;
  }
  void submitted()
  {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      ++generation_;
    }
    changed_.notify_one();
  }
};

class ViewerDisplay final : public DisplayDriver {
  ViewerPresenter &presenter_;
  int size_;

 public:
  int updates = 0;
  bool fallback_requested = false;
  std::chrono::steady_clock::time_point ready;
  explicit ViewerDisplay(ViewerPresenter &presenter, int size) : presenter_(presenter), size_(size)
  {
    graphics_interop_buffer_.assign(GraphicsInteropDevice::VULKAN,
                                    reinterpret_cast<int64_t>(presenter_.export_handle()),
                                    presenter_.buffer_bytes());
  }
  GraphicsInteropDevice graphics_interop_get_device() override
  {
    GraphicsInteropDevice device;
    device.type = GraphicsInteropDevice::VULKAN;
    const auto &uuid = presenter_.device_uuid();
    device.uuid.assign(uuid.begin(), uuid.end());
    return device;
  }
  bool update_begin(const Params &, int width, int height) override
  {
    if (width != size_ || height != size_) {
      return false;
    }
    presenter_.interop_mutex.lock();
    try {
      presenter_.wait_for_reader();
    }
    catch (...) {
      presenter_.interop_mutex.unlock();
      throw;
    }
    return true;
  }
  void update_end() override
  {
    ready = std::chrono::steady_clock::now();
    ++updates;
    presenter_.interop_mutex.unlock();
  }
  half4 *map_texture_buffer() override
  {
    // Fail visibly rather than silently enable a CPU pixel-copy fallback.
    fallback_requested = true;
    return nullptr;
  }
  void unmap_texture_buffer() override {}
  void next_tile_begin() override {}
  void zero() override
  {
    graphics_interop_buffer_.zero();
  }
  void draw(const Params &) override {}
};

struct Viewer {
  unique_ptr<Session> session;
  unique_ptr<RenderCompletion> completion;
  ViewerDisplay *display = nullptr;
  SessionParams params;
  BufferParams buffers;
  photara::realtime::FrameScheduler frames;
  Transform original_camera;
  float3 camera_offset = make_float3(0, 0, 0);
  std::vector<unsigned char> bgra;
  bool have_image = false;
  ViewerPresenter presenter;
  bool present_pending = false;
  bool realtime = true, paused = false, dirty = true, running = false;
  bool closing = false, dragging = false;
  bool smoke_cycle = false;
  bool use_neeat = true;
  int texture_lod_override = -1;
  int mouse_x = 0, mouse_y = 0;
  int size = 512, spp = 1, reference_samples = 256;
  int completed = 0, smoke_frames = 0;
  double last_ms = 0;
  uint64_t title_updates = 0, rendered_frames_presented = 0;
  bool completed_frame_pending = false;
  bool reset_frame_history = true;
  bool dlss = false, camera_sweep = false, reset_temporal = true;
  bool submitted_temporal_reset = true;
  bool stable_mirror_guide = false;
  int camera_cut_frame = -1;
  float sweep_base_x = 0.0f;
  std::shared_ptr<DenoiseTemporalHistory> dlss_history;
  uint64_t submitted_dlss_frames = 0;
  struct PendingFrame {
    int index = 0, samples = 0;
    bool realtime = true;
    double observed_ms = 0, worker_ms = 0;
    std::chrono::steady_clock::time_point submitted;
  } pending_frame;
  std::chrono::steady_clock::time_point last_title;
  std::ofstream metrics;
  string metrics_path;
  string scene_path = "examples/scene_cornell_box.xml";
  string screenshot = "viewer_capture.bmp";
  std::chrono::steady_clock::time_point started;
  const std::chrono::steady_clock::time_point presentation_clock = std::chrono::steady_clock::now();
  HWND window = nullptr;

  ~Viewer()
  {
    if (session && session->is_rendering()) {
      session->cancel(true);
      session->wait();
    }
    completion.reset();
    session.reset();
  }

  void initialize()
  {
    const auto devices = Device::available_devices(DEVICE_MASK(DEVICE_OPTIX));
    for (const auto &device : devices) {
      if (device.meets_driver_requirement) {
        params.device = device;
        break;
      }
    }
    if (params.device.type == DEVICE_NONE) {
      throw std::runtime_error("No supported OptiX device found");
    }
    params.headless = false;
    params.denoise_device = params.device;
    params.background = false;
    params.stop_when_done = true;
    params.tile_size = 0;
    params.use_auto_tile = false;
    params.use_resolution_divider = false;
    params.samples = reference_samples;
    SceneParams scene_params;
    scene_params.auto_texture_cache = true;
    scene_params.texture_cache_path = "photara_tx";
    session = make_unique<Session>(params, scene_params);
    if (!path_exists(scene_path)) {
      scene_path = path_get(scene_path);
    }
    if (!path_exists(scene_path)) {
      throw std::runtime_error("Scene not found: " + scene_path);
    }
    xml_read_file(session->scene.get(), scene_path.c_str());
    auto *scene = session->scene.get();
    if (texture_lod_override >= 0) {
      scene->integrator->set_texture_lod_method(TextureLODMethod(texture_lod_override));
    }
    scene->camera->set_full_width(size);
    scene->camera->set_full_height(size);
    scene->camera->compute_auto_viewplane();
    original_camera = scene->camera->get_matrix();
    scene->integrator->set_use_adaptive_sampling(false);
    scene->integrator->set_use_denoise(false);
    if (dlss) {
      if (!(params.device.denoisers & DENOISER_DLSS)) {
        throw std::runtime_error("DLSS Ray Reconstruction is unavailable on this device/build");
      }
      scene->integrator->set_use_denoise(true);
      scene->integrator->set_denoiser_type(DENOISER_DLSS);
      scene->integrator->set_denoiser_passes(
          DENOISER_PASS_ALBEDO | DENOISER_PASS_SPECULAR_ALBEDO | DENOISER_PASS_NORMAL |
          DENOISER_PASS_ROUGHNESS | DENOISER_PASS_DEPTH | DENOISER_PASS_BACKWARD_MOTION |
          DENOISER_PASS_SPECULAR_MOTION);
      scene->integrator->set_denoise_start_sample(1);
      scene->integrator->set_denoiser_upscale_factor(1.0f);
      scene->integrator->set_use_pixel_jitter(true);
      dlss_history = std::make_shared<DenoiseTemporalHistory>();
      session->set_denoise_temporal_history(dlss_history);
      array<Transform> motion;
      motion.push_back_slow(original_camera);
      scene->camera->set_motion(motion);
    }
    if (stable_mirror_guide && !dlss) {
      throw std::runtime_error("--stable-mirror-guide requires --denoiser DLSS");
    }
    auto *pass = scene->create_node<Pass>();
    pass->set_name(ustring("combined"));
    pass->set_type(PASS_COMBINED);
    if (dlss) pass->set_mode(PassMode::DENOISED);
    buffers.width = buffers.full_width = size;
    buffers.height = buffers.full_height = size;
    frames.configure(spp);
    if (!metrics_path.empty()) {
      metrics.open(metrics_path);
      if (!metrics) {
        throw std::runtime_error("Cannot open metrics output");
      }
      metrics << "frame,mode,spp,observed_ms,worker_to_gpu_display_ms,cpu_pixel_readback_bytes,"
                 "present_attempts,presented_frames,surface_queries,title_updates,rendered_frames_"
                 "presented,present_backpressure_ms,history_samples,present_timestamp_ms,dlss,temporal_reset,camera_x,phase,dlss_frames,dlss_resets,dlss_reuses,stable_mirror_guide\n";
    }
  }

  void restart()
  {
    dirty = true;
    frames.configure(spp);
    if (running) {
      session->request_cancel();
    }
  }

  void title(bool force = true)
  {
    const auto now = std::chrono::steady_clock::now();
    if (!force && now - last_title < std::chrono::milliseconds(250)) {
      return;
    }
    last_title = now;
    char text[512];
    std::snprintf(text,
                  sizeof(text),
                  "Photara Viewer | %s | %d spp/frame | %u %s | %.1f ms worker+GPU display | %s | "
                  "1 RT / 2 Ref | +/- spp | Space pause | WASD/QE move | R reset | F12 BMP",
                  dlss ? "RealTime DLSS RR" : (realtime ? "RealTime baseline" : "Reference"),
                  realtime ? spp : reference_samples,
                  presenter.history_samples(),
                  dlss ? "display spp (DLSS owns history)" : "history spp",
                  last_ms,
                  paused ? "Paused" : (!running && !dirty && !realtime ? "Complete" : "Running"));
    SetWindowTextA(window, text);
    ++title_updates;
  }

  void save()
  {
    if (!have_image) {
      return;
    }
    bgra = presenter.capture();
    BITMAPFILEHEADER file = {};
    file.bfType = 0x4d42;
    file.bfOffBits = sizeof(file) + sizeof(BITMAPINFOHEADER);
    file.bfSize = DWORD(file.bfOffBits + bgra.size());
    BITMAPINFOHEADER info = {};
    info.biSize = sizeof(info);
    info.biWidth = size;
    info.biHeight = -size;
    info.biPlanes = 1;
    info.biBitCount = 32;
    info.biCompression = BI_RGB;
    std::ofstream stream(screenshot, std::ios::binary);
    stream.write(reinterpret_cast<const char *>(&file), sizeof(file));
    stream.write(reinterpret_cast<const char *>(&info), sizeof(info));
    stream.write(reinterpret_cast<const char *>(bgra.data()), bgra.size());
    if (!stream) {
      throw std::runtime_error("Failed to save screenshot: " + screenshot);
    }
  }

  void try_present()
  {
    if (!presenter.present(window)) {
      return;
    }
    present_pending = false;
    if (!completed_frame_pending) {
      return;
    }  // Resize-only redraw.
    completed_frame_pending = false;
    ++rendered_frames_presented;
    if (metrics) {
      const auto &frame = pending_frame;
      const double submit_ms = std::chrono::duration<double, std::milli>(
                                   std::chrono::steady_clock::now() - frame.submitted)
                                   .count();
      metrics << frame.index << ',' << (frame.realtime ? "REALTIME" : "REFERENCE") << ','
              << frame.samples << ',' << frame.observed_ms << ',' << frame.worker_ms << ",0,"
              << presenter.present_attempts << ',' << presenter.presented_frames << ','
              << presenter.surface_queries << ',' << title_updates << ','
              << rendered_frames_presented << ',' << submit_ms - frame.observed_ms << ','
              << presenter.history_samples() << ','
              << std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - presentation_clock).count()
              << ',' << dlss << ',' << submitted_temporal_reset << ',' << camera_offset.x << ','
              << (camera_sweep && frame.index >= 60 && frame.index < 240 ? "motion" : "static")
              << ',' << (dlss_history ? dlss_history->dlss_frames : 0)
              << ',' << (dlss_history ? dlss_history->dlss_resets : 0)
              << ',' << (dlss_history ? dlss_history->dlss_reuses : 0)
              << ',' << (stable_mirror_guide && frame.realtime && frame.samples == 1)
              << '\n';
      if ((rendered_frames_presented & 63) == 0) {
        metrics.flush();
      }
    }
  }

  void tick()
  {
    if (present_pending && !IsIconic(window)) {
      try_present();
    }
    if (running) {
      if (session->is_rendering()) {
        return;
      }
      session->wait();
      running = false;
      if (display->fallback_requested) {
        throw std::runtime_error(
            "CUDA/Vulkan interop failed; viewer cannot use a CPU display fallback");
      }
      if (session->progress.get_error() || session->device->have_error()) {
        throw std::runtime_error(session->progress.get_error() ?
                                     session->progress.get_error_message() :
                                     session->device->error_message());
      }
      if (!dirty && !closing && !session->progress.get_cancel()) {
        if (dlss_history && dlss_history->dlss_frames <= submitted_dlss_frames) {
          throw std::runtime_error("Frame completed without a successful DLSS RR evaluation");
        }
        if (!display->updates) {
          throw std::runtime_error("Render finished without a GPU display update");
        }
        have_image = true;
        last_ms = std::chrono::duration<double, std::milli>(display->ready - started).count();
        const double observed_ms = std::chrono::duration<double, std::milli>(
                                       std::chrono::steady_clock::now() - started)
                                       .count();
        pending_frame = {completed,
                         realtime ? spp : reference_samples,
                         realtime,
                         observed_ms,
                         last_ms,
                         started};
        completed_frame_pending = true;
        // DLSS owns temporal reprojection/rejection; never average its output again.
        presenter.completed_frame(dlss || reset_frame_history, realtime ? spp : reference_samples);
        present_pending = true;
        ++completed;
        try_present();
        // Optional deterministic stress sequence for mode/resize/ownership tests.
        if (smoke_cycle) {
          if (completed == 8 || completed == 24) {
            reset_temporal = true;
            realtime = false;
            restart();
          }
          if (completed == 9 || completed == 25) {
            reset_temporal = true;
            realtime = true;
            restart();
          }
          if (completed == 16) {
            camera_offset.x = .1f;
            restart();
            SetWindowPos(
                window, nullptr, 0, 0, 640, 480, SWP_NOMOVE | SWP_NOZORDER | SWP_NOACTIVATE);
          }
        }
        title(!realtime);
      }
    }
    if (closing) {
      DestroyWindow(window);
      return;
    }
    if (smoke_frames && completed >= smoke_frames) {
      if (present_pending) {
        return;
      }  // Include the final frame's Present submission.
      if (rendered_frames_presented != uint64_t(completed)) {
        throw std::runtime_error("A completed frame was not submitted for presentation");
      }
      save();
      PostMessage(window, WM_CLOSE, 0, 0);
      return;
    }
    // Never overwrite a completed image before its Present submission succeeds.
    if (present_pending) {
      return;
    }
    if (paused || IsIconic(window) || (!realtime && !dirty)) {
      return;
    }
    auto *scene = session->scene.get();
    if (camera_sweep) {
      const float offset = sweep_base_x + ((completed >= 60 && completed < 240) ?
          0.6f * std::sin(float(completed - 60) * 2.0f * 3.14159265f / 180.0f) : 0.0f);
      if (camera_offset.x != offset) {
        camera_offset.x = offset;
        dirty = true;
      }
    }
    if (completed == camera_cut_frame) {
      camera_offset.x += 1.0f;
      sweep_base_x += 1.0f;
      dirty = true;
      reset_temporal = true;
    }
    if (dirty) {
      scene->camera->set_matrix(transform_translate(camera_offset) * original_camera);
      scene->camera->need_flags_update = true;
    }
    SessionParams frame = params;
    scene->film->set_denoising_stable_mirror_guide(stable_mirror_guide && realtime && spp == 1);
    reset_frame_history = dirty || !realtime;
    if (dlss_history) {
      // Camera-only movement keeps valid history; explicit cuts/mode changes reset it.
      dlss_history->reset = reset_temporal || !realtime;
      submitted_dlss_frames = dlss_history->dlss_frames;
    }
    submitted_temporal_reset = reset_temporal || !realtime;
    reset_temporal = false;
    frame.samples = realtime ? frames.sample_period : reference_samples;
    frame.realtime_frame = realtime;
    frame.use_neeat = use_neeat;
    frame.reset_neeat = dirty;
    frame.use_sample_subset = realtime;
    frame.sample_subset_offset = realtime ? frames.offset() : 0;
    frame.sample_subset_length = realtime ? spp : reference_samples;
    scene->integrator->set_aa_samples(frame.samples);
    display->updates = 0;
    session->progress.reset();
    started = std::chrono::steady_clock::now();
    session->reset(frame, buffers);
    session->start();
    completion->submitted();
    if (realtime) {
      frames.advance();
    }
    running = true;
    dirty = false;
    // Keep the submitted reset flag until its frame has been reported.
  }
};

Viewer *viewer = nullptr;

LRESULT CALLBACK window_proc(HWND hwnd, UINT message, WPARAM key, LPARAM data)
{
  auto &v = *viewer;
  try {
    switch (message) {
      case WM_SIZE:
        v.presenter.request_resize();
        v.present_pending = v.have_image;
        return 0;
      case WM_CLOSE:
        v.closing = true;
        if (v.running) {
          v.session->request_cancel();
        }
        else {
          DestroyWindow(hwnd);
        }
        return 0;
      case WM_DESTROY:
        PostQuitMessage(0);
        return 0;
      case WM_KEYDOWN:
        switch (key) {
          case VK_ESCAPE:
            PostMessage(hwnd, WM_CLOSE, 0, 0);
            return 0;
          case '1':
            v.reset_temporal = true;
            v.realtime = true;
            break;
          case '2':
            v.reset_temporal = true;
            v.realtime = false;
            break;
          case VK_SPACE:
            v.paused = !v.paused;
            v.title();
            return 0;
          case VK_OEM_PLUS:
          case VK_ADD:
            if (v.dlss) return 0;
            v.reset_temporal = true;
            v.spp = std::min(64, v.spp * 2);
            break;
          case VK_OEM_MINUS:
          case VK_SUBTRACT:
            if (v.dlss) return 0;
            v.reset_temporal = true;
            v.spp = std::max(1, v.spp / 2);
            break;
          case 'A':
            v.camera_offset.x -= .1f;
            break;
          case 'D':
            v.camera_offset.x += .1f;
            break;
          case 'W':
            v.camera_offset.z += .1f;
            break;
          case 'S':
            v.camera_offset.z -= .1f;
            break;
          case 'Q':
            v.camera_offset.y -= .1f;
            break;
          case 'E':
            v.camera_offset.y += .1f;
            break;
          case 'R':
            v.reset_temporal = true;
            v.camera_offset = make_float3(0, 0, 0);
            break;
          case VK_F12:
            v.save();
            return 0;
          default:
            return 0;
        }
        v.restart();
        v.title();
        return 0;
      case WM_LBUTTONDOWN:
        v.dragging = true;
        v.mouse_x = GET_X_LPARAM(data);
        v.mouse_y = GET_Y_LPARAM(data);
        SetCapture(hwnd);
        return 0;
      case WM_LBUTTONUP:
        v.dragging = false;
        ReleaseCapture();
        return 0;
      case WM_CAPTURECHANGED:
        v.dragging = false;
        return 0;
      case WM_MOUSEMOVE:
        if (v.dragging) {
          const int x = GET_X_LPARAM(data), y = GET_Y_LPARAM(data);
          v.camera_offset.x -= float(x - v.mouse_x) * .005f;
          v.camera_offset.y += float(y - v.mouse_y) * .005f;
          v.mouse_x = x;
          v.mouse_y = y;
          v.restart();
        }
        return 0;
      case WM_ERASEBKGND:
        return 1;
      case WM_PAINT: {
        PAINTSTRUCT paint;
        BeginPaint(hwnd, &paint);
        EndPaint(hwnd, &paint);
        // Vulkan keeps the presented image; painting does not create a new frame.
        return 0;
      }
    }
  }
  catch (const std::exception &error) {
    MessageBoxA(hwnd, error.what(), "Photara Viewer error", MB_ICONERROR);
    DestroyWindow(hwnd);
    PostQuitMessage(1);
    return 0;
  }
  return DefWindowProc(hwnd, message, key, data);
}
}  // namespace

int main(int argc, char **argv)
{
  log_init(nullptr);
  path_init();
  TimerResolution timer_resolution;
  Viewer app;
  viewer = &app;
  try {
    for (int i = 1; i < argc; ++i) {
      const string arg = argv[i];
      auto value = [&]() -> string {
        if (++i == argc) {
          throw std::runtime_error("Missing value for " + arg);
        }
        return argv[i];
      };
      if (arg == "--size") {
        app.size = std::stoi(value());
      }
      else if (arg == "--spp") {
        app.spp = std::stoi(value());
      }
      else if (arg == "--denoiser") {
        const string denoiser = value();
        if (denoiser != "NONE" && denoiser != "DLSS") {
          throw std::runtime_error("Viewer denoiser must be NONE or DLSS");
        }
        app.dlss = denoiser == "DLSS";
      }
      else if (arg == "--camera-sweep") {
        app.camera_sweep = true;
      }
      else if (arg == "--stable-mirror-guide") {
        app.stable_mirror_guide = true;
      }
      else if (arg == "--camera-cut-frame") {
        app.camera_cut_frame = std::stoi(value());
      }
      else if (arg == "--camera-x") {
        app.camera_offset.x = std::stof(value());
      }
      else if (arg == "--reference-samples") {
        app.reference_samples = std::stoi(value());
      }
      else if (arg == "--texture-lod") {
        const string method = value();
        if (method != "RAY_CONES" && method != "RAY_DIFFERENTIALS") {
          throw std::runtime_error("Texture LOD must be RAY_CONES or RAY_DIFFERENTIALS");
        }
        app.texture_lod_override = method == "RAY_CONES" ? 0 : 1;
      }
      else if (arg == "--light-sampler") {
        const string sampler = value();
        if (sampler != "NEE_AT" && sampler != "LIGHT_TREE") {
          throw std::runtime_error("Light sampler must be NEE_AT or LIGHT_TREE");
        }
        app.use_neeat = sampler == "NEE_AT";
      }
      else if (arg == "--smoke-cycle") {
        app.smoke_cycle = true;
      }
      else if (arg == "--smoke-frames") {
        app.smoke_frames = std::stoi(value());
      }
      else if (arg == "--metrics") {
        app.metrics_path = value();
      }
      else if (arg == "--screenshot") {
        app.screenshot = value();
      }
      else if (arg == "--mode") {
        const string mode = value();
        if (mode != "REALTIME" && mode != "REFERENCE") {
          throw std::runtime_error("Invalid mode");
        }
        app.realtime = mode == "REALTIME";
      }
      else if (!arg.empty() && arg[0] != '-') {
        app.scene_path = arg;
      }
      else {
        throw std::runtime_error("Unknown option: " + arg);
      }
    }
    if (app.size < 16 || app.size > 4096 || app.spp < 1 || app.spp > 64 ||
        app.reference_samples < 1 || app.smoke_frames < 0 ||
        (!app.realtime && app.smoke_frames > 1) ||
        (app.smoke_cycle && (!app.realtime || app.smoke_frames < 26)))
    {
      throw std::runtime_error("Invalid size, sample budget or smoke frame count");
    }
    if (app.dlss && app.spp != 1) {
      throw std::runtime_error("DLSS realtime viewer currently requires --spp 1");
    }
    app.initialize();
    WNDCLASSA wc = {};
    wc.lpfnWndProc = window_proc;
    wc.hInstance = GetModuleHandle(nullptr);
    wc.lpszClassName = "PhotaraViewer";
    wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
    RegisterClassA(&wc);
    app.window = CreateWindowA(wc.lpszClassName,
                               "Photara Viewer",
                               WS_OVERLAPPEDWINDOW,
                               CW_USEDEFAULT,
                               CW_USEDEFAULT,
                               app.size + 40,
                               app.size + 80,
                               nullptr,
                               nullptr,
                               wc.hInstance,
                               nullptr);
    if (!app.window) {
      throw std::runtime_error("Cannot create viewer window");
    }
    app.presenter.initialize(app.window, app.size);
    auto driver = make_unique<ViewerDisplay>(app.presenter, app.size);
    if (!app.session->device->should_use_graphics_interop(driver->graphics_interop_get_device(),
                                                          true))
    {
      throw std::runtime_error("OptiX and Vulkan must use the same GPU");
    }
    app.display = driver.get();
    app.session->set_display_driver(std::move(driver));
    app.completion = make_unique<RenderCompletion>(*app.session);
    ShowWindow(app.window, SW_SHOW);
    app.title();
    // Wake directly on finite render completion or input. Retry only a pending
    // swapchain acquisition; idle and paused windows need no periodic wakeups.
    MSG message = {};
    bool quit = false;
    const HANDLE completed = app.completion->event();
    app.tick();
    while (!quit) {
      const DWORD timeout = app.present_pending && !IsIconic(app.window) ? 2 : INFINITE;
      const DWORD result = MsgWaitForMultipleObjectsEx(
          1, &completed, timeout, QS_ALLINPUT, MWMO_INPUTAVAILABLE);
      if (result == WAIT_FAILED) {
        throw std::runtime_error("Viewer completion wait failed");
      }
      while (PeekMessage(&message, nullptr, 0, 0, PM_REMOVE)) {
        if (message.message == WM_QUIT) {
          quit = true;
          break;
        }
        TranslateMessage(&message);
        DispatchMessage(&message);
      }
      if (!quit) {
        app.tick();
      }
    }
    if (app.running) {
      app.session->cancel(true);
      app.session->wait();
    }
    return int(message.wParam);
  }
  catch (const std::exception &error) {
    MessageBoxA(nullptr, error.what(), "Photara Viewer error", MB_ICONERROR);
    return 1;
  }
}
