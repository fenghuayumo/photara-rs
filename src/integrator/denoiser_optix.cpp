/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#ifdef WITH_OPTIX

#  include "integrator/denoiser_optix.h"
#  include "integrator/pass_accessor_gpu.h"

#  include "device/optix/device_impl.h"
#  include "device/optix/queue.h"
#  include "util/log.h"

#  include <optix_denoiser_tiling.h>

CCL_NAMESPACE_BEGIN

OptiXDenoiser::OptiXDenoiser(Device *denoiser_device, const DenoiseParams &params)
    : DenoiserGPU(denoiser_device, params),
      state_(denoiser_device, "__denoiser_state", true),
      temporal_color_(denoiser_device, "__denoiser_previous_color", MEM_READ_WRITE),
      temporal_guide_previous_(denoiser_device, "__denoiser_previous_guide", MEM_READ_WRITE),
      temporal_guide_output_(denoiser_device, "__denoiser_output_guide", MEM_READ_WRITE)
{
}

OptiXDenoiser::~OptiXDenoiser()
{
  /* It is important that the OptixDenoiser handle is destroyed before the OptixDeviceContext
   * handle, which is guaranteed since the local denoising device owning the OptiX device context
   * is deleted as part of the Denoiser class destructor call after this. */
  if (optix_denoiser_ != nullptr) {
    optixDenoiserDestroy(optix_denoiser_);
  }
}

bool OptiXDenoiser::is_device_supported(const DeviceInfo &device)
{
  if (device.type == DEVICE_OPTIX) {
    return device.denoisers & DENOISER_OPTIX;
  }
  return false;
}

bool OptiXDenoiser::denoise_buffer(const BufferParams &buffer_params,
                                   const BufferParams &denoised_buffer_params,
                                   RenderBuffers *render_buffers,
                                   const int num_samples,
                                   const bool allow_inplace_modification,
                                   const float2 pixel_jitter)
{
  OptiXDevice *const optix_device = static_cast<OptiXDevice *>(denoiser_device_);
  const CUDAContextScope scope(optix_device);

  if (temporal_history) {
    const std::vector<int> geometry = {buffer_params.width,
                                       buffer_params.height,
                                       buffer_params.full_x,
                                       buffer_params.full_y,
                                       buffer_params.full_width,
                                       buffer_params.full_height,
                                       buffer_params.window_x,
                                       buffer_params.window_y,
                                       buffer_params.window_width,
                                       buffer_params.window_height,
                                       params_.passes};
    if (temporal_history->geometry != geometry) {
      temporal_history->previous.clear();
      temporal_history->geometry = geometry;
    }
    temporal_history->output.clear();
  }

  const bool success = DenoiserGPU::denoise_buffer(buffer_params,
                                                   denoised_buffer_params,
                                                   render_buffers,
                                                   num_samples,
                                                   allow_inplace_modification && !temporal_history,
                                                   pixel_jitter);
  if (temporal_history && (!success || is_cancelled() || denoiser_device_->have_error())) {
    temporal_history->output.clear();
  }
  return success;
}

bool OptiXDenoiser::denoise_create_if_needed(DenoiseContext &context)
{
  const bool use_pass_albedo = (context.denoise_params.passes & DENOISER_PASS_ALBEDO) != 0;
  const bool use_pass_normal = (context.denoise_params.passes & DENOISER_PASS_NORMAL) != 0;
  const bool use_pass_motion = context.denoise_params.temporally_stable &&
                               context.guiding_params.pass_flow != PASS_UNUSED;
  const bool use_upscale_model = context.denoise_params.upscale_factor == 2.0f;

  const bool recreate_denoiser = (optix_denoiser_ == nullptr) ||
                                 (use_pass_albedo_ != use_pass_albedo) ||
                                 (use_pass_normal_ != use_pass_normal) ||
                                 (use_pass_motion_ != use_pass_motion) ||
                                 (use_upscale_model_ != use_upscale_model);
  if (!recreate_denoiser) {
    return true;
  }

  /* Destroy existing handle before creating new one. */
  if (optix_denoiser_) {
    optixDenoiserDestroy(optix_denoiser_);
  }

  /* Create OptiX denoiser handle on demand when it is first used. */
  OptixDenoiserModelKind model;
  if (use_pass_motion) {
    if (use_upscale_model) {
      model = OPTIX_DENOISER_MODEL_KIND_TEMPORAL_UPSCALE2X;
    }
    else {
      model = temporal_history ? OPTIX_DENOISER_MODEL_KIND_TEMPORAL_AOV :
                                 OPTIX_DENOISER_MODEL_KIND_TEMPORAL;
    }
  }
  else {
    if (use_upscale_model) {
      model = OPTIX_DENOISER_MODEL_KIND_UPSCALE2X;
    }
    else {
      model = OPTIX_DENOISER_MODEL_KIND_AOV;
    }
  }

  OptixDenoiserOptions denoiser_options = {};
  denoiser_options.guideAlbedo = use_pass_albedo;
  denoiser_options.guideNormal = use_pass_normal;
  denoiser_options.denoiseAlpha = OPTIX_DENOISER_ALPHA_MODE_COPY;

  const OptixResult result = optixDenoiserCreate(
      static_cast<OptiXDevice *>(denoiser_device_)->context,
      model,
      &denoiser_options,
      &optix_denoiser_);

  if (result != OPTIX_SUCCESS) {
    set_error("Failed to create OptiX denoiser");
    return false;
  }

  /* OptiX denoiser handle was created with the requested number of input passes. */
  use_pass_albedo_ = use_pass_albedo;
  use_pass_normal_ = use_pass_normal;
  use_pass_motion_ = use_pass_motion;
  use_upscale_model_ = use_upscale_model;

  /* OptiX denoiser has been created, but it needs configuration. */
  is_configured_ = false;

  return true;
}

bool OptiXDenoiser::denoise_configure_if_needed(DenoiseContext &context)
{
  /* Limit maximum tile size denoiser can be invoked with. */
  const int2 tile_size = make_int2(min(context.buffer_params.width, 4096),
                                   min(context.buffer_params.height, 4096));

  if (is_configured_ && (configured_size_.x == tile_size.x && configured_size_.y == tile_size.y)) {
    return true;
  }

  optix_device_assert(
      denoiser_device_,
      optixDenoiserComputeMemoryResources(optix_denoiser_, tile_size.x, tile_size.y, &sizes_));

  const bool tiled = tile_size.x < context.buffer_params.width ||
                     tile_size.y < context.buffer_params.height;

  /* Allocate denoiser state if tile size has changed since last setup. */
  state_.device = denoiser_device_;
  state_.alloc_to_device(sizes_.stateSizeInBytes + sizes_.withOverlapScratchSizeInBytes);

  /* Initialize denoiser state for the current tile size. */
  const OptixResult result = optixDenoiserSetup(
      optix_denoiser_,
      0, /* Work around bug in r495 drivers that causes artifacts when denoiser setup is called
          * on a stream that is not the default stream. */
      tile_size.x + (tiled ? sizes_.overlapWindowSizeInPixels * 2 : 0),
      tile_size.y + (tiled ? sizes_.overlapWindowSizeInPixels * 2 : 0),
      state_.device_pointer,
      sizes_.stateSizeInBytes,
      state_.device_pointer + sizes_.stateSizeInBytes,
      sizes_.withOverlapScratchSizeInBytes);
  if (result != OPTIX_SUCCESS) {
    set_error("Failed to set up OptiX denoiser");
    return false;
  }

  cuda_device_assert(denoiser_device_, cuCtxSynchronize());

  is_configured_ = true;
  configured_size_ = tile_size;

  return true;
}

bool OptiXDenoiser::denoise_run(const DenoiseContext &context, const DenoisePass &pass)
{
  /* Set up input and output layer information. */
  OptixImage2D color_layer = {0};
  OptixImage2D albedo_layer = {0};
  OptixImage2D normal_layer = {0};
  OptixImage2D flow_layer = {0};

  OptixImage2D output_layer = {0};
  OptixImage2D prev_output_layer = {0};

  /* Color pass. */
  {
    const int pass_denoised = pass.denoised_offset;
    const int64_t pass_stride_in_bytes = context.buffer_params.pass_stride * sizeof(float);

    color_layer.data = context.render_buffers->buffer.device_pointer +
                       pass_denoised * sizeof(float);
    color_layer.width = context.buffer_params.width;
    color_layer.height = context.buffer_params.height;
    color_layer.rowStrideInBytes = pass_stride_in_bytes * context.buffer_params.stride;
    color_layer.pixelStrideInBytes = pass_stride_in_bytes;
    color_layer.format = pass.num_components > 3 && use_upscale_model_ ?
                             OPTIX_PIXEL_FORMAT_FLOAT4 :
                             OPTIX_PIXEL_FORMAT_FLOAT3;
  }

  /* Previous output. */
  if (use_pass_motion_ && context.prev_output.offset != PASS_UNUSED) {
    const int64_t pass_stride_in_bytes = context.prev_output.pass_stride * sizeof(float);

    prev_output_layer.data = context.prev_output.device_pointer +
                             context.prev_output.offset * sizeof(float);
    prev_output_layer.width = context.denoised_buffer_params.width;
    prev_output_layer.height = context.denoised_buffer_params.height;
    prev_output_layer.rowStrideInBytes = pass_stride_in_bytes * context.prev_output.stride;
    prev_output_layer.pixelStrideInBytes = pass_stride_in_bytes;
    prev_output_layer.format = OPTIX_PIXEL_FORMAT_FLOAT3;
  }

  /* Optional albedo and color passes. */
  const device_ptr d_guiding_buffer = context.guiding_params.device_pointer;
  const int64_t pixel_stride_in_bytes = context.guiding_params.pass_stride * sizeof(float);
  const int64_t row_stride_in_bytes = context.guiding_params.stride * pixel_stride_in_bytes;

  if (use_pass_albedo_) {
    albedo_layer.data = d_guiding_buffer + context.guiding_params.pass_albedo * sizeof(float);
    albedo_layer.width = context.buffer_params.width;
    albedo_layer.height = context.buffer_params.height;
    albedo_layer.rowStrideInBytes = row_stride_in_bytes;
    albedo_layer.pixelStrideInBytes = pixel_stride_in_bytes;
    albedo_layer.format = OPTIX_PIXEL_FORMAT_FLOAT3;
  }

  if (use_pass_normal_) {
    normal_layer.data = d_guiding_buffer + context.guiding_params.pass_normal * sizeof(float);
    normal_layer.width = context.buffer_params.width;
    normal_layer.height = context.buffer_params.height;
    normal_layer.rowStrideInBytes = row_stride_in_bytes;
    normal_layer.pixelStrideInBytes = pixel_stride_in_bytes;
    normal_layer.format = OPTIX_PIXEL_FORMAT_FLOAT3;
  }

  if (use_pass_motion_) {
    flow_layer.data = d_guiding_buffer + context.guiding_params.pass_flow * sizeof(float);
    flow_layer.width = context.buffer_params.width;
    flow_layer.height = context.buffer_params.height;
    flow_layer.rowStrideInBytes = row_stride_in_bytes;
    flow_layer.pixelStrideInBytes = pixel_stride_in_bytes;
    flow_layer.format = OPTIX_PIXEL_FORMAT_FLOAT2;
  }

  /* Denoise in-place of the noisy input in the render buffers.
   * TODO: This is not valid for alpha with OPTIX_DENOISER_MODEL_KIND_UPSCALE2X +
   * OPTIX_DENOISER_ALPHA_MODE_COPY, since concurrent alpha reads/writes overlap. */
  {
    output_layer = color_layer;
    output_layer.width = context.denoised_buffer_params.width;
    output_layer.height = context.denoised_buffer_params.height;
    output_layer.rowStrideInBytes = output_layer.pixelStrideInBytes *
                                    context.denoised_buffer_params.stride;
  }

  OptixDenoiserGuideLayer guide_layers = {};
  guide_layers.albedo = albedo_layer;
  guide_layers.normal = normal_layer;
  guide_layers.flow = flow_layer;

  OptixDenoiserLayer image_layers = {};
  image_layers.input = color_layer;
  image_layers.previousOutput = prev_output_layer;
  image_layers.output = output_layer;

  /* Finally run denoising. */
  OptixDenoiserParams params = {}; /* All parameters are disabled/zero. */

  bool use_history = false;
  size_t guide_size = 0;
  if (use_pass_motion_ && temporal_history) {
    const size_t num_pixels = size_t(output_layer.width) * output_layer.height;
    guide_size = num_pixels * sizes_.internalGuideLayerPixelSizeInBytes;
    auto previous = temporal_history->previous.find(int(pass.type));
    use_history = previous != temporal_history->previous.end() &&
                  previous->second.color.size() == num_pixels * 3 &&
                  previous->second.guide.size() == guide_size;

    /* Both buffers must be provided even on the first frame. OptiX ignores previous data when
     * temporalModeUsePreviousLayers is zero, and initializes its internal guide history. */
    float *color = temporal_color_.alloc(num_pixels * 3);
    unsigned char *guide = temporal_guide_previous_.alloc(guide_size);
    if (use_history) {
      memcpy(color, previous->second.color.data(), num_pixels * 3 * sizeof(float));
      memcpy(guide, previous->second.guide.data(), guide_size);
    }
    else {
      memset(color, 0, num_pixels * 3 * sizeof(float));
      memset(guide, 0, guide_size);
    }
    temporal_color_.copy_to_device();
    temporal_guide_previous_.copy_to_device();
    memset(temporal_guide_output_.alloc(guide_size), 0, guide_size);
    temporal_guide_output_.copy_to_device();

    image_layers.previousOutput = output_layer;
    image_layers.previousOutput.data = temporal_color_.device_pointer;
    image_layers.previousOutput.pixelStrideInBytes = 3 * sizeof(float);
    image_layers.previousOutput.rowStrideInBytes = output_layer.width * 3 * sizeof(float);
    image_layers.previousOutput.format = OPTIX_PIXEL_FORMAT_FLOAT3;

    OptixImage2D internal_guide = {};
    internal_guide.width = output_layer.width;
    internal_guide.height = output_layer.height;
    internal_guide.pixelStrideInBytes = sizes_.internalGuideLayerPixelSizeInBytes;
    internal_guide.rowStrideInBytes = internal_guide.width * internal_guide.pixelStrideInBytes;
    internal_guide.format = OPTIX_PIXEL_FORMAT_INTERNAL_GUIDE_LAYER;
    internal_guide.data = temporal_guide_previous_.device_pointer;
    guide_layers.previousOutputInternalGuideLayer = internal_guide;
    internal_guide.data = temporal_guide_output_.device_pointer;
    guide_layers.outputInternalGuideLayer = internal_guide;
    params.temporalModeUsePreviousLayers = use_history;
    LOG_INFO << "OptiX temporal denoising: " << (use_history ? "reuse history" : "reset history")
             << ", pass " << int(pass.type) << ", " << output_layer.width << "x"
             << output_layer.height;
  }

  optix_device_assert(denoiser_device_,
                      optixUtilDenoiserInvokeTiled(
                          optix_denoiser_,
                          static_cast<OptiXDeviceQueue *>(denoiser_queue_.get())->stream(),
                          &params,
                          state_.device_pointer,
                          sizes_.stateSizeInBytes,
                          &guide_layers,
                          &image_layers,
                          1,
                          state_.device_pointer + sizes_.stateSizeInBytes,
                          sizes_.withOverlapScratchSizeInBytes,
                          sizes_.overlapWindowSizeInPixels,
                          configured_size_.x,
                          configured_size_.y));

  if (use_pass_motion_ && temporal_history) {
    if (!denoiser_queue_->synchronize() || denoiser_device_->have_error()) {
      return false;
    }
    DenoiseTemporalHistory::Layer &output = temporal_history->output[int(pass.type)];
    output.color.resize(size_t(output_layer.width) * output_layer.height * 3);
    CUDA_MEMCPY2D copy = {};
    copy.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    copy.srcDevice = output_layer.data;
    copy.dstMemoryType = CU_MEMORYTYPE_HOST;
    copy.dstHost = output.color.data();
    /* Output pixels are interleaved with other passes, so gather RGB without copying the
     * complete render buffer. A CUDA 2D copy gathers one RGB pixel per row. */
    copy.srcPitch = output_layer.pixelStrideInBytes;
    copy.dstPitch = 3 * sizeof(float);
    copy.WidthInBytes = 3 * sizeof(float);
    copy.Height = size_t(output_layer.width) * output_layer.height;
    cuda_device_assert(denoiser_device_, cuMemcpy2DUnaligned(&copy));
    temporal_guide_output_.copy_from_device();
    output.guide.assign(temporal_guide_output_.data(), temporal_guide_output_.data() + guide_size);
    if (denoiser_device_->have_error()) {
      return false;
    }
  }

  return true;
}

CCL_NAMESPACE_END

#endif
