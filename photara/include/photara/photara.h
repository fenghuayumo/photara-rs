/* SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include <stddef.h>

#ifdef _WIN32
#  ifdef PHOTARA_BUILD_SHARED
#    define PHOTARA_API __declspec(dllexport)
#  elif defined(PHOTARA_USE_SHARED)
#    define PHOTARA_API __declspec(dllimport)
#  else
#    define PHOTARA_API
#  endif
#else
#  define PHOTARA_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Photara renderer C API.
 *
 * The API is intentionally small and stateful: create a context, describe a
 * scene (materials, meshes, lights, camera, world), render, then read the named
 * render passes. The scene can be modified and rendered repeatedly, which is
 * the intended path for batch and sim2real use. All data is copied during the
 * call, so the caller's pointers only need to stay valid for the duration of
 * the function.
 *
 * Meshes, instances and lights return non-negative handles. A handle stays
 * valid until that object or light is removed, its group is cleared, or the
 * context is destroyed.
 *
 * Coordinates are right-handed, +Y up, distances in meters. Camera and light
 * transforms use the same convention as Cycles: local +Z is the camera view
 * direction and the area-light emission direction is local -Z.
 */

typedef struct PhotaraContext PhotaraContext;

typedef enum PhotaraDevice {
  PHOTARA_DEVICE_AUTO = 0,
  PHOTARA_DEVICE_OPTIX = 1,
} PhotaraDevice;

/* REALTIME is currently a frame-scheduled OptiX baseline. ReSTIR/cache
 * acceleration is not enabled by this mode yet. */
typedef enum PhotaraRenderMode {
  PHOTARA_RENDER_REFERENCE = 0,
  PHOTARA_RENDER_REALTIME = 1,
} PhotaraRenderMode;

typedef struct PhotaraOptions {
  int width;
  int height;
  int samples;
  int seed;
  int max_bounces;
  /* Denoise the combined pass with the device denoiser (OptiX or OIDN). */
  int use_denoise;
  /* Use Cycles adaptive sampling. Disable for a fixed, reproducible sample count. */
  int use_adaptive_sampling;
  PhotaraDevice device;
  /*
   * Directory that contains the renderer's `lib/kernel_*.zst` binaries. When
   * using the Python package this is set automatically by the wrapper. May be
   * NULL, in which case the process working directory is used.
   */
  const char *resource_dir;
} PhotaraOptions;

typedef struct PhotaraMaterialDesc {
  float base_color[3];
  float metallic;
  float roughness;
  float ior;
  float transmission;
  float emission_color[3];
  float emission_strength;
} PhotaraMaterialDesc;

typedef struct PhotaraMeshDesc {
  /* Vertex attributes, 3 floats per vertex. */
  const float *positions;
  /* Optional per-vertex normals, 3 floats per vertex. */
  const float *normals;
  /* Optional per-vertex UVs, 2 floats per vertex. */
  const float *uvs;
  /* Triangle indices, 3 ints per triangle. */
  const int *indices;
  int num_vertices;
  int num_triangles;
  /*
   * Optional 4x4 row-major transform, column-vector convention, i.e. the
   * translation is in elements 3, 7, 11. NULL means identity.
   */
  const float *matrix;
  /* Index returned by photara_add_material, or -1 for the default material. */
  int material_index;
  /* Value written to the "object_id" pass. */
  int object_id;
} PhotaraMeshDesc;

/* Fixed-topology mesh update. Vertex/triangle counts and indices are kept. */
typedef struct PhotaraMeshUpdateDesc {
  /* Vertex positions, 3 floats per vertex. Required. */
  const float *positions;
  /* Optional per-vertex normals, 3 floats per vertex. */
  const float *normals;
  /* Optional per-vertex UVs, 2 floats per vertex. */
  const float *uvs;
  /* Must match the vertex count of the mesh being updated. */
  int num_vertices;
} PhotaraMeshUpdateDesc;

/* Mesh topology replacement. The mesh object and its handle stay valid. */
typedef struct PhotaraMeshReplaceDesc {
  const float *positions;
  const float *normals;
  const float *uvs;
  const int *indices;
  int num_vertices;
  int num_triangles;
  /* -1 keeps the current material, otherwise an index from photara_add_material. */
  int material_index;
} PhotaraMeshReplaceDesc;

typedef struct PhotaraCameraDesc {
  float position[3];
  float target[3];
  /* Optional up hint, defaults to (0, 1, 0). */
  float up[3];
  float fov_degrees;
  float near_clip;
  float far_clip;
} PhotaraCameraDesc;

typedef struct PhotaraAreaLightDesc {
  /* Center of the rectangular light. */
  float position[3];
  /* Point the light faces. */
  float target[3];
  /* Optional up hint for the rectangle orientation, defaults to (0, 1, 0). */
  float up[3];
  /* Full extents of the rectangle in meters. */
  float size_u;
  float size_v;
  float color[3];
  float strength;
} PhotaraAreaLightDesc;

typedef struct PhotaraSunLightDesc {
  /* Direction in which the light travels (from the sun towards the scene). */
  float direction[3];
  float color[3];
  float strength;
  /* Angular diameter in degrees. Zero produces a delta light. */
  float angle_degrees;
} PhotaraSunLightDesc;

typedef struct PhotaraPointLightDesc {
  float position[3];
  float color[3];
  float strength;
  /* Zero is a delta point light. Positive values make it a sphere light. */
  float radius;
} PhotaraPointLightDesc;

typedef struct PhotaraSpotLightDesc {
  float position[3];
  /* Direction in which the light travels (the beam points along local -Z). */
  float direction[3];
  /* Optional up hint, defaults to (0, 1, 0). */
  float up[3];
  float color[3];
  float strength;
  /* Zero is a delta light; positive values make the spot a sphere light. */
  float radius;
  /* Full cone angle in degrees, strictly between 0 and 180. */
  float angle_degrees;
  /* Cone edge softness. Values are clamped to [1e-4, 1]. */
  float smooth;
} PhotaraSpotLightDesc;

/* Lifecycle. Returns NULL on failure; see photara_last_error(NULL). */
PHOTARA_API PhotaraContext *photara_create(const PhotaraOptions *options);
PHOTARA_API PhotaraContext *photara_create_viewport(const PhotaraOptions *options);
/* Returns 0 for a new frame, 1 if unchanged/unavailable, -1 on error. */
PHOTARA_API int photara_attach_display(PhotaraContext *ctx, void *driver);
PHOTARA_API int photara_draw_display(PhotaraContext *ctx);
PHOTARA_API int photara_copy_preview(PhotaraContext *ctx, float *rgba, float *depth,
                                    size_t pixels, unsigned long long *revision);
PHOTARA_API void photara_destroy(PhotaraContext *ctx);

typedef enum PhotaraTextureInterpolation {
  PHOTARA_TEXTURE_LINEAR = 0,
  PHOTARA_TEXTURE_CLOSEST = 1,
  PHOTARA_TEXTURE_CUBIC = 2,
  PHOTARA_TEXTURE_SMART = 3
} PhotaraTextureInterpolation;

typedef enum PhotaraTextureExtension {
  PHOTARA_TEXTURE_REPEAT = 0,
  PHOTARA_TEXTURE_EXTEND = 1,
  PHOTARA_TEXTURE_CLIP = 2,
  PHOTARA_TEXTURE_MIRROR = 3
} PhotaraTextureExtension;

typedef struct PhotaraTextureDesc {
  /* Image file path. Absolute paths are recommended. */
  const char *filename;
  /* NULL/"" = auto, otherwise "sRGB", "scene_linear" or "data". */
  const char *colorspace;
  /* PhotaraTextureInterpolation. */
  int interpolation;
  /* PhotaraTextureExtension. Ignored by environment textures. */
  int extension;
} PhotaraTextureDesc;

/* Binary alpha clipping from the base-color image alpha. Negative cutoff disables.
 * Independent from hardware acceleration; OMM activates automatically if eligible. */
PHOTARA_API int photara_set_material_alpha_clip(PhotaraContext *ctx, int material, float cutoff);
PHOTARA_API int photara_get_omm_stats(PhotaraContext *ctx, int *supported, int *enabled,
                                     int *meshes, unsigned long long *transparent_states);

/* Materials. Return a non-negative material index, or -1 on error. */
PHOTARA_API int photara_add_material(PhotaraContext *ctx, const PhotaraMaterialDesc *desc);
/* Update a material in place. Meshes referencing it change on the next render. */
PHOTARA_API int photara_update_material(PhotaraContext *ctx,
                                        int material_index,
                                        const PhotaraMaterialDesc *desc);
/* Use an image as the Principled Base Color input. The material's base_color
 * remains a multiplier over the texture. */
PHOTARA_API int photara_set_material_base_color_texture(PhotaraContext *ctx,
                                                        int material_index,
                                                        const PhotaraTextureDesc *desc);
PHOTARA_API int photara_clear_material_base_color_texture(PhotaraContext *ctx,
                                                          int material_index);

/* Mesh objects. Return a non-negative object handle, or -1 on error. */
/* Arrays: N*3 positions/scales, N*4 wxyz rotations and encoded RGB+opacity,
 * optional N*45 coefficient-major RGB SH rest (degrees 0..3, zero padded). */
PHOTARA_API int photara_add_gaussian_splats(PhotaraContext *ctx, int count,
    const float *positions, const float *scales, const float *rotations,
    const float *colors_opacity, const float *sh_rest, const float *matrix, int object_id);
PHOTARA_API int photara_add_mesh(PhotaraContext *ctx, const PhotaraMeshDesc *desc);
/* Add an object sharing the mesh of source_object (instancing). The new object
 * inherits the source material; matrix and object_id follow the mesh desc
 * conventions. Returns an object handle, or -1 on error. */
PHOTARA_API int photara_add_instance(PhotaraContext *ctx,
                                     int source_object,
                                     const float *matrix,
                                     int object_id);
/* Replace an object transform. matrix has the same layout as PhotaraMeshDesc. */
PHOTARA_API int photara_set_object_transform(PhotaraContext *ctx,
                                             int object,
                                             const float matrix[16]);
/* Non-zero makes the object visible to all rays; zero hides it entirely. */
PHOTARA_API int photara_set_object_visible(PhotaraContext *ctx, int object, int visible);
/* Set the value written to the object_id pass for this object. */
PHOTARA_API int photara_set_object_id(PhotaraContext *ctx, int object, int object_id);
/* Assign a material index (or -1 for the default material) to the object's
 * mesh. Instances sharing the mesh are affected as well. */
PHOTARA_API int photara_set_object_material(PhotaraContext *ctx,
                                            int object,
                                            int material_index);
/* Update vertex data without changing topology. Any object sharing the mesh
 * updates it for all users. */
PHOTARA_API int photara_update_mesh(PhotaraContext *ctx,
                                    int object,
                                    const PhotaraMeshUpdateDesc *desc);
/* Replace vertex data and topology in place. The mesh/object handle stays valid. */
PHOTARA_API int photara_replace_mesh(PhotaraContext *ctx,
                                     int object,
                                     const PhotaraMeshReplaceDesc *desc);
PHOTARA_API int photara_remove_object(PhotaraContext *ctx, int object);
/* Remove every mesh object (including instances). Materials are kept. */
PHOTARA_API int photara_clear_objects(PhotaraContext *ctx);

/* Camera. photara_set_camera places a perspective camera by look-at. */
PHOTARA_API int photara_set_camera(PhotaraContext *ctx, const PhotaraCameraDesc *desc);
/* Set the camera from a 4x4 transform using the PhotaraMeshDesc matrix layout.
 * Camera local +Z is the view direction. fov <= 0 keeps the current FOV;
 * non-positive clip values keep the current clipping range. */
PHOTARA_API int photara_set_camera_matrix(PhotaraContext *ctx,
                                          const float matrix[16],
                                          float fov_degrees,
                                          float near_clip,
                                          float far_clip);

/* Lights. Return a non-negative light handle, or -1 on error. */
PHOTARA_API int photara_add_area_light(PhotaraContext *ctx, const PhotaraAreaLightDesc *desc);
PHOTARA_API int photara_add_sun_light(PhotaraContext *ctx, const PhotaraSunLightDesc *desc);
PHOTARA_API int photara_add_point_light(PhotaraContext *ctx,
                                        const PhotaraPointLightDesc *desc);
PHOTARA_API int photara_add_spot_light(PhotaraContext *ctx,
                                       const PhotaraSpotLightDesc *desc);
PHOTARA_API int photara_update_area_light(PhotaraContext *ctx,
                                          int light,
                                          const PhotaraAreaLightDesc *desc);
PHOTARA_API int photara_update_sun_light(PhotaraContext *ctx,
                                         int light,
                                         const PhotaraSunLightDesc *desc);
PHOTARA_API int photara_update_point_light(PhotaraContext *ctx,
                                           int light,
                                           const PhotaraPointLightDesc *desc);
PHOTARA_API int photara_update_spot_light(PhotaraContext *ctx,
                                          int light,
                                          const PhotaraSpotLightDesc *desc);
/* Replace only the light object's transform (same layout as PhotaraMeshDesc). */
PHOTARA_API int photara_set_light_transform(PhotaraContext *ctx,
                                            int light,
                                            const float matrix[16]);
/* Non-zero enables the light; zero turns it off. Light objects stay hidden
 * from camera rays while enabled. */
PHOTARA_API int photara_set_light_visible(PhotaraContext *ctx, int light, int visible);
PHOTARA_API int photara_remove_light(PhotaraContext *ctx, int light);
PHOTARA_API int photara_clear_lights(PhotaraContext *ctx);

PHOTARA_API int photara_set_background(PhotaraContext *ctx,
                                       const float color[3],
                                       float strength);
/* Use an equirectangular environment image. The constant color remains stored
 * and is restored by photara_clear_background_texture. */
PHOTARA_API int photara_set_background_texture(PhotaraContext *ctx,
                                               const PhotaraTextureDesc *desc);
PHOTARA_API int photara_clear_background_texture(PhotaraContext *ctx);

/* Remove all mesh objects and lights. Materials, camera and world are kept. */
PHOTARA_API int photara_clear_scene(PhotaraContext *ctx);

/* Texture LOD applies to both Reference and RealTime. The alternative retains
 * the existing compact differential estimator, not full vector differentials. */
typedef enum PhotaraTextureLOD {
  PHOTARA_TEXTURE_LOD_RAY_CONES = 0,
  PHOTARA_TEXTURE_LOD_RAY_DIFFERENTIALS = 1
} PhotaraTextureLOD;
PHOTARA_API int photara_set_texture_lod(PhotaraContext *ctx, PhotaraTextureLOD method);
PHOTARA_API int photara_get_texture_lod(PhotaraContext *ctx, PhotaraTextureLOD *method);

/* Per-frame render controls. */
PHOTARA_API int photara_set_seed(PhotaraContext *ctx, int seed);
PHOTARA_API int photara_set_samples(PhotaraContext *ctx, int samples);
/* Configure an idle context. Reference sample settings are retained.
 * RealTime uses fresh films and consecutive sample subsets per render job.
 * Reconfiguration resets the subset sequence; denoiser choice is unchanged. */
/* Experimental temporal analytic-light proposal, RealTime only. Default on.
 * Mesh emitters retain tree sampling. Changing this resets feedback history. */
PHOTARA_API int photara_set_neeat(PhotaraContext *ctx, int enabled);
/* Idle-context diagnostics: active analytic count and largest learned probability. */
PHOTARA_API int photara_get_neeat_stats(PhotaraContext *ctx, int *lights, float *max_pdf);
/* Screen-cache diagnostics; probabilities are measured before proposal mixing. */
PHOTARA_API int photara_get_neeat_local_stats(PhotaraContext *ctx, int *tiles, float *max_pdf);
PHOTARA_API int photara_set_render_mode(PhotaraContext *ctx,
                                       PhotaraRenderMode mode,
                                       int samples_per_frame);


/* Asynchronous render jobs. Job ids are non-zero and monotonically increasing;
 * one job may be active per context. Scene mutation is rejected while a job is
 * running. */
typedef enum PhotaraRenderState {
  PHOTARA_RENDER_RUNNING = 0,
  PHOTARA_RENDER_DONE = 1,
  PHOTARA_RENDER_CANCELLED = 2,
  PHOTARA_RENDER_FAILED = 3
} PhotaraRenderState;

typedef struct PhotaraRenderProgress {
  /* PhotaraRenderState value. */
  int state;
  float progress;
  int current_sample;
  int num_samples;
  /* Strings are owned by the context and valid until the next poll/begin. */
  const char *status;
  const char *substatus;
} PhotaraRenderProgress;

/* Start an asynchronous render and return its job id. */
PHOTARA_API int photara_render_begin(PhotaraContext *ctx,
                                     unsigned long long *job);
/* Poll a job. Returns 0 for a valid snapshot, -1 for an invalid/superseded job.
 * After a terminal state, photara_copy_pass can be used when the state is DONE. */
PHOTARA_API int photara_render_poll(PhotaraContext *ctx,
                                    unsigned long long job,
                                    PhotaraRenderProgress *progress);
/* Cancel a running job and wait for the renderer to stop. */
/* Request cancellation without waiting. Poll until the job reaches a terminal state
 * before modifying the scene or starting another job. */
PHOTARA_API int photara_render_request_cancel(PhotaraContext *ctx, unsigned long long job);

/* Queue the latest perspective camera on an existing viewport job. Scene
 * mutation happens on the render thread; intermediate camera updates coalesce.
 * The job remains valid and converges normally when updates stop. */
PHOTARA_API int photara_queue_camera_matrix(PhotaraContext *ctx,
                                          unsigned long long job,
                                          const float matrix[16],
                                          float fov_degrees,
                                          float near_clip,
                                          float far_clip);

PHOTARA_API int photara_render_cancel(PhotaraContext *ctx, unsigned long long job);
/* Finish a job. Returns 0 only for a successful completed render, -1 while
 * running, cancelled, failed, or for an invalid job. */
PHOTARA_API int photara_render_finish(PhotaraContext *ctx, unsigned long long job);

/* Render synchronously and store the requested passes. */
PHOTARA_API int photara_render(PhotaraContext *ctx);

/* Render result access. */
PHOTARA_API int photara_render_width(const PhotaraContext *ctx);
PHOTARA_API int photara_render_height(const PhotaraContext *ctx);
/* Number of float components for a pass, or -1 if the pass is unknown. */
PHOTARA_API int photara_pass_channels(const PhotaraContext *ctx, const char *name);
/*
 * Copy a pass into `dst`. Rows are bottom-up, matching Cycles render buffers.
 * `dst_count` must be at least width * height * channels.
 */
/* Configure an idle session's denoiser. Type: 0 automatic, 2 OptiX, 4 OIDN, 8 DLSS.
 * Passes: 0 color, 1 color+albedo, 5 color+albedo+normal.
 * Prefilter: 1 none, 2 fast, 3 accurate. Quality: 1 high, 2 balanced, 3 fast. */
/* Film exposure is a linear multiplier. Filter: 0 box, 1 Gaussian, 2 Blackman-Harris. */
/* Resize an idle session without rebuilding geometry, materials or device. */
PHOTARA_API int photara_resize(PhotaraContext *ctx, int width, int height);

PHOTARA_API int photara_configure_film(PhotaraContext *ctx, float exposure, int transparent,
                                     int transparent_glass, float roughness_threshold,
                                     int filter_type, float filter_width);

/* DLSS upscale factor: 1 native, 1.5 quality, 1/0.58 balanced, 2 performance, 3 ultra. */
/* Sampling controls use seconds for time_limit, matching Blender's UI. */
PHOTARA_API int photara_configure_sampling(PhotaraContext *ctx, float threshold,
                                         int min_samples, double time_limit,
                                         int denoise_start_sample);
PHOTARA_API int photara_configure_dlss(PhotaraContext *ctx, float factor);
PHOTARA_API int photara_auto_denoiser(void);
PHOTARA_API int photara_configure_denoiser(PhotaraContext *ctx, int type, int passes,
                                         int prefilter, int quality, int use_gpu);

PHOTARA_API int photara_copy_pass(PhotaraContext *ctx,
                                  const char *name,
                                  float *dst,
                                  size_t dst_count);

PHOTARA_API const char *photara_device_name(const PhotaraContext *ctx);
/* Last error for `ctx`, or the last context creation error when `ctx` is NULL. */
PHOTARA_API const char *photara_last_error(const PhotaraContext *ctx);

#ifdef __cplusplus
} /* extern "C" */
#endif
