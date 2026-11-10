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
 * scene (materials, meshes, lights, camera, world), render once, then read the
 * named render passes. All data is copied during the call, so the caller's
 * pointers only need to stay valid for the duration of the function.
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

/* Lifecycle. Returns NULL on failure; see photara_last_error(NULL). */
PHOTARA_API PhotaraContext *photara_create(const PhotaraOptions *options);
PHOTARA_API void photara_destroy(PhotaraContext *ctx);

/* Scene construction. Return 0 on success, -1 on error.
 * photara_add_material returns a non-negative material index instead. */
PHOTARA_API int photara_add_material(PhotaraContext *ctx, const PhotaraMaterialDesc *desc);
PHOTARA_API int photara_add_mesh(PhotaraContext *ctx, const PhotaraMeshDesc *desc);
PHOTARA_API int photara_set_camera(PhotaraContext *ctx, const PhotaraCameraDesc *desc);
PHOTARA_API int photara_add_area_light(PhotaraContext *ctx, const PhotaraAreaLightDesc *desc);
PHOTARA_API int photara_add_sun_light(PhotaraContext *ctx, const PhotaraSunLightDesc *desc);
PHOTARA_API int photara_set_background(PhotaraContext *ctx,
                                       const float color[3],
                                       float strength);

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
