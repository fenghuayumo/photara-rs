/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "kernel/types.h"

#ifndef KERNEL_DATA_ARRAY
#  define KERNEL_DATA_ARRAY(type, name)
#endif
#ifndef KERNEL_DATA_ARRAY_WRITABLE
#  define KERNEL_DATA_ARRAY_WRITABLE(type, name) KERNEL_DATA_ARRAY(type, name)
#endif

/* BVH2, not used for OptiX or Embree. */
KERNEL_DATA_ARRAY(float4, bvh_nodes)
KERNEL_DATA_ARRAY(float4, bvh_leaf_nodes)
KERNEL_DATA_ARRAY(uint, prim_type)
KERNEL_DATA_ARRAY(uint, prim_visibility)
KERNEL_DATA_ARRAY(uint, prim_index)
KERNEL_DATA_ARRAY(uint, prim_object)
KERNEL_DATA_ARRAY(uint, object_node)
KERNEL_DATA_ARRAY(float2, prim_time)

/* objects */
KERNEL_DATA_ARRAY(KernelObject, objects)
KERNEL_DATA_ARRAY(Transform, object_motion_pass)
KERNEL_DATA_ARRAY(DecomposedTransform, object_motion)
KERNEL_DATA_ARRAY(uint, object_flag)
KERNEL_DATA_ARRAY(uint, object_prim_offset)

/* cameras */
KERNEL_DATA_ARRAY(DecomposedTransform, camera_motion)

/* triangles */
KERNEL_DATA_ARRAY(uint, tri_shader)
KERNEL_DATA_ARRAY(packed_uint3, tri_vindex)
KERNEL_DATA_ARRAY(packed_float3, tri_verts)

/* curves */
KERNEL_DATA_ARRAY(KernelCurve, curves)
KERNEL_DATA_ARRAY(float4, curve_keys)
KERNEL_DATA_ARRAY(KernelCurveSegment, curve_segments)

/* pointclouds */
KERNEL_DATA_ARRAY(float4, points)
KERNEL_DATA_ARRAY(uint, points_shader)

/* Camera-projected Gaussians and sorted tile lists. */
/* Projected splats, followed by the receiver grid at gsplat_raster.receiver_offset.
 * Receiver records use center = (inverse depth, raster derivatives x/y, confidence). */
KERNEL_DATA_ARRAY(KernelGSplat, gsplat_raster_splats)
KERNEL_DATA_ARRAY(uint2, gsplat_raster_tiles)
KERNEL_DATA_ARRAY(uint, gsplat_raster_indices)
KERNEL_DATA_ARRAY(float4, gsplat_shader_shadow_transmission)

/* attributes */
KERNEL_DATA_ARRAY(AttributeMap, attributes_map)
KERNEL_DATA_ARRAY(float, attributes_float)
KERNEL_DATA_ARRAY(float2, attributes_float2)
KERNEL_DATA_ARRAY(packed_float3, attributes_float3)
KERNEL_DATA_ARRAY(float4, attributes_float4)
KERNEL_DATA_ARRAY(uchar4, attributes_uchar4)
KERNEL_DATA_ARRAY(packed_normal, attributes_normal)
KERNEL_DATA_ARRAY(Quaternion, attributes_quaternion)
KERNEL_DATA_ARRAY(PackedSphericalHarmonicsRest, attributes_spherical_harmonics_rest)

/* lights */
KERNEL_DATA_ARRAY(KernelLightDistribution, light_distribution)
KERNEL_DATA_ARRAY(KernelLight, lights)
KERNEL_DATA_ARRAY(float2, light_background_marginal_cdf)
KERNEL_DATA_ARRAY(float2, light_background_conditional_cdf)

/* light tree */
KERNEL_DATA_ARRAY(float2, neeat_distribution)
KERNEL_DATA_ARRAY_WRITABLE(float, neeat_feedback)
KERNEL_DATA_ARRAY(KernelNEEATLight, neeat_lights)
KERNEL_DATA_ARRAY(KernelNEEATCluster, neeat_clusters)
KERNEL_DATA_ARRAY(KernelNEEATLookup, neeat_lookup)
KERNEL_DATA_ARRAY(KernelLightTreeNode, light_tree_nodes)
KERNEL_DATA_ARRAY(KernelLightTreeEmitter, light_tree_emitters)
KERNEL_DATA_ARRAY(uint, light_to_tree)
KERNEL_DATA_ARRAY(uint, object_lookup_offset)
KERNEL_DATA_ARRAY(uint, triangle_to_tree)

/* particles */
KERNEL_DATA_ARRAY(KernelParticle, particles)

/* shaders */
KERNEL_DATA_ARRAY(uint, svm_nodes)
KERNEL_DATA_ARRAY(KernelShader, shaders)

/* lookup tables */
KERNEL_DATA_ARRAY(float, lookup_table)

/* tabulated Sobol sample pattern */
KERNEL_DATA_ARRAY(float, sample_pattern_lut)

/* ies lights */
KERNEL_DATA_ARRAY(float, ies)

/* Volume. */
KERNEL_DATA_ARRAY(KernelOctreeNode, volume_tree_nodes)
KERNEL_DATA_ARRAY(KernelOctreeRoot, volume_tree_roots)
KERNEL_DATA_ARRAY(int, volume_tree_root_ids)
KERNEL_DATA_ARRAY(float, volume_step_size)

/* image textures */
KERNEL_DATA_ARRAY(KernelImageTexture, image_textures)
KERNEL_DATA_ARRAY_WRITABLE(KernelTileDescriptor, image_texture_tile_descriptors)
KERNEL_DATA_ARRAY_WRITABLE(uint8_t, image_texture_tile_access_state)
KERNEL_DATA_ARRAY(KernelImageUDIM, image_texture_udims)
KERNEL_DATA_ARRAY(KernelImageInfo, image_info)

#undef KERNEL_DATA_ARRAY
#undef KERNEL_DATA_ARRAY_WRITABLE
