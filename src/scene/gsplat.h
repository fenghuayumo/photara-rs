/* SPDX-FileCopyrightText: 2026 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "scene/object.h"
#include "scene/pointcloud.h"

CCL_NAMESPACE_BEGIN

/* This predicate must be independent of camera and integrator settings: OptiX
 * caches its instance masks when the TLAS is built. Unsupported cameras use
 * the ordinary camera visibility bit instead of the raster mesh bit. */
inline bool gsplat_raster_object_supported(const Object *object)
{
  const Geometry *geometry = object->get_geometry();
  for (Node *node : geometry->get_used_shaders()) {
    if (!static_cast<const Shader *>(node)->is_default_gsplat) {
      return false;
    }
  }
  return geometry->is_pointcloud() && geometry->primitive_type() == PRIMITIVE_GSPLAT &&
         !object->use_motion() && !object->get_use_holdout() && !object->get_is_shadow_catcher() &&
         object->get_lightgroup().empty() && !object->has_light_linking();
}

CCL_NAMESPACE_END
