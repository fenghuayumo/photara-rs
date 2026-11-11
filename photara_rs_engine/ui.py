# SPDX-License-Identifier: Apache-2.0

import bpy


class RENDER_PT_photara_sampling(bpy.types.Panel):
    bl_label = "Sampling"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = "render"
    bl_parent_id = "RENDER_PT_context"

    @classmethod
    def poll(cls, context):
        return context.scene and context.scene.render.engine == 'PHOTARA_RS'

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False

        settings = context.scene.photara
        layout.prop(settings, "samples")
        layout.prop(settings, "seed")
        layout.prop(settings, "max_bounces")
        layout.prop(settings, "denoise")
        layout.prop(settings, "adaptive_sampling")


class RENDER_PT_photara_viewport(bpy.types.Panel):
    bl_label = "Viewport"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = "render"
    bl_parent_id = "RENDER_PT_context"

    @classmethod
    def poll(cls, context):
        return context.scene and context.scene.render.engine == 'PHOTARA_RS'

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False

        settings = context.scene.photara
        layout.prop(settings, "viewport_samples")
