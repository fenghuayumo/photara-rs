# SPDX-License-Identifier: Apache-2.0
"""Render settings arranged like Cycles, backed by Photara properties."""

import bpy


class PhotaraButtonsPanel:
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = 'render'

    @classmethod
    def poll(cls, context):
        return context.scene and context.scene.render.engine == 'PHOTARA_RS'


def draw_device(self, context):
    if context.scene and context.scene.render.engine == 'PHOTARA_RS':
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        layout.prop(context.scene.photara, 'device')
        layout.label(text='NVIDIA OptiX')


class RENDER_PT_photara_sampling(PhotaraButtonsPanel, bpy.types.Panel):
    bl_label = 'Sampling'
    bl_order = -20

    def draw(self, context):
        pass


class RENDER_PT_photara_viewport(PhotaraButtonsPanel, bpy.types.Panel):
    bl_label = 'Viewport'
    bl_parent_id = 'RENDER_PT_photara_sampling'

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        settings = context.scene.photara
        layout.prop(settings, 'viewport_adaptive_sampling', text='Adaptive Sampling')
        layout.prop(settings, 'viewport_samples', text='Max Samples' if settings.viewport_adaptive_sampling else 'Samples')


class RENDER_PT_photara_render(PhotaraButtonsPanel, bpy.types.Panel):
    bl_label = 'Render'
    bl_parent_id = 'RENDER_PT_photara_sampling'

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        settings = context.scene.photara
        layout.prop(settings, 'adaptive_sampling')
        layout.prop(settings, 'samples', text='Max Samples' if settings.adaptive_sampling else 'Samples')


class PhotaraDenoisePanel:
    bl_label = 'Denoise'
    bl_options = {'DEFAULT_CLOSED'}
    setting = 'denoise'

    def draw_header(self, context):
        self.layout.prop(context.scene.photara, self.setting, text='')

    def draw(self, context):
        layout = self.layout
        layout.active = getattr(context.scene.photara, self.setting)
        layout.use_property_split = True
        layout.use_property_decorate = False
        settings = context.scene.photara
        prefix = 'viewport_' if self.setting == 'viewport_denoise' else ''
        layout.prop(settings, prefix + 'denoiser')
        layout.prop(settings, prefix + 'denoise_passes')
        if getattr(settings, prefix + 'denoiser') == 'OPENIMAGEDENOISE':
            layout.prop(settings, prefix + 'denoise_prefilter')
            layout.prop(settings, prefix + 'denoise_quality')
            layout.prop(settings, prefix + 'denoise_use_gpu')


class RENDER_PT_photara_viewport_denoise(PhotaraDenoisePanel, PhotaraButtonsPanel, bpy.types.Panel):
    bl_parent_id = 'RENDER_PT_photara_viewport'
    setting = 'viewport_denoise'


class RENDER_PT_photara_render_denoise(PhotaraDenoisePanel, PhotaraButtonsPanel, bpy.types.Panel):
    bl_parent_id = 'RENDER_PT_photara_render'


class RENDER_PT_photara_sampling_advanced(PhotaraButtonsPanel, bpy.types.Panel):
    bl_label = 'Advanced'
    bl_parent_id = 'RENDER_PT_photara_sampling'
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        self.layout.use_property_split = True
        self.layout.use_property_decorate = False
        self.layout.prop(context.scene.photara, 'seed')


class RENDER_PT_photara_light_paths(PhotaraButtonsPanel, bpy.types.Panel):
    bl_label = 'Light Paths'
    bl_order = -10
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        pass


class RENDER_PT_photara_max_bounces(PhotaraButtonsPanel, bpy.types.Panel):
    bl_label = 'Max Bounces'
    bl_parent_id = 'RENDER_PT_photara_light_paths'

    def draw(self, context):
        self.layout.use_property_split = True
        self.layout.use_property_decorate = False
        self.layout.prop(context.scene.photara, 'max_bounces', text='Total')


class RENDER_PT_photara_film(PhotaraButtonsPanel, bpy.types.Panel):
    bl_label = 'Film'
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        self.layout.use_property_split = True
        self.layout.prop(context.scene.photara, 'film_exposure')


class RENDER_PT_photara_film_transparency(PhotaraButtonsPanel, bpy.types.Panel):
    bl_label = 'Transparent'
    bl_parent_id = 'RENDER_PT_photara_film'

    def draw_header(self, context):
        self.layout.prop(context.scene.render, 'film_transparent', text='')

    def draw(self, context):
        self.layout.use_property_split = True
        self.layout.active = context.scene.render.film_transparent
        settings = context.scene.photara
        self.layout.prop(settings, 'film_transparent_glass')
        sub = self.layout.column()
        sub.active = settings.film_transparent_glass
        sub.prop(settings, 'film_transparent_roughness')


class RENDER_PT_photara_film_filter(PhotaraButtonsPanel, bpy.types.Panel):
    bl_label = 'Pixel Filter'
    bl_parent_id = 'RENDER_PT_photara_film'

    def draw(self, context):
        self.layout.use_property_split = True
        settings = context.scene.photara
        self.layout.prop(settings, 'pixel_filter_type')
        if settings.pixel_filter_type != 'BOX':
            self.layout.prop(settings, 'filter_width')


class RENDER_PT_photara_volumes(PhotaraButtonsPanel, bpy.types.Panel):
    bl_label = 'Volumes'
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        self.layout.label(text='Volume rendering is not yet supported.', icon='INFO')
        self.layout.label(text='Volume materials and grids are not exported.')


_color_panels = []

def register_color_management():
    from bl_ui import properties_render
    for name in dir(properties_render):
        cls = getattr(properties_render, name)
        if name.startswith('RENDER_PT_color_management') and hasattr(cls, 'COMPAT_ENGINES'):
            if 'PHOTARA_RS' not in cls.COMPAT_ENGINES:
                cls.COMPAT_ENGINES.add('PHOTARA_RS')
                _color_panels.append(cls)


def unregister_color_management():
    for cls in _color_panels:
        cls.COMPAT_ENGINES.discard('PHOTARA_RS')
    _color_panels.clear()


classes = (
    RENDER_PT_photara_sampling,
    RENDER_PT_photara_viewport,
    RENDER_PT_photara_viewport_denoise,
    RENDER_PT_photara_render,
    RENDER_PT_photara_render_denoise,
    RENDER_PT_photara_sampling_advanced,
    RENDER_PT_photara_light_paths,
    RENDER_PT_photara_max_bounces,
    RENDER_PT_photara_film,
    RENDER_PT_photara_film_transparency,
    RENDER_PT_photara_film_filter,
    RENDER_PT_photara_volumes,
)
