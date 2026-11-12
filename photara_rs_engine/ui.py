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
        layout.prop(context.scene.photara_rs, 'device')
        layout.label(text='NVIDIA OptiX')


class RENDER_PT_photara_sampling(PhotaraButtonsPanel, bpy.types.Panel):
    bl_label = 'Sampling'
    bl_order = -20

    def draw(self, context):
        pass


def draw_sampling(layout, settings, viewport=False):
    prefix = 'viewport_' if viewport else ''
    enabled = getattr(settings, prefix + 'adaptive_sampling')
    dlss = viewport and settings.viewport_denoise and settings.viewport_denoiser == 'DLSS'
    heading = layout.column(align=True, heading="Noise Threshold")
    heading.active = not dlss
    row = heading.row(align=True)
    row.prop(settings, prefix + 'adaptive_sampling', text='')
    sub = row.row()
    sub.active = enabled
    sub.prop(settings, prefix + 'adaptive_threshold', text='')
    col = layout.column(align=True)
    col.prop(settings, 'viewport_samples' if viewport else 'samples', text='Max Samples' if enabled else 'Samples')
    if enabled:
        row = col.row()
        row.active = not dlss
        row.prop(settings, prefix + 'adaptive_min_samples', text='Min Samples')
    if not viewport:
        col.prop(settings, 'time_limit')


class RENDER_PT_photara_viewport(PhotaraButtonsPanel, bpy.types.Panel):
    bl_label = 'Viewport'
    bl_parent_id = 'RENDER_PT_photara_sampling'

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        settings = context.scene.photara_rs
        draw_sampling(layout, settings, viewport=True)


class RENDER_PT_photara_render(PhotaraButtonsPanel, bpy.types.Panel):
    bl_label = 'Render'
    bl_parent_id = 'RENDER_PT_photara_sampling'

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        settings = context.scene.photara_rs
        draw_sampling(layout, settings)


class PhotaraDenoisePanel:
    bl_label = 'Denoise'
    bl_options = {'DEFAULT_CLOSED'}
    setting = 'denoise'

    def draw_header(self, context):
        self.layout.prop(context.scene.photara_rs, self.setting, text='')

    def draw(self, context):
        # Match Cycles sampling_viewport_denoise / sampling_render_denoise:
        # Start Sample belongs to the viewport, final renders denoise at the end.
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        settings = context.scene.photara_rs
        viewport = self.setting == 'viewport_denoise'
        prefix = 'viewport_' if viewport else ''
        col = layout.column()
        col.active = getattr(settings, self.setting)
        sub = col.column()
        sub.prop(settings, prefix + 'denoiser', text='Denoiser')
        denoiser = getattr(settings, prefix + 'denoiser')
        if denoiser == 'AUTO':
            import photara_rs
            denoiser = photara_rs.auto_denoiser()
        if not viewport and denoiser == 'OPTIX':
            row = col.row()
            row.enabled = False
            row.prop(settings, 'temporal_denoising_unavailable', text='Temporal Denoising')
            col.label(text='Animation history is not yet supported', icon='INFO')
        if denoiser == 'DLSS':
            if not viewport:
                row = col.row()
                row.enabled = False
                row.prop(settings, 'animation_history_unavailable', text='Keep Animation History')
            col.prop(settings, prefix + 'denoise_upscale_quality',
                     text='Upscale Mode' if viewport else 'Upscaling')
            if not viewport:
                col.label(text='Use low render samples', icon='INFO')
                if settings.denoise_upscale_quality != 'NONE':
                    col.label(text='Disable extra render passes for upscaling', icon='INFO')
                    col.label(text='Shadow catchers / 3DGS: native resolution')
                col.label(text='Only the combined image is reconstructed')
                col.label(text='Animation history is not yet supported', icon='INFO')
                if context.scene.render.use_motion_blur:
                    col.label(text='Disable motion blur for DLSS final rendering', icon='ERROR')
            return
        col.prop(settings, prefix + 'denoise_passes', text='Passes')
        if denoiser == 'OPENIMAGEDENOISE':
            col.prop(settings, prefix + 'denoise_prefilter', text='Prefilter')
            col.prop(settings, prefix + 'denoise_quality', text='Quality')
        if viewport:
            col.prop(settings, 'viewport_denoise_start_sample', text='Start Sample')
        if denoiser == 'OPENIMAGEDENOISE':
            row = col.row()
            row.prop(settings, prefix + 'denoise_use_gpu', text='Use GPU')


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
        self.layout.prop(context.scene.photara_rs, 'seed')


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
        self.layout.prop(context.scene.photara_rs, 'max_bounces', text='Total')


class RENDER_PT_photara_film(PhotaraButtonsPanel, bpy.types.Panel):
    bl_label = 'Film'
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        self.layout.use_property_split = True
        self.layout.prop(context.scene.photara_rs, 'film_exposure')


class RENDER_PT_photara_film_transparency(PhotaraButtonsPanel, bpy.types.Panel):
    bl_label = 'Transparent'
    bl_parent_id = 'RENDER_PT_photara_film'

    def draw_header(self, context):
        self.layout.prop(context.scene.render, 'film_transparent', text='')

    def draw(self, context):
        self.layout.use_property_split = True
        self.layout.active = context.scene.render.film_transparent
        settings = context.scene.photara_rs
        self.layout.prop(settings, 'film_transparent_glass')
        sub = self.layout.column()
        sub.active = settings.film_transparent_glass
        sub.prop(settings, 'film_transparent_roughness')


class RENDER_PT_photara_film_filter(PhotaraButtonsPanel, bpy.types.Panel):
    bl_label = 'Pixel Filter'
    bl_parent_id = 'RENDER_PT_photara_film'

    def draw(self, context):
        self.layout.use_property_split = True
        settings = context.scene.photara_rs
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
