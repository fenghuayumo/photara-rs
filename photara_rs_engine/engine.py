# SPDX-License-Identifier: Apache-2.0
"""The Photara Blender render engine.

Final renders (F12) write combined and depth passes into Blender's render
result. Viewport renders display asynchronous frames at increasing sample
counts, restarting whenever the camera or scene changes.
"""

from __future__ import annotations

import time
import weakref

import numpy as np

import bpy

photara_rs = None

from . import sync


def _srgb_encode(rgb):
    """Apply the sRGB transfer curve to linear RGB (viewport fallback)."""
    rgb = np.asarray(rgb, dtype=np.float32)
    lo = rgb * 12.92
    hi = 1.055 * np.power(np.maximum(rgb, 1e-8), 1.0 / 2.4) - 0.055
    return np.where(rgb <= 0.0031308, lo, hi)


def _load_photara():
    from . import ensure_photara_module

    error = ensure_photara_module()
    if error:
        raise RuntimeError(f"photara_rs is not importable: {error}")


def _denoiser_options(settings, viewport=False):
    prefix = 'viewport_' if viewport else ''
    return {name: getattr(settings, prefix + name) for name in (
        'denoiser', 'denoise_passes', 'denoise_prefilter',
        'denoise_quality', 'denoise_use_gpu')}


def _film_options(scene):
    settings = scene.photara
    return dict(exposure=settings.film_exposure,
                transparent=scene.render.film_transparent,
                transparent_glass=settings.film_transparent_glass,
                roughness_threshold=settings.film_transparent_roughness,
                filter_type=settings.pixel_filter_type, filter_width=settings.filter_width)


class PhotaraRenderEngine(bpy.types.RenderEngine):
    bl_idname = 'PHOTARA_RS'
    bl_label = "Photara"

    # EEVEE renders the Material Preview mode; "Rendered" uses view_draw below.
    bl_use_eevee_viewport = True
    bl_use_preview = False
    bl_use_postprocess = True
    bl_use_shading_properties_custom = False
    bl_use_eevee_freestyle = False

    def __init__(self, *args, **kwargs):
        # Blender 5.x instantiates render engines with RNA arguments.
        super().__init__(*args, **kwargs)
        self._viewport = None
        self._viewport_failed = False

    def __del__(self):
        try:
            viewport = getattr(self, "_viewport", None)
        except ReferenceError:
            return
        if viewport is not None:
            viewport.close()
            self._viewport = None

    # ------------------------------------------------------------------
    # Render passes
    # ------------------------------------------------------------------

    def update_render_passes(self, scene, render_layer):
        self.register_pass(scene, render_layer, "Combined", 4, 'RGBA', 'COLOR')
        if render_layer.use_pass_z:
            self.register_pass(scene, render_layer, "Depth", 1, 'Z', 'VALUE')

    # ------------------------------------------------------------------
    # Final render
    # ------------------------------------------------------------------

    def render(self, depsgraph):
        blender_scene = depsgraph.scene_eval
        settings = blender_scene.photara
        render = blender_scene.render
        scale = render.resolution_percentage / 100.0
        width = max(1, int(render.resolution_x * scale))
        height = max(1, int(render.resolution_y * scale))

        camera_object = blender_scene.camera
        if camera_object is None:
            self.report({'ERROR'}, "Scene has no camera")
            return

        scene = None
        try:
            _load_photara()
            if camera_object.data.type != 'PERSP':
                raise ValueError("Photara currently supports perspective cameras only")
            if render.use_border:
                raise ValueError("Photara does not yet support border rendering; disable Render Region")
            scene = photara_rs.Scene(
                width=width,
                height=height,
                samples=settings.samples,
                seed=settings.seed,
                max_bounces=settings.max_bounces,
                denoise=settings.denoise,
                adaptive_sampling=settings.adaptive_sampling,
                device=settings.device,
                **_denoiser_options(settings),
            )
            scene.configure_film(**_film_options(blender_scene))
            state = sync.SceneSyncState(scene)
            state.sync(depsgraph)
            aspect = (width / height) * (render.pixel_aspect_x / max(1e-6, render.pixel_aspect_y))
            state.sync_camera(camera_object, aspect)

            self.update_stats("Photara", "Starting render")
            job = scene.render_async()
            while True:
                progress = job.poll()
                if progress.state == "done":
                    break
                if progress.state == "cancelled":
                    return
                if progress.state == "failed":
                    self.report({'ERROR'}, progress.status or "Photara render failed")
                    return
                self.update_progress(progress.progress)
                self.update_stats(
                    "Photara",
                    f"Sample {progress.current_sample}/{progress.num_samples}"
                    f" ({progress.substatus or progress.status})",
                )
                if self.test_break():
                    job.cancel()
                    return
                time.sleep(0.05)

            frame = job.wait()
            self._write_render_result(frame, depsgraph, width, height)
            self.update_progress(1.0)
        except Exception as exc:
            import traceback

            detail = "".join(traceback.format_exception_only(type(exc), exc)).strip()
            self.report({'ERROR'}, f"Photara render failed: {detail}")
            traceback.print_exc()
        finally:
            if scene is not None:
                scene.close()

    def _write_render_result(self, frame, depsgraph, width, height):
        result = self.begin_result(0, 0, width, height, layer=depsgraph.view_layer.name)
        try:
            self._fill_render_result(result, frame, depsgraph.view_layer)
        finally:
            self.end_result(result)

    def _fill_render_result(self, result, frame, view_layer):
        layer = result.layers[0]
        # Blender passes are bottom-up; Photara's arrays are top-down.
        layer.passes["Combined"].rect.foreach_set(
            np.ascontiguousarray(np.flipud(frame.beauty), dtype=np.float32).ravel()
        )
        if getattr(view_layer, "use_pass_z", False) and "Depth" in layer.passes:
            layer.passes["Depth"].rect.foreach_set(
                np.ascontiguousarray(np.flipud(frame.depth), dtype=np.float32).ravel()
            )

    # ------------------------------------------------------------------
    # Viewport render
    # ------------------------------------------------------------------

    def view_update(self, context, depsgraph):
        if self._viewport_failed:
            return

        try:
            _load_photara()
            viewport = self._ensure_viewport(context)
            if viewport is None:
                return
            viewport.sync(depsgraph)
            viewport.update_camera(context)
            viewport.restart_render()
        except Exception as exc:
            self._viewport_failed = True
            self._report_viewport_failure(exc)

    def view_draw(self, context, depsgraph):
        if self._viewport_failed:
            return

        try:
            _load_photara()
            viewport = self._ensure_viewport(context)
            if viewport is None:
                return
            if not viewport.sync_state._full_sync_done:
                viewport.sync(depsgraph)

            if viewport.camera_changed(context):
                viewport._cancel_job()
                viewport.update_camera(context)
                viewport.restart_render()
            viewport.draw(context)
            viewport.ensure_timer()
        except Exception as exc:
            self._viewport_failed = True
            self._report_viewport_failure(exc)

    def _report_viewport_failure(self, exc):
        self._viewport_failed = True
        # Disable the viewport path after the first error so a broken API
        # call cannot flood the console with a traceback per redraw.
        try:
            self.report({'ERROR'}, f"Photara viewport disabled: {exc}")
        except Exception:
            pass

    def _ensure_viewport(self, context):
        region = getattr(context, "region", None)
        region_data = getattr(context, "region_data", None)
        if region is None or region_data is None:
            return None

        if self._viewport is None:
            self._viewport = ViewportState(self, region.width, region.height)
        elif (self._viewport.size_changed(context)
              or self._viewport.settings_version != _settings_version()):
            self._viewport.close()
            self._viewport = ViewportState(self, region.width, region.height)
        return self._viewport


class ViewportState:
    """Owns the viewport photara scene, render job and display texture."""

    POLL_INTERVAL = 0.05

    def __init__(self, engine, width, height):
        import photara_rs

        self.engine = weakref.ref(engine)
        self.width = max(1, int(width))
        self.height = max(1, int(height))
        self.settings_version = _settings_version()
        self.camera_matrix = None
        self.job = None
        self.texture = None
        self.timer_running = False

        blender_scene = bpy.context.scene
        settings = blender_scene.photara
        self.samples = max(1, int(settings.viewport_samples))
        self.current_samples = 1
        self.seed = int(settings.seed)

        self.scene = photara_rs.Scene(
            width=self.width,
            height=self.height,
            samples=self.samples,
            seed=self.seed,
            max_bounces=max(1, int(settings.max_bounces)),
            denoise=bool(settings.viewport_denoise),
            adaptive_sampling=bool(settings.viewport_adaptive_sampling),
            device=settings.device,
            **_denoiser_options(settings, viewport=True),
        )
        self.scene.configure_film(**_film_options(blender_scene))
        self.sync_state = sync.SceneSyncState(self.scene)

    def close(self):
        self._cancel_job()
        if bpy.app.timers.is_registered(self._poll_timer):
            bpy.app.timers.unregister(self._poll_timer)
        self.timer_running = False
        self.texture = None
        if getattr(self, "scene", None) is not None:
            self.scene.close()
            self.scene = None
        self.sync_state = None

    def _cancel_job(self):
        self.pending_frame = None
        job = getattr(self, "job", None)
        if job is not None:
            try:
                job.cancel()
            except Exception:
                pass
            self.job = None

    # -- sync / camera -----------------------------------------------------

    def sync(self, depsgraph):
        self._cancel_job()
        self.sync_state.sync(depsgraph)

    def update_camera(self, context):
        self._cancel_job()
        self.scene.configure_film(**_film_options(context.scene))
        self._sync_viewport_world(context)
        region_data = context.region_data
        perspective = getattr(region_data, "is_perspective", None)
        if perspective is None:
            perspective = region_data.view_perspective != 'ORTHO'
        self.orthographic = not perspective

        if perspective:
            view_mode = region_data.view_perspective
            if view_mode == 'CAMERA':
                # Looking through a scene camera: use its actual lens/sensor
                # so the image matches the camera view framing.
                camera_object = bpy.context.scene.camera
                if camera_object is not None:
                    aspect = self.width / max(1.0, float(self.height))
                    self.sync_state.sync_camera(camera_object, aspect)
                else:
                    self.sync_state.sync_camera_view(
                        region_data, self.width, self.height
                    )
            else:
                self.sync_state.sync_camera_view(
                    region_data, self.width, self.height
                )
        self.camera_matrix = self._camera_signature(context)

    @staticmethod
    def _camera_signature(context):
        region_data = context.region_data
        return (
            region_data.view_perspective,
            tuple(value for row in region_data.view_matrix for value in row),
            tuple(value for row in region_data.window_matrix for value in row),
            ViewportState._world_signature(context),
            tuple(_film_options(context.scene).values()),
        )

    @staticmethod
    def _world_signature(context):
        shading = context.space_data.shading
        return (shading.use_scene_world_render, shading.use_scene_lights_render,
                shading.studio_light, shading.studiolight_intensity,
                shading.studiolight_background_alpha,
                context.scene.render.film_transparent)

    def _sync_viewport_world(self, context):
        # Match BlenderViewportParameters in Cycles' blender/viewport.cpp.
        shading = context.space_data.shading
        self.background_mix = 1.0
        self.background_color = tuple(context.scene.world.color) if context.scene.world else (0, 0, 0)
        self.transparent_background = context.scene.render.film_transparent
        if shading.use_scene_world_render:
            sync.sync_world(self.sync_state, context.scene.world)
        else:
            studio = shading.selected_studio_light
            self.scene.world.environment_texture = None
            self.scene.world.color = (1.0, 1.0, 1.0)
            self.scene.world.strength = shading.studiolight_intensity
            if studio and studio.path:
                self.scene.world.environment_texture = {"path": studio.path}
            self.background_mix = shading.studiolight_background_alpha
        for light in self.scene.lights:
            light.visible = shading.use_scene_lights_render

    def camera_changed(self, context):
        return self.camera_matrix != self._camera_signature(context)

    def size_changed(self, context):
        return (
            self.width != max(1, int(context.region.width))
            or self.height != max(1, int(context.region.height))
        )

    # -- rendering ---------------------------------------------------------

    def restart_render(self):
        self._cancel_job()
        if getattr(self, "scene", None) is None or self.scene.closed:
            return
        blender_scene = bpy.context.scene
        settings = blender_scene.photara
        if _settings_version() != self.settings_version:
            self.samples = max(1, int(settings.viewport_samples))
            self.seed = int(settings.seed)
            self.scene.samples = self.samples
            self.settings_version = _settings_version()
        self.current_samples = 1
        self.scene.samples = self.current_samples
        self.scene.seed = self.seed
        self.job = self.scene.render_async()

    # -- polling -----------------------------------------------------------

    def ensure_timer(self):
        if self.timer_running or getattr(self, "job", None) is None:
            return
        self.timer_running = True
        bpy.app.timers.register(self._poll_timer, first_interval=self.POLL_INTERVAL)

    def _poll_timer(self):
        engine = self.engine()
        if engine is None or getattr(self, "job", None) is None:
            self.timer_running = False
            return None

        try:
            progress = self.job.poll()
            if progress.state == "running":
                return self.POLL_INTERVAL
            if progress.state == "done":
                self.pending_frame = self.job.wait()
                self.pending_samples = self.current_samples
                self.job = None
                if self.current_samples < self.samples:
                    self.current_samples = min(self.samples, self.current_samples * 2)
                    self.scene.samples = self.current_samples
                    self.job = self.scene.render_async()
                engine.tag_redraw()
                if self.job is not None:
                    return self.POLL_INTERVAL
            else:
                engine._report_viewport_failure(RuntimeError(progress.status or progress.state))
                self.job = None
        except Exception as exc:
            engine._report_viewport_failure(exc)
            self.job = None

        self.timer_running = False
        return None

    def _upload_result(self, frame):
        import gpu

        # GPUTexture data is bottom-up OpenGL order; flip the top-down result.
        rgba = np.flipud(frame.beauty).astype(np.float32)
        background = np.flipud(~frame.depth_valid)
        if self.background_mix < 1.0 and not self.transparent_background:
            rgba[background, :3] = (rgba[background, :3] * self.background_mix
                                    + np.asarray(self.background_color) * (1.0 - self.background_mix))
        # Keep pixels linear; Blender applies the display transfer at presentation.
        rgba[..., :3] *= 2.0 ** float(bpy.context.scene.view_settings.exposure)
        if self.transparent_background:
            # Composite transparent film against a neutral checker, as Blender
            # does for its viewport, using the native premultiplied film alpha.
            yy, xx = np.indices((self.height, self.width))
            checker = np.where(((xx // 8 + yy // 8) % 2) == 0, 0.18, 0.28)
            alpha = rgba[..., 3:4]
            rgba[..., :3] += checker[..., None] * (1.0 - alpha)
        rgba[..., 3] = 1.0

        pixels = np.ascontiguousarray(rgba, dtype=np.float32).ravel()
        buffer = gpu.types.Buffer('FLOAT', pixels.size, pixels)
        self.texture = gpu.types.GPUTexture(
            size=(self.width, self.height), format='RGBA16F', data=buffer
        )

    # -- drawing -----------------------------------------------------------

    def draw(self, context):
        import gpu
        from gpu_extras.presets import draw_texture_2d
        from mathutils import Matrix

        # GPU resources must be created in view_draw's graphics context,
        # never from the application timer callback.
        frame = getattr(self, "pending_frame", None)
        if frame is not None:
            self._upload_result(frame)
            self.pending_frame = None

        gpu.matrix.push()
        gpu.matrix.load_identity()
        gpu.matrix.load_projection_matrix(Matrix((
            (2.0 / self.width, 0.0, 0.0, -1.0),
            (0.0, 2.0 / self.height, 0.0, -1.0),
            (0.0, 0.0, -1.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        )))
        blend = gpu.state.blend_get()
        depth_test = gpu.state.depth_test_get()
        try:
            gpu.state.depth_test_set('NONE')
            gpu.state.blend_set('NONE')
            if self.texture is not None:
                draw_texture_2d(self.texture, (0.0, 0.0), self.width, self.height)
            else:
                self._draw_placeholder()
        finally:
            gpu.state.blend_set(blend)
            gpu.state.depth_test_set(depth_test)
            gpu.matrix.pop()

    def _draw_placeholder(self):
        import gpu
        from gpu_extras.batch import batch_for_shader

        shader = gpu.shader.from_builtin('UNIFORM_COLOR')
        vertices = (
            (0.0, 0.0),
            (self.width, 0.0),
            (self.width, self.height),
            (0.0, self.height),
        )
        batch = batch_for_shader(
            shader, 'TRIS', {"pos": vertices}, indices=((0, 1, 2), (0, 2, 3))
        )
        shader.uniform_float("color", (0.05, 0.05, 0.05, 1.0))
        batch.draw(shader)


def _settings_version():
    scene = bpy.context.scene
    if scene is None:
        return None
    settings = scene.photara
    return (
        settings.viewport_samples,
        settings.seed,
        settings.max_bounces,
        settings.viewport_denoise,
        settings.viewport_adaptive_sampling,
        tuple(_denoiser_options(settings, viewport=True).values()),
    )
