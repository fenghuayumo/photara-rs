# SPDX-License-Identifier: Apache-2.0
"""Import standard trained Gaussians using Blender's native point-cloud importer."""
import bpy
from bpy_extras.io_utils import ImportHelper
from pathlib import Path
import tempfile

class IMPORT_SCENE_OT_photara_gaussians(bpy.types.Operator, ImportHelper):
    bl_idname = "import_scene.photara_gaussians"
    bl_label = "Import Gaussian Splatting"
    bl_options = {'REGISTER', 'UNDO'}
    filename_ext = ".ply"
    filter_glob: bpy.props.StringProperty(default="*.ply;*.splat", options={'HIDDEN'})

    def execute(self, context):
        from . import ensure_photara_module
        error = ensure_photara_module()
        if error:
            self.report({'ERROR'}, error)
            return {'CANCELLED'}
        try:
            from photara_rs.gsplat import load_gaussians, save_gaussian_ply
            data = load_gaussians(self.filepath)
            with tempfile.TemporaryDirectory(prefix='photara_gsplat_') as directory:
                path = Path(directory) / (Path(self.filepath).stem + '.ply')
                save_gaussian_ply(path, **data)
                bpy.ops.wm.ply_import(filepath=str(path))
            obj = context.object
            if obj is None or obj.type != 'POINTCLOUD':
                raise RuntimeError("This Blender build lacks native Gaussian point-cloud PLY import")
            return {'FINISHED'}
        except (ValueError, OSError, RuntimeError) as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}

def menu_import(self, context):
    self.layout.operator(IMPORT_SCENE_OT_photara_gaussians.bl_idname,
                         text="Gaussian Splatting (.ply/.splat)")
