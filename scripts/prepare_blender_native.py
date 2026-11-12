"""Build Photara's own renderer with the complete vendored Blender integration.

Generate an isolated source tree with distinct C++ symbols/CMake targets, then
add it to a compatible Blender build. This links Blender APIs once in the host;
no Blender-owned scene or shader allocations cross into the standalone wheel.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent.parent


def write_if_changed(path, text):
    data = text.encode('utf-8')
    if not path.exists() or path.read_bytes() != data:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


def patch_once(path, before, after):
    text = path.read_text(encoding='utf-8')
    if after in text:
        return
    if text.count(before) != 1:
        raise RuntimeError(f'Unsupported Blender integration anchor: {path}')
    write_if_changed(path, text.replace(before, after))


def prepare(source: Path, build: Path):
    stage = ROOT / 'build' / 'blender_native_source'
    functions = ('CCL_python_module_init', 'CCL_log_init', 'CCL_implicit_sharing_init',
                 'CCL_resolve_texture_cache', 'CCL_generate_texture_cache')
    for path in (ROOT / 'src').rglob('*'):
        if not path.is_file() or path.suffix not in {'.cpp', '.h', '.c', '.cu', '.cuh', '.cmake', '.txt', '.py', '.json', '.tables', '.inc', '.mm', '.metal', '.osl'}:
            continue
        relative = path.relative_to(ROOT / 'src')
        if '__pycache__' in relative.parts:
            continue
        text = path.read_text(encoding='utf-8')
        if path.suffix in {'.cmake', '.txt'}:
            text = text.replace('cycles_', 'photara_').replace('bf_intern_cycles', 'bf_intern_photara')
            text = text.replace('zstd_compress', 'photara_zstd_compress').replace('photara_zstd_compress.cpp', 'zstd_compress.cpp')
            text = text.replace('namespace\\ ccl\\ {', 'namespace\\ photara_native\\ {')
        else:
            text = text.replace('ccl::', 'photara_native::')
            text = text.replace('namespace ccl', 'namespace photara_native')
        text = text.replace('"../../guardedalloc/MEM_guardedalloc.h"', '"MEM_guardedalloc.h"')
        for name in functions:
            text = text.replace(name, name.replace('CCL_', 'PHOTARA_'))
        if relative.as_posix() == 'device/optix/util.h':
            text = text.replace('#pragma once', '#pragma once\n\n#define g_optixFunctionTable photara_optixFunctionTable')
        if relative.as_posix() == 'integrator/denoiser_dlss.cpp':
            text = text.replace('#  define NVSDK_NGX_HEADER_ONLY', '/* Reuse the host SDK loader; renderer/denoiser state remains independent. */')
        if relative.parts[0] == 'blender':
            if path.suffix == '.cpp':
                text = text.replace('#include \"blender/util.h\"', '#include \"blender/util.h\"\n#include \"DEG_depsgraph_query.hh\"')
                text = text.replace('RNA_id_pointer_create(&b_scene.id)', 'RNA_id_pointer_create(DEG_get_original_id(&b_scene.id))')
                text = text.replace('RNA_id_pointer_create(&b_scene->id)', 'RNA_id_pointer_create(DEG_get_original_id(&b_scene->id))')
            # Scene controls are independent. Per-material/object/world/viewport
            # RNA keeps Blender's complete existing Cycles property schema.
            text = text.replace('RNA_pointer_get(&scene_rna_ptr, "cycles")',
                                'RNA_pointer_get(&scene_rna_ptr, "photara_native")')
            if relative.name == 'python.cpp':
                text = text.replace('"_cycles"', '"_photara_native"')
                text = text.replace('#include <Python.h>', '#include <Python.h>\n#include \"scene/camera.h\"\n#include \"scene/shader.h\"\n#include \"scene/shader_graph.h\"')
                diagnostics = (ROOT/'src/blender/photara_diagnostics.inc').read_text()
                text = text.replace('static PyMethodDef methods[] = {', diagnostics + '\nstatic PyMethodDef methods[] = {\n    {\"session_stats\", photara_session_stats_func, METH_O, \"Photara scene and resolution diagnostics\"},')
            if relative.name == 'logging.cpp':
                text = text.replace('{"cycles"}', '{"photara_native"}')
            if relative.name == 'sync.cpp':
                # Full viewport pixels; DLSS's explicit upscale factor remains
                # the only permitted reduction in internal render resolution.
                # Insert before the final return in get_session_params only.
                start = text.index('SessionParams BlenderSync::get_session_params(')
                end = text.index('\nDenoiseParams ', start)
                section = text[start:end]
                section = section.replace('  return params;',
                    '  params.use_resolution_divider = false;\n'
                    '  params.pixel_size = 1;\n'
                    '  params.preview_update_interval = background ? 0.0 : 1.0 / 60.0;\n'
                    '  return params;')
                text = text[:start] + section + text[end:]
            if relative.name == 'device.cpp':
                begin = text.index('DeviceInfo blender_device_info(')
                body = text.index('\n{', begin)
                finish = text.index('\nCCL_NAMESPACE_END', body)
                text = text[:body] + '''
{
  (void)b_preferences;
  (void)b_scene;
  (void)background;
  const auto devices = Device::available_devices(DEVICE_MASK_OPTIX);
  for (const auto &device : devices) {
    if (device.meets_driver_requirement) {
      preferences_device = device;
      if (preview) {
        preferences_device.kernel_optimization_level = KERNEL_OPTIMIZATION_LEVEL_OFF;
      }
      return preferences_device;
    }
  }
  return preferences_device;
}
''' + text[finish:]
            if relative.name == 'CMakeLists.txt':
                text = text.replace('../../../source/blender/makesrna', '${CMAKE_SOURCE_DIR}/source/blender/makesrna')
                text = re.sub(r'delayed_install\([^\n]+ADDON_FILES[^\n]+\)', '', text)
        if relative.as_posix() == 'CMakeLists.txt':
            text = text.replace('  include(cmake/external_libs.cmake)\n', '')
            text = text.replace('add_subdirectory(app)\n', '').replace('add_subdirectory(doc)\n', '')
            text = text.replace('  add_subdirectory(test)\n', '')
            text = text.replace('"scripts/addons_core/cycles"', '"scripts/addons_core/photara_native"')
        write_if_changed(stage / relative, text)

    wrapper = stage / 'embedded' / 'CMakeLists.txt'
    write_if_changed(wrapper, f'''# Generated by prepare_blender_native.py.
set(CYCLES_STANDALONE_REPOSITORY FALSE)
set(WITH_CYCLES_STANDALONE OFF)
set(WITH_CYCLES_STANDALONE_GUI OFF)
set(WITH_CYCLES_OSL OFF)
set(WITH_CYCLES_USD OFF)
set(WITH_USD OFF)
set(WITH_CYCLES_HYDRA_RENDER_DELEGATE OFF)
set(WITH_CYCLES_DEVICE_HIP OFF)
set(WITH_CYCLES_DEVICE_HIPRT OFF)
set(WITH_CYCLES_DEVICE_METAL OFF)
set(WITH_CYCLES_DEVICE_ONEAPI OFF)
set(WITH_CYCLES_CUDA_BINARIES OFF)
set(WITH_CYCLES_OPTIX_BINARIES OFF)
set(WITH_GTESTS OFF)
add_definitions(-DWITH_SYSTEM_PUGIXML)
include_directories("{(ROOT/'third_party/atomic').as_posix()}"
                    "{(source/'extern/cuew/include').as_posix()}"
                    "{(source/'intern/guardedalloc').as_posix()}"
                    "{(source/'intern/sky/include').as_posix()}"
                    "{(source/'intern/mikktspace').as_posix()}")
add_subdirectory("{stage.as_posix()}" "${{CMAKE_CURRENT_BINARY_DIR}}/renderer")
''')
    addon = ROOT / 'photara_rs_engine'
    for name, destination in (('properties.py', 'native_properties.py'),
                              ('engine.py', 'native_engine.py'),
                              ('camera.py', 'native_camera.py')):
        code = (ROOT / 'src/blender/addon' / name).read_text(encoding='utf-8')
        code = code.replace('import _cycles', 'import _photara_native as _cycles')
        code = code.replace('from . import engine', 'from . import native_engine as engine')
        code = code.replace('from . import camera', 'from . import native_camera as camera')
        code = code.replace('CyclesRenderSettings', 'PhotaraNativeRenderSettings')
        code = code.replace('bpy.types.Scene.cycles', 'bpy.types.Scene.photara_native')
        code = code.replace('scene.cycles', 'scene.photara_native')
        code = code.replace('context.preferences.addons[__package__]', "context.preferences.addons['cycles']")
        if name == 'engine.py':
            code = code.replace('path = os.path.dirname(__file__)',
                                'from photara_rs._backend import resource_dir\n    path = resource_dir()')
        write_if_changed(addon / destination, code)

    # Benchmark controls on the existing Cycles backend are opt-in and do not
    # change its normal renderer behavior. Both sides can report actual pixels.
    host_python = source / 'intern/cycles/blender/python.cpp'
    host_stats = (ROOT/'src/blender/photara_diagnostics.inc').read_text()
    host_code = host_python.read_text(encoding='utf-8')
    if 'photara_session_stats_func' not in host_code:
        host_code = host_code.replace('#include <Python.h>', '#include <Python.h>\n#include "scene/camera.h"\n#include "scene/shader.h"\n#include "scene/shader_graph.h"')
        host_code = host_code.replace('static PyMethodDef methods[] = {', host_stats + '\nstatic PyMethodDef methods[] = {\n    {"session_stats", photara_session_stats_func, METH_O, "Scene and resolution diagnostics"},')
        write_if_changed(host_python, host_code)
    host_sync = source / 'intern/cycles/blender/sync.cpp'
    host_code = host_sync.read_text(encoding='utf-8')
    if 'CYCLES_VIEWPORT_FULL_RESOLUTION' not in host_code:
        start = host_code.index('SessionParams BlenderSync::get_session_params(')
        end = host_code.index('\nDenoiseParams ', start)
        section = host_code[start:end].replace('  return params;',
            '  if (!background && std::getenv("CYCLES_VIEWPORT_FULL_RESOLUTION")) {\n'
            '    params.use_resolution_divider = false;\n'
            '    params.pixel_size = 1;\n'
            '  }\n  return params;')
        host_code = host_code[:start]+section+host_code[end:]
        write_if_changed(host_sync, host_code)

    # Separate library/module, independently owned Photara renderer and scene.
    patch_once(source / 'intern/CMakeLists.txt',
               '  add_subdirectory(cycles)\n',
               '  add_subdirectory(cycles)\n'
               '  if(PHOTARA_NATIVE_SOURCE)\n'
               '    add_subdirectory("${PHOTARA_NATIVE_SOURCE}" "${CMAKE_BINARY_DIR}/photara_native")\n'
               '  endif()\n')
    patch_once(source / 'source/blender/python/intern/CMakeLists.txt',
               '    bf_intern_cycles\n',
               '    bf_intern_cycles\n'
               '    $<$<BOOL:${PHOTARA_NATIVE_SOURCE}>:bf_intern_photara>\n')
    path = source / 'source/blender/python/intern/bpy_interface.cc'
    patch_once(path, '#  include "CCL_api.h"',
               '#  include "CCL_api.h"\n'
               '#  ifdef WITH_PHOTARA_NATIVE\n'
               'namespace blender { void *PHOTARA_python_module_init(); }\n'
               '#  endif')
    patch_once(path, '    {"_cycles", CCL_initPython},',
               '    {"_cycles", CCL_initPython},\n'
               '#  ifdef WITH_PHOTARA_NATIVE\n'
               '    {"_photara_native", []() -> PyObject * {\n'
               '       return (PyObject *)blender::PHOTARA_python_module_init();\n'
               '     }},\n'
               '#  endif')
    patch_once(source / 'source/blender/python/intern/CMakeLists.txt',
               '  add_definitions(-DWITH_CYCLES)\n',
               '  add_definitions(-DWITH_CYCLES)\n'
               '  if(PHOTARA_NATIVE_SOURCE)\n'
               '    add_definitions(-DWITH_PHOTARA_NATIVE)\n'
               '  endif()\n')
    print('Configure Blender with: -DPHOTARA_NATIVE_SOURCE=' + wrapper.parent.as_posix())
    return wrapper.parent


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--blender-source', type=Path, required=True)
    parser.add_argument('--blender-build', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.blender_source.resolve(), args.blender_build.resolve())
