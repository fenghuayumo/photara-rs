# Native synchronization in 0.2

## RealTime final images and animation

Requires the rebuilt custom Blender with `_photara_native.realtime_stats` and
matching staged CUDA/OptiX kernels. An ordinary Blender using the compatibility
Python converter reports an error for RealTime final output instead of silently
using an independent-frame fallback. Reference and the existing viewport remain available.

In **Render Properties → Sampling**, select **RealTime** using **Render Mode**
above the **Viewport** and **Render** subpanels. Final-output controls remain
under **Render**:

- **Samples** is the new SPP per output frame; history does not increase this budget.
- **History Frames** bounds the previous raw frames available to GPU reprojection (0 disables it).
- **Warm-up Frames** adds internal frames before a still or whenever animation history
  resets. Only the final image for that timeline frame is saved.
- **NEE-AT** enables RealTime's adaptive light selection.
- **Reset History** discards history before the next output frame.

Use **F12** for a still and **Ctrl+F12** for animation. Set the camera animation,
frame range, resolution, frame rate and output path in Blender as usual. Select
PNG/OpenEXR for a frame sequence, or Video/MPEG-4/H.264 for a video (older Blender
versions use the FFmpeg file-format selector). Color management and encoding
are handled by Blender. RealTime does not promise realtime encoding or a fixed FPS.

```python
scene.render.engine = 'PHOTARA_RS'
p = scene.photara_rs
p.render_mode = 'REALTIME'
p.samples = 4
p.history_samples = 8
p.realtime_warmup_frames = 8
p.denoiser = 'OPTIX'  # OPENIMAGEDENOISE and DLSS also supported
scene.render.filepath = '//frames/camera_'
bpy.ops.render.render(animation=True)
```

Persistent Data is enabled automatically and remains enabled when switching back
to Reference; it retains the renderer/GPU context between animation frames.
Each render job starts fresh. Consecutive camera-only frames preserve history;
frame jumps, camera-object cuts, scene edits, a manual reset, or a history-limit
change invalidate it. Non-camera dependency-graph updates conservatively clear
history; animated geometry/materials/lights can therefore trigger more warm-up work.
Multiple layers use separate resets rather than sharing radiance across layers.

Reprojection runs on GPU before denoising, using the shared bounded raw-frame
implementation. There is no reconstruction pixel readback; Blender's final result
transfer and video encoding still have their usual costs. Native-resolution
OptiX, OIDN, DLSS and output without denoising are supported. Experimental
stable-plane decomposition is not implicitly enabled by selecting DLSS. This is temporal
reconstruction, with the core's reflection/transmission and visibility limitations.
Perspective, single-view output is currently required; motion blur, render border
and DLSS upscaling report errors. Adaptive stopping and time limits are disabled
for RealTime final frames so Samples remains the fixed per-frame budget.

Validation: `scripts/test_blender_realtime_output.py` exercises a still, three-frame
PNG sequence, H.264 video, all three denoisers, and return to Reference. GPU history
diagnostics are read explicitly by the test, outside normal frame output.

The matching custom Blender build now provides `_photara_native`, compiled
from Photara's renderer and the complete Blender/Cycles scene synchronization
layer. This path supports complete SVM material node graphs, textures and
dependency-graph updates. See [native build and validation](../docs/BLENDER_NATIVE_SYNC.md).
The earlier standalone-wheel display bridge below remains the fallback for
Blender builds without this module and has more limited material conversion.

Renderer settings use `scene.photara_rs` (for example,
`scene.photara_rs.samples = 32`). Saved settings from the earlier
`scene.photara` namespace are copied automatically when enabling the add-on
or opening a blend file. The `photara_render` dataset export add-on retains
its own `scene.photara` settings; both add-ons can be enabled together.

# Photara Blender 插件

在 Blender 中直接使用 Photara（OptiX / Metal 路径追踪）渲染。插件通过
`photara_rs` Python 包驱动渲染器核心，无需修改或重新编译 Blender。

## 支持的 Blender 版本

- 插件面向 Blender 4.2 LTS 及以上，当前实测版本为下述本地 5.3 构建。
  wheel 的原生扩展必须匹配 Blender 内置 Python；Python 3.11 用 cp311，
  本地 Python 3.13 用 cp313。请以具体构建检测结果为准。
- Windows x64 + NVIDIA RTX GPU（OptiX）
- Apple Silicon macOS（Metal）；已验证 Blender 5.3 / Python 3.13 的独立插件路径。

## 安装

### 一键安装（推荐）

先关闭 Blender，在仓库根目录执行：

```bat
python scripts\install_blender.py --blender "D:\ProgramCode\C++\build_windows\bin"
```

脚本自动检测 Blender 内置 Python，选择 `dist/` 里匹配的 wheel，安装
Python 包和插件，检查 OptiX 设备并保存启用状态。重新打开 Blender，
在渲染引擎下拉框选择 **Photara**。重复执行可更新插件。

已在上述目录的 **Blender 5.3.0 Beta / Python 3.13.13 / RTX 5090 D v2**
实测：独立 Python 渲染、F12 的 Combined/Depth、多层 EXR、集合实例、
共享数据的独立灯光、材质/位置更新、渲染视口及视口参数变化后重新渲染。
验证场景位于 `test_outputs/photara_demo.blend`。

```bat
"D:\ProgramCode\C++\build_windows\bin\blender.exe" --background --python-exit-code 1 --python scripts\test_blender.py
"D:\ProgramCode\C++\build_windows\bin\blender.exe" --factory-startup --python-exit-code 1 --python scripts\test_blender.py -- --viewport
```

以下为手动安装和构建方法。

### 第 1 步：把 `photara_rs` 装进 Blender 的 Python

```bat
:: 找到 Blender 自带的 python.exe，例如：
:: C:\Program Files\Blender Foundation\Blender 4.2\4.2\python\bin\python.exe

"Blender的python.exe" -m pip install dist\photara_rs-<version>-cp311-cp311-win_amd64.whl
```

wheel 由仓库根目录的 `scripts\build_python_windows.bat` 等流程产出
（见根目录 `README.md` 的 Python package 一节）。

对本地 Blender 5.3（Python 3.13），可用 Blender 的 Python 重新编译绑定
（切换解释器只需重编 Python 绑定）：

```bat
cmake -S . -B build -U "_Python*" -U "PYBIND11_*" -U "PYTHON_*" ^
  -DPython_EXECUTABLE="<blender的python.exe>" ^
  -DPython_INCLUDE_DIR="<blender源码 lib\windows_x64\python\313\include>" ^
  -DPython_LIBRARY="<blender源码 lib\windows_x64\python\313\libs\python313.lib>"
cmake --build build --target photara_python
"<blender的python.exe>" scripts\stage_python_package.py
"<blender的python.exe>" -m pip wheel --no-build-isolation --no-deps -w dist photara\python
```

以上构建命令在 VS 开发者命令行中运行。切换 ABI 前，先把包目录里旧的
`_photara*.pyd` 移到其他目录备份；stage 和 wheel 构建会拒绝混合 ABI。
打包使用的 Python 必须与刚编译的绑定一致，不能用普通 Python 3.11
替 Python 3.13 的扩展打包。

### 第 2 步：安装插件

```bat
python scripts\package_blender_addon.py
```

生成 `dist\photara_blender_addon-<版本>.zip`，然后在 Blender 中
`Edit > Preferences > Add-ons > Install...` 选择该 zip 并启用
**Render: Photara Renderer**。

如果 `photara_rs` 没有装进 Blender 的 Python，也可以在插件偏好设置里
填写包含 `photara_rs` 包的目录。

## 使用

1. `Render Properties` 中把渲染引擎切换为 **Photara**。
2. F12 最终渲染：输出 `Combined`（RGBA，线性）与 `Depth`（勾选
   `Z` pass 时）。
3. 视口切换到 `Rendered`：一个原生渲染任务持续累积采样，通过核心
   `OutputDriver.update_render_tile` 输出中间结果；约 50 ms 的调度间隔为
   目标值，实际刷新速度取决于单次 GPU 工作耗时。达到采样上限后停止。
   视角/场景变化时异步取消旧任务并重置采样，保留同一原生 Session、几何
   和设备；只有改变会话级设置才重建会话。所有帧保持完整视口分辨率。
   不再用多次独立短渲染模拟累积，也不再主动清空旧画面制造闪烁。
   颜色和深度一起绘制，由 Blender 使用深度处理覆盖层遮挡。

性能检查可通过 Blender `--python scripts/benchmark_blender_viewport.py`
运行。`scripts/test_blender_alignment.py` 检查多个视角的渲染投影对齐。
本地自定义 Blender 已启用原生 DisplayDriver 桥接，直接复用 Cycles 的
Vulkan/CUDA 图形互操作路径，显示像素不经过 CPU。未打桥接补丁的 Blender
仍使用兼容的 CPU 传输路径。构建、ABI 和所有权说明见 [GPU_DISPLAY.md](GPU_DISPLAY.md)。
专项验证记录在 `test_outputs/gpu_display_verification.json`，包括 GPU 更新、
CPU 映射和三个视角的像素对齐检查。
4. Render Properties 按 Cycles 布局组织：顶部 `Device: GPU Compute`；
   `Sampling > Viewport / Render` 分别设置采样数、自适应采样和降噪；
   `Sampling > Advanced` 设置种子；`Light Paths > Max Bounces` 设置总反弹次数。
   当前仅提供 OptiX 渲染设备；展开 Denoise 可选择 Automatic、OptiX 或
   OpenImageDenoise，并设置 Passes。OIDN 另有 Prefilter、Quality 和 Use GPU。
   视口和最终渲染分别保存降噪设置。尚未开放的噪声阈值、最小采样数、
   各类反弹上限等选项不显示。

视口遵循着色弹出面板的 `Scene World` / `Scene Lights` 开关。关闭
`Scene World` 时使用 Blender 所选的预览 HDRI，并使用其强度和背景
可见度；开启时显示场景 World（World 本身较暗时，背景也会较暗）。
`Film` 提供曝光倍率、Transparent、Transparent Glass / Roughness Threshold，
以及 Box / Gaussian / Blackman-Harris 像素滤镜。透明 Film 由原生核心产生
预乘 RGBA，视口用该 alpha 合成棋盘格。

`Color Management` 使用 Blender 原生面板；Display、View Transform、Look、
Exposure、Gamma 等控制最终渲染的显示和输出。视口目前只有曝光和显示转换，
尚未完整应用 AgX / Look / 曲线等设置。`Volumes` 显示尚未支持提示，体积
材质和网格导出尚未实现，不能通过调节采样步长启用体积渲染。

## 已同步的数据

| Blender | Photara |
|---|---|
| Mesh（含修改器计算结果、实例/dupli） | 三角化网格，顶点法线、UV、`matrix_world` |
| Principled BSDF | base color / metallic / roughness / IOR / transmission / emission |
| 面光 / 点光 / 平行光 / 聚光 | 强度（W）、颜色、尺寸、半径、角度、penumbra |
| 世界（Background / Environment Texture .hdr） | 背景色 / 等距柱状环境贴图 |
| 相机（fov、裁剪、sensor fit、像素比） | 透视相机 |
| 物体 `pass_index` | `object_id` 通道 |

同步是增量的（基于 `depsgraph.updates`，与 Cycles 的 `BlenderSync`
同一策略）：只移动对象时只更新 4x4 矩阵，不触碰几何；只改材质参数时
只推送着色器数值；未变化的网格零开销。批量数据（顶点/三角形/UV/法线）
通过 `foreach_get` + numpy 直接复制，为 C 级速度。

## 已知限制

- 正交视口暂不支持；最终渲染遇到正交或 panorama 相机会明确报错。
- 多材质网格使用第一个材质槽（photara 材质按网格整体指定）。
- 原生 GPU 路径直接使用本地 Cycles 的显示驱动、着色器和同步机制；兼容
  CPU 路径的视口颜色管理仍是近似。最终渲染由 Blender 处理颜色管理。
- 预览 HDRI 的旋转和背景模糊尚未同步。
- 渲染边框（Border render）、panorama 相机、体积、毛发等
  暂未同步。
- 当前保证透视相机、方形像素及无偏移的取景；camera shift、非方形像素、
  景深和运动模糊尚未同步。材质按活动 Material Output 选择 Principled，支持数值参数、RGB/Value 常量及通过 reroute 连接的 Base Color 图像纹理（外部或打包文件、平面投影、活动 UV）。贴图断开时会清除旧纹理；自定义坐标映射及完整节点图尚未支持。
- 每帧动画渲染会重建场景（保证正确性）；后续可复用会话。

## 架构说明

上游 Cycles 把同步层（`intern/cycles/blender/`）编译进 Blender 本体；
本项目通过公开的 `photara_rs` API 在插件内完成同样的事，换来零编译
依赖。若未来需要进一步压低同步开销，`photara/include/photara/photara.h`
的 C ABI 也支持把 sync 层下沉为 C++ 实现（路径与 Cycles 一致），插件
结构无需改变。

## macOS Metal 演示场景

先按 [macOS 构建说明](../BUILDING.md#macos-apple-silicon) 编译原生扩展，
运行 `python scripts/package_blender_addon.py`，在 Blender 中安装生成的插件 zip。
在插件偏好设置的 **photara_rs Path** 中选择本仓库的 `photara/python` 目录。
当前开发安装会使用这里的原生扩展和 `install/source` 运行时源码。

用 Blender 执行 `scripts/create_blender_metal_demo.py` 可生成并渲染三灯棚拍场景：

```sh
/Applications/Blender.app/Contents/MacOS/Blender --background --python-exit-code 1 \
  --python scripts/create_blender_metal_demo.py
```

结果位于 `test_outputs/blender_metal_demo/`。打开 `photara_metal_studio.blend`，
渲染引擎已设为 **Photara**，设备为 **Apple Metal**，Reference 模式、128 Samples、
OpenImageDenoise。按 **F12** 渲染，**F11** 查看或返回渲染结果。
场景使用基础 Principled 材质，包含陶瓷、金属、玻璃、三盏矩形面光和无缝背景。
