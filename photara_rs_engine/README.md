# Photara Blender 插件

在 Blender 中直接使用 Photara（OptiX 路径追踪）渲染。插件通过
`photara_rs` Python 包驱动渲染器核心，无需修改或重新编译 Blender。

## 支持的 Blender 版本

- 插件面向 Blender 4.2 LTS 及以上，当前实测版本为下述本地 5.3 构建。
  wheel 的原生扩展必须匹配 Blender 内置 Python；Python 3.11 用 cp311，
  本地 Python 3.13 用 cp313。请以具体构建检测结果为准。
- Windows x64 + NVIDIA RTX GPU（OptiX）

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

"Blender的python.exe" -m pip install dist\photara_rs-0.1.0-cp311-cp311-win_amd64.whl
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
3. 视口切换到 `Rendered` 着色模式：插件先显示低采样画面，再用更高
   采样数刷新；相机或场景变化会自动重新渲染。当前通过多次独立渲染
   实现渐进预览，每次需要重新采样，因此性能和 Cycles 的原生累积不同。
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
- 视口颜色管理是近似（线性纹理 + 曝光，由 Blender 视口进行显示转换）；最终渲染由 Blender 的视图变换
  正确处理。
- 预览 HDRI 的旋转和背景模糊尚未同步。
- 渲染边框（Border render）、panorama 相机、体积、毛发等
  暂未同步。
- 当前保证透视相机、方形像素及无偏移的取景；camera shift、非方形像素、
  景深和运动模糊尚未同步。材质仅同步 Principled 数值参数，不转换完整节点图。
- 每帧动画渲染会重建场景（保证正确性）；后续可复用会话。

## 架构说明

上游 Cycles 把同步层（`intern/cycles/blender/`）编译进 Blender 本体；
本项目通过公开的 `photara_rs` API 在插件内完成同样的事，换来零编译
依赖。若未来需要进一步压低同步开销，`photara/include/photara/photara.h`
的 C ABI 也支持把 sync 层下沉为 C++ 实现（路径与 Cycles 一致），插件
结构无需改变。
