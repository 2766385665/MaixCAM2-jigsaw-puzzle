# MaixCAM2 拼图装置

这是一个用 MaixCAM2 摄像头识别碎片，再计算拼接位置，最后通过 UART0 控制 STM32 滑轨和电磁铁搬运碎片的项目。

当前代码默认完成的是“3-5块碎片拼成长方形”。代码已经预留任务接口，后续可以更换为完全不同的任务。

建议必须要完整看完这份文件以及代码结构解析


## 先看哪一个文件

如果你第一次接触，请按这个顺序阅读：

1. 本 README：确认项目用途、下载哪些文件、如何运行。
2. [代码结构解析.md](代码结构解析.md)：查看每个文件的作用，以及以后应该改哪个文件。
3. [提交历史记录.md](提交历史记录.md)：了解以前每次提交做了什么。
4. `puzzle_motion_v2/maix_project/main.py`：从设备入口开始读实际运行流程。

代码结构解析文件是详细说明，README 只负责帮助你快速上手，避免在 GitHub 下载后面对一堆文件不知道从哪里开始。

##建议先学习maixcam2的基本操作（1h）



## MaixCAM 端必须放哪些文件

在 MaixVision 中打开并下载整个目录：

`puzzle_motion_v2/maix_project/`

当前主程序实际运行需要下面这些文件：

```text
app.yaml                         MaixVision 打包清单
app.png                          应用图标
main.py                          程序入口、相机、触摸屏、显示和 UART
runtime_pipeline.py              公共流程：坐标转换、绘图、保存
task_pipeline.py                 任务接口和当前矩形拼图任务
piece_vision.py                  A4 矫正和碎片识别
camera_calibration.py            镜头/相机标定参数
pixel_to_pulse_calibration.py    屏幕像素到滑轨脉冲的标定
puzzle_solver_v3.py              当前矩形拼图求解器
puzzle_solver.py                 V3 失败时使用的旧版回退求解器
texture_matcher.py               图案/颜色连续性判断
motion_protocol.py                吸取点、净空检查和 STM32 指令
```

最简单可靠的做法是直接下载整个 `maix_project` 文件夹，不要手动挑选后漏掉依赖文件。

### 也可以放进 MaixCAM，但不是主流程必须的文件

```text
uart0_test.py                    单独测试 UART0，不运行拼图主程序
jiuzheng.py                     单独测试曝光和相机画面
make_demo_plan.py                生成 PC 仿真用的演示 JSON
```

这些文件在 `app.yaml` 中已经列出，随目录一起下载不会影响主程序；如果设备空间很紧，也可以不放，但调试硬件时建议保留。

### 不需要下载到 MaixCAM 的文件

```text
puzzle_motion_v2/pc_tests/       PC 回归测试脚本
puzzle_motion_v2/pc_simulator/   PC 滑轨动画仿真器
puzzle_solver_validation.py      早期 PC 几何验证
puzzle_visual_demo.py            PC 动画演示
requirements-puzzle.txt          PC 端依赖清单
*.md                             说明文档，不参与程序运行
*.jpg / *.png / *.json / *.mp4   样例、测试结果和生成文件
maixcam2_first_version/          第一版识别程序和备份
maix_project/sg/                 独立标定实验程序
dist/                            以前生成的安装包
*.zip / *.docx                   压缩备份和设计报告
```

这些文件对 PC 测试、查历史和写文档有用，但不是 MaixCAM 主程序的运行依赖。GitHub 克隆后不要把所有实验图片、视频和压缩包都复制进设备。

## MaixCAM 如何运行

1. 相机固定在 A4 工作面上方，画面能看到完整 A4 四角。
2. 在 MaixVision 打开 `puzzle_motion_v2/maix_project`，入口选择 `main.py`。
3. 确认 UART0 已连接 STM32，滑轨和电磁铁处于安全状态。
4. 确认 `pixel_to_pulse_calibration.py` 里的标定数据对应当前相机和滑轨。
5. 触摸 START/DETECT，程序会执行：拍照 -> A4 矫正 -> 识别碎片 -> 求解 -> 生成运动计划 -> UART 发送。
6. 触摸 SAVE 可保存本次原图、矫正图、二值图、标注图、运动 JSON 和求解诊断。

设备端默认保存到 `/root/puzzle_motion_output`。一次完整结果通常包括：

```text
*_raw.jpg / *_rectified.jpg / *_binary.png / *_annotated.jpg
*_motion.json                 运动计划
*_stm32.txt                   实际发送帧的文字记录
*_solver_diagnostics.json     求解器诊断信息
```

## PC 端验证和仿真

PC 仿真器使用的就是项目中的同一套几何、纹理和运动计划算法，再把生成的 JSON 画成电脑动画。因此它可以验证：

- 求解器能否找到合理布局；
- 吸取点和目标点是否在范围内；
- 碎片移动、旋转和指令顺序是否正确；
- 运动计划 JSON 和 STM32 帧格式是否基本正确。

但 PC 仿真不能证明真实设备一定安全或准确。它没有模拟电机回差、加减速、滑动、吸盘偏心、残磁、（相机噪声畸变）和 UART 反馈。真实硬件仍必须做限位、低速单块和多块测试。

安装 PC 依赖并运行：

```powershell
python -m pip install -r requirements-puzzle.txt
python puzzle_solver_validation.py --trials 1000 --negative-trials 500
python puzzle_motion_v2/pc_tests/validate_global_solver_v3_synthetic.py
python puzzle_motion_v2/pc_tests/validate_motion_clearance.py
python puzzle_motion_v2/pc_tests/validate_pixel_to_pulse_calibration.py
```

需要动画时运行根目录的 `启动拼图可视化演示.cmd`，需要滑轨仿真时运行 `puzzle_motion_v2/pc_simulator/启动滑轨搬运仿真.cmd`。

详细的测试文件说明见 [代码结构解析.md](代码结构解析.md) 的“PC 测试和仿真文件”部分。

## 更换成七巧板等新任务

不要直接修改 `puzzle_solver_v3.py` 里的矩形规则。推荐做法：

1. 新建 `puzzle_motion_v2/maix_project/tangram_task.py`。
2. 实现 `task_pipeline.py` 中 `TaskAdapter` 要求的任务函数。
3. 在 `main.py` 中把 `ACTIVE_TASK = RectanglePuzzleTask()` 换成 `TangramTask()`。
4. 把 `tangram_task.py` 加入 `app.yaml` 的 `files` 列表。
5. 先在 PC 上验证，再下载到 MaixCAM。

七巧板可以有不同的碎片数量、目标形状和求解方法，但仍应输出统一的 `plan["pieces"]`、`plan["commands"]` 和 `plan["stm32_frames"]`，这样显示、发送和保存功能可以继续复用。

具体函数和修改位置请直接看 [代码结构解析.md](代码结构解析.md)，不要靠猜文件名。

## 我的后续修改建议（重点）

下面四项最值得优先处理的工作，按重要性排序：

### 1. 重新解决摄像头畸变

当前代码使用 `piece_vision.py` 中的自定义稀疏畸变修正，主要是当时时间不足的临时方案，效果和稳定性都不够理想。畸变会直接影响 A4 角点、碎片边长、接缝位置和最后的吸取坐标，因此这是最重要的改进项。

建议：

- 优先评估 MaixCAM 内置的镜头矫正函数，尽量使用成熟的整幅图矫正；
- 解决整幅图矫正带来的内存分配问题，可以考虑降低处理分辨率、复用缓冲区、分阶段处理或只在 START 时矫正一次；
- 使用棋盘格/标定板重新采集相机参数，记录矫正前后的直线和角点误差；
- 不要只看画面是否“看起来变直”，要用 A4 四角和已知长度做数值验证；
- 重新矫正后要同步检查 `piece_vision.py`、纹理采样、屏幕映射和脉冲标定。

### 2. 重新标定屏幕像素与实际脉冲的关系

当前 `pixel_to_pulse_calibration.py` 使用固定参数。更换相机位置、屏幕缩放、滑轨安装或机械结构后，旧参数都会失效。

建议重新采集覆盖四边和中心的标定点，分别记录屏幕坐标、实际脉冲和重复测量误差；再重新拟合映射，并验证边界点、角点和吸取点。标定结果要写入版本记录，不能只在本地临时修改数字。

### 3. 重新选择背景纸板并调整识别参数

背景纸板应该和待识别物块在亮度、颜色或纹理上有明显区别，同时避免反光、阴影和印刷图案干扰。

建议先固定相机曝光和纸板，再重新调整 `piece_vision.py` 中的阈值、背景颜色判断、碎片面积范围和边缘参数。每次只改一组参数，并用同一批真实照片做前后对比，避免“换纸板和换算法”同时发生导致无法判断原因。
必要时直接重新写碎片提取部分，因为这一部分目前写法较为复杂冗余。

### 4. 根据被吸起碎片大小重新设计安全预留

当前吸取安全主要依赖统一的磁铁半径和固定边缘余量。碎片大小、形状、厚度和磁铁接触面不同，安全余量不应该完全一样。

建议按碎片实际尺寸动态计算：磁铁半径 + 边缘预留 + 定位误差 + 旋转/搬运误差。对于太小、太尖或无法满足余量的碎片，应明确标记为不可自动搬运，而不是勉强生成指令。必要时可以完全重写 `motion_protocol.safe_pickup_point()` 和相关净空检查，并增加不同尺寸碎片的专门测试。

## 建议的改进顺序

```text
相机畸变重新标定
    -> 重新标定像素/脉冲
    -> 更换并固定背景纸板
    -> 重新调整视觉参数
    -> 按碎片大小重写吸取安全策略
    -> PC 同算法验证
    -> 无负载和低速实机验证
```

每项改动都建议单独提交，并在提交说明中写清：改了哪些文件、使用了哪组样例、PC 测试结果和真实硬件结果。

## 版本提示

- `24c7d5f`：上一版可运行检查点。
- `210fab7`：易读版重构。
- `8197ac1`：任务适配器模块化。
- `796de64`：按文件重写交接结构说明。

当前 README 和结构文档属于交接资料；实验图片、视频、压缩包和构建输出不要当成主程序源代码。
