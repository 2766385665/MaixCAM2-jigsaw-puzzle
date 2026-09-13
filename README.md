# MaixCAM2 拼图装置（易读版）

这是一个“相机识别碎片 -> 任务求解 -> 规划滑轨搬运 -> UART0 控制 STM32”的端到端项目。当前默认任务是矩形拼图；任务流程已通过 `TaskAdapter` 解耦，七巧板等新任务可替换自己的识别后处理、求解器和运动策略。当前提交是代码整理后的“易读版”；上一提交 `24c7d5f` 保留为可运行基线。

## 快速定位

- MaixCAM 入口：`puzzle_motion_v2/maix_project/main.py`
- 运行时流程：`puzzle_motion_v2/maix_project/runtime_pipeline.py`
- 可替换任务接口：`puzzle_motion_v2/maix_project/task_pipeline.py`
- 视觉识别：`puzzle_motion_v2/maix_project/piece_vision.py`
- V3 几何求解：`puzzle_motion_v2/maix_project/puzzle_solver_v3.py`
- 运动协议：`puzzle_motion_v2/maix_project/motion_protocol.py`
- 代码结构说明：[`代码结构解析.md`](代码结构解析.md)
- 提交历史：[`提交历史记录.md`](提交历史记录.md)

## 运行

### MaixCAM2

在 MaixVision 中打开 `puzzle_motion_v2/maix_project`，入口选择 `main.py`。相机需能看到完整 A4 工作面；运行前确认 UART0 已连接 STM32，且 `pixel_to_pulse_calibration.py` 中的标定范围与实际滑轨一致。触摸 `START/DETECT` 后执行一次识别和搬运，`SAVE` 保存调试会话。

### PC 验证

PC 验证依赖 Python、NumPy、OpenCV、Shapely 和 PyQt5：

```powershell
python -m pip install -r requirements-puzzle.txt
python puzzle_solver_validation.py --trials 1000 --negative-trials 500
python puzzle_visual_demo.py --headless
```

更多设备端和仿真操作见 `puzzle_motion_v2/README_拼图求解与滑轨仿真.md`。MaixCAM 固件环境使用系统自带 `maix`，不要在 PC 环境安装同名替代包。

## 更换任务

`main.py` 中的 `ACTIVE_TASK` 是任务插槽，默认指向 `RectanglePuzzleTask`。
接手者实现 `TaskAdapter`（见 `代码结构解析.md`）后，将该变量替换为七巧板适配器，
即可保留相机、触摸屏、显示和 UART 外壳。新任务可以使用完全不同的碎片数量、
目标形状、求解器和运动规划，但应返回统一的运动计划字典供 UI 和保存模块使用。

## 输出文件

设备端默认写入 `/root/puzzle_motion_output`：原始/矫正/二值/标注图，运动计划 `*_motion.json`，STM32 帧 `*_stm32.txt`，以及可复现求解过程的 `*_solver_diagnostics.json`。

## 当前不足与风险

- V3 求解仍是启发式搜索；相同图案或多解扑克牌布局可能需要纹理评分和人工复核，23 秒超时后不保证有解。
- 视觉阈值、A4 角点和稀疏几何标定依赖现场光照、相机安装姿态；换设备后必须重新验证。
- 像素到脉冲的标定目前是固定参数，未自动学习温漂、机械回差、加减速和负载变化。
- 旋转轴与电磁铁中心的偏心补偿尚未闭环；吸盘磨损、碎片翘曲、滑动和残磁也未在软件中建模。
- UART 发送前有边界和净空检查，但没有真实 STM32 的反馈确认、急停回执或断电恢复流程。
- PC 仿真只验证几何轨迹和协议时序，不等价于真实电机、电磁铁和相机的硬件验收。
- 当前模块化主要覆盖运行时编排；`piece_vision.py`、`puzzle_solver_v3.py` 和 `motion_protocol.py` 仍然较大，后续应以回归测试为先逐步拆分。

## 交接建议

先运行 PC 验证，再在无负载状态测试 UART 和滑轨限位，最后使用已保存的真实采集样例做 MaixCAM 回归。任何坐标系、标定参数或协议字段变更，都应在提交说明中记录输入样例、验证命令和硬件结果。
