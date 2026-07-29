# MaixCAM 2拼图求解与滑轨仿真

## 1. 文件结构

```text
maix_project/
├─ main.py               Maix应用入口、触摸交互
├─ piece_vision.py       v1.1碎片识别和多边形拟合
├─ puzzle_solver.py      无Shapely几何拼图搜索
├─ motion_protocol.py    安全吸取点及搬运指令JSON
└─ make_demo_plan.py     PC侧生成固定测试指令

pc_simulator/
├─ pc_motion_simulator.py
├─ demo_motion_plan.json
├─ demo_motion_simulation.mp4
├─ demo_motion_contact_sheet.jpg
└─ 启动滑轨搬运仿真.cmd
```

Maix端只依赖系统自带的`maix`、`numpy`和`cv2`，不需要Shapely、SciPy、
PyQt或YOLO。

## 2. Maix端流程

```text
摄像头图像
→ A4透视矫正
→ 识别1～4块3～5边形
→ 转换为A4厘米坐标
→ 枚举等长接缝并做非重叠回溯
→ 检查9～12 cm × 5～9 cm矩形
→ 把矩形移动到A4下半区
→ 计算每块安全吸取点
→ 输出XY/Z/θ搬运指令JSON
```

坐标系固定为：

```text
原点：A4左上角
+X：向右
+Y：向下
角度正方向：图像中的顺时针
长度单位：cm
```

## 3. 在MaixVision运行

在MaixVision中打开`maix_project`整个文件夹，并使用“运行项目”，入口必须是：

```text
main.py
```

操作顺序：

1. 让完整A4四角对准绿色框。
2. 点`DETECT`进入实时识别。
3. 确认所有碎片都有正确编号和边数。
4. 点`FREEZE`，程序立即搜索矩形。
5. 成功时，下半区显示目标轮廓、吸取点、箭头和旋转角。
6. 点`SAVE`保存图像及搬运JSON。

文件保存在：

```text
/root/puzzle_motion_output
```

其中最重要的是：

```text
pieces_时间_motion.json
```

## 4. 搬运指令

每块碎片的核心指令包含：

```json
{
  "piece_id": 1,
  "pick_x_cm": 4.2,
  "pick_y_cm": 6.1,
  "place_x_cm": 12.3,
  "place_y_cm": 19.4,
  "rotate_deg_clockwise": -37.5,
  "pickup_clearance_cm": 1.2,
  "pickup_safe": true
}
```

动作顺序为：

```text
MOVE_XY_PICK
Z_DOWN
MAGNET_ON
Z_UP
ROTATE
MOVE_XY_PLACE
Z_DOWN
MAGNET_OFF
Z_UP
ROTATE_HOME
```

吸取点不是强制使用轮廓中心。程序先寻找满足“电磁铁半径+安全余量”的区域，
再选择最靠近面积质心的位置。放置吸取点通过完整刚体变换计算，因此吸取点
改变不会破坏最终拼图位置。

默认工具参数：

```python
MAGNET_RADIUS_CM = 0.50
PICKUP_MARGIN_CM = 0.20
```

需要按实际电磁铁接触面修改。

## 5. PC滑轨模拟

双击：

```text
pc_simulator\启动滑轨搬运仿真.cmd
```

默认播放内置四块混合形状测试。也可以把Maix导出的`*_motion.json`直接拖到
该CMD文件上。

键盘：

```text
Space  暂停/继续
R      重新播放
Esc    退出
```

动画显示：

- XY空载移动；
- Z下降、吸取和抬升；
- 绕实际吸取点旋转；
- 携带碎片移动到目标吸取点；
- 释放并依次拼成矩形；
- 机械头轨迹、磁铁状态、角度和最终指令误差。

生成无窗口视频：

```powershell
D:\label\python\python.exe pc_motion_simulator.py demo_motion_plan.json `
  --headless --video motion_simulation.mp4
```

## 6. 当前限制

- 当前几何搜索以“完整接缝边两两匹配”为主；T形接缝中“一条长边同时对应
  两条短边”的情况需要后续增加边分段搜索。
- 扑克牌图案存在几何多解时，仍需增加接缝纹理连续性评分。
- 指令坐标是A4厘米坐标，接真实滑轨前必须标定到电机毫米坐标。
- 尚未补偿旋转轴与电磁铁中心的偏心；实机应增加TCP偏心标定。
- PC动画验证的是数学运动，不包含电机回差、加速度、碎片滑动和磁铁剩磁。

## 7. 当前软件验证

- OpenCV/Numpy求解器：1000组随机合法拼图全部求解成功，1000组不相关
  负样本错误接受为0。
- 1000组运动计划的吸取点刚体映射及目标区域边界全部通过；其中6组因默认
  5 mm半径电磁铁加2 mm余量无法满足而被正确标记为`pickup_safe=false`。
- 固定混合形状演示：三角形、四边形和五边形最终恢复为10×6 cm矩形。
- 模拟器报告的最大最终顶点指令误差约0.001 mm（JSON四位小数舍入造成）。
