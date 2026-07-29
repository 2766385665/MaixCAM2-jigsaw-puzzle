# 拼图算法测试与可视化

## 直接观看拼接过程

双击：

`启动拼图可视化演示.cmd`

程序会自动：

1. 生成一组随机四片拼图；
2. 把碎片随机旋转、散放在 A4 上半区域；
3. 提取轮廓并求解目标矩形；
4. 动画显示每片碎片移动、旋转并完成拼接；
5. 将本次过程保存为 `puzzle_validation_output/puzzle_demo.mp4`。

窗口按键：

- `Space`：暂停或继续；
- `R`：生成新的随机测试图例；
- `Q` 或 `Esc`：退出。

## 批量算法验证

在 PowerShell 中进入本目录后运行：

```powershell
D:\label\python\python.exe puzzle_solver_validation.py --trials 1000 --negative-trials 500
```

结果保存在：

`puzzle_validation_output/summary.json`

## 只生成演示视频

```powershell
D:\label\python\python.exe puzzle_visual_demo.py --headless
```
