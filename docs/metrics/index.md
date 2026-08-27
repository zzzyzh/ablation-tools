# Metrics

`Metrics/` 保存可以跨 General、MLLM、Embodied 复用的指标计算。目录按指标族组织，而不是按任务类型组织。

| 指标族 | 代码路径 | 当前实现 |
| --- | --- | --- |
| [IoU](iou.md) | `Metrics/IoU/` | per-class、micro、mIoU、frequency-weighted IoU |
| [Dice](dice.md) | `Metrics/Dice/` | per-class、micro、mean、frequency-weighted、generalized Dice |
| [Confusion Matrix](confusion_matrix.md) | `Metrics/ConfusionMatrix/` | matrix/count、Precision、Recall、F1 |

## 统一约定

- 指标函数接收定义明确的 tensor，不自动执行 argmax、sigmoid、threshold 或任务匹配。
- 类别数、类别顺序、ignore/background 和 zero-division/absent-class 策略必须显式固定。
- 先汇总全局 count 或混淆矩阵，再计算 dataset-level 指标；不要平均不同 batch 的比率。
- 计数与分数的 dtype/device 行为由各指标页面明确说明。
- 当前 IoU 是 hard-label set overlap，不是 bbox IoU；当前 Dice 不是 soft/probability Dice。

具体的 absent class 约定见 [IoU](iou.md) 与 [Dice](dice.md)，混淆矩阵方向和 zero division 见 [Confusion Matrix](confusion_matrix.md)。
