# Metrics

`Metrics/` 保存可以跨 General、MLLM、Embodied 复用的指标计算。目录按指标族组织，而不是按任务类型组织。

| 指标族 | 代码路径 | 当前实现 |
| --- | --- | --- |
| [IoU](iou.md) | `Metrics/IoU/` | per-class、micro、mIoU、frequency-weighted IoU |
| [Dice](dice.md) | `Metrics/Dice/` | per-class、micro、mean、frequency-weighted、generalized Dice |
| [Confusion Matrix](confusion_matrix.md) | `Metrics/ConfusionMatrix/` | matrix/count、Precision、Recall、F1 |
| [MAE](mae.md) | `Metrics/MAE/` | absolute error、mean/sum/dimension reductions |
| [MSE](mse.md) | `Metrics/MSE/` | squared error、mean/sum/dimension reductions |

## 统一约定

- 指标函数接收定义明确的 tensor，不自动执行 argmax、sigmoid、threshold、mask 或任务匹配。
- 类别数、类别顺序、ignore/background、zero-division、有效样本与 reduction 策略必须显式固定。
- 先汇总全局 count/误差或混淆矩阵，再计算 dataset-level 指标；不要平均口径不一致的 batch 比率。
- 计数与分数的 dtype/device 行为由各指标页面明确说明。
- 当前 IoU 是 hard-label set overlap，不是 bbox IoU；当前 Dice 不是 soft/probability Dice。

具体约定进入各指标页面。
