# Metrics

`Metrics/` 保存可以跨 General、MLLM、Embodied 复用的指标计算。目录按指标族组织，而不是按任务类型组织。

| 指标族 | 代码路径 | 当前实现 |
| --- | --- | --- |
| [IoU](iou.md) | `Metrics/IoU/` | per-class、micro、mIoU、frequency-weighted IoU |
| [Dice](dice.md) | `Metrics/Dice/` | per-class、micro、mean、frequency-weighted、generalized Dice |

## 统一约定

- overlap 接口接收相同 shape/device 的 hard class-index tensor，不自动执行 argmax、sigmoid、threshold 或 one-hot。
- `num_classes` 必须显式给出，不能从当前 batch 最大 label 推断，否则 absent class 会从评估中消失。
- 所有有效 label 必须位于 `[0, num_classes - 1]`；`ignore_index` 只作用于 target，对应 prediction 在范围校验前一起移除。
- 全部有效元素先汇总成全局 class count，再计算指标；这不是逐样本指标的平均。
- `class_indices` 只决定 aggregate 纳入哪些类，不删除像素，也不改变 per-class count。
- 计数为 `int64`，分数为 `float64`，device 与输入保持一致。
- 当前 IoU 是 hard-label set overlap，不是 bbox IoU；当前 Dice 不是 soft/probability Dice。

## Absent class

当某类 prediction 与 target 都为空时分母为零，由 `absent_class` 明确控制：

- `"ignore"`：per-class 返回 NaN，并从 macro mean 排除；
- `"zero"`：填 0 并纳入 mean；
- `"one"`：填 1 并纳入 mean；
- `"raise"`：只要选中类存在未定义指标就报错。

正式比较必须固定该策略、类别顺序、background 是否纳入以及 ignore_index。
