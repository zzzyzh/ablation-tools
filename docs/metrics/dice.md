# Dice 指标族

代码路径：

- `Metrics/Dice/dice.py`：per-class 与 micro Dice；
- `Metrics/Dice/mean_dice.py`：mean 与 frequency-weighted Dice；
- `Metrics/Dice/generalized_dice.py`：显式 class weighting 的 generalized Dice。

## 定义

```text
Dice_c = 2 * intersection_c / (prediction_count_c + target_count_c)
```

对非空类，`Dice = 2 * IoU / (1 + IoU)`。mean Dice 对选中类别做 macro 平均；micro Dice 先汇总计数；frequency-weighted Dice 按 target support 加权。

Generalized Dice 在选中类别上计算：

```text
2 * sum(weight_c * intersection_c)
--------------------------------------
sum(weight_c * (prediction_c + target_c))
```

`class_weight` 支持 `"uniform"`、`"inverse"` 和 `"inverse_square"`。后两种只给 target support 大于零的类赋权，避免 absent target 产生无限权重；默认 `inverse_square`。

## 使用

```python
from Metrics.Dice import compute_dice, compute_generalized_dice, compute_mean_dice

classwise = compute_dice(predictions, targets, num_classes=4)
aggregated = compute_mean_dice(
    predictions,
    targets,
    num_classes=4,
    class_indices=[1, 2, 3],
    absent_class="ignore",
)
generalized = compute_generalized_dice(
    predictions,
    targets,
    num_classes=4,
    class_indices=[1, 2, 3],
    class_weight="inverse_square",
)
```

## 适用边界

- 当前实现只接受 hard labels，不接收 logits、probability map 或 soft target。
- generalized Dice 的权重定义必须随结果报告；不同权重不能只统称为 Dice。
- macro、micro、frequency-weighted 与 generalized 回答的问题不同，不应挑选最好看的一个替代预定义主指标。
- absent policy、background、ignore_index、类别选择和统计范围必须在 baseline/ablation 之间保持一致。
