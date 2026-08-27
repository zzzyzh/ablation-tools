# IoU 与 mIoU

代码按职责拆分为：

- `Metrics/IoU/iou.py`：`compute_iou`，返回 per-class 与 micro IoU；
- `Metrics/IoU/miou.py`：`compute_mean_iou`，增加 mIoU 与 frequency-weighted IoU。

## 定义

对类别 `c`：

```text
intersection_c = count(prediction == c and target == c)
union_c        = prediction_count_c + target_count_c - intersection_c
IoU_c          = intersection_c / union_c
```

`mean_iou` 是选中且按 absent policy 纳入类别的算术平均；`micro_iou` 先对选中类别的 intersection/union 求和再相除；`frequency_weighted_iou` 按 target support 加权。

## 使用

```python
from Metrics.IoU import compute_iou, compute_mean_iou

classwise = compute_iou(
    predictions,
    targets,
    num_classes=4,
    ignore_index=255,
    absent_class="ignore",
)

aggregated = compute_mean_iou(
    predictions,
    targets,
    num_classes=4,
    ignore_index=255,
    class_indices=[1, 2, 3],  # aggregate 排除 background，但不删除其像素
)

per_class_iou = classwise.per_class_iou
miou = aggregated.mean_iou
micro_iou = aggregated.micro_iou
frequency_weighted_iou = aggregated.frequency_weighted_iou
```

## 公平比较

- 固定 hard prediction 的生成方式；不同 threshold/argmax 规则属于独立实验变量。
- 固定 `num_classes`、class order、background、ignore_index、class_indices 与 absent policy。
- 数据集级 mIoU 应先累积全量 count 再计算，不要平均不同 batch 的 mIoU。
- `frequency_weighted_iou` 会放大高频类，不能代替 mIoU；应按研究问题选择并明确命名。
- 对 bbox overlap 使用未来明确命名的 box IoU，实现不会依据 `[N,4]` shape 自动猜测。
