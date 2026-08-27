# Confusion Matrix、Precision、Recall 与 F1

代码位于 `Metrics/ConfusionMatrix/`：

- `counts.py`：构建/校验混淆矩阵，并提取 TP、FP、FN、TN；
- `precision.py`、`recall.py`、`f1.py`：分别计算单项指标；
- `precision_recall_f1.py`：从同一矩阵一次返回 per-class、macro、micro 结果。

## 矩阵方向

矩阵固定为：

```text
confusion_matrix[target_class, predicted_class]
```

即行是真实类别，列是预测类别。对类别 `c`：

```text
TP_c = matrix[c, c]
FN_c = row_sum_c - TP_c
FP_c = column_sum_c - TP_c
TN_c = matrix.sum() - TP_c - FN_c - FP_c
```

不要在不同实验中交换行列方向；非对称错误会让 Precision 与 Recall 互换含义。

```text
Precision_c = TP_c / (TP_c + FP_c)
Recall_c    = TP_c / (TP_c + FN_c)
F1_c        = 2 * TP_c / (2 * TP_c + FP_c + FN_c)
```


## 构建与计算

```python
from Metrics.ConfusionMatrix import (
    compute_confusion_matrix,
    compute_precision_recall_f1,
)

matrix = compute_confusion_matrix(
    predictions,
    targets,
    num_classes=4,
    ignore_index=255,
)

metrics = compute_precision_recall_f1(matrix, zero_division=0.0)
per_class_precision = metrics.per_class_precision
per_class_recall = metrics.per_class_recall
per_class_f1 = metrics.per_class_f1
macro_f1 = metrics.macro_f1
micro_f1 = metrics.micro_f1
```

也可以分别调用：

```python
from Metrics.ConfusionMatrix import compute_f1_score, compute_precision, compute_recall

precision = compute_precision(matrix, average="none")
macro_recall = compute_recall(matrix, average="macro")
micro_f1 = compute_f1_score(matrix, average="micro")
```

## Average 与 zero division

- `average="none"`：返回每类 `[C]`；
- `average="macro"`：先计算每类指标，再做算术平均；
- `average="micro"`：先汇总所有类的 numerator/denominator，再相除。

`zero_division` 只允许 `0.0` 或 `1.0`，用于某类没有预测、没有 target 或整个矩阵为空时的未定义比率。macro 会把填充值纳入平均，因此该设置必须随结果报告。

在标准闭集、单标签、方阵分类中，micro Precision、Recall、F1 都等于 accuracy；实现仍保留各自名称，以便统计口径清晰。

## 输入边界

- 输入混淆矩阵必须是非空方阵、非负、finite；允许整数 count 或浮点 sample weight。
- 整数矩阵统一为 `int64` count；float16/bfloat16 提升到 float32；所有指标输出为 float64，device 保持不变。
- `compute_confusion_matrix` 只接收 hard class index，不自动 argmax/threshold；ignore_index 只作用于 target。
- 本模块不定义 background 类，也不执行 bbox matching、IoU threshold、AP/mAP 或 multilabel thresholding。

## 公平比较

- 固定 class order、num_classes、ignore_index、hard prediction 生成方式和 zero_division。
- 先在完整评估集上累积一个混淆矩阵，再计算指标；不要平均不同 batch 的 macro/micro 指标。
- 同时报告 per-class support，防止 macro 指标掩盖类别样本量差异。
- 若输入来自 detection，box matching 与 background 规则必须在调用侧先完成并单独记录；本模块只消费最终分类混淆矩阵。
