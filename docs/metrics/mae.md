# Mean Absolute Error（MAE）

代码位于 `Metrics/MAE/mae.py`，公共名称为 `compute_mae` 和完整别名 `compute_mean_absolute_error`。

```text
absolute_error_i = |prediction_i - target_i|
MAE              = mean(absolute_error)
```

## 使用

```python
from Metrics.MAE import compute_mae

mae = compute_mae(predictions, targets)
per_element = compute_mae(predictions, targets, reduction="none")
last_dimension_mae = compute_mae(predictions, targets, dim=-1)
summed = compute_mae(predictions, targets, reduction="sum", dim=(0, 1))
```

`predictions` 与 `targets` 必须 shape/device 相同、非空、real 且 finite。整数会在减法前提升到 float64；为保证减法精确，整数绝对值限制在 `2**52` 内，超出时 fail-fast。float16/bfloat16 提升到 float32，其余浮点按 PyTorch dtype promotion 处理。

## Reduction

- `reduction="none"`：返回逐元素绝对误差，此时不能再给 `dim` 或 `keepdim`；
- `reduction="mean"`：默认对全部元素平均，也可指定单个或多个 dim；
- `reduction="sum"`：对全部或指定 dim 求和；
- `keepdim` 只在 mean/sum reduction 时生效。

## Ablation 比较

- 固定 target 定义、单位、归一化/反归一化、有效 mask 和统计维度。
- 明确是对所有 element 全局平均、先对每个 sample 平均再汇总，还是对时间/动作维分别报告；这些口径不等价。
- MAE 保留原变量单位，对大误差的惩罚弱于 MSE；不能因为数值更小就认为模型更好。
- 缺失值、NaN/Inf 不会被静默忽略，应在调用前按预先定义的 mask 过滤。
- 同时报告有效样本数与误差分布；单一均值可能掩盖少数失败样本。
