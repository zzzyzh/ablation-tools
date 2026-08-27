# Mean Squared Error（MSE）

代码位于 `Metrics/MSE/mse.py`，公共名称为 `compute_mse` 和完整别名 `compute_mean_squared_error`。

```text
squared_error_i = (prediction_i - target_i)^2
MSE             = mean(squared_error)
```

## 使用

```python
from Metrics.MSE import compute_mse

mse = compute_mse(predictions, targets)
per_element = compute_mse(predictions, targets, reduction="none")
per_dimension = compute_mse(predictions, targets, dim=0)
```

shape/device、dtype promotion、finite 检查和 reduction 契约与 [MAE](mae.md) 完全一致。平方后若计算 dtype 无法表示结果，接口会 fail-fast，提示使用更高精度，而不是返回 Inf。

## Ablation 比较

- MSE 的单位是原变量单位的平方，对大误差更敏感；它与 MAE 回答的风险偏好不同。
- 固定输出/target 的归一化、各维量纲和权重。不同量纲直接求平均时，高尺度维会主导 MSE。
- 对 action chunk、时间序列或多关节输出，应明确 batch、time、dimension 的 reduction 顺序并同时报告分维结果。
- 如果需要 RMSE，应在最终定义好的 MSE aggregate 上开平方；不要先逐元素开平方，否则会退化为绝对误差。
- 不在看完结果后切换 MAE/MSE 作为主指标；主指标与辅助指标应预先声明。
