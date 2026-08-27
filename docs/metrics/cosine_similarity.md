# Cosine Similarity Score

Cosine Similarity 比较两个向量的方向，不考虑整体长度：

```text
cosine(x, y) = dot(x, y) / (||x|| * ||y||)
```

实现位于 `Metrics/CosineSimilarity/`：

- `compute_cosine_similarity`：严格对齐、相同 shape 的成对向量；
- `compute_pairwise_cosine_similarity`：两个显式二维向量集合的全组合矩阵。

## 成对计算

```python
from Metrics.CosineSimilarity import compute_cosine_similarity

scores = compute_cosine_similarity(
    prediction_embeddings,
    target_embeddings,
    dim=-1,
    zero_vector="zero",
    reduction="none",
)
mean_score = compute_cosine_similarity(
    prediction_embeddings,
    target_embeddings,
    reduction="mean",
)
```

输入必须 shape/device 完全相同；即使 shape 可以 broadcast，也会直接报错，避免把 paired comparison 意外变成跨样本比较。`reduction="none"` 只移除 feature 维并保留其余维顺序；mean/sum 对所有 paired scores 汇总。

## Pairwise 计算

```python
from Metrics.CosineSimilarity import compute_pairwise_cosine_similarity

# first: [N, D], second: [M, D] -> [N, M]
similarity_matrix = compute_pairwise_cosine_similarity(first, second)

# 也支持 feature-first：[D, N] 与 [D, M]
similarity_matrix = compute_pairwise_cosine_similarity(
    first_feature_first,
    second_feature_first,
    dim=0,
)
```

Pairwise API 只接受两个二维 tensor，不隐式 flatten batch。pairwise mean 是所有 `N*M` 组合的平均，不等同于 aligned paired mean。

## 零向量与 eps

- `zero_vector="zero"`：任一侧为精确零向量时 score 固定为 0，包括 zero-vs-zero；
- `zero_vector="raise"`：遇到零向量立即报错；
- `eps` 是 norm 的分母下限，不是零向量判定阈值。非零但 norm 小于 eps 的向量会得到衰减 score。

不提供 zero-vs-zero=1 的策略，因为这会奖励表示坍缩；也不返回 NaN。baseline 与 ablation 必须使用相同 eps 和 zero policy。

## Dtype 与数值契约

- 输入必须 real、finite、非空；同 device，不发生隐式迁移。
- 只要一侧是整数，就在 `[-2**52, 2**52]` 精确范围内提升到 float64；超出时报错。
- float16/bfloat16 提升到 float32；其他浮点按 PyTorch promotion。
- norm、normalize、score 或 reduction 无法在计算 dtype 表示时 fail-fast。
- 输出会在 finite 检查后 clamp 到 `[-1, 1]`，只修正浮点舍入误差。

## Ablation 比较

- 固定 feature extraction checkpoint、层、token 范围、pooling、mask、样本顺序和配对关系。
- 区分 aligned comparison 与 all-pairs distribution；不能把两种平均值用同一个名称报告。
- Cosine 丢弃向量幅度；如幅度可能承载信息，应同时报告 norm 或欧氏距离。
- 对时间序列或 action chunk 明确 feature dim 与 score 聚合维，避免先平均 embedding 和先平均 score 混淆。
- 报告样本数、均值、离散程度与分布，不只报告单个 mean score。
