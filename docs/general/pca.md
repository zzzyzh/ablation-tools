# Principal Component Analysis（PCA）

PCA 是确定性的线性降维方法，用正交主轴解释特征中的最大方差方向。它适合压缩高维表示、检查有效秩、绘制 explained-variance 曲线，以及为后续可视化提供统一的线性预处理。

实现位于 `General/dimensionality_reduction/pca.py`，只依赖 PyTorch。

## 公共接口

| API | 作用 |
| --- | --- |
| `fit_pca` | 在训练/参考特征上拟合不可变 `FittedPCA` 状态 |
| `FittedPCA.transform` | 用相同 mean、scale、components 投影新特征 |
| `FittedPCA.inverse_transform` | 从 component score 重建原特征空间 |
| `compute_pca` | 一次完成 fit 和训练特征 projection |

## 使用

```python
from General.dimensionality_reduction import compute_pca, fit_pca

projection = compute_pca(
    reference_features,       # [samples, features]
    n_components=2,
    standardize=False,
    whiten=False,
)

embedding = projection.embedding
explained_ratio = projection.fitted_pca.explained_variance_ratio

# 避免数据泄漏：只在 reference/train 上 fit，再复用到其他组。
model = fit_pca(train_features, n_components=0.95)
validation_embedding = model.transform(validation_features)
validation_reconstruction = model.inverse_transform(validation_embedding)
```

整数 `n_components` 表示固定维数；浮点 `(0, 1]` 表示选择达到目标累计 explained-variance ratio 的最少维数；`None` 保留数值 `effective_rank` 内的全部方向。

## Center、standardize 与 whiten

- PCA 始终使用训练特征 mean 做中心化。
- `standardize=True` 进一步按训练集每个原始 feature 的标准差缩放，等价于对相关系数结构做 PCA；常量 feature 的 scale 设为 1，不产生除零。
- `whiten=True` 在 component 空间再除以主成分标准差，使训练 projection 的选中维度方差约为 1；它会丢失原始方差尺度，不应作为无成本默认值。

三种设置回答的问题不同，baseline 与 ablation 必须完全一致。不要根据二维图效果临时切换 standardize 或 whiten。

## 输出契约

`FittedPCA` 保存：

- `mean`、`scale` 与 `components [K, D]`；
- `singular_values`、`explained_variance`、`explained_variance_ratio`；
- `cumulative_explained_variance_ratio` 与 effective-rank 内、`n_components` 截断前的 `total_variance`；
- fit 时的 sample/feature 数、`effective_rank`、`correction`、`rank_rtol`、standardize 与 whiten。

float16/bfloat16 输入提升到 float32；其他浮点 dtype 与 device 保持。拟合状态会 detach，因此不会长期保留训练特征的 SVD autograd 图，但 transform 对新输入仍可求导。SVD 主轴的符号本来不唯一，实现把每个 component 最大绝对 loading 固定为正；相等或接近的奇异值仍允许整个子空间旋转。

## Ablation 比较规范

- 先明确一行特征代表 sample、token、frame 还是 episode；不同统计单位不能直接混合。
- 比较 baseline 与 ablation 时，应在联合参考集或预先固定的 train/reference 集上 fit 一套 PCA，再分别 transform；不要各自 fit 后直接比较坐标。
- 若联合 fit，应保存每组 sample 数并避免数量严重不平衡主导主轴；必要时先做分层等量采样。
- 固定上游 pooling、归一化、层、token 范围、dtype 和异常值处理。
- 同时报告 explained variance 与原任务指标；二维可视分离不等同于任务机制或因果贡献。
- `n_components=0.95` 等数据驱动选择只能在 reference split 上确定，不能读取 test/ablation 结果后调整。

## 常见问题

- **所有特征没有方差**：PCA 无可解释方向，实现会直接报错。
- **白化失败**：选中了零方差/超出有效秩的 component；减少组件或关闭 whiten。
- **不同实验坐标翻转**：确认是否复用同一个 `FittedPCA`；即使符号稳定，不同数据重新 fit 的主轴也不是同一坐标系。
- **重建不完全**：截断 PCA 本来就是有损压缩；保留全部 effective-rank 方向时，也只保证在 `rank_rtol` 定义的数值秩容差内重建。
