# t-Distributed Stochastic Neighbor Embedding（t-SNE）

t-SNE 是非线性、随机的局部邻域可视化方法。它适合诊断高维样本的局部邻居是否在不同 ablation 组之间发生变化，但二维/三维图不能单独证明类别可分、机制存在或因果贡献。

实现位于 `General/dimensionality_reduction/tsne.py`，通过惰性导入调用 scikit-learn；没有安装 scikit-learn 时，PCA 等其他模块仍可正常导入。

## 环境准备

```bash
python -m pip install -r setup/requirements.txt
python -m pip install -e . --no-deps
```

也可以只安装对应 extra：

```bash
python -m pip install -e '.[dimensionality-reduction]'
```

参考环境固定 `scikit-learn==1.5.2`。适配层同时识别新版 `max_iter` 和旧版 `n_iter` 参数，但结果复现仍应固定 scikit-learn、NumPy/BLAS 与线程环境。

## 公共接口

| API | 作用 |
| --- | --- |
| `TSNEConfig` | 不可变的 t-SNE 与可选 PCA 预处理配置 |
| `fit_tsne` | 对一个固定样本集合执行一次 fit-transform |
| `fit_joint_tsne` | 先合并多个命名组，共享一次 PCA/t-SNE，再按组切回 |
| `TSNEResult` | CPU embedding、KL、迭代数、effective LR 与预处理元数据 |

t-SNE 没有可靠的 out-of-sample `transform`。接口使用 `fit_*` 命名，不提供容易误导的 transform。

## 基本使用

```python
from General.dimensionality_reduction import TSNEConfig, fit_tsne

config = TSNEConfig(
    perplexity=30.0,
    random_state=0,
    init="pca",
    learning_rate="auto",
    max_iter=1000,
    pca_components=None,  # 基础接口不隐式改变特征
)

result = fit_tsne(features, config=config)  # features: [samples, features]
embedding = result.embedding               # CPU [samples, 2]
```

对很高维的 dense feature，可显式启用常见的 PCA→t-SNE 流程：

```python
config = TSNEConfig(
    perplexity=30.0,
    random_state=0,
    pca_components=50,
    pca_standardize=False,
)
result = fit_tsne(features, config=config)
print(result.pca_components_used)
print(result.pca_explained_variance_ratio)
```

只有当原始 feature 数大于 `pca_components` 时才执行 PCA；若数据有效秩更低，实际维数会缩到有效秩并记录在 result，不会默默白化。

## Baseline 与 ablation 联合拟合

跨组叠图默认使用一次 joint fit：

```python
from General.dimensionality_reduction import fit_joint_tsne

joint = fit_joint_tsne(
    {
        "baseline": baseline_features,
        "ablation": ablation_features,
    },
    config=config,
)
baseline_xy = joint.embeddings["baseline"]
ablation_xy = joint.embeddings["ablation"]
```

mapping 按插入顺序 concat/split，perplexity 按合并后的总样本数校验。分别 fit 两组会产生两个独立坐标系，即使 seed 与参数相同，也不能比较旋转、尺度、轴方向或簇间距离。

新增/删除任何组都会改变 joint t-SNE 的全部坐标。若要量化 paired sample 的漂移，应回到原始特征或固定 PCA 空间计算距离。

## 常见做法

1. 固定 checkpoint、层、token/patch 范围、pooling 和 sample ID；canonical 输入是一行一个统计单位的 `[N, F]`。
2. 预先确定每组样本数、类别配额和采样顺序；paired ablation 复用相同样本。
3. scaling、L2 normalization 或 PCA 均作为显式配置，并在 joint/reference population 上只 fit 一次。
4. `perplexity` 是邻域尺度而不是聚类数，必须满足 `0 < perplexity < N_total`；接口不会自动 clamp。
5. 固定 `init`、metric、learning rate、early exaggeration、max_iter、method、angle 与版本。
6. 使用预注册的多个 seed（例如 0–4），展示全部或报告稳定性，不只保留最好看的图。

```python
results = [
    fit_tsne(features, config=TSNEConfig(perplexity=30, random_state=seed))
    for seed in range(5)
]
```

`learning_rate="auto"` 依赖总样本数；如果组间总 N 改变，effective learning rate 也会改变。优先等量采样，或显式固定 learning rate。`kl_divergence` 只适合同一数据、N 和配置下检查优化，不可跨 perplexity 或数据集解释成“表示更好”。

## 输出与记录

`TSNEResult` 保存：

- CPU `embedding`、`kl_divergence`、`n_iter`、`effective_learning_rate`；
- 完整 `TSNEConfig` 与输入 `[N, F]`；
- t-SNE 实际输入 feature 数；
- 可选 PCA 的实际维数与 explained-variance ratio。

同时应保存每行 sample ID、group/label、采样索引、源 checkpoint/layer、pooling、原始 shape、scikit-learn 版本与绘图参数。图片不能覆盖原始 float embedding。

## 解释边界

- t-SNE 主要保持局部邻域；轴方向、正负、绝对尺度、簇面积、密度、空白和远距离都不能直接当定量证据。
- 图中成簇不等于存在离散类别，也不等于下游性能提升。
- 结果对样本组成、perplexity、seed 和初始化敏感；必须与原空间距离、预注册的 kNN/linear probe、trustworthiness 或任务指标配合。
- 当前实现只接收 dense finite floating feature matrix，不支持 precomputed distance matrix。
- scikit-learn t-SNE 是 CPU 路径，大样本仍然昂贵；输入 tensor 会 detach 并移动到 CPU。
