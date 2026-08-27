# Dimensionality Reduction

高维特征的通用降维与诊断方法：

- `pca.py`：纯 PyTorch PCA，支持 explained variance、标准化、白化、复用 transform 与 inverse transform。
- `tsne.py`：惰性 scikit-learn t-SNE、显式 PCA 预降维与 baseline/ablation joint fit。

方法论与使用方式见 [`docs/general/pca.md`](../../docs/general/pca.md) 和 [`docs/general/tsne.md`](../../docs/general/tsne.md)。
