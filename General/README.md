# General

跨模型与跨任务可复用的 ablation 方法。这里的实现只依赖明确的张量契约，不硬编码模型名称、任务数据路径、图像尺寸或 token 规则。

当前主题：

- [`attention_visualization/`](attention_visualization/)：Grad-CAM、attention Grad-CAM、Q/K/V 特征图与热力图渲染。
- [`roi_extraction/`](roi_extraction/)：Grounding DINO ROI bbox 候选提取、坐标几何与显式选择策略。
- [`dimensionality_reduction/`](dimensionality_reduction/)：PCA 高维特征压缩、解释方差与可复用投影。

方法论与迁移说明见 [`docs/general/`](../docs/general/index.md)。
