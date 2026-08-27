# Attention Visualization

本目录提供不绑定具体模型的 attention 特征图分析代码：

- `gradcam.py`：激活 Grad-CAM、attention Grad-CAM 与通用前向/梯度捕获。
- `qkv.py`：QK logits、QKV attention 重算、统一 Q/K/V 特征图，以及支持不同 token 网格的单路 `compute_token_feature_map`。
- `visualization.py`：缩放、着色、叠图与图像保存。

所有计算接口均接收已经抽取出的 `torch.Tensor`。模型的层定位、输出字段选择、特殊 token 定义和 patch 网格推导属于迁移适配，应由调用者显式提供。

环境、完整方法说明和使用方式见 [`docs/general/attention_visualization.md`](../../docs/general/attention_visualization.md)。
