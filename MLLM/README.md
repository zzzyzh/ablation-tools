# MLLM

多模态大语言模型 ablation 主线。模型特有的视觉 token、文本 token、跨模态连接与生成策略适配留在调用侧，公共计算保持可复用。

当前方法：

- [`rouge_l/`](rouge_l/)：基于最长公共子序列的 ROUGE-L 生成文本重合评分。

方法论、使用与未来规划见 [`docs/mllm/`](../docs/mllm/index.md)。
