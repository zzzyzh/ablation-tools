# MLLM

MLLM 主线用于整理多模态大语言模型的 ablation 方法，重点是分离模态输入、编码器、投影/采样模块、多模态 token 与语言解码器的影响。

## 当前实现

| 方法 | 代码路径 | 当前实现 |
| --- | --- | --- |
| [ROUGE-L](rouge_l.md) | `MLLM/rouge_l/` | LCS、单/多参考和 aligned batch 生成文本评分 |
| [MiniLM Semantic Cosine](sentence_similarity.md) | `MLLM/sentence_similarity/` | paraphrase-MiniLM-L6-v2 aligned sentence similarity |

## 拟整理的主题

| 候选主题 | 典型对照 | 主要风险 |
| --- | --- | --- |
| 模态输入贡献 | 完整输入 vs. 遮蔽/替换一种模态 | 替换值可能引入分布外输入 |
| 视觉 token 数量与采样 | 相同图像、不同 token 预算 | 分辨率、计算量与位置编码同时变化 |
| 融合/投影路径 | 原始模块 vs. 冻结、置换或简化模块 | 参数量与训练预算不可比 |
| 跨模态注意力 | 保留 vs. 屏蔽指定 token 路径 | mask 语义和缓存位置容易出错 |
| prompt 与模态排列 | 只改变模板或输入顺序 | tokenization 与解码策略造成混杂 |
| 多图/视频时序信息 | 不同帧数、顺序或时间间隔 | 帧采样与总 token 预算耦合 |

## 方法命名约定

代码目录和文件使用小写 `snake_case`。函数名描述实际操作和输出，不使用 `run_ablation` 之类宽泛名称。

## 公平比较的最小要求

- 固定原始媒体、文本 prompt、chat template、tokenizer 和特殊 token 配置。
- 记录视觉/音频预处理的 resize、crop、采样率、帧数和归一化参数。
- 分开报告输入 token 数、生成 token 数、FLOPs/时延与任务指标。
- 若替换或遮蔽一种模态，设置零值、均值或打乱等 control，检查分布偏移。
- 生成任务固定 decoding 参数；使用采样时报告多个 seed 的均值和离散度。
- 文本指标固定 decode 后清洗、tokenization、reference 集合和空输出策略。

## 接口边界

MLLM 专用层只负责把模型输出、模态布局和文本对象转换为统一中间表示。能够直接对张量完成的计算继续复用 [General](../general/index.md) 与 [Metrics](../metrics/index.md)，避免复制算法。
