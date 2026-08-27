# General

General 主线收录与模型家族和任务类型无关的 ablation 操作。实现尽量只依赖张量语义，将“怎样从具体模型取得输入”留给项目侧薄适配层。

## 方法目录

| 主题 | 可回答的问题 | 状态 |
| --- | --- | --- |
| [注意力特征图可视化](attention_visualization.md) | 目标输出依赖哪些空间位置？Q/K 匹配与 V 聚合呈现什么结构？ | 可用：Grad-CAM 与 Q/K/V |
| [Grounding DINO ROI BBox 提取](roi_extraction.md) | 如何从文本和图像生成可复用、可筛选的 ROI bbox？ | 可用：候选提取、坐标处理与选择 |
| [Principal Component Analysis](pca.md) | 表征的主要线性变化方向和有效秩如何随 ablation 改变？ | 可用：fit/transform/inverse、解释方差、白化 |

## 通用边界

一个 General 方法应该：

- 接收明确的张量、维度约定和可选 mask，不在内部猜测模型层名或任务对象。
- 区分候选生成、纯计算与实验策略；任务启发式由调用者显式配置。
- 不默认聚合 batch、head、token、时间或候选维；聚合必须由调用方选择。
- 保留 dtype、device 和梯度语义，除非 API 文档明确声明转换。
- 对 shape、mask、坐标和非有限值尽早报错，让迁移错误在小样本阶段暴露。

## 结果组织建议

除最终图或 selected bbox 外，每个样本还应保留原始候选、输入 prompt、模型/processor revision、阈值、选择配置、原始分辨率和坐标变换。展示归一化或整数 bbox 不能覆盖原始 float 结果。

## 新方法接入约定

新增方法时提供最小数值验证、边界验证和与本页结构一致的方法论说明。如果算法必须感知具体任务对象，应把该逻辑留在调用侧或对应任务主线，而不是扩大 General 的责任。
