# Ablation Tools 方法论手册

`ablation-tools` 将常用实验操作拆成边界清晰、可迁移、可验证的张量级参考实现，不绑定具体模型、数据集或推理框架。

## 三条主线

| 主线 | 代码路径 | 关注对象 | 当前状态 |
| --- | --- | --- | --- |
| [General](general/index.md) | `General/` | 与任务类型无关的通用分析 | attention visualization、ROI extraction、PCA、t-SNE |
| [MLLM](mllm/index.md) | `MLLM/` | 多模态输入、token 与融合路径 | ROUGE-L 文本序列重合评估 |
| [Embodied](embodied/index.md) | `Embodied/` | 感知、记忆、决策与控制链路 | 方法论骨架 |

跨主线共享指标见 [Metrics](metrics/index.md)，当前包含 IoU、Dice、Confusion Matrix、MAE 与 MSE。

## Ablation 的最小闭环

1. **假设**：说明要验证的机制和预期变化，不在看完结果后反向改写问题。
2. **基线**：固定模型/权重、数据划分、预处理、解码或控制参数与评估脚本。
3. **单一干预**：一个对照组只改变一个明确因素；组合干预同时保留对应单因素组。
4. **测量**：主任务指标与诊断指标分开，预先确定统计单位、聚合方式和异常样本规则。
5. **重复**：随机过程使用相同 seed 集合，报告均值、离散程度与有效样本数。
6. **记录**：保留配置、代码/模型版本、原始结果与失败样本；展示归一化不能覆盖原始值。

可视化与 ROI 定位首先是诊断证据，不自动构成因果结论。实际干预与共享指标的统计口径都必须在实验前定义。

## 目录与 API 规范

- Ablation 代码按 `General`、`MLLM`、`Embodied` 组织；共享指标在 `Metrics/` 下按 IoU、Dice 等数学定义组织，不按任务重复实现。
- 一个目录表达一个研究问题或指标族，一个文件表达一个清晰方法；不建立不断膨胀的 `misc.py` 或 `utils.py`。
- 公共名称采用“动作 + 对象”：`compute_*` 表示无副作用张量计算，`*Capture`/`*Extractor` 表示明确生命周期，`select_*` 表示实验策略。
- 公共 docstring 明确 shape、dtype、device、数值范围、坐标/mask 方向、归一化与返回值；不支持的布局应尽早报错。
- 同类方法统一参数词汇，例如 `num_classes`、`class_indices`、`absent_class`、`score_threshold`、`boxes_xyxy`。

## 模型无关边界

通用实现只处理定义明确的张量。模型层路径、任务对象、数据路径、缓存和 task-specific fallback 由迁移侧显式适配。指标函数不自动推断类别数，不把 logits 变成 hard labels，也不依据 shape 猜测 mask overlap 或 bbox overlap。

## 新方法交付检查

- 方法名称脱离上下文仍能直接表达输入、动作或输出。
- 代码能用合成张量或 fake backend 验证，不要求下载模型、数据或权重。
- 对照变量、统计口径、适用边界、已知局限和不能支持的结论均有文档。
- 公式、shape、坐标/mask、异常输入、dtype/device 与资源释放完成临时验证。
- 临时测试路径和输出在验证后删除，代码目录只保留可复用实现。
- 索引、根导航、公共导出与文档示例和真实 API 一致。
- commit 保持单一 scope，body 记录张量契约、验证结果和迁移注意事项。
