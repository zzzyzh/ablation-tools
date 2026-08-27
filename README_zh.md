<h1 align="center">
  <img src="assets/ablation-tools-icon.png" width="96" alt="ablation-tools 图标"><br>
  ablation-tools
</h1>

<p align="center">
  <a href="README.md">English</a> | <a href="README_zh.md">中文</a>
</p>

面向实验消融与诊断分析的通用参考仓库。项目沉淀模型无关的张量级实现、共享指标与方法论文档，方便将通用 ablation 操作迁移到具体模型和任务。

## 项目导航

| 主线 | 代码 | 方法论与使用说明 | 当前内容 |
| --- | --- | --- | --- |
| General | [`General/`](General/) | [`docs/general/`](docs/general/index.md) | attention visualization、ROI extraction、PCA、t-SNE |
| MLLM | [`MLLM/`](MLLM/) | [`docs/mllm/`](docs/mllm/index.md) | ROUGE-L 生成文本序列重合评估 |
| Embodied | [`Embodied/`](Embodied/) | [`docs/embodied/`](docs/embodied/index.md) | 具身系统方法论骨架 |

跨主线共享指标位于 [`Metrics/`](Metrics/)；参考环境与依赖版本见 [`setup/`](setup/README.md)。

## General 方法

- [`General/attention_visualization/`](General/attention_visualization/)：Grad-CAM、Q/K/V 分析与热力图渲染。
- [`General/roi_extraction/`](General/roi_extraction/)：Grounding DINO ROI bbox 提取、坐标几何与显式候选选择。
- [`General/dimensionality_reduction/`](General/dimensionality_reduction/)：PCA 全局线性投影与 t-SNE 局部邻域可视化。

## MLLM 方法

- [`MLLM/rouge_l/`](MLLM/rouge_l/)：基于最长公共子序列的 ROUGE-L 生成文本评分。

## 共享指标

- [`Metrics/IoU/`](Metrics/IoU/)：classwise/micro IoU、mIoU 与 frequency-weighted IoU。
- [`Metrics/Dice/`](Metrics/Dice/)：classwise、mean、micro、frequency-weighted 与 generalized Dice。
- [`Metrics/ConfusionMatrix/`](Metrics/ConfusionMatrix/)：混淆矩阵、Precision、Recall 与 F1。
- [`Metrics/MAE/`](Metrics/MAE/)：具有显式 reduction 的 Mean Absolute Error。
- [`Metrics/MSE/`](Metrics/MSE/)：具有显式 reduction 的 Mean Squared Error。

统一实验方法论与命名规范见 [`docs/index.md`](docs/index.md)。具体使用说明统一放在对应方法文档中，而不是根 README。

## 新增 ablation 的流程

1. Ablation 方法在 `General`、`MLLM`、`Embodied` 中选择主线；跨任务指标直接按数学指标族加入 `Metrics/`。
2. 实现前定义假设、baseline、唯一变化项、张量契约与公平比较所需控制变量。
3. 实现模型无关的核心计算；模型层定位、token 划分与任务对象转换留给调用侧薄适配层。
4. 在对应 `docs/` 路径增加同名说明，覆盖使用方式、参数语义、迁移检查、局限与不能支持的结论。
5. 用临时合成测试验证公式、shape、异常输入、dtype/device 和资源释放；验证后删除测试路径与产物。
6. 更新索引、公共导出和根导航，检查代码、文档与真实 API 完全一致。
7. 提交前确认仓库不包含缓存、生成图片、模型/数据文件、临时脚本或调试文件。

## Commit 要求

- 一个 commit 只引入一个边界清楚的方法或不可分割的指标族，代码、文档和公共导出同提交完成。
- 标题采用 `<type>(<scope>): <summary>`，例如 `feat(general): add t-SNE dimensionality reduction`。
- commit body 至少说明新增方法、张量契约、模型无关边界、验证结果和已知限制。
- 不提交无关格式化、运行产物、下载权重、数据集或临时测试。
- 提交前检查 staged diff，确认导航、命名、链接与行为变化准确。

## 引用

如果本仓库对你的研究有帮助，请使用 [`CITATION.cff`](CITATION.cff) 或以下 BibTeX：

```bibtex
@software{yan_2026_ablation_tools,
  author  = {Yan, Zhonghao},
  title   = {ablation-tools: Reusable Ablation Methods and Metrics},
  year    = {2026},
  version = {0.1.0},
  url     = {https://github.com/zzzyzh/ablation-tools}
}
```

## License

版权所有 © 2026 Zhonghao Yan。

本项目采用 [Apache License 2.0](LICENSE)。该许可证允许商业使用、修改和再分发，但须遵守许可证及声明保留条件，并提供明确的专利授权。
