# ablation-tools

面向实验消融与诊断分析的通用参考仓库。项目沉淀模型无关的张量级实现与方法论文档，方便将通用 ablation 操作迁移到具体模型和任务。

## 项目导航

| 主线 | 代码 | 方法论与使用说明 | 当前内容 |
| --- | --- | --- | --- |
| General | [`General/`](General/) | [`docs/general/`](docs/general/index.md) | 通用表示与分析；已实现 attention visualization |
| MLLM | [`MLLM/`](MLLM/) | [`docs/mllm/`](docs/mllm/index.md) | 多模态大语言模型；当前为方法骨架 |
| Embodied | [`Embodied/`](Embodied/) | [`docs/embodied/`](docs/embodied/index.md) | 具身系统；当前为方法骨架 |

参考环境与依赖版本见 [`setup/`](setup/README.md)。

当前首个方法位于 [`General/attention_visualization/`](General/attention_visualization/)：

- `gradcam.py`：激活 Grad-CAM、attention Grad-CAM 与通用 hook 捕获；
- `qkv.py`：QK logits、QKV attention 重算与 Q/K/V token 特征图；
- `visualization.py`：热力图缩放、着色、叠图与保存。

统一实验方法论与命名规范见 [`docs/index.md`](docs/index.md)，具体使用见 [`docs/general/attention_visualization.md`](docs/general/attention_visualization.md)。

## 新增 ablation 的流程

1. 在 `General`、`MLLM`、`Embodied` 中选择主线，并用小写 `snake_case` 建立问题明确的方法目录。
2. 先定义假设、baseline、唯一变化项、输入输出张量契约与公平比较所需控制变量。
3. 在对应主线实现模型无关的核心计算；模型层定位、token 划分与任务对象转换留给调用侧。
4. 在 `docs/<track>/` 增加同名方法说明，包含适用问题、使用步骤、参数语义、迁移检查、局限与不能支持的结论。
5. 用合成张量临时验证公式、shape、异常输入、dtype/device 和资源释放；验证结束后删除临时测试路径与产物。
6. 更新该主线索引、公共导出和根 README 导航，检查代码、文档与真实 API 完全一致。
7. 提交前确认仓库不包含缓存、输出图片、模型/数据文件、临时脚本或调试文件。

详细规范见 [`docs/index.md`](docs/index.md)。

## Commit 要求

- 一个 commit 只引入一个边界清楚的方法或一组不可分割的配套修改，代码、文档和公共导出应同提交完成。
- 标题采用 `<type>(<track>): <summary>`，例如 `feat(general): add attention visualization ablation methods`。
- commit body 至少说明：新增方法、公共张量契约、模型无关边界、验证结果，以及已知限制或迁移注意事项。
- 不提交运行产物、缓存、临时测试目录、下载权重或数据集；不把无关格式化和其他方法混入同一 commit。
- 提交前检查 staged diff，确保导航、方法名称和文档链接准确，且不存在未说明的行为变化。
