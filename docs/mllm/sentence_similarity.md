# Paraphrase MiniLM Cosine Similarity

该方法使用 `paraphrase-MiniLM-L6-v2` 将 prediction/reference 文本编码为 384 维 sentence embedding，再复用 `Metrics/CosineSimilarity` 计算 aligned semantic cosine score。

实现位于 `MLLM/sentence_similarity/paraphrase_minilm.py`。模型加载使用惰性 `sentence-transformers` 依赖，未安装时不会影响其他 MLLM 或 Metrics 模块。

## 环境与本地 checkpoint

参考依赖固定在 `setup/requirements.txt`：

```bash
conda activate ablation-tools
python -m pip install -r setup/requirements.txt
python -m pip install -e . --no-deps
```

本地参考 checkpoint：

```text
/mnt/nas/yanzhonghao/ckpts/paraphrase-MiniLM-L6-v2
```

checkpoint 配置为 6 层 MiniLM、384 维 embedding、attention-mask mean pooling、`max_seq_length=128`，模型卡许可证为 Apache-2.0。库实现不硬编码该机器路径，调用者必须显式传入，避免迁移后静默加载错误模型。

## 使用

```python
from MLLM.sentence_similarity import ParaphraseMiniLMCosineScorer

scorer = ParaphraseMiniLMCosineScorer.from_pretrained(
    "/mnt/nas/yanzhonghao/ckpts/paraphrase-MiniLM-L6-v2",
    device="cuda:0",
    local_files_only=True,
)

result = scorer.score(
    predictions,
    references,
    batch_size=64,
    normalize_embeddings=False,
)

per_sample_scores = result.scores
mean_score = result.mean_score
```

prediction 与 reference 必须一一 aligned 且数量相同。实现把两组文本合并为一次 encode，随后按原顺序切分，保证相同 tokenizer、padding/truncation、batch 和模型状态。

默认不返回 embedding，减少结果对象占用；诊断时可设：

```python
result = scorer.score(
    predictions,
    references,
    return_embeddings=True,
)
prediction_embeddings = result.prediction_embeddings
```

## 输出契约

`MiniLMCosineSimilarityResult` 包含：

- `scores [N]` 与 scalar `mean_score`；
- prediction 数、embedding 维数和 `normalize_embeddings`；
- checkpoint path/ID；
- 可选 prediction/reference embeddings。

embedding 必须是 finite floating `[text_count, embedding_dimension]`。默认保留 checkpoint 的原始 embedding，再由通用 cosine 完成归一化；`normalize_embeddings` 若开启必须作为实验配置记录。学习到精确零 embedding 时默认 `zero_vector="raise"`，也可显式设为 `"zero"`。空字符串会传给模型，不会被库层丢弃；实验协议必须定义空输出是否保留。

## 与 ROUGE-L 的区别

- ROUGE-L 检查 token 顺序与表面序列重合；
- MiniLM cosine 比较 sentence embedding 方向，可以给 paraphrase 较高分；
- 两者都不能单独证明事实正确、视觉 grounding、指令遵循或任务成功。

建议同时报告 ROUGE-L、semantic cosine 和任务指标，不在看完结果后选择最有利的一个。

## 公平比较

- 固定 checkpoint 文件、sentence-transformers/transformers 版本、device/dtype 和 max sequence length。
- 固定模型输入前的 chat template 清理、Unicode/空白处理、prompt 去除与截断规则。
- 固定 `normalize_embeddings`、batch size、空文本策略和样本配对顺序。
- score 是 aligned mean，不是 all-pairs semantic retrieval score；不能混用。
- 模型主要面向 paraphrase/sentence similarity，语言和领域变化可能产生系统偏差；跨语言结果需单独验证。
- Cosine 丢弃 embedding norm；若 norm 可能反映异常或坍缩，应同时记录 norm 分布。

## 模型引用

该 checkpoint 来自 Sentence-Transformers `paraphrase-MiniLM-L6-v2`。使用模型结果时应同时引用 Sentence-BERT：

```bibtex
@inproceedings{reimers-2019-sentence-bert,
  title     = {Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks},
  author    = {Reimers, Nils and Gurevych, Iryna},
  booktitle = {Proceedings of EMNLP-IJCNLP},
  year      = {2019}
}
```
