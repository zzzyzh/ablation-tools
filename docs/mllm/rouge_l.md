# ROUGE-L

ROUGE-L 使用 prediction 与 reference token 序列的最长公共子序列（Longest Common Subsequence, LCS）衡量生成内容的顺序重合。它不要求连续 n-gram，但会保留 token 顺序。

实现位于 `MLLM/rouge_l/rouge_l.py`，纯 Python、无外部依赖。

## 定义

```text
P_lcs = LCS(prediction, reference) / len(prediction)
R_lcs = LCS(prediction, reference) / len(reference)

F_beta = (1 + beta^2) * P_lcs * R_lcs
         --------------------------------
         R_lcs + beta^2 * P_lcs
```

默认 `beta=1`，即 Precision 与 Recall 等权。实现使用滚动 DP，时间复杂度 `O(m*n)`，额外空间 `O(min(m,n))`，不重建 alignment。

## 单参考使用

```python
from MLLM.rouge_l import compute_rouge_l

score = compute_rouge_l(
    prediction="the cat was under the bed",
    reference="the cat was found under the bed",
)

precision = score.precision
recall = score.recall
f1 = score.fmeasure
```

默认 raw string 使用 whitespace tokenizer，不自动 lowercase、Unicode normalize、去标点、stemming 或删除特殊 token。`lowercase=True` 是显式实验选项。

## 多语言 tokenization

无空格语言不能依赖默认 whitespace split。可以传入项目 tokenizer：

```python
score = compute_rouge_l(
    prediction_text,
    reference_text,
    tokenizer=lambda text: tokenizer.tokenize(text),
)
```

也提供简单 character baseline：

```python
from MLLM.rouge_l import compute_rouge_l, tokenize_rouge_characters

score = compute_rouge_l(
    "猫坐垫",
    "猫在垫",
    tokenizer=tokenize_rouge_characters,
)
```

character helper 按 Unicode code point 切分并默认跳过 whitespace；它不是 grapheme、词或语素 tokenizer。正式多语言结果应固定同一个 tokenizer/normalizer。

也可直接传入 string token 或 integer token ID 序列：

```python
score = compute_rouge_l([101, 2023, 2003], [101, 2023, 2001])
```

## 多参考与 batch

```python
from MLLM.rouge_l import compute_rouge_l_batch, compute_rouge_l_multi_reference

best = compute_rouge_l_multi_reference(
    prediction,
    [reference_a, reference_b, reference_c],
)
print(best.reference_index, best.score.fmeasure)

scores = compute_rouge_l_batch(predictions, aligned_references)
mean_f1 = sum(score.fmeasure for score in scores) / len(scores)
```

多参考接口选择 `fmeasure` 最高的完整 score；相同时依次比较 Recall、Precision，仍相同则稳定选择最早 reference。batch 接口只返回 aligned 单参考 score，不隐藏 macro/micro 聚合。

## 空序列

默认 `zero_division=0.0`：空 prediction/reference 的未定义比率填 0。设为 `1.0` 时，两侧都空得到满分；只有一侧为空时 F-score 仍为 0。该策略必须在实验之间固定并报告。

## MLLM 公平比较

- 在评分前固定是否移除 system/user prompt、chat template、BOS/EOS、image token 和截断 padding。
- 固定 decode 参数、max tokens、stop condition、大小写、Unicode/标点处理和 tokenizer revision。
- prediction 与 reference 必须使用同一 tokenization；改变 tokenizer 本身会改变 ROUGE-L。
- 多参考数量与选择策略保持一致；不能只给部分实验组更多 reference。
- batch 汇总应同时报告样本数、均值和分布；不要只选最高样本或最高 reference。
- ROUGE-L 衡量表面序列重合，不等价于事实正确性、语义相似度、视觉 grounding 或任务成功率，应与语义和任务指标配合。

## 适用边界

- 当前实现是 sentence-level ROUGE-L，不是 sentence-union LCS 的 ROUGE-Lsum。
- 不提供 stemming、bootstrap confidence interval 或语言专用 normalize。
- 超长序列仍需 `O(m*n)` 时间；应记录截断规则并避免不同组使用不同长度上限。
