# Inference Timing

具身策略的通用推理计时与频率换算：

- `benchmark.py`：warmup、同步 wall-clock 计时、分位延迟与 batch throughput。
- `frequency.py`：latency↔Hz、control rate→replanning rate 和有效重规划频率。

迁移伪代码、计时边界和报告规范见 [`docs/embodied/inference_timing.md`](../../docs/embodied/inference_timing.md)。
