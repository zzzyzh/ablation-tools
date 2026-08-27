# Metrics

跨任务复用的指标实现按数学定义直接组织，不按 segmentation、classification 或 detection 再分层：

- [`IoU/`](IoU/)：classwise/micro IoU、mIoU 与 frequency-weighted IoU。
- [`Dice/`](Dice/)：classwise/micro Dice、mean/frequency-weighted Dice 与 generalized Dice。

当前 overlap 指标只接收 hard class-index tensor。使用约定见 [`docs/metrics/`](../docs/metrics/index.md)。
