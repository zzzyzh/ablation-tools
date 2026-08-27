# ROI Extraction

本目录整理文本引导的 ROI bbox 提取流程，并把模型候选生成、bbox 几何与实验选择策略分离：

- `grounding_dino.py`：Grounding DINO 批量候选提取与惰性 Transformers 加载。
- `bounding_boxes.py`：绝对像素 XYXY、clip、有效框、面积、触边和向外整数化。
- `roi_selection.py`：显式 score、面积、边界、label 与 top-k 选择，不隐藏 fallback 或相机策略。

具体环境、张量契约和迁移方式见 [`docs/general/roi_extraction.md`](../../docs/general/roi_extraction.md)。
