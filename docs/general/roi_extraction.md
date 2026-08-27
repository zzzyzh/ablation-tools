# Grounding DINO ROI BBox 提取

该方法把文本引导的 ROI bbox 提取拆成三层：Grounding DINO 候选生成、通用 bbox 几何、显式候选选择。实现参考 `native-lwam/action` 的实际检测流程，但不包含其中的相机名称、视频读取、黑框遮挡、背景 control 或 action 评估。

> Bbox 提取本身是观测步骤，不是因果消融。使用 ROI 做 mask、crop、替换或扰动时，需要把干预方式和同面积/同分布 control 单独定义并报告。

## 环境准备

参考环境在 `setup/requirements.txt` 中固定 `transformers==4.57.1`。该版本 Grounding DINO 后处理的候选阈值参数名是 `threshold`。

```bash
python -m pip install -r setup/requirements.txt
python -m pip install -e . --no-deps
```

也可以只安装对应 extra：

```bash
python -m pip install -e '.[grounding-dino]'
```

## 方法边界

| 层 | 公共接口 | 责任 |
| --- | --- | --- |
| 模型适配 | `GroundingDINOBoxExtractor` | processor/model 加载、批量推理、HF 后处理 |
| Bbox 几何 | `clip_bounding_boxes_xyxy`、`round_bounding_boxes_xyxy_outward` | 坐标裁剪、有效性、面积、触边和整数化 |
| 候选选择 | `select_roi_detections` | score、面积、label、边界与 top-k 的显式策略 |

通用层不会硬编码 checkpoint、设备、prompt、camera、fallback 或 NMS。参考流程本身没有额外 NMS，因此本实现也不会隐式抑制重叠候选。

## 输出契约

每张图返回一个 `ROIDetections`：

- `boxes_xyxy`：浮点 `[N, 4]`，绝对原图像素坐标 `[left, top, right, bottom]`；
- `scores`：与 bbox 对齐的 `[N]` 置信度；
- `labels`：与 bbox 对齐的文本 label tuple；
- `image_size`：固定为 `(height, width)`；
- `areas` / `area_fractions`：连续 bbox 面积与全图占比。

坐标采用 half-open 语义，`right`/`bottom` 可以等于图像宽高。向图像切片前使用 `round_bounding_boxes_xyxy_outward`：左上取 `floor`，右下取 `ceil`，随后可直接用于 `image[y0:y1, x0:x1]` 或 Pillow `crop((x0, y0, x1, y1))`。

HF 后处理可能给出轻微越界的 float box。实现先 clip 到 `[0, width] × [0, height]`，再丢弃 clip 后零面积框；NaN/Inf、shape 错误和 batch 对不齐均 fail-fast。

## 基本使用

```python
from PIL import Image

from General.roi_extraction import (
    GroundingDINOBoxExtractor,
    round_bounding_boxes_xyxy_outward,
    select_roi_detections,
)

extractor = GroundingDINOBoxExtractor.from_pretrained(
    model_name_or_path="/path/to/grounding-dino-base",
    device="cuda:0",
    local_files_only=True,
)

image = Image.open("frame.png").convert("RGB")
candidates = extractor.extract_roi_boxes(
    [image],
    "excavator toy.",
    box_threshold=0.20,  # 保留后续 fallback 可能需要的低分候选
    text_threshold=0.20,
)[0]

selection = select_roi_detections(
    candidates,
    score_threshold=0.25,
    exclude_border=True,
    border_margin=1,
    maximum_area_fraction=0.8,
    max_detections=1,
)

if not selection.detections.is_empty:
    box_int = round_bounding_boxes_xyxy_outward(
        selection.detections.boxes_xyxy,
        selection.detections.image_size,
    )[0]
    roi = image.crop(tuple(box_int.tolist()))
```

`extract_roi_boxes` 接受 PIL image，或 CPU `uint8` RGB tensor `[3, H, W]`。单个 prompt 会广播到整个 batch；prompt 序列必须与图片等长。默认 `normalize_prompts=True` 会小写、压缩空格并补上结尾 `.`，以避免 Transformers 把“不含点号的字符串列表”误判为单张图片的候选 label 列表。该开关属于实验配置，必须记录。

## 复现参考项目的选择策略

参考项目先以 `box_threshold=0.20` 生成候选，再在调用侧执行：

- primary：剔除距离图像边界 1px 内的框，先取 `score >= 0.25` top-1；为空时再以 `score >= 0.20` 选择，并显式记录 fallback；
- wrist / wrist_right：允许触边，只取 `score >= 0.25` top-1，不使用 fallback。

General 不包含 camera 分支。fallback 应由两次显式选择表达：

```python
selected = select_roi_detections(
    candidates,
    score_threshold=0.25,
    exclude_border=True,
    border_margin=1,
    max_detections=1,
)
fallback_used = False
if selected.detections.is_empty:
    selected = select_roi_detections(
        candidates,
        score_threshold=0.20,
        exclude_border=True,
        border_margin=1,
        max_detections=1,
    )
    fallback_used = not selected.detections.is_empty
```

注意 Transformers 4.57.1 的后处理保留 `score > box_threshold` 的候选，而 `select_roi_detections` 使用 `score >= score_threshold`。因此候选生成阈值应不高于所有后续选择阈值。

## 公平比较与记录要求

- 固定 Grounding DINO checkpoint/revision、Transformers 版本、device/dtype 与 processor 配置。
- 固定原始分辨率和色彩顺序；不要把 BGR ndarray 当作 RGB 输入。
- 固定原始 prompt、是否规范化、box/text threshold 和 label 解释方式。
- 固定 border margin、面积上下限、允许 label 与 top-k；不要看完结果后临时挑框。
- 保存全部候选，而不只保存 selected bbox，便于重算选择策略。
- 漏检应作为显式空 `ROIDetections` 记录，不应静默丢帧。
- 改 prompt、threshold 或 checkpoint 后不得复用旧候选缓存，避免混合配置。
- ROI 面积过大时，mask/crop 可能成为强分布外干预；应报告面积分布并设置合理 control。

## 已知限制

- Grounding DINO label 可能是完整 phrase，也可能是 token 片段；默认不做 exact-label 二次过滤。
- 本方法不执行跨帧跟踪、跨相机融合、NMS、视频同步或结果落盘。
- `text_threshold` 控制 token 到 phrase 的解码，不等价于 bbox score threshold。
- PIL 绘制矩形的右下端点常按包含式处理，而本仓库 bbox/crop 采用 half-open；可视化时需避免多画 1px。
