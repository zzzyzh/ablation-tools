# 注意力特征图可视化

本主题提供两类互补的通用方法：

- **Grad-CAM** 用目标标量对中间激活或 attention weights 的梯度，估计“哪些位置对当前目标更重要”。
- **Q/K/V 分析** 直接检查 query-key 兼容性、softmax 后的注意力和 value 聚合，用于回答“注意力内部是怎样匹配与传递信息的”。

实现位于 `General/attention_visualization/`。它们只处理已抽取的 `torch.Tensor`；层定位、模型输出解包、特殊 token 识别和 patch 网格推导属于迁移代码。

> 这些图描述模型内部的关联或局部敏感性，不能单独证明因果贡献。若研究问题是“移除该区域是否改变输出”，还需要遮挡、置换或干预实验。

## 环境准备

仓库参考环境固定在 `setup/requirements.txt`：Python 3.10+、PyTorch 2.6.0，并使用 Pillow 保存图片。

```bash
python -m pip install -r setup/requirements.txt
python -m pip install -e . --no-deps
```

若已有与硬件匹配的 PyTorch 环境，可只安装项目本身；计算与渲染接口均不在内部选择设备。


## 方法选择

| 方法/API | 需要的信息 | 返回的东西 | 适合的问题 |
| --- | --- | --- | --- |
| `GradCAMCapture` + `compute_gradcam` | 中间激活及目标标量对它的梯度 | 沿 channel 加权聚合的空间/token CAM | 目标输出对哪些中间位置敏感？ |
| `compute_attention_gradcam` | attention weights 及其梯度 | 融合 head 后的 query-key CAM | 对当前目标而言，哪些 Q→K 连接更重要？ |
| `compute_qk_attention_logits` | Q、K 和可选 attention mask | softmax 之前的 QK logits | 哪些 token 在表示空间中更兼容？ |
| `compute_qkv_attention` | Q、K、V 和可选 attention mask | logits、probabilities 和 context | softmax 如何改变匹配，V 如何被聚合？ |
| `compute_qkv_feature_maps` | Q、K、V、token 网格信息 | Q/K/V 各自的 token 强度图 | 三类投影的空间活跃度有何差异？ |
| `compute_token_feature_map` | 单个 token 特征与其独立网格 | 单路 token 强度图 | cross-attention 两侧布局不同时如何分别成图？ |
| `render_heatmap` / `overlay_heatmap` | 标量热力图，可选原图 | RGB 张量 | 如何用统一色图渲染或叠加结果？ |

Grad-CAM 依赖一个明确的目标标量，因此可以对不同类别、token 或时间步得到不同图。Q/K/V 特征图不使用目标梯度，更适合检查层/head 内部结构。两者的数值语义不同，不应将颜色深浅作直接横向比较。

## 张量约定

### Grad-CAM

`compute_gradcam(activations, gradients, ...)` 要求两个张量形状完全相同。对图像特征，常见形状为 `[batch, channels, height, width]`，默认 `channel_dim=1`，返回 `[batch, height, width]`。对 token 特征 `[batch, tokens, channels]`，使用 `channel_dim=-1`，返回 `[batch, tokens]`。

`compute_attention_gradcam(attention_weights, gradients, head_dim=1)` 的典型输入是 `[batch, heads, query_tokens, key_tokens]`，返回 `[batch, query_tokens, key_tokens]`。它沿 head 维融合，不会自动删除 prefix token。

两个 API 默认使用 ReLU 保留正贡献，并对每张 CAM 做 min-max 归一化。若要研究负贡献或跨样本幅度，显式设置 `relu=False` 或 `normalize=False`，并保存原始结果。

### Q/K/V

Q/K/V 计算接收 canonical `[..., tokens, channels]` 布局。既可传入 `[batch, tokens, channels]`，也可传入多头 `[batch, heads, tokens, head_channels]`。Q 和 K 的 token 数可以不同；K 和 V 的 token 数必须一致。

`compute_qk_attention_logits` 和 `compute_qkv_attention` 默认按 `1 / sqrt(channels)` 缩放 QK 点积。这个缩放只应执行一次；如果传入的 Q 已经在模型内缩放，需要设置 `scale=False`。

`attention_mask` 支持两种明确语义：布尔 mask 中 `True` 表示可见、`False` 表示屏蔽；浮点 mask 作为 additive bias 加到 logits，常用 `0`/`-inf` 表示可见/屏蔽。mask 必须能广播到完整的 `[..., query_tokens, key_tokens]` 形状且不得引入额外维度。例如对 `[batch, heads, query_tokens, key_tokens]` logits，`[key_tokens]`、`[query_tokens, key_tokens]` 和 `[batch, 1, 1, key_tokens]` 都是常见布局。完全被屏蔽的 query 行在 `.probabilities` 中返回全零，避免生成 NaN。

`compute_qkv_attention` 返回一个结果对象：

- `.logits`：`[..., query_tokens, key_tokens]`，为应用 mask 后的 softmax 前分数；
- `.probabilities`：最后一维 softmax 后的注意力概率；
- `.context`：`probabilities @ value` 的聚合结果。

`compute_qkv_feature_maps` 先在 channel 上将每个 token 约化为标量，再融合可选 head、删除 prefix token 并重排为空间网格。`feature_reduction="l2"` 表示 token 向量范数；`head_fusion` 可显式选择 `"mean"`、`"max"` 或 `"sum"`。返回对象的 `.query`、`.key` 和 `.value` 使用相同聚合口径。

例如，对 `[batch, heads, tokens, head_channels]` 输入使用 `head_dim=1`、`prefix_tokens=1` 和 `spatial_shape=(patch_height, patch_width)` 时，若 `tokens - 1 == patch_height * patch_width`，三个返回图的形状均为 `[batch, patch_height, patch_width]`。

`compute_qkv_feature_maps` 对 Q/K/V 共用一套 `prefix_tokens` 与 `spatial_shape`，适合三路 token 布局相同的 self-attention。cross-attention 的 query 与 key/value 常有不同前缀或网格，应分别调用单张量接口：

```python
from General.attention_visualization import compute_token_feature_map

query_map = compute_token_feature_map(
    query, head_dim=1, prefix_tokens=query_prefix_tokens,
    spatial_shape=query_spatial_shape,
)
key_map = compute_token_feature_map(
    key, head_dim=1, prefix_tokens=key_prefix_tokens,
    spatial_shape=key_spatial_shape,
)
```


## 示例：捕获模型中间层的 Grad-CAM

下面的 `target_module`、`select_target_score` 和 prefix token 数是唯一需要按模型调整的部分。示例假设该层输出 `[batch, tokens, channels]`，且剩余 token 可排列为 `patch_height × patch_width`。

```python
import torch

from General.attention_visualization import GradCAMCapture, overlay_heatmap

model.eval()
model.zero_grad(set_to_none=True)

with GradCAMCapture(target_module) as capture:
    prediction = model(model_inputs)
    target_score = select_target_score(prediction)  # 必须是标量
    target_score.backward()

    token_cam = capture.compute(channel_dim=-1)  # [batch, tokens]

patch_cam = token_cam[:, num_prefix_tokens:].reshape(
    token_cam.shape[0], patch_height, patch_width
)
overlay = overlay_heatmap(input_image, patch_cam, alpha=0.5, colormap="turbo")
```

如果模块返回 tuple、dict 或自定义对象，通过 `output_selector` 只选出需要的 Tensor：

```python
with GradCAMCapture(
    target_module,
    output_selector=lambda output: output[0],
) as capture:
    prediction = model(model_inputs)
    select_target_score(prediction).backward()
    cam = capture.compute(channel_dim=-1)
```

`GradCAMCapture` 在 forward 后提供 `capture.activations`，在 backward 后提供 `capture.gradients`。也可直接调用纯函数，便于对已保存张量做离线分析：

```python
from General.attention_visualization import compute_gradcam

cam = compute_gradcam(
    activations,
    gradients,
    channel_dim=-1,
    relu=True,
    normalize=True,
)
```

上下文管理器会在退出时移除 hook。若不使用 `with`，在实验结束后调用 `close()`；多次前向之间可用 `clear()` 清理已捕获状态。

## 示例：attention Grad-CAM

attention Grad-CAM 需要实际参与目标计算图的 attention weights。如果该 Tensor 不是叶子节点，在 backward 之前调用 `retain_grad()`：

```python
from General.attention_visualization import compute_attention_gradcam

attention_weights.retain_grad()
target_score.backward()

attention_cam = compute_attention_gradcam(
    attention_weights,
    attention_weights.grad,
    head_dim=1,
)  # [batch, query_tokens, key_tokens]
```

这个二维矩阵的横纵轴是 key/query token，不是图像的 height/width。要叠加到图像上，必须先固定一个 query，再删除 key 中的特殊 token，最后按已知 patch 网格 reshape。

## 示例：重算 Q/K/V attention

```python
from General.attention_visualization import (
    compute_qk_attention_logits,
    compute_qkv_attention,
)

# query/key/value: [batch, heads, tokens, head_channels]
logits = compute_qk_attention_logits(
    query,
    key,
    scale=True,
    attention_mask=attention_mask,
)

attention = compute_qkv_attention(
    query,
    key,
    value,
    scale=True,
    attention_mask=attention_mask,
)

assert torch.allclose(logits, attention.logits)
probabilities = attention.probabilities
context = attention.context
```

这是一个用于分析的纯 attention 重算。在将它与模型内部输出比较之前，需核对模型是否另外应用 Q/K normalization、位置旋转、logit soft-cap、attention bias、dropout 或特殊缩放。分析结果不一致时，不要默认通用重算就是模型真实路径。

## 示例：Q/K/V token 特征图

```python
from General.attention_visualization import (
    compute_qkv_feature_maps,
    render_heatmap,
)

feature_maps = compute_qkv_feature_maps(
    query,
    key,
    value,
    head_dim=1,
    head_fusion="mean",
    feature_reduction="l2",
    spatial_shape=(patch_height, patch_width),
    prefix_tokens=num_prefix_tokens,
    normalize=False,
)

# 分析时保留 feature_maps.query 的原始幅度，展示时再归一化。
query_rgb = render_heatmap(
    feature_maps.query,
    colormap="turbo",
    normalize=True,
)
```

`spatial_shape` 必须来自实际 tokenizer/patch embedding 配置，不应通过 `sqrt(tokens)` 猜测。非方形输入、动态分辨率、多图输入和视频都可以产生非方形或分段 token 布局。

## 示例：统一渲染

```python
from General.attention_visualization import (
    overlay_heatmap,
    render_heatmap,
    save_rgb_image,
)

heatmap_rgb = render_heatmap(
    patch_cam,
    colormap="turbo",
    normalize=True,
)  # [batch, 3, height, width]

overlay_rgb = overlay_heatmap(
    input_image,
    patch_cam,
    alpha=0.5,
    colormap="turbo",
    normalize=True,
)

# Pillow 是可选依赖；需要落盘时安装 `.[visualization]`。
save_rgb_image(overlay_rgb[0], "outputs/attention_overlay.png")
```

`render_heatmap` 接收 `[height, width]` 或 `[batch, height, width]`，返回 channel-first RGB 浮点张量，值域为 `[0, 1]`。`overlay_heatmap` 接收 `[channels, height, width]` 或 batch 版本的原图，会将热力图缩放到原图尺寸。输入必须是可展示色域；如果是模型归一化后的图像，先用对应 mean/std 反归一化。

`overlay_heatmap` 中的 `alpha` 是最大不透明度，实际像素不透明度还会乘以归一化后的热力图强度；零强度区域不会改变原图。

## 公平比较与控制变量

| 项目 | 必须固定或明确报告的内容 |
| --- | --- |
| 模型状态 | 权重、`train`/`eval` 模式、dropout、梯度检查点、编译/融合 attention 路径 |
| 目标 | 同一标量定义，例如同一类别 logit 或同一生成 token 的 pre-softmax logit |
| 层与 head | 同一层索引、同一 head 索引或融合策略；不在看到结果后临时挑图 |
| token 布局 | prefix/suffix token 数、query/key 范围、patch 网格与顺序 |
| attention 计算 | Q/K 缩放、mask/bias 语义、softmax dtype、dropout 与位置编码所在阶段 |
| CAM 约化 | channel/head 维、ReLU 开关、head 融合、feature reduction |
| 归一化 | 逐样本还是全局，是否使用相同数值范围；原始幅度必须另存 |
| 渲染 | 色图、alpha、插值方式、输出分辨率与原图反归一化 |
| 数据 | 同一样本、预处理、随机种子和样本顺序；尽量使用成对比较 |

单独对每张图做 min-max 归一化会让弱信号与强信号看起来同样显著。它适合展示图内分布，但不适合比较样本或 ablation 组之间的绝对幅度。跨组比较时，应用预先定义的公共范围，并报告未归一化统计量。

## 迁移检查清单

1. 写明研究问题，在看图前固定目标、层、head、query 和聚合策略。
2. 定位真正参与计算的模块或 Q/K/V Tensor，确认 hook 不是安装在未执行的替代分支上。
3. 打印并记录张量形状，明确 batch、head、token、channel 和空间维；不依赖默认猜测。
4. 根据 tokenizer 或 patch embedding 推导 prefix token 和网格，并验证 `remaining_tokens == height * width`。
5. Grad-CAM 路径关闭 `torch.no_grad()`/`torch.inference_mode()`，定义标量 target，每次 backward 前清空旧梯度。
6. Q/K/V 路径核对 projection 之后的 reshape/transpose、缩放、mask、bias、位置编码和 normalization。
7. 用一个小张量手算 QK 点积与 softmax，用单层小样本检查 CAM 的 shape、有限性与非零梯度。
8. 分开保存原始张量、聚合结果、归一化参数、渲染图和完整元数据。
9. 在同一样本上运行 baseline 和 ablation，先比较原始数值统计，再比较图像。
10. 用参数随机化、标签随机化或局部遮挡做 sanity check，并与任务指标的变化一起报告。

## 常见问题

### CAM 全零或没有梯度

检查 forward 是否在 `no_grad`/`inference_mode` 中，target 是否真正依赖捕获层，以及 fused attention/编译是否跳过了所选模块。若只在 `relu=True` 时全零，可以用 `relu=False` 诊断是否主要为负梯度，但必须明确更改后的语义。

### 热力图位置错位

最常见原因是 prefix token 计数、row-major/column-major 顺序、crop 后尺寸或插值目标错误。先用按索引递增的人工网格验证 reshape 和叠加路径，再渲染真实结果。

### 重算 attention 与模型输出不同

逐项核对 Q/K normalization、RoPE 时机、scale、additive bias、bool/additive mask、softmax dtype、dropout 和 grouped/multi-query attention 的 head 映射。测试时优先使用高精度和无 dropout 的小张量，再逐步还原模型配置。

### 不同图看起来都很显著

通常是逐图 min-max 归一化隐藏了幅度差异，或者先归一化每个 head 再聚合改变了权重。对比原始张量的 min/max/mean/std、零值比例和分位数，并固定“先聚合还是先归一化”的顺序。

### 把 attention 当成解释结论

attention probability 是某一层内的路由权重，还会受 residual、MLP、后续层和 value 内容影响。Q/K/V 范数图只表示表示强度，不是 token 贡献。把可视化结果解释为证据时，至少搭配一项干预结果和一项任务指标。

## 建议报告模板

每个图或表的 caption 至少写明：方法名、target 定义、层/head/query 索引、token 筛选、聚合顺序、ReLU/归一化、色图/插值和样本数。正文同时报告 baseline 与 ablation 的原始幅度统计、任务指标、种子间离散度和预先定义的对比准则。
