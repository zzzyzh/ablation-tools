"""Model-agnostic attention attribution, Q/K/V analysis, and rendering."""

from .gradcam import (
    GradCAMCapture,
    compute_attention_gradcam,
    compute_gradcam,
    normalize_heatmap,
)
from .qkv import (
    QKVAttentionResult,
    QKVFeatureMaps,
    compute_qk_attention_logits,
    compute_qkv_attention,
    compute_qkv_feature_maps,
    compute_token_feature_map,
)
from .visualization import (
    ColormapName,
    overlay_heatmap,
    render_heatmap,
    resize_heatmap,
    save_rgb_image,
)

__all__ = [
    "ColormapName",
    "GradCAMCapture",
    "QKVAttentionResult",
    "QKVFeatureMaps",
    "compute_attention_gradcam",
    "compute_gradcam",
    "compute_qk_attention_logits",
    "compute_qkv_attention",
    "compute_qkv_feature_maps",
    "compute_token_feature_map",
    "normalize_heatmap",
    "overlay_heatmap",
    "render_heatmap",
    "resize_heatmap",
    "save_rgb_image",
]
