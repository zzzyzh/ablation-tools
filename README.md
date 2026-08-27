<h1 align="center">
  <img src="assets/ablation-tools-icon.png" width="96" alt="ablation-tools icon"><br>
  ablation-tools
</h1>

<p align="center">
  <a href="README.md">English</a> | <a href="README_zh.md">中文</a>
</p>

A model-agnostic reference repository for ablation studies and diagnostic analysis. It collects reusable tensor-level implementations, shared metrics, and methodology documents that can be migrated to concrete models and tasks.

## Project navigation

| Track | Code | Methodology and usage | Current scope |
| --- | --- | --- | --- |
| General | [`General/`](General/) | [`docs/general/`](docs/general/index.md) | Attention visualization, ROI extraction, PCA, t-SNE |
| MLLM | [`MLLM/`](MLLM/) | [`docs/mllm/`](docs/mllm/index.md) | Methodology scaffold for multimodal LLMs |
| Embodied | [`Embodied/`](Embodied/) | [`docs/embodied/`](docs/embodied/index.md) | Methodology scaffold for embodied systems |

Cross-track metrics live in [`Metrics/`](Metrics/). Reproducible environment requirements are documented in [`setup/`](setup/README.md).

## General methods

- [`General/attention_visualization/`](General/attention_visualization/): Grad-CAM, Q/K/V analysis, and heatmap rendering.
- [`General/roi_extraction/`](General/roi_extraction/): Grounding DINO ROI box extraction, box geometry, and explicit candidate selection.
- [`General/dimensionality_reduction/`](General/dimensionality_reduction/): PCA for reusable linear projections and t-SNE for local-neighborhood visualization.

## Shared metrics

- [`Metrics/IoU/`](Metrics/IoU/): classwise/micro IoU, mIoU, and frequency-weighted IoU.
- [`Metrics/Dice/`](Metrics/Dice/): classwise, mean, micro, frequency-weighted, and generalized Dice.
- [`Metrics/ConfusionMatrix/`](Metrics/ConfusionMatrix/): confusion matrices, Precision, Recall, and F1.
- [`Metrics/MAE/`](Metrics/MAE/): Mean Absolute Error with explicit reductions.
- [`Metrics/MSE/`](Metrics/MSE/): Mean Squared Error with explicit reductions.

See [`docs/index.md`](docs/index.md) for repository-wide methodology and naming conventions. Method-specific usage belongs in the corresponding documentation page rather than in the root README.

## Adding an ablation method

1. Place an ablation method under `General`, `MLLM`, or `Embodied`; place cross-task metrics directly under `Metrics/` by mathematical metric family.
2. Define the hypothesis, baseline, single changed factor, tensor contract, and fair-comparison controls before implementation.
3. Implement a model-agnostic core. Keep model layer lookup, token partitioning, and task-object conversion in the caller-side adapter.
4. Add a matching page under `docs/` covering usage, parameter semantics, migration checks, limitations, and unsupported conclusions.
5. Validate formulas, shapes, invalid inputs, dtype/device behavior, and resource lifecycles with temporary synthetic tests; remove temporary test paths and artifacts afterward.
6. Update indexes, public exports, and root navigation, then verify that code, documentation, and actual APIs agree.
7. Ensure the repository contains no caches, generated images, model/data files, temporary scripts, or debug probes before committing.

## Commit requirements

- Keep one method or indivisible metric family per commit; code, documentation, and public exports should land together.
- Use `<type>(<scope>): <summary>`, for example `feat(general): add t-SNE dimensionality reduction`.
- The commit body should state the new method, tensor contract, model-independent boundary, validation performed, and known limitations.
- Do not mix unrelated formatting, generated artifacts, downloaded weights, datasets, or temporary tests into the same commit.
- Inspect the staged diff and verify navigation, naming, links, and behavioral changes before committing.

## Citation

If this repository supports your research, please cite it using [`CITATION.cff`](CITATION.cff) or:

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

Copyright 2026 Zhonghao Yan.

Licensed under the [Apache License 2.0](LICENSE). This license permits use, modification, and redistribution, including commercial use, subject to its notice and license conditions, and includes an explicit patent grant.
