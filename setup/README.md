# Setup

本目录记录仓库参考实现的验证环境：Python 3.10+、PyTorch 2.6.0、Pillow 11.3.0、Grounding DINO 适配使用的 Transformers 4.57.1，以及 t-SNE 使用的 scikit-learn 1.5.2。

```bash
python -m pip install -r setup/requirements.txt
python -m pip install -e . --no-deps
```

`requirements.txt` 固定的是可复现参考版本。若项目已有与硬件/CUDA 对应的 PyTorch 环境，可保留兼容版本并只安装其余依赖；通用实现不在内部选择设备。
