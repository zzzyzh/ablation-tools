# Setup

参考 Conda 环境固定命名为 `ablation-tools`，Python 版本为 3.10：

```bash
conda create -n ablation-tools python=3.10 -y
conda activate ablation-tools
```

随后安装固定参考依赖与 editable package：

```bash
python -m pip install -r setup/requirements.txt
python -m pip install -e . --no-deps
```

当前参考依赖包括 PyTorch 2.6.0、Pillow 11.3.0、Transformers 4.57.1、t-SNE 使用的 scikit-learn 1.5.2，以及 MiniLM sentence similarity 使用的 sentence-transformers 3.4.1。

`requirements.txt` 固定的是可复现参考版本。若项目已有与硬件/CUDA 对应的 PyTorch 环境，可在同名 Conda 环境中保留兼容版本并只安装其余依赖；通用实现不在内部选择设备。
