# Setup

本目录记录仓库参考实现的验证环境。当前基线为 Python 3.10+、PyTorch 2.6.0；`Pillow` 用于把 RGB tensor 保存为图片。

```bash
python -m pip install -r setup/requirements.txt
python -m pip install -e . --no-deps
```

`requirements.txt` 固定的是可复现参考版本。若项目已有与硬件/CUDA 对应的 PyTorch 环境，可保留兼容版本并只执行 editable install；通用实现本身不绑定 CUDA，也不会在内部选择设备。
