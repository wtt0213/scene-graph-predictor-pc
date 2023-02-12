# 点云场景图预测使用说明

# 1. 环境依赖

CUDA版本: 11.3
其他依赖库的安装命令如下：

```bash
conda create -n py38 python=3.8
```

```bash
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.12.1+cu113.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.12.1+cu113.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.12.1+cu113.html
pip install torch-geometric
```

# 2. 下载安装

```
pip install scene-graph-predictor-pc
```

# 3. 使用说明