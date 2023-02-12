# 点云场景图预测使用说明

# 1. 模型介绍


# 2. 环境依赖

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

# 3. 下载安装

```bash
pip install scene-graph-predictor-pc
```

# 4. 使用说明
```python
import scene_graph_predictor_pc
import trimesh

plydata=trimesh.load('scene/scene0/labels.instances.align.annotated.v2.ply', process=False)
model = scene_graph_predictor_pc.SceneGraphPredictor()
# model 加载数据
model.load('checkpoint')
# 模型预测
res = model.inference(plydata, 10)
```