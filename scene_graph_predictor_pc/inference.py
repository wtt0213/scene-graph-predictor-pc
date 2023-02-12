import trimesh
import numpy as np
import os
from scene_graph_predictor_pc.src.model.model import MMGNet
from scene_graph_predictor_pc.src.utils.config import Config
from scene_graph_predictor_pc.src.utils import util, define, util, op_utils, util_ply
from itertools import product
import torch
import torch.nn as nn
   
def load_config():
    r"""loads model config

    """
    local_file_path = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(local_file_path, 'config/vlsat.json')
    
    # load config file
    config = Config(config_path)     
    return config

def generate_data(plydata):

    # get file path
    local_file_path = os.path.dirname(os.path.abspath(__file__))

    classNames = util.read_txt_to_list(os.path.join(local_file_path, 'data/classes.txt'))
    # read relationship class
    relationNames = util.read_relationships(os.path.join(local_file_path, 'data/relationships.txt'))
    
    points = np.array(plydata.vertices)
    instances = util_ply.read_labels(plydata).flatten()

    nodes = list(np.unique(instances))
    if 0 in nodes: # remove background
        nodes.remove(0)
    
    edge_indices = list(product(list(range(len(nodes))), list(range(len(nodes)))))
    edge_indices = [i for i in edge_indices if i[0]!=i[1]]

    instances_box = dict()
    dim_point = points.shape[-1]
    obj_points = torch.zeros([len(nodes), 128, dim_point])
    descriptor = torch.zeros([len(nodes), 11])
    
    for i, instance_id in enumerate(nodes):
        # get node point
        try:
            obj_pointset = points[np.where(instances == instance_id)[0]]
        except:
            print('error')
        min_box = np.min(obj_pointset[:,:3], 0) - 0.2
        max_box = np.max(obj_pointset[:,:3], 0) + 0.2
        instances_box[instance_id] = (min_box,max_box)  
        choice = np.random.choice(len(obj_pointset), 128, replace=True)
        obj_pointset = obj_pointset[choice, :]
        descriptor[i] = op_utils.gen_descriptor(torch.from_numpy(obj_pointset)[:,:3])
        obj_pointset = torch.from_numpy(obj_pointset.astype(np.float32))
        obj_pointset[:,:3] = zero_mean(obj_pointset[:,:3])
        obj_points[i] = obj_pointset


    return obj_points, descriptor, edge_indices, classNames, relationNames, nodes

def zero_mean(point):
    mean = torch.mean(point, dim=0)
    point -= mean.unsqueeze(0)
    return point  

class SceneGraphPredictor(nn.Module):
    
    def __init__(self) -> None:
        super().__init__()
        config = load_config()
        self.model = MMGNet(config)
    
    def load(self, path: str) -> None:
        self.model.load(path)

    @torch.no_grad()
    def inference(self, plydata: trimesh, topk: int) -> list:
        # inference
        obj_points, descriptor, edge_indices, classNames, relationNames, instance_ids = generate_data(plydata)
        res = self.model.inference(obj_points, descriptor, edge_indices, topk)
        
        res_list = []
        for i in res:
            res_list.append({
                "object_id":instance_ids[i[0].item()], 
                "object_class":classNames[i[1].item()], 
                "subject_id":instance_ids[i[2].item()], 
                "subject_class":classNames[i[3].item()], 
                "relation_class":relationNames[i[4].item()], 
                "confidence":i[5].item()
            })
        
        return res_list
