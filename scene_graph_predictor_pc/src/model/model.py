# if __name__ == '__main__' and __package__ is None:
#     from os import sys
#     sys.path.append('../')
import os
from scene_graph_predictor_pc.src.model.vlsat.model import Baseline


class MMGNet():
    def __init__(self, config):
        self.config = config
        self.num_obj_class = 160
        self.num_rel_class = 26
        
        ''' Build Model '''
        self.model = Baseline(self.config, self.num_obj_class, self.num_rel_class)

    def load(self, path=''):
        return self.model.load(path)
    
    def inference(self, obj_points, descriptor, edge_indices, topk):
        return self.model.inference(obj_points, descriptor, edge_indices, topk)