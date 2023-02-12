import torch
from scene_graph_predictor_pc.src.model.model_utils.model_base import BaseModel
from scene_graph_predictor_pc.src.utils import op_utils
from scene_graph_predictor_pc.src.utils.eval_utils import inference_triplet
from scene_graph_predictor_pc.src.model.model_utils.network_GNN import GraphEdgeAttenNetworkLayers
from scene_graph_predictor_pc.src.model.model_utils.network_PointNet import PointNetfeat, PointNetCls, PointNetRelCls, PointNetRelClsMulti

class Baseline(BaseModel):
    """
    512 + 256 baseline
    """
    def __init__(self, config, num_obj_class, num_rel_class, dim_descriptor=11):
        super().__init__('Mmgnet', config)

        self.mconfig = mconfig = config.MODEL
        with_bn = mconfig.WITH_BN

        dim_point = 3
        if mconfig.USE_RGB:
            dim_point +=3
        if mconfig.USE_NORMAL:
            dim_point +=3
        
        dim_f_spatial = dim_descriptor
        dim_point_rel = dim_f_spatial

        self.dim_point=dim_point
        self.dim_edge=dim_point_rel
        self.num_class=num_obj_class
        self.num_rel=num_rel_class
        self.flow = 'target_to_source'
        self.clip_feat_dim = self.config.MODEL.clip_feat_dim

        dim_point_feature = 512
        
        if self.mconfig.USE_SPATIAL:
            dim_point_feature -= dim_f_spatial-3 # ignore centroid
        
        # Object Encoder
        self.obj_encoder = PointNetfeat(
            global_feat=True, 
            batch_norm=with_bn,
            point_size=dim_point, 
            input_transform=False,
            feature_transform=mconfig.feature_transform,
            out_size=dim_point_feature)      
        
        # Relationship Encoder
        self.rel_encoder = PointNetfeat(
            global_feat=True,
            batch_norm=with_bn,
            point_size=dim_point_rel,
            input_transform=False,
            feature_transform=mconfig.feature_transform,
            out_size=mconfig.edge_feature_size)
        
        self.gcn = GraphEdgeAttenNetworkLayers(512,
                            256,
                            self.mconfig.DIM_ATTEN,
                            self.mconfig.N_LAYERS, 
                            self.mconfig.NUM_HEADS,
                            self.mconfig.GCN_AGGR,
                            flow=self.flow,
                            attention=self.mconfig.ATTENTION,
                            use_edge=self.mconfig.USE_GCN_EDGE,
                            DROP_OUT_ATTEN=self.mconfig.DROP_OUT_ATTEN)


        self.obj_predictor = PointNetCls(num_obj_class, in_size=512,
                                 batch_norm=with_bn, drop_out=True)

        if mconfig.multi_rel_outputs:
            self.rel_predictor = PointNetRelClsMulti(
                num_rel_class, 
                in_size=mconfig.edge_feature_size, 
                batch_norm=with_bn,drop_out=True)
        else:
            self.rel_predictor = PointNetRelCls(
                num_rel_class, 
                in_size=mconfig.edge_feature_size, 
                batch_norm=with_bn,drop_out=True)

    def forward(self, obj_points, edge_indices, descriptor=None, batch_ids=None, istrain=False):

        obj_feature = self.obj_encoder(obj_points)
        tmp = descriptor[:,3:].clone()
        tmp[:,6:] = tmp[:,6:].log() # only log on volume and length
        obj_feature = torch.cat([obj_feature, tmp],dim=1)
        
        ''' Create edge feature '''
        with torch.no_grad():
            edge_feature = op_utils.Gen_edge_descriptor(flow=self.flow)(descriptor, edge_indices)
        
        rel_feature = self.rel_encoder(edge_feature)
        
        gcn_obj_feature, gcn_rel_feature, _ = self.gcn(obj_feature, rel_feature, edge_indices)

        rel_cls = self.rel_predictor(gcn_rel_feature)

        obj_logits = self.obj_predictor(gcn_obj_feature)

        return obj_logits, rel_cls
            
    def inference(self, obj_points, descriptor, edge_indices, topk):

        edge_indices = torch.tensor(edge_indices).long()
        obj_points = obj_points.permute(0,2,1).contiguous()

        with torch.no_grad():
            obj_pred, rel_pred = self(obj_points, edge_indices.t().contiguous(), descriptor, istrain=False)
        
        print("Start inference")
        predicts = inference_triplet(obj_pred, rel_pred, edge_indices, topk)

        return predicts
