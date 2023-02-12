import numpy as np
import torch.nn.functional as F
import torch

def inference_triplet(objs_pred, rels_pred, edges, topk=10):
    objs_pred = np.exp(objs_pred)
    size_o, size_r = objs_pred.shape[1], rels_pred.shape[1]
    sample_score_list = []
    for edge in range(len(edges)):
        edge_from = edges[edge][0]
        edge_to = edges[edge][1]
        rel_predictions = rels_pred[edge]
        sub = objs_pred[edge_from]
        obj = objs_pred[edge_to]

        node_score = torch.einsum('n,m->nm',sub,obj)
        conf_matrix = torch.einsum('nl,m->nlm',node_score,rel_predictions)
        conf_matrix_1d = conf_matrix.reshape(-1)

        curr_topk_conf_matrix, curr_topk_conf_id = conf_matrix_1d.topk(1, largest=True)
        sample_score_list.append((curr_topk_conf_matrix, curr_topk_conf_id, edge_from, edge_to))
    
    # sorted by confidence
    sample_score_list = sorted(sample_score_list, key=lambda x: x[0], reverse=True)
    
    res = []
    
    for i in range(topk):
        curr_topk_conf_matrix, curr_topk_conf_id, subjectid, objectid = sample_score_list[i]
        idx = np.unravel_index(curr_topk_conf_id, (size_o, size_o, size_r))
        res.append((subjectid, idx[0], objectid, idx[1], idx[2], curr_topk_conf_matrix))
    
    return res


