import numpy as np
import torch
import torch.nn as nn

from models.detr_modules.detr import MLP
from models.detr_modules.transformer import TransformerDecoder, TransformerDecoderLayer
from models.graph_transformer_dense import GraphTransformerLayerDense



import networkx as nx
import einops as es
import einops.layers.torch as el
from detectron2.config import configurable

class PairwiseConcatLayer(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, y):
        d1, d2 = x.shape[-2], y.shape[-2]
        grid_x, grid_y = torch.meshgrid(torch.arange(d1, device=x.device), torch.arange(d2, device=y.device), indexing='ij')
        res = torch.concat([torch.index_select(x, dim=-2, index=grid_x.flatten()), torch.index_select(y, dim=-2, index=grid_y.flatten())], dim=-1)
        res = es.rearrange(res, '... (L1 L2) C -> ... L1 L2 C', L1=d1, L2=d2)
        return res
        

class DummyHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
    
    def forward(self, hs):
        hs = hs ** 2

class BaseGHead(nn.Module):
    
    @configurable
    def __init__(
        self,
        num_layers=1,
        in_dim=256,
        hidden_dim=256,
        num_heads=1,
        num_nodes=100,
        edge_features='constant_one',
        extra_edge_feature_dim=0,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_nodes = num_nodes
        self.num_layers = num_layers
        self.edge_features = edge_features
        self.extra_edge_feature_dim = int(extra_edge_feature_dim)
        if self.extra_edge_feature_dim < 0:
            raise ValueError("extra_edge_feature_dim must be non-negative")
        
        if edge_features == 'concat':
            out_proj_edge = hidden_dim // 2
            self.pairwise_layer = PairwiseConcatLayer()
        else:
            out_proj_edge = hidden_dim 

        
        self.proj_e1 = nn.Linear(in_dim, out_proj_edge)
        self.proj_e2 = nn.Linear(in_dim, out_proj_edge)
        
        self.proj_node_input = nn.Linear(in_dim, hidden_dim)
        self.edge_features = edge_features
        self.hidden_dim = hidden_dim
        if self.extra_edge_feature_dim > 0:
            self.extra_edge_proj = nn.Sequential(
                nn.Linear(self.extra_edge_feature_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
            )
        else:
            self.extra_edge_proj = None
    
    @classmethod
    def from_config(cls, cfg):
        cfg = cfg.MODEL.GRAPH_HEAD
        return {
            'num_layers': cfg.NUM_LAYERS,
            'hidden_dim': cfg.HIDDEN_DIM,
            'num_heads': cfg.NUM_HEADS,
            'edge_features': cfg.EDGE_FEATURES,
            'extra_edge_feature_dim': getattr(cfg, 'EXTRA_EDGE_FEATURE_DIM', 0),
        }
    
    
    def _prepare_extra_edge_features(self, edge_extra_features, *, layers, batch_size, num_nodes, device, dtype):
        if self.extra_edge_feature_dim == 0:
            if edge_extra_features is not None:
                raise ValueError("edge_extra_features were provided but extra_edge_feature_dim is 0")
            return None
        if edge_extra_features is None:
            raise ValueError("edge_extra_features are required when extra_edge_feature_dim is positive")
        extra = edge_extra_features.to(device=device, dtype=dtype)
        if extra.shape[-1] != self.extra_edge_feature_dim:
            raise ValueError(
                f"edge_extra_features last dim must be {self.extra_edge_feature_dim}, got {extra.shape[-1]}"
            )
        if extra.dim() == 3:
            if extra.shape[:2] != (num_nodes, num_nodes):
                raise ValueError(f"edge_extra_features must have shape [Q,Q,F], got {tuple(extra.shape)}")
            extra = extra.unsqueeze(0).unsqueeze(0).expand(layers, batch_size, -1, -1, -1)
        elif extra.dim() == 4:
            if extra.shape[0] != batch_size or extra.shape[1:3] != (num_nodes, num_nodes):
                raise ValueError(f"edge_extra_features must have shape [B,Q,Q,F], got {tuple(extra.shape)}")
            extra = extra.unsqueeze(0).expand(layers, -1, -1, -1, -1)
        elif extra.dim() == 5:
            if extra.shape[:4] != (layers, batch_size, num_nodes, num_nodes):
                raise ValueError(f"edge_extra_features must have shape [L,B,Q,Q,F], got {tuple(extra.shape)}")
        else:
            raise ValueError(f"edge_extra_features must have 3, 4, or 5 dims, got {tuple(extra.shape)}")
        return extra

    def _compute_edge_features(self, features, edge_extra_features=None):
        # features L B Q C 
        L, B, Q, C = features.shape
        device = features.device
        C = self.hidden_dim
        e1, e2 = self.proj_e1(features), self.proj_e2(features)
        if self.edge_features == 'concat':
            e = self.pairwise_layer(e1, e2)
        elif self.edge_features == 'sum':
            e = e1[:, :, None, :, :] + e2[:, :, :, None, :]
        elif self.edge_features == 'diff': 
            e = e1[:, :, None, :, :] - e2[:, :, :, None, :]
        elif self.edge_features == 'div':
            e = e1[:, :, None, :, :] / e2[:, :, :, None, :]
        elif self.edge_features == 'mul':
            e = e1[:, :, None, :, :] * e2[:, :, :, None, :]
        else:
            raise NotImplementedError(f'{self.edge_features} aggregations not implemented')
        extra = self._prepare_extra_edge_features(
            edge_extra_features,
            layers=L,
            batch_size=B,
            num_nodes=Q,
            device=device,
            dtype=features.dtype,
        )
        if extra is not None:
            e = e + self.extra_edge_proj(extra)
        return e
    

class DenseGraphTransformerHead(BaseGHead):
    
    @configurable
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.graph_transformer_layers = nn.ModuleList([
            GraphTransformerLayerDense(in_dim=self.hidden_dim, out_dim=self.hidden_dim, num_heads=self.num_heads, layer_norm=True, batch_norm=False)
        for _ in range(self.num_layers)
        ])
        self.edge_cls = MLP(input_dim=self.hidden_dim, hidden_dim=self.hidden_dim//2, output_dim=1, num_layers=3)
    
    @classmethod
    def from_config(cls, cfg):
        cfg = cfg.MODEL.GRAPH_HEAD
        return {
            'num_layers': cfg.NUM_LAYERS,
            'hidden_dim': cfg.HIDDEN_DIM,
            'num_heads': cfg.NUM_HEADS,
            'edge_features': cfg.EDGE_FEATURES,
            'extra_edge_feature_dim': getattr(cfg, 'EXTRA_EDGE_FEATURE_DIM', 0),
        }
    
    def forward(self, hs: torch.Tensor, edge_extra_features=None):
        """ _summary_
        Args:
            hs (torch.Tensor): L x B x Q x C
        """
        L, B, Q, C = hs.shape 
        
        e = self._compute_edge_features(features=hs, edge_extra_features=edge_extra_features)
        hs = self.proj_node_input(hs)
        # hs = es.rearrange(hs, 'B L Q C -> (B L) Q C')
        
        
        # e = es.rearrange(e, '(B Q1 Q2) C -> B Q1 Q2 C', B=L*B, Q1=Q, Q2=Q, C=C)
        hs = es.rearrange(hs, 'L B Q C -> (L B) Q C')
        e = es.rearrange(e, 'L B Q1 Q2 C -> (L B) Q1 Q2 C')
        
        for layer in self.graph_transformer_layers:
            hs, e = layer(hs, e)
        e = self.edge_cls(e)
        e = es.rearrange(e, "(L B) Q1 Q2 C -> L B Q1 Q2 C", C=1, L=L, Q1=Q, Q2=Q, B=B)
        return e


def build_graph_head(cfg):
    name = cfg.MODEL.GRAPH_HEAD.NAME
    head = {
        'DummyHead': DummyHead,
        'GraphTransformerDense': DenseGraphTransformerHead,
    }[name](cfg)
    return head
    
