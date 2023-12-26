from pathlib import Path
from types import SimpleNamespace
import warnings
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional, List, Callable, Tuple
from torchrl.modules import MLP

FLASH_AVAILABLE = False

class LearnablePositionalEncoding(nn.Module):
    #                  M=2     dim = 256/4 = 64   gamma = 1.0
    def __init__(self, M: int, dim: int, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma
        self.Wr = nn.Linear(M, dim // 2, bias=False)
        nn.init.normal_(self.Wr.weight.data, mean=0, std=self.gamma ** -2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ encode position vector """
        projected = self.Wr(x)
        cosines, sines = torch.cos(projected), torch.sin(projected)
        emb = torch.stack([cosines, sines], 0)
        return emb

class TokenConfidence(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.token = nn.Sequential(
            nn.Linear(dim, 1),
            nn.Sigmoid()
        )

    def forward(self, 
                desc0: torch.Tensor, 
                desc1: torch.Tensor):
        """ get confidence tokens """
        return (
            self.token(desc0.detach()).squeeze(-1),
            self.token(desc1.detach()).squeeze(-1))



class Attention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.has_sdp = hasattr(F, 'scaled_dot_product_attention')

    def forward(self, q, k, v, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.has_sdp:
            args = [x.contiguous() for x in [q, k, v]]
            v = F.scaled_dot_product_attention(*args, attn_mask = mask)
            return v if mask is None else v.nan_to_num()

class SelfBlock(nn.Module):
    def __init__(self,
                 embed_dim: int, 
                 num_heads: int,
                 bias: bool = True)-> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.toqkv = nn.Linear(embed_dim, 3*embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.inner_attn = Attention()
    def forward(self, 
                descriptor: torch.Tensor,
                encoding: torch.Tensor,):
        qkv = self.toqkv(descriptor)
        qkv = qkv.unflatten(-1, (self.num_heads, -1, 3)).transpose(1, 2)
        q, k, v = qkv[..., 0], qkv[..., 1], qkv[..., 2]
        


class CrossBlock(nn.Module):
    def __init__(self, 
                 embed_dim: int, 
                 num_heads: int,
                 bias: bool = True) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads


class MyTransformerLayer(nn.Module):
    def __init__(self, *args, **kwargs):
        super(MyTransformerLayer, self).__init__()
        self.self_attn = SelfBlock(*args, **kwargs)
        self.cross_attn = CrossBlock(*args, **kwargs)
    
    def forward(self,
                desc0    , desc1, 
                encoding0, encoding1):
        desc0 = self.self_attn(desc0, encoding0)
        desc1 = self.self_attn(desc1, encoding1)
        return self.cross_attn(desc0, desc1)

class MatchAssignment(nn.Module):
    def __init__(self,dim: int):
        super().__init__()
        self.dim = dim
        self.matchability = nn.Linear(dim, 1, bias=True)
        self.final_proj = nn.Linear(dim, dim, bias=True)
    def forward(self, desc0: torch.Tensor, desc1: torch.Tensor):
        mdesc0, mdesc1 = self.final_proj(desc0), self.final_proj(desc1)
        _, _, d = mdesc0.shape
        mdesc0, mdesc1 = mdesc0 / d**.25, mdesc1 / d**.25
        sim = torch.einsum('bmd,bnd->bmn', mdesc0, mdesc1)
        



class fullyGlue(nn.Module):

    pruning_keypoint_thresholds = 1024

    def __init__(self) -> None:
        super().__init__()
        input_dim = 3
        self.input_proj = nn.Linear(input_dim, input_dim, bias=True)
        # head_dim = 256 // 4
        output_dims = [3, 16, 32, 64, 128, 256]
        num_layers = len(output_dims)

        self.transformers = nn.ModuleList(
            [MyTransformerLayer(output_dims[i], output_dims[i+1]) for i in range(num_layers)]
        )
        self.log_assignment = nn.ModuleList(
            [MatchAssignment(output_dims[i]) for i in range(num_layers)])
        self.token_confidence = nn.ModuleList(
            [TokenConfidence(output_dims[i]) for i in range(num_layers)])
    def forward(self, data: dict) -> dict:
        data0, data1 = data['image0'], data['image1']
        kpts0, kpts1 = data0['keypoints'], data1['keypoints']
        batch, point0_num, _ = kpts0.shape
        batch, point1_num, _ = kpts1.shape

