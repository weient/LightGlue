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
        emb = torch.stack([cosines, sines], 0).unsqueeze(-3)
        return emb.repeat_interleave(2, dim=-1)

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

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x = x.unflatten(-1, (-1, 2)) # [1, 4, 2048, 32, 2]
    x1, x2 = x.unbind(dim=-1) # [1, 4, 2048, 32]
    # print("in rotate_half x1 shape: ",x1.shape) # [1, 4, 2048, 32]
    # print("in rotate_half x2 shape: ",x2.shape) # [1, 4, 2048, 32]
    # print("in rotate_half stack shape:",torch.stack((-x2, x1), dim=-1).shape)
    output = torch.stack((-x2, x1), dim=-1).flatten(start_dim=-2)
    return output

def apply_cached_rotary_emb(encoding: torch.Tensor, qk: torch.Tensor) -> torch.Tensor:
    # encoding[0] [1, 1, 2048, 2]
    # qk          [1, 2, 2048, 2]
    # rotate qk   [1, 2, 2048, 2]
    # print("in apply_cached_rotary_emb encoding[0] shape ",encoding[0].shape)
    # print("in apply_cached_rotary_emb qk shape          ",qk.shape)
    # print("in apply_cached_rotary_emb rotate qk shape   ",rotate_half(qk).shape)
    return (qk * encoding[0]) + (rotate_half(qk) * encoding[1])

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
        self.ffn = nn.Sequential(
            nn.Linear(2*embed_dim, 2*embed_dim),
            nn.LayerNorm(2*embed_dim, elementwise_affine=True),
            nn.GELU(),
            nn.Linear(2*embed_dim, embed_dim)
        )

    def forward(self, 
                descriptor: torch.Tensor,
                encoding: torch.Tensor,):
        qkv = self.toqkv(descriptor)
        qkv = qkv.unflatten(-1, (self.num_heads, -1, 3)).transpose(1, 2)
        q, k, v = qkv[..., 0], qkv[..., 1], qkv[..., 2]
        q = apply_cached_rotary_emb(encoding, q)
        k = apply_cached_rotary_emb(encoding, k)
        
        # [1, 2, 2048, 2]
        context = self.inner_attn(q, k, v)
        # print('context shape: ',context.shape)
        message = self.out_proj(
            context.transpose(1, 2).flatten(start_dim=-2))
        # print('message shape: ',message.shape)
        out = descriptor + self.ffn(torch.cat([descriptor, message], -1))
        # print('out shape: ',out.shape)
        return out

class CrossBlock(nn.Module):
    def __init__(self, 
                 embed_dim: int, 
                 num_heads: int,
                 bias: bool = True) -> None:
        super().__init__()
        self.num_heads = num_heads
        dim_head = embed_dim // num_heads
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * num_heads

        self.to_qk = nn.Linear(embed_dim, inner_dim, bias=bias)
        self.to_v = nn.Linear(embed_dim, inner_dim, bias=bias)
        self.to_out = nn.Linear(inner_dim, embed_dim, bias=bias)

        self.ffn = nn.Sequential(
            nn.Linear(2*embed_dim, 2*embed_dim),
            nn.LayerNorm(2*embed_dim, elementwise_affine=True),
            nn.GELU(),
            nn.Linear(2*embed_dim, embed_dim)
        )
        self.flash = None
    def map(self, func: Callable, x0: torch.Tensor, x1: torch.Tensor):
        return func(x0), func(x1)
    def forward(self, x0: torch.Tensor, x1: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> List[torch.Tensor]:
        qk0, qk1 = self.map(self.to_qk, x0, x1)
        v0, v1 = self.map(self.to_v, x0, x1)
        qk0, qk1, v0, v1 = map(
            lambda t: t.unflatten(-1, (self.num_heads, -1)).transpose(1, 2),
            (qk0, qk1, v0, v1))

        ## cross attention
        qk0, qk1 = qk0 * self.scale**0.5, qk1 * self.scale**0.5
        sim = torch.einsum('bhid, bhjd -> bhij', qk0, qk1)
        if mask is not None:
            sim = sim.masked_fill(~mask, -float('inf'))
        attn01 = F.softmax(sim, dim=-1)
        attn10 = F.softmax(sim.transpose(-2, -1).contiguous(), dim=-1)
        m0 = torch.einsum('bhij, bhjd -> bhid', attn01, v1)
        m1 = torch.einsum('bhji, bhjd -> bhid', attn10.transpose(-2, -1), v0)
        if mask is not None:
            m0, m1 = m0.nan_to_num(), m1.nan_to_num()
        m0, m1 = self.map(lambda t: t.transpose(1, 2).flatten(start_dim=-2), m0, m1)
        m0, m1 = self.map(self.to_out, m0, m1)

        x0 = x0 + self.ffn(torch.cat([x0, m0], -1))
        x1 = x1 + self.ffn(torch.cat([x1, m1], -1))

        return x0,x1

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
        desc0, desc1 = self.cross_attn(desc0, desc1)
        concatenated_tensor = torch.cat((desc0, desc1), dim=2)
        return concatenated_tensor , concatenated_tensor



def sigmoid_log_double_softmax(
        sim: torch.Tensor, z0: torch.Tensor, z1: torch.Tensor) -> torch.Tensor:
    """ create the log assignment matrix from logits and similarity"""
    b, m, n = sim.shape
    certainties = F.logsigmoid(z0) + F.logsigmoid(z1).transpose(1, 2)
    scores0 = F.log_softmax(sim, 2)
    scores1 = F.log_softmax(
        sim.transpose(-1, -2).contiguous(), 2).transpose(-1, -2)

    scores = sim.new_full((b, m+1, n+1), 0)
    scores[:, :m, :n] = (scores0 + scores1 + certainties)
    scores[:, :-1, -1] = F.logsigmoid(-z0.squeeze(-1))
    scores[:, -1, :-1] = F.logsigmoid(-z1.squeeze(-1))
    scores_no = F.sigmoid(scores.clone())
    return scores, scores_no

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
        z0 = self.matchability(desc0)
        z1 = self.matchability(desc1)
        scores, scores_no = sigmoid_log_double_softmax(sim, z0, z1)
        return scores, sim, scores_no

    def get_matchability(self, desc: torch.Tensor):
        return torch.sigmoid(self.matchability(desc)).squeeze(-1)

def filter_matches(scores: torch.Tensor, th: float):
    """ obtain matches from a log assignment matrix [Bx M+1 x N+1]"""
    max0, max1 = scores[:, :-1, :-1].max(2), scores[:, :-1, :-1].max(1)
    m0, m1 = max0.indices, max1.indices
    
    indices0 = torch.arange(m0.shape[1], device=m0.device)[None]
    indices1 = torch.arange(m1.shape[1], device=m1.device)[None]
    
    mutual0 = indices0 == m1.gather(1, m0)
    mutual1 = indices1 == m0.gather(1, m1)
    
    max0_exp = max0.values.exp()
    zero = max0_exp.new_tensor(0)

    mscores0 = torch.where(mutual0, max0_exp, zero)
    mscores1 = torch.where(mutual1, mscores0.gather(1, m1), zero)
    
    valid0 = mutual0 & (mscores0 > th)
    valid1 = mutual1 & valid0.gather(1, m1)
    
    m0 = torch.where(valid0, m0, -1)
    m1 = torch.where(valid1, m1, -1)
    #np.savetxt('mutual.txt', mutual0.cpu().numpy())
    return m0, m1, mscores0, mscores1





class myGlue(nn.Module):

    pruning_keypoint_thresholds = 1024

    def __init__(self) -> None:
        super().__init__()
        input_dim = 3
        self.input_proj = nn.Linear(input_dim, 4, bias=True)
        output_dims = [4, 8, 16, 32, 64, 128, 256]
        num_layers = len(output_dims)
        self.num_layers = num_layers-1
        self.num_heads = 4

        self.transformers = nn.ModuleList(
            [MyTransformerLayer(output_dims[i], 2) for i in range(num_layers)])
        self.log_assignment = nn.ModuleList(
            [MatchAssignment(output_dims[i+1]) for i in range(num_layers-1)])
        self.token_confidence = nn.ModuleList(
            [TokenConfidence(output_dims[i+1]) for i in range(num_layers-1)])
        # print('hhh ',output_dims[0]//2)
        self.poes_encoding = nn.ModuleList([LearnablePositionalEncoding(2, output_dims[i]//2) for i in range(num_layers)])
        
        self.register_buffer('confidence_thresholds', 
            torch.Tensor([self.confidence_threshold(i) for i in range(num_layers)]))

    def forward(self, data: dict) -> dict:
        with torch.autocast(enabled=True, device_type='cuda'):
            return self._forward(data)

    def _forward(self, data: dict) -> dict:
        data0, data1 = data['image0'], data['image1']
        kpts0, kpts1 = data0['keypoints'], data1['keypoints']
        batch, kpts0_num, _ = kpts0.shape
        batch, kpts1_num, _ = kpts1.shape

        desc0 = data0['descriptors'].detach().contiguous()
        desc1 = data1['descriptors'].detach().contiguous()
        device = kpts0.device

        if torch.is_autocast_enabled():
            desc0 = desc0.half()
            desc1 = desc1.half()

        desc0 = self.input_proj(desc0)
        desc1 = self.input_proj(desc1)

        pruning_th_num = 1024
        self.depth_confidence = 0.95
        self.width_confidence = 0.99
        filter_threshold = 0.1

        do_early_stop = self.depth_confidence > 0
        do_point_pruning = self.width_confidence > 0

        if do_point_pruning:
            ind0 = torch.arange(0, kpts0_num, device=device)[None]
            ind1 = torch.arange(0, kpts1_num, device=device)[None]
            prune0 = torch.ones_like(ind0)
            prune1 = torch.ones_like(ind1)
        
        token0, token1 = None, None
        for i in range(self.num_layers):
            encoding0, encoding1 = self.poes_encoding[i](kpts0), self.poes_encoding[i](kpts1)
            # print('descriptor0 shape',desc0.shape)
            # print('encode0: ',encoding0.shape)
            desc0, desc1 = self.transformers[i](desc0, desc1,
                                                encoding0,encoding1)
            # print('desc0 shape: ',desc0.shape)
            if i == self.num_layers - 1:
                continue  # no early stopping or adaptive width at last layer
            
            if do_early_stop:
                token0, token1 = self.token_confidence[i](desc0, desc1)
                if self.check_if_stop(token0[..., :kpts0_num, :], token1[..., :kpts1_num, :]\
                                      , i, kpts0_num+kpts1_num):
                    break
            
            if do_point_pruning and desc0.shape[-2] > pruning_th_num:
                scores0 = self.log_assignment[i].get_matchability(desc0)
                prunemask0 = self.get_pruning_mask(token0, scores0, i)
                keep0 = torch.where(prunemask0)[1]
                ind0      =      ind0.index_select(1, keep0)
                desc0     =     desc0.index_select(1, keep0)
                encoding0 = encoding0.index_select(-2,keep0)
                prune0[:, ind0] += 1
            
            if do_point_pruning and desc1.shape[-2] > pruning_th_num:
                scores1 = self.log_assignment[i].get_matchability(desc1)
                prunemask1 = self.get_pruning_mask(token1, scores1, i)
                keep1 = torch.where(prunemask1)[1]
                ind1      =      ind1.index_select(1,  keep1)
                desc1     =     desc1.index_select(1,  keep1)
                encoding1 = encoding1.index_select(-2, keep1)
                prune1[:, ind1] += 1
        
        desc0, desc1 = desc0[..., :kpts0_num, :], desc1[..., :kpts1_num, :]
        scores, _, _ = self.log_assignment[i](desc0, desc1)
        # print('desc0_all shape: ',desc0.shape)
        # print('desc1_all shape: ',desc1.shape)
        m0, m1, mscores0, mscores1 = filter_matches(scores, filter_threshold)
        # print('m0 shape: ',m0[0,:5])
        # print('m1 shape: ',m1[0,:5])
        # print('mscore0 shape: ',mscores0[0,:5])
        # print('mscore1 shape: ',mscores1[0,:5])
        matches, mscores = [], []
        for k in range(batch):
            valid = m0[k] > -1
            m_indices_0 = torch.where(valid)[0]
            m_indices_1 = m0[k][valid]
            if do_point_pruning:
                m_indices_0 = ind0[k, m_indices_0]
                m_indices_1 = ind1[k, m_indices_1]
            matches.append(torch.stack([m_indices_0, m_indices_1], -1))
            mscores.append(mscores0[k][valid])
        # TODO: Remove when hloc switches to the compact format.
        if do_point_pruning:
            m0_ = torch.full((batch, kpts0_num), -1, device=m0.device, dtype=m0.dtype)
            m1_ = torch.full((batch, kpts1_num), -1, device=m1.device, dtype=m1.dtype)
            m0_[:, ind0] = torch.where(
                m0 == -1, -1, ind1.gather(1, m0.clamp(min=0)))
            m1_[:, ind1] = torch.where(
                m1 == -1, -1, ind0.gather(1, m1.clamp(min=0)))
            mscores0_ = torch.zeros((batch, kpts0_num), device=mscores0.device)
            mscores1_ = torch.zeros((batch, kpts1_num), device=mscores1.device)
            mscores0_[:, ind0] = mscores0
            mscores1_[:, ind1] = mscores1
            m0, m1, mscores0, mscores1 = m0_, m1_, mscores0_, mscores1_
        else:
            prune0 = torch.ones_like(mscores0) * self.num_layers
            prune1 = torch.ones_like(mscores1) * self.num_layers
        
        pred = {
            'matches0': m0,
            'matches1': m1,
            'matching_scores0': mscores0,
            'matching_scores1': mscores1,
            'stop': i+1,
            'matches': matches,
            'scores': mscores,
            'prune0': prune0,
            'prune1': prune1
        }
        return pred
    
    def get_pruning_mask(self, confidences: torch.Tensor, scores: torch.Tensor,
                         layer_index: int) -> torch.Tensor:
        """ mask points which should be removed """
        keep = scores > (1 - self.width_confidence)
        if confidences is not None:  # Low-confidence points are never pruned.
            keep |= confidences <= self.confidence_thresholds[layer_index]
        return keep
    
    def confidence_threshold(self, layer_index: int) -> float:
        """ scaled confidence threshold """
        threshold = 0.8 + 0.1 * np.exp(-4.0 * layer_index / self.num_layers)
        return np.clip(threshold, 0, 1)

    def check_if_stop(self,
                      confidences0: torch.Tensor, confidences1: torch.Tensor,
                      layer_index: int          , num_points: int            ) -> torch.Tensor:
        """ evaluate stopping condition"""
        confidences = torch.cat([confidences0, confidences1], -1)
        threshold = self.confidence_thresholds[layer_index]
        ratio_confident = 1.0 - (confidences < threshold).float().sum() / num_points
        return ratio_confident > self.depth_confidence
