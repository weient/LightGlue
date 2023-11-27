from pathlib import Path
from lightglue import LightGlue, SuperPoint, DISK
from lightglue.utils import load_image, rbd
from lightglue import viz2d
import torch
from torch import nn
import numpy as np
from torchrl.modules import MLP
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard.writer import SummaryWriter
import torch.nn.functional as F
from datetime import datetime
import itertools
import os


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
    exp = False
    if exp:
        scores_no = sim.new_full((b, m+1, n+1), 0)
        c_exp = F.sigmoid(z0) + F.sigmoid(z1).transpose(1, 2)
        s0_exp = F.softmax(sim, 2)
        s1_exp = F.softmax(sim.transpose(-1, -2).contiguous(), 2).transpose(-1, -2)
        scores_no[:, :m, :n] = (s0_exp + s1_exp + c_exp)
        scores_no[:, :-1, -1] = F.sigmoid(-z0.squeeze(-1))
        scores_no[:, -1, :-1] = F.sigmoid(-z1.squeeze(-1))
    else:
        scores_no = F.sigmoid(scores.clone())
    return scores, scores_no

class MatchAssignment(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.matchability = nn.Linear(dim, 1, bias=True)
        self.final_proj = nn.Linear(dim, dim, bias=True)

    def forward(self, desc0: torch.Tensor, desc1: torch.Tensor):
        """ build assignment matrix from descriptors """
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

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        # ,norm_class=nn.BatchNorm1d
        in_features = 256
        hidden_sizes = [128, 64, 32, 16]
        output_size = 3
        activation_class=nn.ReLU
        
        layers = []
        prev_size = in_features

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(activation_class())
            layers.append(nn.BatchNorm1d(hidden_size))
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, output_size))

        self.mlp = nn.Sequential(*layers)
    
    
    def forward(self, inputs: torch.Tensor):
        codes = self.mlp(inputs)
        return codes

class Decoder(nn.Module):
    def __init__(self, in_features=3 , out_features=256, hidden_sizes= [16, 32, 64, 128], activation_class=nn.ReLU):
        super(Decoder, self).__init__()

        layers = []
        prev_size = in_features


        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(activation_class())
            layers.append(nn.BatchNorm1d(hidden_size))
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, out_features))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class MLP_module(nn.Module):
    def __init__(self):
        super(MLP_module, self).__init__()
        # Encoder
        self.encoder = Encoder()
        # Decoder
        self.decoder = Decoder()
        
        dim = 256
        n_layers = 1
        self.log_assignment = nn.ModuleList(
            [MatchAssignment(dim) for _ in range(n_layers)])
    def forward(self, desc0: torch.Tensor, desc1: torch.Tensor):
            desc0_mlp = self.encoder(desc0)
            desc1_mlp = self.encoder(desc1)
            # scores_mlp, _, scores_no = self.log_assignment[0](desc0_mlp, desc1_mlp)
            desc0_back = self.decoder(desc0_mlp)
            desc1_back = self.decoder(desc1_mlp)
            scores_mlp, _, scores_no = self.log_assignment[0](desc0_back, desc1_back)
            return scores_no, desc0_back, desc1_back


class MLPDataset(Dataset):
    def __init__(self):
        self.data_root = '/mnt/home_6T/public/koki/scannet_dataset/scans/scene0000_00/rgb/color/'


def image_name_to_txt():
    folder_path = '/mnt/home_6T/public/koki/scannet_dataset/scans/scene0000_00/rgb/color/'
    files = os.listdir(folder_path)
    jpg_files = [file for file in files if file.endswith('.jpg')]

    jpg_files.sort()
    output_txt_path = 'scene0000_00.txt'

    with open(output_txt_path,'w') as f:
        for file_name in jpg_files:
            f.write(file_name + '\n')
    print('done')





if __name__ == '__main__':
    # image_name_to_txt()
    mlp = MLP_module()
    print(mlp)





'''
export CUDA_VISIBLE_DEVICES=0
/mnt/home_6T/public/koki/scannet_dataset/scans/scene0000_00/
'''