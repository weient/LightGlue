from pathlib import Path
from lightglue import LightGlue, SuperPoint, DISK
from lightglue.utils import load_image, rbd
from lightglue import viz2d
import torch
import numpy as np
import os
import itertools
from torchrl.modules import MLP
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split

torch.set_grad_enabled(False)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 'mps', 'cpu'
extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)  # load the extractor

class MLP_module(nn.Module):
    def __init__(self):
        super().__init__()
        self.MLP = MLP(in_features=256, out_features=3, num_cells=[128, 64, 32, 16])
        self.MLP_de = MLP(in_features=3, out_features=256, num_cells=[16, 32, 64, 128])
    def forward(self, desc0: torch.Tensor, desc1: torch.Tensor):
        desc0_mlp = self.MLP(desc0)
        desc1_mlp = self.MLP(desc1)
        desc0_back = self.MLP_de(desc0_mlp)
        desc1_back = self.MLP_de(desc1_mlp)
        return desc0_mlp, desc1_mlp

model = MLP_module().to(device)
checkpoint = torch.load("/mnt/home_6T/public/koki/MLP_data/MLP_checkpoint/model_20231212_171923_7")

model.load_state_dict(checkpoint)
model.to(device)


class Embedding_Dataset(Dataset):
    def __init__(self,image_path):
        self.files = os.listdir(image_path)
        self.image_path = image_path
    def __getitem__(self, index):
        image = load_image(self.image_path+f'{index}.jpg')
        return image
    def __len__(self):
        return len(self.files)

index = 1
for index in range(2,31):
    image_path = '/home/koki/LightGlue/scannet_dataset/scannet_image' + f'/scene{index:04}_00/color/'
    dataset = Embedding_Dataset(image_path)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)

    from tqdm import tqdm

    data = []
    for image in tqdm(dataloader, desc="Loading data"):
        feats = extractor.extract(image.to(device))    
        feats['descriptors'] , _ = model(feats['descriptors'],feats['descriptors'])
        split_tensors = torch.split(feats['descriptors'], 1, dim=0)
        split_tensors_list = list(split_tensors)
        data.extend(split_tensors_list)
    torch.save(data, f'/home/koki/LightGlue/scannet_dataset/scannet_embed/scene{index:04}_00.pth')

# files = os.listdir(image_path)
# print(len(files))
# file_length = len(files)
# data = []
# for i in tqdm(range(file_length), desc='Processing images'):
#     image = load_image(image_path+f'{i}.jpg')
#     feats = extractor.extract(image.to(device))
#     feats['descriptors'] , _ = model(feats['descriptors'],feats['descriptors'])
#     data.append(feats)

# torch.save(data, '/home/koki/LightGlue/scannet_dataset/scannet_embed/scene0000_00.pth')