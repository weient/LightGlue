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

        return desc0_back, desc1_back

class MLPDataset(Dataset):
    # data loading
    def __init__(self):
        
        self.data_root = "/mnt/home_6T/public/koki/scannet_dataset/color_images/"
        self.img_pairs = "/mnt/home_6T/public/koki/scannet_dataset/image_pairs.txt"
        with open(self.img_pairs) as f:
            self.line = f.read().splitlines()
        
    # working for indexing
    def __getitem__(self, index):
        # /mnt/home_6T/public/koki/scannet_dataset/color_images/scene0000_00/color/
        scene = self.line[index].split(' ')[0]
        img0_id =  self.line[index].split(' ')[1]
        img1_id =  self.line[index].split(' ')[2]
        img0 = load_image(os.path.join(self.data_root, 'scene'+scene+'_00/'+'color/'+img0_id+'.jpg'),resize=648)
        img1 = load_image(os.path.join(self.data_root, 'scene'+scene+'_00/'+'color/'+img1_id+'.jpg'),resize=648)
        return img0, img1
    # return the length of our dataset
    def __len__(self):
        return len(self.line)



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
model = MLP_module().to(device)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)  # load the extractor
matcher = LightGlue(features='superpoint').eval().to(device)

best_vloss = 1_000_000.
seed = 100
torch.manual_seed(seed)
dataset = MLPDataset()
train_set, val_set = random_split(dataset, [0.75, 0.25])
train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
val_loader = DataLoader(val_set, batch_size=8, shuffle=True)


def train_one_epoch(epoch_index, tb_writer):
    running_loss = 0.
    last_loss = 0.

    for i, data in enumerate(train_loader):
        # Every data instance is an input + label pair
        with torch.no_grad():
            img0, img1 = data
            feats0 = extractor.extract(img0.clone().detach().to(device))
            feats1 = extractor.extract(img1.clone().detach().to(device))
            matches01 = matcher({'image0': feats0, 'image1': feats1})
            label = matches01['label']
        #print("label : ", label.shape)
        # Make predictions for this batch
        feats0_des = feats0['descriptors'].clone().detach()
        feats1_des = feats1['descriptors'].clone().detach()
        d0_back, d1_back = model(feats0_des, feats1_des)
        feats0['descriptors'] = d0_back
        feats1['descriptors'] = d1_back
        with torch.no_grad():
            matches_mlp = matcher({'image0': feats0, 'image1': feats1})
            outputs = matches_mlp['label']

        # Zero your gradients for every batch!
        optimizer.zero_grad()        
        
        # Compute the loss and its gradients
        loss = loss_fn(outputs, label)
        loss_d0 = loss_fn(d0_back, feats0_des)
        loss_d1 = loss_fn(d1_back, feats1_des)
        loss_total  = 5000*loss + loss_d0 + loss_d1
        
        #with torch.autograd.detect_anomaly():
        loss_total.backward()

        # gradient clipping
        #torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=5, norm_type=2)
        
        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss_total.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            #tb_x = epoch_index * len(train_loader) + i + 1
            #tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.
    return last_loss

# Iterate through the dataset in batches
for batch in train_loader:
    # The batch variable contains a tuple of (batch_img0, batch_img1)
    batch_img0, batch_img1 = batch
    print(batch_img0.shape)
    with torch.no_grad():
        feats0 = extractor.extract(batch_img0.clone().detach().to(device))
        print(feats0['descriptors'].shape)
    break
   
# 648 484

'''
export CUDA_VISIBLE_DEVICES=0
/mnt/home_6T/public/koki/scannet_dataset/scans/scene0000_00/
'''