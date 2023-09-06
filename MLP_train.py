from pathlib import Path
from lightglue import LightGlue, SuperPoint, DISK
from lightglue.utils import load_image, rbd
from lightglue import viz2d
import torch
from torch import nn
import numpy as np
from torchrl.modules import MLP
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
import torch.nn.functional as F
from datetime import datetime
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
    return scores

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
        scores = sigmoid_log_double_softmax(sim, z0, z1)
        return scores, sim

    def get_matchability(self, desc: torch.Tensor):
        return torch.sigmoid(self.matchability(desc)).squeeze(-1)
 
class MLP_module(nn.Module):
    def __init__(self):
        super().__init__()
        dim = 3
        n_layers = 1
        self.MLP = MLP(in_features=256, out_features=3, num_cells=[128, 64, 32, 16])
        self.log_assignment = nn.ModuleList(
            [MatchAssignment(dim) for _ in range(n_layers)])
    def forward(self, desc0: torch.Tensor, desc1: torch.Tensor):

        desc0_mlp = self.MLP(desc0)
        desc1_mlp = self.MLP(desc1)
        scores_mlp, _ = self.log_assignment[0](desc0_mlp, desc1_mlp)
        return scores_mlp

class MLPDataset(Dataset):

    # data loading
    def __init__(self):
        self.data_path = '/mnt/home_6T/public/weien/MLP_data.pt'
        self.data = torch.load(self.data_path)
    # working for indexing
    def __getitem__(self, index):
        return self.data[index][0], self.data[index][1], self.data[index][2]
    # return the length of our dataset
    def __len__(self):
        return len(self.data)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
train_set = MLPDataset()
val_set = MLPDataset()
train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
val_loader = DataLoader(train_set, batch_size=1, shuffle=True)
model = MLP_module().to(device)
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.00001, momentum=0.9)


def train_one_epoch(epoch_index, tb_writer):
    running_loss = 0.
    last_loss = 0.

    for i, data in enumerate(train_loader):
        # Every data instance is an input + label pair
        input0, input1, label = data

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(input0, input1)
        # Compute the loss and its gradients
        loss = loss_fn(outputs, label)
        #with torch.autograd.detect_anomaly():
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        #print(loss.item())
        if i % 10 == 9:
            last_loss = running_loss / 10 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(train_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss


# Initializing in a separate cell so we can easily add more epochs to the same run
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter('/mnt/home_6T/public/weien/MLP_checkpoint/runs/fashion_trainer_{}'.format(timestamp))
epoch_number = 0

EPOCHS = 500

best_vloss = 1_000_000.

for epoch in range(EPOCHS):
    print('EPOCH {}:'.format(epoch_number + 1))

    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    avg_loss = train_one_epoch(epoch_number, writer)


    running_vloss = 0.0
    # Set the model to evaluation mode, disabling dropout and using population
    # statistics for batch normalization.
    model.eval()

    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        for i, vdata in enumerate(val_loader):
            vin0, vin1, vlabels = vdata
            voutputs = model(vin0, vin1)
            vloss = loss_fn(voutputs, vlabels)
            running_vloss += vloss

    avg_vloss = running_vloss / (i + 1)
    print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

    # Log the running loss averaged per batch
    # for both training and validation
    writer.add_scalars('Training vs. Validation Loss',
                    { 'Training' : avg_loss, 'Validation' : avg_vloss },
                    epoch_number + 1)
    writer.flush()

    # Track best performance, and save the model's state
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        model_path = '/mnt/home_6T/public/weien/MLP_checkpoint/model_{}_{}'.format(timestamp, epoch_number)
        torch.save(model.state_dict(), model_path)

    epoch_number += 1