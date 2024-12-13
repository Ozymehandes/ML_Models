import torch
import torch.nn as nn
import numpy as np
from torchvision.models import resnet18
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import Dataset

class Encoder(nn.Module):
    def __init__(self, D=128, device='cuda'):
        super(Encoder, self).__init__()
        self.resnet = resnet18(pretrained=False).to(device)
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=1)
        self.resnet.maxpool = nn.Identity()
        self.resnet.fc = nn.Linear(512, 512)
        self.fc = nn.Sequential(nn.BatchNorm1d(512), nn.ReLU(inplace=True), nn.Linear(512, D))

    def forward(self, x):
        x = self.resnet(x)
        x = self.fc(x)
        return x

    def encode(self, x):
        return self.forward(x)


class Projector(nn.Module):
    def __init__(self, D, proj_dim=512):
        super(Projector, self).__init__()
        self.model = nn.Sequential(nn.Linear(D, proj_dim),
                                   nn.BatchNorm1d(proj_dim),
                                   nn.ReLU(inplace=True),
                                   nn.Linear(proj_dim, proj_dim),
                                   nn.BatchNorm1d(proj_dim),
                                   nn.ReLU(inplace=True),
                                   nn.Linear(proj_dim, proj_dim)
                                   )

    def forward(self, x):
        return self.model(x)


class VicReg(nn.Module):
    def __init__(self, D=128, proj_dim=512, device='cuda'):
        super(VicReg, self).__init__()
        self.encoder = Encoder(D, device)
        self.projector = Projector(D, proj_dim)
        self.device = device

    def forward(self, x1, x2):
        z1 = self.encoder(x1)
        z2 = self.encoder(x2)
        z1 = self.projector(z1)
        z2 = self.projector(z2)
        return z1, z2

    def encode(self, x):
        return self.encoder.encode(x)

    def proj(self, x):
        return self.projector(x)

class LinearProbe(nn.Module):
    def __init__(self, encoder, D=128, num_classes=10):
        super().__init__()
        self.encoder = encoder
        self.fc = nn.Linear(D, num_classes)

    def forward(self, x):
        with torch.no_grad():
            features = self.encoder.encode(x)
        return self.fc(features)
    
class NeighborDataset(Dataset):
    def __init__(self, original_dataset, representations, n_neighbors=4):
        self.original_dataset = original_dataset
        self.representations = representations
        
        self.nn = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean')
        self.nn.fit(self.representations)
        
        self.neighbors = self.nn.kneighbors(self.representations, return_distance=False)
    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        img1, label = self.original_dataset[idx]
        neighbor_idx = np.random.choice(self.neighbors[idx][1:])  # Exclude the first neighbor (original image)
        img2, _ = self.original_dataset[neighbor_idx]
        return img1, img2, label