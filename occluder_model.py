import torch
import torch.nn as nn
import torch.nn.functional as F

class OccluderNet(nn.Module):
    def __init__(self, num_classes=3):
        super(OccluderNet, self).__init__()
        # Input: 3x64x64
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        
        # 64x64 -> 32x32 -> 16x16 -> 8x8
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) # 32x32
        x = self.pool(F.relu(self.conv2(x))) # 16x16
        x = self.pool(F.relu(self.conv3(x))) # 8x8
        
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
