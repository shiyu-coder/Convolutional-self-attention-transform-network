import torch
import torch.nn as nn


class NVIDIA_ORIGIN(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.BatchNorm2d(3)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=24, kernel_size=(5, 5), stride=(2, 2))
        self.conv2 = nn.Conv2d(in_channels=24, out_channels=36, kernel_size=(5, 5), stride=(2, 2))
        self.conv3 = nn.Conv2d(in_channels=36, out_channels=48, kernel_size=(5, 5), stride=(2, 2))
        self.conv4 = nn.Conv2d(in_channels=48, out_channels=64, kernel_size=(3, 3), stride=(1, 1))
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1))
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(31680, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 10)
        self.fc4 = nn.Linear(10, 1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, X):
        X = X.squeeze(1)
        X = self.norm(X)
        X = self.conv1(X)
        X = self.relu(X)
        X = self.conv2(X)
        X = self.relu(X)
        X = self.conv3(X)
        X = self.relu(X)
        X = self.conv4(X)
        X = self.relu(X)
        X = self.conv5(X)
        X = self.relu(X)
        X = self.flat(X)
        X = self.fc1(X)
        X = self.relu(X)
        X = self.fc2(X)
        X = self.relu(X)
        X = self.fc3(X)
        X = self.relu(X)
        X = self.fc4(X)
        X = X.unsqueeze(1)
        return X


if __name__ == '__main__':
    X = torch.rand(size=(8, 1, 3, 180, 320))
    net = NVIDIA_ORIGIN()
    X = net(X)
    print(X.shape)
