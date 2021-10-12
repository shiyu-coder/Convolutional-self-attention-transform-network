import torch
import torch.nn as nn


class NVIDIA_ORIGIN(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.BatchNorm2d(3)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=24, kernel_size=(5, 5), stride=(2, 2))
        self.conv2 = nn.Conv2d(in_channels=24, out_channels=36, kernel_size=(5, 5), stride=(2, 2))
        self.conv3 = nn.Conv2d(in_channels=36, out_channels=48, kernel_size=(5, 5), stride=(2, 2))
        self.conv4 = nn.Conv2d(in_channels=48, out_channels=64, kernel_size=(3, 3), stride=(2, 2))
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1))
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(7168, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 10)
        self.fc4 = nn.Linear(10, 1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        seq_len = x.shape[1]
        x = x.reshape(-1, x.shape[2], x.shape[3], x.shape[4])
        x = self.norm(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.conv5(x)
        x = self.relu(x)
        x = self.flat(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        x = x.reshape(-1, seq_len, x.shape[1])
        return x


class CNN_LSTM(nn.Module):

    def __init__(self):
        super(CNN_LSTM, self).__init__()
        self.norm = nn.BatchNorm2d(3)
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=24, kernel_size=(5, 5), stride=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(in_channels=24, out_channels=36, kernel_size=(5, 5), stride=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(in_channels=36, out_channels=48, kernel_size=(5, 5), stride=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(in_channels=48, out_channels=64, kernel_size=(3, 3), stride=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(7168, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
        )
        self.lstm = nn.LSTM(128, 1)

    def forward(self, x):
        seq_len = x.shape[1]
        x = x.reshape(-1, x.shape[2], x.shape[3], x.shape[4])
        x = self.norm(x)
        x = self.cnn(x)
        x = x.reshape(-1, seq_len, x.shape[1])
        x, _ = self.lstm(x)
        return x


class TDCNN_LSTM(nn.Module):

    def __init__(self):
        super(TDCNN_LSTM, self).__init__()
        self.norm = nn.BatchNorm3d(3)
        self.cnn = nn.Sequential(
            nn.Conv3d(3, 4, kernel_size=(3, 12, 12), stride=(1, 2, 2)),
            nn.ReLU(),
            nn.Conv3d(4, 9, kernel_size=(2, 5, 5), stride=(1, 2, 2)),
            nn.ReLU(),
            nn.Conv3d(9, 12, kernel_size=(2, 5, 5), stride=(1, 2, 2)),
            nn.ReLU(),
            nn.Conv3d(12, 16, kernel_size=(2, 5, 5)),
            nn.ReLU(),
            nn.Conv3d(16, 20, kernel_size=(2, 5, 5)),
            nn.ReLU(),
        )
        self.dense = nn.Sequential(
            nn.Linear(6160, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, 128)
        )
        self.lstm = nn.LSTM(128, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4)
        x = self.norm(x)
        x = self.cnn(x)
        x = x.view(x.shape[0], x.shape[2], -1)
        x = self.dense(x)
        x, _ = self.lstm(x)
        return x


if __name__ == '__main__':
    X = torch.rand(size=(2, 12, 3, 180, 320))
    net = TDCNN_LSTM()
    X = net(X)
    print(X.shape)
