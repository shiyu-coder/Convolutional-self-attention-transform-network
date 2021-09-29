import torch
import torch.nn as nn
from models.encoder import PCNNEncoder, SACNNEncoder, CNNEncoder
from models.decoder import Decoder
from models.cnn import PCNN, SelfAttentionConv, CNNLayer


class CSATNet(nn.Module):
    def __init__(self, num_hiddens=128, num_heads=4, seq_len=4, cnn_layer1_num=2, cnn_layer2_num=0,
                 enc_layer_num=2, dec_layer_num=2, label_size=1, drop_out=0.1, min_output_size=32):
        super(CSATNet, self).__init__()
        # self.enc = CNNEncoder(num_hiddens, num_heads, seq_len, cnn_layer1_num,
        #                       cnn_layer2_num, enc_layer_num, input_size, drop_out, min_output_size, attention)
        self.enc = CNNEncoder(num_hiddens, num_heads, seq_len, cnn_layer1_num,
                              cnn_layer2_num, enc_layer_num, drop_out, min_output_size)
        self.key_size = self.enc.key_size
        self.dec = Decoder(dec_layer_num, self.key_size, num_hiddens, num_heads, seq_len, drop_out)

        self.dense = nn.Linear(num_hiddens, label_size)

    def forward(self, x):
        x, cross = self.enc(x)
        output = self.dec(x, cross)
        output = self.dense(output)
        return output


class PSACNN(nn.Module):
    def __init__(self, num_hiddens=128, cnn_layer1_num=2, cnn_layer2_num=0,
                 input_size=(88, 200), label_size=1, attention=False):
        super(PSACNN, self).__init__()
        self.cnn = PCNN(num_hiddens, cnn_layer1_num, cnn_layer2_num, input_size, attention)
        self.dense = nn.Sequential(
            nn.Linear(num_hiddens, 128),
            nn.ELU(),
            nn.Linear(128, 32),
            nn.ELU(),
            nn.Linear(32, label_size)
        )

    def forward(self, x):
        batch_num = x.shape[0]
        x = x.reshape(-1, x.shape[2], x.shape[3], x.shape[4])
        x = self.cnn(x)
        x = self.dense(x)
        x = x.reshape(batch_num, -1, x.shape[1])
        return x


class SACNN(nn.Module):
    def __init__(self, cnn_layer1_num=3, cnn_layer2_num=2, label_size=1):
        super(SACNN, self).__init__()
        self.cnn = nn.Sequential()
        in_channels = [3, 24, 36, 48, 64, 80, 128, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256]
        for i in range(0, cnn_layer1_num):
            self.cnn.add_module("layer1-"+str(i), nn.Conv2d(in_channels[i], in_channels[i+1], kernel_size=5, stride=2))
            self.cnn.add_module("actFun-" + str(i), nn.ELU())
        self.cnn.add_module("pool1", nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        for i in range(cnn_layer1_num, cnn_layer1_num + cnn_layer2_num):
            self.cnn.add_module("layer1-" + str(i), SelfAttentionConv(in_channels[i], in_channels[i+1]//2, in_channels[i+1]))
        self.cnn.add_module("pool2", nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        in_channel = in_channels[cnn_layer1_num + cnn_layer2_num]
        self.dense = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channel * 3, 128),
            nn.ELU(),
            nn.Linear(128, 32),
            nn.ELU(),
            nn.Linear(32, label_size)
        )

    def forward(self, x):
        batch_num = x.shape[0]
        x = x.reshape(-1, x.shape[2], x.shape[3], x.shape[4])
        x = self.cnn(x)
        x = self.dense(x)
        x = x.reshape(batch_num, -1, x.shape[1])
        return x


class FSACNN(nn.Module):
    def __init__(self, cnn_layer1_num=3, cnn_layer2_num=2, label_size=1):
        super(FSACNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2),
            nn.ELU(),
        )
        in_channels = [24, 36, 48, 64, 80, 128, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256]
        for i in range(0, cnn_layer1_num):
            self.cnn.add_module("layer1-"+str(i), SelfAttentionConv(in_channels[i], in_channels[i+1]//2, in_channels[i+1]))
            self.cnn.add_module("actFun-" + str(i), nn.ELU())
        self.cnn.add_module("pool1", nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        for i in range(cnn_layer1_num, cnn_layer1_num + cnn_layer2_num):
            self.cnn.add_module("layer1-" + str(i), SelfAttentionConv(in_channels[i], in_channels[i+1]//2, in_channels[i+1]))
        self.cnn.add_module("pool2", nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        in_channel = in_channels[cnn_layer1_num + cnn_layer2_num]
        self.cnn.add_module("layer3-cnn", nn.Conv2d(in_channel, in_channels[cnn_layer1_num + cnn_layer2_num+1],
                                                    kernel_size=3, stride=2))
        in_channel = in_channels[cnn_layer1_num + cnn_layer2_num + 1]

        self.dense = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channel * 60, 128),
            nn.ELU(),
            nn.Linear(128, 32),
            nn.ELU(),
            nn.Linear(32, label_size)
        )

    def forward(self, x):
        batch_num = x.shape[0]
        x = x.reshape(-1, x.shape[2], x.shape[3], x.shape[4])
        x = self.cnn(x)
        x = self.dense(x)
        x = x.reshape(batch_num, -1, x.shape[1])
        return x


class CNN(nn.Module):
    def __init__(self, cnn_layer1_num=3, cnn_layer2_num=2, label_size=1):
        super(CNN, self).__init__()
        self.cnn = CNNLayer(256, cnn_layer1_num, cnn_layer2_num)
        self.dense = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 32),
            nn.ELU(),
            nn.Linear(32, label_size)
        )

    def forward(self, x):
        batch_num = x.shape[0]
        x = x.reshape(-1, x.shape[2], x.shape[3], x.shape[4])
        x = self.cnn(x)
        x = self.dense(x)
        x = x.reshape(batch_num, -1, x.shape[1])
        return x


if __name__ == '__main__':
    X = torch.rand(size=(8, 16, 3, 88, 200))
    net = CSATNet(128, 4, 16, 3, 2, 3, 3, 1, 0.1, 32)
    # net = SACNN(3, 2, 1)
    X = net(X)
    print(X.shape)

