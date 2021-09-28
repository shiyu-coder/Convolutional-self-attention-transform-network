import torch.nn as nn
import torch
import math
from attention import ParallelInception, MultiHeadAttention
from utils.positionalEncoding import PositionalEncoding
from utils.addNorm import AddNorm
from cnn import PCNN


class SingleEncoder(nn.Module):

    def __init__(self, layer_num=2, num_hiddens=128, num_heads=4, seq_len=4, drop_out=0.1, min_output_size=32):
        super(SingleEncoder, self).__init__()
        self.layer_num = layer_num

        self.mha = []
        self.an = []
        self.li = []

        seq_size = num_hiddens
        for i in range(layer_num):
            self.mha.append(MultiHeadAttention(seq_size, seq_size, seq_size, seq_size, num_heads, drop_out))
            self.an.append(AddNorm((seq_len, seq_size), drop_out))
            self.li.append(nn.Linear(seq_size, max(min_output_size, seq_size//2)))
            seq_size = max(min_output_size, seq_size//2)

        self.actFun = nn.ELU()
        self.key_size = seq_size

    def forward(self, x):
        for i in range(self.layer_num):
            x = self.an[i](x, self.mha[i](x, x, x))
            x = self.actFun(self.li[i](x))
        return x


class CNNEncoder(nn.Module):

    def __init__(self, num_hiddens=128, num_heads=4, seq_len=4, cnn_layer1_num=2, cnn_layer2_num=0,
                 enc_layer_num=2, input_size=(88, 200), drop_out=0.1, min_output_size=32):
        super(CNNEncoder, self).__init__()
        self.num_hiddens = num_hiddens

        self.cnn = PCNN(num_hiddens, cnn_layer1_num, cnn_layer2_num, input_size)
        self.pe = PositionalEncoding(num_hiddens, 0)
        self.enc = SingleEncoder(enc_layer_num, num_hiddens, num_heads, seq_len, drop_out, min_output_size)

        self.key_size = self.enc.key_size

    def forward(self, x):
        batch_num = x.shape[0]
        x = x.reshape(-1, x.shape[2], x.shape[3], x.shape[4])
        x = self.cnn(x)
        x = x.reshape(batch_num, -1, x.shape[1])
        x = self.pe(x * math.sqrt(self.num_hiddens))
        y = self.enc(x)

        return x, y


if __name__ == '__main__':
    X = torch.rand(size=(8, 4, 3, 88, 200))
    net = CNNEncoder(num_hiddens=128, num_heads=4, seq_len=4, enc_layer_num=2, cnn_layer1_num=3, cnn_layer2_num=1)
    X = net(X)
    print(X[0].shape)
    print(X[1].shape)
