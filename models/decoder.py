import torch.nn as nn
from attention import MaskedMultiHeadAttention
from utils.addNorm import AddNorm


class DecoderLayer(nn.Module):
    def __init__(self, key_size, num_hiddens, num_heads, norm_shape, drop_out):
        super(DecoderLayer, self).__init__()

        self.dense1 = nn.Linear(num_hiddens, num_hiddens)
        self.dense2 = nn.Linear(num_hiddens, num_hiddens)
        self.at1 = MaskedMultiHeadAttention(num_hiddens, num_hiddens, num_hiddens, num_hiddens, num_heads)
        self.at2 = MaskedMultiHeadAttention(key_size, num_hiddens, key_size, num_hiddens, num_heads)
        self.norm1 = AddNorm(norm_shape, drop_out)
        self.norm2 = AddNorm(norm_shape, drop_out)
        self.norm3 = AddNorm(norm_shape, drop_out)
        self.actFun = nn.ELU()

    def forward(self, x, cross):
        y = self.at1(x, x, x)
        x = self.norm1(x, y)
        x = self.norm2(x, self.at2(x, cross, cross))

        x = self.norm3(x, self.actFun(self.dense2(self.actFun(self.dense1(x)))))

        return x


class Decoder(nn.Module):
    def __init__(self, key_size, num_hiddens, num_heads, norm_shape, drop_out):
        super(Decoder, self).__init__()
        # self.pe = PositionalEncoding(key_size, 0)
        self.a1 = DecoderLayer(key_size, num_hiddens, num_heads, norm_shape, drop_out)
        self.a2 = DecoderLayer(key_size, num_hiddens, num_heads, norm_shape, drop_out)

    def forward(self, x, cross):
        # print(x.shape)
        # x = self.pe(x)

        x = self.a1(x, cross)
        x = self.a2(x, cross)

        return x

