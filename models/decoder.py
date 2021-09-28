import torch.nn as nn
from models.attention import MaskedMultiHeadAttention
from utils.addNorm import AddNorm


class DecoderLayer(nn.Module):
    def __init__(self, key_size=32, num_hiddens=128, num_heads=4, seq_len=4, drop_out=0.1):
        super(DecoderLayer, self).__init__()

        self.dense1 = nn.Linear(num_hiddens, num_hiddens)
        self.dense2 = nn.Linear(num_hiddens, num_hiddens)
        self.at1 = MaskedMultiHeadAttention(num_hiddens, num_hiddens, num_hiddens, num_hiddens, num_heads)
        self.at2 = MaskedMultiHeadAttention(key_size, num_hiddens, key_size, num_hiddens, num_heads)
        self.norm1 = AddNorm((seq_len, num_hiddens), drop_out)
        self.norm2 = AddNorm((seq_len, num_hiddens), drop_out)
        self.norm3 = AddNorm((seq_len, num_hiddens), drop_out)
        self.actFun = nn.ELU()

    def forward(self, x, cross):
        x = self.norm1(x, self.at1(x, x, x))
        x = self.norm2(x, self.at2(x, cross, cross))

        x = self.norm3(x, self.actFun(self.dense2(self.actFun(self.dense1(x)))))

        return x


class Decoder(nn.Module):
    def __init__(self, layer_num=2, key_size=32, num_hiddens=128, num_heads=4, seq_len=4, drop_out=0.1):
        super(Decoder, self).__init__()
        self.layer_num = layer_num
        self.dec = []
        for i in range(layer_num):
            self.dec.append(DecoderLayer(key_size, num_hiddens, num_heads, seq_len, drop_out))

    def forward(self, x, cross):
        for i in range(self.layer_num):
            x = self.dec[i](x, cross)

        return x

