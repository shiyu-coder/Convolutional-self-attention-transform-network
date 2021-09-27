import torch.nn as nn
import math
from attention import AttentionInception, MultiHeadAttention
from utils.positionalEncoding import PositionalEncoding
from utils.addNorm import AddNorm


class CNNEncoder(nn.Module):

    def __init__(self, num_hiddens, num_heads, norm_shape, drop_out):
        super(CNNEncoder, self).__init__()
        self.num_hiddens = num_hiddens

        self.b1 = nn.Sequential(nn.BatchNorm2d(3),
                                nn.Conv2d(3, 4, kernel_size=3, stride=2, padding=3),
                                nn.ELU(),
                                nn.MaxPool2d(kernel_size=2, stride=2, padding=1))

        self.b2 = nn.Sequential(AttentionInception(4, 8, (4, 8), (4, 8)),
                                nn.ELU(),
                                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                                AttentionInception(24, 16, (16, 32), (16, 32)),
                                nn.ELU(),
                                nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        # 模块3
        self.b3 = nn.Sequential(nn.Flatten(),
                                nn.Linear(6240, num_hiddens),
                                nn.ELU(),)

        # 位置编码
        self.pe = PositionalEncoding(num_hiddens, 0)

        # 自注意力模块
        self.actFun = nn.ELU()

        self.at1 = MultiHeadAttention(num_hiddens, num_hiddens, num_hiddens, num_hiddens, num_heads, drop_out)
        self.norm1 = AddNorm(norm_shape, drop_out)
        self.dense1 = nn.Linear(num_hiddens, 64)
        self.at2 = MultiHeadAttention(64, 64, 64, 64, num_heads, drop_out)
        self.norm2 = AddNorm((norm_shape[0], 64), drop_out)
        self.dense2 = nn.Linear(64, 32)

        self.tmp_dense = nn.Linear(32, 1)

    def forward(self, x):
        batch_num = x.shape[0]
        x = x.reshape(-1, x.shape[2], x.shape[3], x.shape[4])
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)

        x = x.reshape(batch_num, -1, x.shape[1])
        x = self.pe(x * math.sqrt(self.num_hiddens))

        y = self.norm1(x, self.at1(x, x, x))
        y = self.actFun(self.dense1(y))
        y = self.norm2(y, self.at2(y, y, y))
        y = self.actFun(self.dense2(y))

        # x = self.tmp_dense(x)

        return x, y
