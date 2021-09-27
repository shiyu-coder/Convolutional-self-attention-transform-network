import torch
import torch.nn as nn
from encoder import CNNEncoder
from decoder import Decoder


class CSATNet(nn.Module):
    def __init__(self, key_size, num_hiddens, num_heads, norm_shape, label_size, drop_out):
        super(CSATNet, self).__init__()
        self.en1 = CNNEncoder(num_hiddens, num_heads, norm_shape, drop_out)
        self.de1 = Decoder(key_size, num_hiddens, num_heads, norm_shape, drop_out)

        self.dense = nn.Linear(num_hiddens, label_size)

    def forward(self, x):
        x, cross = self.en1(x)
        output = self.de1(x, cross)
        output = self.dense(output)
        return output


if __name__ == '__main__':
    X = torch.rand(size=(8, 16, 3, 88, 200))
    net = CSATNet(32, 128, 4, (16, 128), 2, 0.1)
    X = net(X)
    print(X.shape)

