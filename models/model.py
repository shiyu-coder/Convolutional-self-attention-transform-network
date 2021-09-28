import torch
import torch.nn as nn
from encoder import CNNEncoder
from decoder import Decoder


class CSATNet(nn.Module):
    def __init__(self, num_hiddens=128, num_heads=4, seq_len=4, cnn_layer1_num=2, cnn_layer2_num=0,
                 enc_layer_num=2, dec_layer_num=2, input_size=(88, 200), label_size=1,
                 drop_out=0.1, min_output_size=32, attention=False):
        super(CSATNet, self).__init__()
        self.enc = CNNEncoder(num_hiddens, num_heads, seq_len, enc_layer_num, cnn_layer1_num,
                              cnn_layer2_num, input_size, drop_out, min_output_size, attention)
        self.key_size = self.enc.key_size
        self.dec = Decoder(dec_layer_num, self.key_size, num_hiddens, num_heads, seq_len, drop_out)

        self.dense = nn.Linear(num_hiddens, label_size)

    def forward(self, x):
        x, cross = self.enc(x)
        output = self.dec(x, cross)
        output = self.dense(output)
        return output


if __name__ == '__main__':
    X = torch.rand(size=(8, 16, 3, 88, 200))
    net = CSATNet(128, 4, 16, 3, 2, 3, 3, (88, 200), 1, 0.1, 32, True)
    X = net(X)
    print(X.shape)

