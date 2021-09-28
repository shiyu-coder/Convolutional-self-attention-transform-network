import torch
import torch.nn as nn
from attention import ParallelInception


class PCNN(nn.Module):

    def __init__(self, num_hiddens=128, layer1_num=2, layer2_num=0, input_size=(88, 200)):
        super(PCNN, self).__init__()
        self.num_hiddens = num_hiddens

        self.b1 = nn.Sequential(nn.BatchNorm2d(3),
                                nn.Conv2d(3, 4, kernel_size=3, stride=2, padding=1),
                                nn.ELU())

        self.b2 = nn.Sequential()

        channel_sum = 4

        for i in range(layer1_num):
            self.b2.add_module("layer1-AI-"+str(i),
                               ParallelInception(channel_sum, 4*(i+2), (4*(i+1), 4*(i+2)), (4*(i+1), 4*(i+2))))
            self.b2.add_module("layer1-actFun-"+str(i), nn.ELU())
            channel_sum = 12*i+24

        self.b2.add_module("layer1-pool", nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        for i in range(layer2_num):
            self.b2.add_module("layer2-AI-" + str(i),
                               ParallelInception(channel_sum, 4*(layer1_num+1),
                                                 (4*layer1_num, 4*(layer1_num+1)), (4*layer1_num, 4*(layer1_num+1))))
            self.b2.add_module("layer2-actFun-" + str(i), nn.ELU())

        self.b2.add_module("layer2-pool", nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        self.b3 = nn.Sequential(nn.Flatten(),
                                nn.Linear(channel_sum * (input_size[0] * input_size[1]) // 64, num_hiddens),
                                nn.ELU(), )

    def forward(self, x):
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        return x


if __name__ == "__main__":
    X = torch.rand(size=(8, 3, 88, 200))
    net = PCNN(num_hiddens=128, layer1_num=3, layer2_num=1)
    X = net.b1(X)
    for layer in net.b2:
        X = layer(X)
        print('output shape:\t', X.shape)
    X = net.b3(X)
    print(X.shape)


