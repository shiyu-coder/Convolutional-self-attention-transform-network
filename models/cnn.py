import torch
import torch.nn as nn


class ParallelInception(nn.Module):

    def __init__(self, in_channels, c1, c2, c3):
        super(ParallelInception, self).__init__()
        # 单1x1卷积层
        self.p1_1 = nn.Conv2d(in_channels, c1, kernel_size=1)
        # 线路2，1 x 1卷积层后接3 x 3卷积层
        self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
        # 1x1卷积层 + 5x5卷积层
        self.p3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)

        self.actFun = nn.ELU()

    def forward(self, x):
        p1 = self.actFun(self.p1_1(x))
        p2 = self.actFun(self.p2_2(self.actFun(self.p2_1(x))))
        p3 = self.actFun(self.p3_2(self.actFun(self.p3_1(x))))

        # 在通道维度上连接输出
        output = torch.cat((p1, p2, p3), dim=1)
        return output


class ParallelAttentionInception(nn.Module):

    def __init__(self, in_channels, c1, c2, c3, c4):
        super(ParallelAttentionInception, self).__init__()
        # 单1x1卷积层
        self.p1_1 = nn.Conv2d(in_channels, c1, kernel_size=1)
        # 1 x 1卷积层后接3 x 3卷积层
        self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
        # 1x1卷积层 + 5x5卷积层
        self.p3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)
        # 1x1卷积 + 3x3卷积， 1x1卷积，两者相乘
        self.p4_1 = nn.Conv2d(in_channels, c4[0], kernel_size=1)
        self.p4_2 = nn.Conv2d(c4[0], c4[1], kernel_size=3, padding=1)
        self.p4_3 = nn.Conv2d(in_channels, c4[1], kernel_size=1)

        self.actFun = nn.ELU()

    def forward(self, x):
        p1 = self.actFun(self.p1_1(x))
        p2 = self.actFun(self.p2_2(self.actFun(self.p2_1(x))))
        p3 = self.actFun(self.p3_2(self.actFun(self.p3_1(x))))
        p41 = self.p4_2(self.actFun(self.p4_1(x)))
        p42 = self.actFun(self.p4_3(x))
        p4 = p41 * p42

        # 在通道维度上连接输出
        output = torch.cat((p1, p2, p3, p4), dim=1)
        return output


class PCNN(nn.Module):

    def __init__(self, num_hiddens=128, layer1_num=2, layer2_num=0, input_size=(88, 200), attention=False):
        super(PCNN, self).__init__()
        self.num_hiddens = num_hiddens

        self.b1 = nn.Sequential(nn.BatchNorm2d(3),
                                nn.Conv2d(3, 4, kernel_size=3, stride=2, padding=1),
                                nn.ELU())

        self.b2 = nn.Sequential()

        channel_sum = 4

        for i in range(layer1_num):
            self.b2.add_module("layer1-PI-"+str(i),
                               ParallelInception(channel_sum, 4*(i+2), (4*(i+1), 4*(i+2)), (4*(i+1), 4*(i+2))))
            self.b2.add_module("layer1-actFun-"+str(i), nn.ELU())
            channel_sum = 12*i+24

        self.b2.add_module("layer1-pool", nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        if attention:
            for i in range(layer2_num):
                self.b2.add_module("layer2-PAI-" + str(i),
                                   ParallelAttentionInception(channel_sum, 4 * (layer1_num + 1),
                                                              (2 * (layer1_num + 1), 4 * (layer1_num + 1)),
                                                              (2 * (layer1_num + 1), 4 * (layer1_num + 1)),
                                                              (2 * (layer1_num + 1), 4 * (layer1_num + 1))))
                self.b2.add_module("layer2-actFun-" + str(i), nn.ELU())
                if i == 0:
                    channel_sum = 16 * (layer1_num + 1)
        else:
            for i in range(layer2_num):
                self.b2.add_module("layer2-AI-" + str(i),
                                   ParallelInception(channel_sum, 4 * (layer1_num + 1),
                                                     (2 * (layer1_num + 1), 4 * (layer1_num + 1)),
                                                     (2 * (layer1_num + 1), 4 * (layer1_num + 1))))
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
    # X = torch.rand(size=(8, 3, 88, 200))
    # net = PCNN(num_hiddens=128, layer1_num=3, layer2_num=1)
    # X = net.b1(X)
    # for layer in net.b2:
    #     X = layer(X)
    #     print('output shape:\t', X.shape)
    # X = net.b3(X)
    # print(X.shape)
    X = torch.rand(size=(8, 8, 88, 200))
    net = ParallelAttentionInception(8, c1=8, c2=(4, 8), c3=(4, 8), c4=(4, 8))
    X = net(X)
    print(X.shape)


