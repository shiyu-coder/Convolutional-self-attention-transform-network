import torch
import numpy as np
import torch.nn as nn


class SelfAttentionConv(nn.Module):
    def __init__(self, in_channels, c1, c2):
        super(SelfAttentionConv, self).__init__()
        self.p1 = nn.Conv2d(in_channels, c1, kernel_size=1)
        self.p2 = nn.Conv2d(c1, c2, kernel_size=3, padding=1)
        self.p3 = nn.Conv2d(in_channels, c2, kernel_size=1)
        self.actFun = nn.ELU()

    def forward(self, x):
        x1 = self.p2(self.actFun(self.p1(x)))
        x2 = self.actFun(self.p3(x))
        x = x1 * x2
        return x


class BlurPool(nn.Module):

    def __init__(self, c_in, c_out):
        super(BlurPool, self).__init__()
        self.conv_op = nn.Conv2d(c_in, c_out, 3, stride=2, bias=False)
        kernel = np.array([[1., 2., 1.], [2., 4., 2.], [1., 2., 1.]], dtype='float32')
        kernel = kernel.reshape((1, 1, 3, 3))
        kernel = torch.from_numpy(kernel)
        kernel = kernel / torch.sum(kernel)
        kernel = kernel.repeat(c_in, c_out, 1, 1)
        self.conv_op.weight.data = kernel
        self.conv_op.weight.requires_grad = False

    def forward(self, x):
        return self.conv_op(x)


class ChannelExpansionConv(nn.Module):

    def __init__(self, c_in, c_out):
        super(ChannelExpansionConv, self).__init__()
        self.conv_op = nn.Conv2d(c_in, c_out, 3, padding=1, bias=False)

    def forward(self, x):
        return self.conv_op(x)


class SACNNLayer(nn.Module):
    def __init__(self, num_hiddens=128, cnn_layer1_num=3, cnn_layer2_num=2):
        super(SACNNLayer, self).__init__()
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
            nn.Linear(in_channel * 12, 256),
            nn.ELU(),
            nn.Linear(256, num_hiddens),
        )

    def forward(self, x):
        x = self.cnn(x)
        x = self.dense(x)
        return x


class ChannelParallelismCNN(nn.Module):

    def __init__(self, num_hiddens=128, cnn_layer1_num=2, cnn_layer2_num=2):
        super(ChannelParallelismCNN, self).__init__()
        self.actFun = nn.ELU()
        self.cnn_layer1_num = cnn_layer1_num
        self.cnn_layer2_num = cnn_layer2_num
        self.c1 = []
        self.c2 = []
        if torch.cuda.is_available():
            for i in range(cnn_layer1_num):
                self.c1.append(nn.Conv2d(3*2**i, 3*2**i, kernel_size=3, padding=1).cuda())
        else:
            for i in range(cnn_layer1_num):
                self.c1.append(nn.Conv2d(3*2**i, 3*2**i, kernel_size=3, padding=1))

        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        if torch.cuda.is_available():
            for i in range(cnn_layer1_num, cnn_layer1_num + cnn_layer2_num):
                self.c2.append(nn.Conv2d(3*2**i, 3*2**i, kernel_size=3, padding=1).cuda())
        else:
            for i in range(cnn_layer1_num, cnn_layer1_num + cnn_layer2_num):
                self.c2.append(nn.Conv2d(3*2**i, 3*2**i, kernel_size=3, padding=1))

        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        channel = 3 * 2**(cnn_layer1_num + cnn_layer2_num)
        self.c3 = nn.Sequential(
            nn.Conv2d(channel, 96, kernel_size=3, stride=2),
            nn.ELU(),
            nn.Conv2d(96, 128, kernel_size=3, stride=2),
            nn.ELU(),
        )

        self.dense = nn.Sequential(
            nn.Flatten(),
            nn.Linear(5632, 256),
            nn.ELU(),
            nn.Linear(256, 150),
            nn.ELU(),
            nn.Linear(150, num_hiddens),
        )

    def forward(self, x):
        for i in range(self.cnn_layer1_num):
            y = self.c1[i](x)
            x = torch.cat((x, y), dim=1)
            x = self.actFun(x)
        x = self.pool1(x)
        for i in range(self.cnn_layer2_num):
            y = self.c2[i](x)
            x = torch.cat((x, y), dim=1)
            x = self.actFun(x)
        x = self.pool2(x)
        x = self.c3(x)
        x = self.dense(x)
        return x


class BlurCNNLayer(nn.Module):
    def __init__(self, num_hiddens=128, cnn_layer1_num=3, cnn_layer2_num=2, laplace=False):
        super(BlurCNNLayer, self).__init__()
        self.cnn = nn.Sequential()
        # in_channels = [3, 24, 36, 48, 64, 80, 128, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256]
        # in_channels = [1, 24, 36, 48, 64, 80, 128, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256]
        self.laplace = laplace
        if laplace:
            self.laplace = ChannelExpansionConv(3, 3)
            in_channels = [6, 24, 36, 48, 64, 80, 128, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256]
        else:
            in_channels = [3, 24, 36, 48, 64, 80, 128, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256]
        for i in range(0, cnn_layer1_num):
            self.cnn.add_module("layer1-"+str(i), nn.Conv2d(in_channels[i], in_channels[i+1], kernel_size=3))
            self.cnn.add_module("actFun-" + str(i), nn.ELU())
            self.cnn.add_module("blurPool1-" + str(i), BlurPool(in_channels[i+1], in_channels[i+1]))
        # self.cnn.add_module("pool1", nn.MaxPool2d(kernel_size=3, stride=1, padding=1))
        # self.cnn.add_module("blurpool1", BlurPool(in_channels[cnn_layer1_num], in_channels[cnn_layer1_num]))
        for i in range(cnn_layer1_num, cnn_layer1_num + cnn_layer2_num):
            self.cnn.add_module("layer2-" + str(i), nn.Conv2d(in_channels[i], in_channels[i+1], kernel_size=3))
            self.cnn.add_module("actFun-" + str(i), nn.ELU())
            self.cnn.add_module("blurPool2-" + str(i), BlurPool(in_channels[i + 1], in_channels[i + 1]))
        # self.cnn.add_module("pool2", nn.MaxPool2d(kernel_size=3, stride=1, padding=1))
        # self.cnn.add_module("blurpool2", BlurPool(in_channels[cnn_layer1_num + cnn_layer2_num],
        #                                           in_channels[cnn_layer1_num + cnn_layer2_num]))
        in_channel = in_channels[cnn_layer1_num + cnn_layer2_num]
        self.dense = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channel * 14, 256),
            nn.ELU(),
            nn.Linear(256, num_hiddens),
        )

    def forward(self, x):
        if self.laplace:
            lap = self.laplace(x)
            x = torch.cat((x, lap), dim=1)
        x = self.cnn(x)
        x = self.dense(x)
        return x


class CNNLayer(nn.Module):
    def __init__(self, num_hiddens=128, cnn_layer1_num=3, cnn_layer2_num=2, channel_expansion=False):
        super(CNNLayer, self).__init__()
        self.cnn = nn.Sequential()
        # in_channels = [3, 24, 36, 48, 64, 80, 128, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256]
        # in_channels = [1, 24, 36, 48, 64, 80, 128, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256]
        self.channel_expansion = channel_expansion
        if channel_expansion:
            self.channel_expansion = ChannelExpansionConv(3, 3)
            in_channels = [6, 24, 36, 48, 64, 80, 128, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256]
        else:
            in_channels = [3, 24, 36, 48, 64, 80, 128, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256]
        for i in range(0, cnn_layer1_num):
            self.cnn.add_module("layer1-"+str(i), nn.Conv2d(in_channels[i], in_channels[i+1], kernel_size=5, stride=2))
            self.cnn.add_module("actFun-" + str(i), nn.ELU())
        self.cnn.add_module("pool1", nn.MaxPool2d(kernel_size=3, stride=1, padding=1))
        for i in range(cnn_layer1_num, cnn_layer1_num + cnn_layer2_num):
            self.cnn.add_module("layer2-" + str(i), nn.Conv2d(in_channels[i], in_channels[i+1], kernel_size=3))
            self.cnn.add_module("actFun-" + str(i), nn.ELU())
        self.cnn.add_module("pool2", nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        in_channel = in_channels[cnn_layer1_num + cnn_layer2_num]
        self.dense = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channel * 136, 256),
            nn.ELU(),
            nn.Linear(256, num_hiddens),
        )

    def forward(self, x):
        if self.channel_expansion:
            ce = self.channel_expansion(x)
            x = torch.cat((x, ce), dim=1)
        x = self.cnn(x)
        x = self.dense(x)
        return x


class FastCNNLayer(nn.Module):
    def __init__(self, num_hiddens=128, cnn_layer1_num=3, cnn_layer2_num=2, channel_expansion=False):
        super(FastCNNLayer, self).__init__()
        self.cnn = nn.Sequential()
        # in_channels = [3, 24, 36, 48, 64, 80, 128, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256]
        # in_channels = [1, 24, 36, 48, 64, 80, 128, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256]
        self.channel_expansion = channel_expansion
        if channel_expansion:
            self.channel_expansion = ChannelExpansionConv(3, 3)
            in_channels = [6, 24, 36, 48, 64, 80, 128, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256]
        else:
            in_channels = [3, 24, 36, 48, 64, 80, 128, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256]
        for i in range(0, cnn_layer1_num):
            self.cnn.add_module("layer11-" + str(i), nn.Conv2d(in_channels[i], in_channels[i] // 2, kernel_size=1))
            self.cnn.add_module("actFun1-" + str(i), nn.ELU())
            self.cnn.add_module("layer12-"+str(i), nn.Conv2d(in_channels[i] // 2, in_channels[i+1] // 2, kernel_size=5, stride=2))
            self.cnn.add_module("actFun2-" + str(i), nn.ELU())
            self.cnn.add_module("layer13-" + str(i), nn.Conv2d(in_channels[i+1] // 2, in_channels[i+1], kernel_size=1))
            self.cnn.add_module("actFun3-" + str(i), nn.ELU())
        self.cnn.add_module("pool1", nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        for i in range(cnn_layer1_num, cnn_layer1_num + cnn_layer2_num):
            self.cnn.add_module("layer21-" + str(i), nn.Conv2d(in_channels[i], in_channels[i] // 2, kernel_size=1))
            self.cnn.add_module("actFun1-" + str(i), nn.ELU())
            self.cnn.add_module("layer22-" + str(i), nn.Conv2d(in_channels[i] // 2, in_channels[i+1] // 2, kernel_size=3, stride=1))
            self.cnn.add_module("actFun2-" + str(i), nn.ELU())
            self.cnn.add_module("layer23-" + str(i), nn.Conv2d(in_channels[i+1] // 2, in_channels[i + 1], kernel_size=1))
            self.cnn.add_module("actFun3-" + str(i), nn.ELU())
        self.cnn.add_module("pool2", nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        in_channel = in_channels[cnn_layer1_num + cnn_layer2_num]
        self.dense = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channel * 30, 256),
            nn.ELU(),
            nn.Linear(256, num_hiddens),
        )

    def forward(self, x):
        if self.channel_expansion:
            ce = self.channel_expansion(x)
            x = torch.cat((x, ce), dim=1)
        x = self.cnn(x)
        x = self.dense(x)
        return x


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
    X = torch.rand(size=(8, 3, 88, 200))
    net = BlurCNNLayer()
    X = net(X)
    print(X.shape)
    a = np.array([1., 2., 1.])
    filt = torch.Tensor(a[:, None] * a[None, :])
    print(filt.shape)
    print(filt)


