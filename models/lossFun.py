import torch
import torch.nn as nn


class SteeringLoss(nn.Module):

    def __init__(self, a, r, b):
        super(SteeringLoss, self).__init__()
        self.a = a
        self.r = r
        self.b = b

    def forward(self, input, target):
        ret = (1 + self.a * torch.abs(target) ** self.r) ** self.b * (input - target) ** 2 / 2
        ret = torch.mean(ret)
        return ret


class UnbalancedLoss(nn.Module):

    def __init__(self, a, r):
        super(UnbalancedLoss, self).__init__()
        self.a = a
        self.r = r

    def forward(self, input, target):
        ret1 = torch.exp((1 + self.a * torch.abs(target)) ** self.r - 1)
        t = torch.abs(input - target)
        ret2 = torch.where(t < 1, 0.5 * t ** 2, t - 0.5)
        ret = ret1 * ret2
        return torch.mean(ret)


if __name__ == '__main__':
    x = torch.randn((8, 1))
    y = torch.randn((8, 1))
    lossFun = SteeringLoss(1, 1, 1)
    loss = lossFun(x, y)
    print(loss.shape)
