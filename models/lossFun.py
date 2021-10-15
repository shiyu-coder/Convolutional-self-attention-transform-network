import torch
import torch.nn as nn
import matplotlib.pyplot as plt


class SmartMSE(nn.Module):
    def __init__(self, mu):
        super(SmartMSE, self).__init__()
        self.mu2 = mu ** 2
        self.gs = []

    def plot_g(self):
        bins = plt.hist(self.gs, bins=40, facecolor="blue", edgecolor="black", alpha=0.7)
        plt.show()

    def forward(self, input, target):
        mse = torch.mean((input - target) ** 2)
        g = torch.mean(torch.abs(input - target)) / torch.sqrt(mse + self.mu2)
        self.gs.append(float(g.detach().numpy()))
        return mse



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

    def __init__(self, a, r, m):
        super(UnbalancedLoss, self).__init__()
        self.a = a
        self.r = r
        self.m = m

    def forward(self, input, target):
        t = torch.abs(input - target)
        ret1 = torch.where(t < self.m, torch.exp((1 + self.a * torch.abs(target)) ** (self.r + 2) - 1),
                           torch.exp((1 + self.a * torch.abs(1 - target)) ** self.r - 1))
        ret2 = torch.where(t < 1, 0.5 * t ** 2, t - 0.5)
        ret = ret1 * ret2
        return torch.mean(ret)


if __name__ == '__main__':
    x = torch.randn((8, 1))
    y = torch.randn((8, 1))
    lossFun = SteeringLoss(1, 1, 1)
    loss = lossFun(x, y)
    print(loss.shape)
