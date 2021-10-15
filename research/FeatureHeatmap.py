import torch
import cv2
import numpy as np
import torch.nn as nn
import torch
from data.data_loader import ADHDataset
from compares.cmp_model import NVIDIA_ORIGIN
from models.model import CSATNet, PSACNN, SACNN, FSACNN, CNN, CSATNet_v2
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm

transform = transforms.Compose([
            transforms.ToTensor(),
        ])

seq_len = 5
data_set = ADHDataset('../../test_dataset', seq_len, transform=transform, mode='train')
data_loader = DataLoader(data_set, batch_size=1, shuffle=False)


model = CSATNet_v2(128, 4, seq_len, 3, 2, 3, 3, 32, 1, 0.05, 32, False, True)
model.load_state_dict(torch.load('../checkpoints/CSATNet_v2-ADHDataset-mse_0.8_0.8-nhi128-nhe4-sl5-cl1n3-cl2n2-eln3-dln3-vn32-is(180, 320)-ls1-do0.05-mos32-aFalse-ceTrue-0/checkpoint.pth'))

model.eval()
feature_maps = []
for i, (batch_x, batch_y) in enumerate(data_loader):
    if i < 102: # 102, 150, 300, 500, 520, 580, 600
        continue
    if torch.cuda.is_available():
        batch_x, batch_y = batch_x.cuda(), batch_y.cuda()

    batch_num = batch_x.shape[0]
    x = batch_x.reshape(-1, batch_x.shape[2], batch_x.shape[3], batch_x.shape[4])
    x = model.norm(x)
    if model.cnn.channel_expansion:
        f0 = model.cnn.actFun(model.cnn.ce1(x))
        f0_ = f0.detach().cpu().mean(dim=1)
        f0_ = f0_.unsqueeze(1)
        feature_maps.append(f0_)
        f1 = model.cnn.actFun(model.cnn.ce2(f0))
        f1_ = f1.detach().cpu().mean(dim=1)
        f1_ = f1_.unsqueeze(1)
        feature_maps.append(f1_)
        f2 = model.cnn.actFun(model.cnn.ce3(x))
        f2_ = f2.detach().cpu().mean(dim=1)
        f2_ = f2_.unsqueeze(1)
        feature_maps.append(f2_)
        x = torch.cat((x, f1, f2), dim=1)
    nonConvLayers = [1, 3, 5, 6, 8, 10, 11]
    count = 0
    for layer in model.cnn.cnn:
        x = layer(x)
        if count not in nonConvLayers:
            x_ = x.detach().cpu().mean(dim=1)
            x_ = x_.unsqueeze(1)
            feature_maps.append(x_)
        count += 1

    ct1 = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=3, stride=1,
                             output_padding=0, bias=False)
    ct1.weight.requires_grad = False
    ct1.weight = nn.Parameter(torch.ones(1, 1, 3, 3))

    feature_maps[-2] = feature_maps[-2] * ct1(feature_maps[-1])
    feature_maps[-3] = feature_maps[-3] * ct1(feature_maps[-2])

    pool1 = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=[1, 1],
                               output_padding=0, bias=False)
    pool1.weight.requires_grad = False
    pool1.weight = nn.Parameter(torch.ones(1, 1, 3, 3) / 9)

    ct2 = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=5, stride=2,
                             output_padding=[1, 0], bias=False)
    ct2.weight.requires_grad = False
    ct2.weight = nn.Parameter(torch.ones(1, 1, 5, 5))
    ct3 = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=5, stride=2,
                             output_padding=1, bias=False)
    ct3.weight.requires_grad = False
    ct3.weight = nn.Parameter(torch.ones(1, 1, 5, 5))
    ct4 = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1,
                             output_padding=0, bias=False)
    ct4.weight.requires_grad = False
    ct4.weight = nn.Parameter(torch.ones(1, 1, 3, 3))

    # for i in range(len(feature_maps)):
    #     print(feature_maps[i].shape)

    feature_maps[-4] = feature_maps[-4] * ct2(pool1(feature_maps[-3]))
    # feature_maps[-5] = feature_maps[-5] * ct3(feature_maps[-4]) # 1
    inp1 = ct3(feature_maps[-5])
    # feature_maps[-6] = feature_maps[-6] * ct3(feature_maps[-5]) # 1
    inp2 = ct4(feature_maps[-6])
    # feature_maps[1] = feature_maps[1] * ct4(feature_maps[2])    # 1
    feature_maps[0] = feature_maps[0] * ct4(feature_maps[1])
    inp3 = feature_maps[0]
    inp1 = inp1.repeat(1, 3, 1, 1)
    inp2 = inp2.repeat(1, 3, 1, 1)
    inp3 = inp3.repeat(1, 3, 1, 1)
    inp1 = inp1.permute(0, 2, 3, 1)
    inp2 = inp2.permute(0, 2, 3, 1)
    inp3 = inp3.permute(0, 2, 3, 1)
    batch_x = batch_x.squeeze(0)
    imgs = batch_x.permute(0, 2, 3, 1)
    imgs = (imgs * 255).int().detach().numpy()
    res = []
    for i in range(1):
        img = cv2.cvtColor(np.uint8(imgs[i]), cv2.IMREAD_COLOR)
        tmp1 = inp1[i]
        tmp1 = (tmp1 - torch.min(tmp1)) / (torch.max(tmp1) - torch.min(tmp1))
        tmp1 = np.uint8((tmp1 * 255).int().detach().numpy())
        heatmap1 = cv2.applyColorMap(tmp1, cv2.COLORMAP_JET)
        fin1 = cv2.addWeighted(heatmap1, 0.0, img, 1, 0)
        # fin1 = cv2.addWeighted(heatmap1, 0.4, img, 0.6, 0)
        res.append(fin1)

        tmp2 = inp2[i]
        tmp2 = (tmp2 - torch.min(tmp2)) / (torch.max(tmp2) - torch.min(tmp2))
        tmp2 = np.uint8((tmp2 * 255).int().detach().numpy())
        heatmap2 = cv2.applyColorMap(tmp2, cv2.COLORMAP_JET)
        fin2 = cv2.addWeighted(heatmap2, 1, img, 0, 0)
        # fin2 = cv2.addWeighted(heatmap2, 0.4, img, 0.6, 0)
        res.append(fin2)

        tmp3 = inp3[i]
        tmp3 = (tmp3 - torch.min(tmp3)) / (torch.max(tmp3) - torch.min(tmp3))
        tmp3 = np.uint8((tmp3 * 255).int().detach().numpy())
        heatmap3 = cv2.applyColorMap(tmp3, cv2.COLORMAP_JET)
        fin3 = cv2.addWeighted(heatmap3, 1, img, 0, 0)
        # fin3 = cv2.addWeighted(heatmap3, 0.4, img, 0.6, 0)
        res.append(fin3)
    p_img = np.hstack(res)
    cv2.imwrite("../../3_channel_view.jpg", p_img)
    cv2.imshow("", p_img)
    cv2.waitKey()

    break





