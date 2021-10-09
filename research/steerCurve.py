import torch
from data.data_loader import ADHDataset
from compares.cmp_model import NVIDIA_ORIGIN
from models.model import CSATNet, PSACNN, SACNN, FSACNN, CNN, CSATNet_v2
import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils.data import DataLoader
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm


def get_steer_curve(model, data_loader):
    model.eval()
    preds = []
    trues = []
    for i, (batch_x, batch_y) in enumerate(data_loader):
        if torch.cuda.is_available():
            batch_x, batch_y = batch_x.cuda(), batch_y.cuda()

        outputs = model(batch_x)

        batch_y = batch_y.float()

        pred = outputs.detach().cpu().numpy()
        true = batch_y.detach().cpu().numpy()

        pred = pred[:, -1, 0][0]
        true = true[:, -1, 0][0]

        if i % 200 == 0:
            print(str(i) + " :: " + str(len(data_loader)))

        preds.append(pred)
        trues.append(true)
        if i >= 1000:
            break
    return preds, trues


transform = transforms.Compose([
            transforms.ToTensor(),
        ])

seq_len = 6
data_set = ADHDataset('../../test_dataset', seq_len, transform=transform, mode='valid')
data_loader = DataLoader(data_set, batch_size=1, shuffle=False)


model1 = CSATNet_v2(128, 4, seq_len, 3, 2, 3, 3, 32, 1, 0.05, 32, False, True)
model1.load_state_dict(torch.load('../checkpoints/CSATNet_v2-ADHDataset-unbalancedLoss-nhi128-nhe4-sl6-cl1n3-cl2n2-eln3-dln3-vn32-is(180, 320)-ls1-do0.05-mos32-aFalse-ceTrue-0/checkpoint.pth'))

model2 = CSATNet_v2(128, 4, seq_len, 3, 2, 3, 3, 32, 1, 0.05, 32, False, True)
model2.load_state_dict(torch.load('../checkpoints/CSATNet_v2-ADHDataset-mse-nhi128-nhe4-sl6-cl1n3-cl2n2-eln3-dln3-vn32-is(180, 320)-ls1-do0.05-mos32-aFalse-ceTrue-0/checkpoint.pth'))


preds1, trues = get_steer_curve(model1, data_loader)
preds2, _ = get_steer_curve(model1, data_loader)

plt.plot(preds1, label='unbalanced loss')
plt.plot(preds2, label='mse')
plt.plot(trues, label='label')
plt.legend()
plt.show()

