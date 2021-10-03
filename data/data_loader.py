import torch
import torchvision.transforms as transforms
import os
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import h5py


class ADDataset(Dataset):

    def __init__(self, data_dir, label_dir, seq_len, transform=None, mode='train', multitask=False):
        super(ADDataset, self).__init__()
        self.data = self.get_data(data_dir)
        self.labels = self.get_labels(label_dir)
        self.transform = transform
        self.seq_len = seq_len
        self.mulitask = multitask
        if len(self.data) <= 10000:
            if mode == 'train':
                self.data = self.data[:len(self.data) * 80 // 100]
                self.labels = self.labels[:len(self.labels) * 80 // 100]
            elif mode == 'valid':
                self.data = self.data[len(self.data) * 80 // 100:len(self.data) * 90 // 100]
                self.labels = self.labels[len(self.labels) * 80 // 100:len(self.labels) * 90 // 100]
            elif mode == 'test':
                self.data = self.data[len(self.data) * 90 // 100:]
                self.labels = self.labels[len(self.labels) * 90 // 100:]
        else:
            if mode == 'train':
                self.data = self.data[:len(self.data) * 90 // 100]
                self.labels = self.labels[:len(self.labels) * 90 // 100]
            elif mode == 'valid':
                self.data = self.data[len(self.data) * 90 // 100:len(self.data) * 95 // 100]
                self.labels = self.labels[len(self.labels) * 90 // 100:len(self.labels) * 95 // 100]
            elif mode == 'test':
                self.data = self.data[len(self.data) * 95 // 100:]
                self.labels = self.labels[len(self.labels) * 95 // 100:]

    def get_data(self, data_dir):
        data = []
        for root, dirs, _ in os.walk(data_dir):
            for dir in dirs:
                next_path = os.path.join(root, dir)
                for new_root, _, files in os.walk(next_path):
                    for file in files:
                        img_dir = os.path.join(str(new_root), str(file))
                        data.append(img_dir)
        return data

    def get_labels(self, label_dir):
        labels = []
        labels_csv = pd.read_csv(label_dir)
        for i, item in labels_csv.iterrows():
            labels.append([item[0], ])
        return labels

    def __getitem__(self, index):
        dirs = self.data[index:index+self.seq_len]
        imgs = []
        for dir in dirs:
            img = np.array(Image.open(dir))
            if self.transform is not None:
                img = self.transform(img)
            imgs.append(img)

        seq_img = torch.stack(imgs, 0)
        if self.mulitask:
            single_label = self.labels[index + 1:index + self.seq_len + 1]
            single_label.extend(self.labels[index + 1:index + self.seq_len + 1])
            seq_label = torch.tensor(single_label)
        else:
            seq_label = torch.tensor(self.labels[index+1:index+self.seq_len+1])
        return seq_img, seq_label

    def __len__(self):
        return len(self.data) - self.seq_len


class ADHDataset(Dataset):
    def __init__(self, data_dir, seq_len, transform=None, mode='train'):
        super(ADHDataset, self).__init__()
        if mode == "train":
            data_dir = os.path.join(data_dir, "SeqTrain")
        elif mode == 'valid':
            data_dir = os.path.join(data_dir, "SeqVal")
        elif mode == 'test':
            data_dir = os.path.join(data_dir, "SeqTest")
        self.data_h5s = self.get_data(data_dir)
        self.transform = transform
        self.seq_len = seq_len

    def get_data(self, data_dir):
        data_h5s = []
        for root, _, files in os.walk(data_dir):
            for file in files:
                h5_path = os.path.join(root, file)
                data_h5s.append(h5_path)
        return data_h5s

    def __getitem__(self, index):
        pos = index // 200
        rel_pos = index % 200
        seq_label = []
        seq_img = []
        if rel_pos + self.seq_len + 1 > 200:
            with h5py.File(self.data_h5s[pos], 'r') as f:
                for i in range(rel_pos, 200):
                    img = f['img'][i]
                    if self.transform is not None:
                        img = self.transform(img)
                    if i != rel_pos:
                        seq_label.append([f['steer'][i], ])
                    seq_img.append(img)
            with h5py.File(self.data_h5s[pos+1], 'r') as f:
                for i in range(0, rel_pos + self.seq_len - 200 + 1):
                    if i != rel_pos + self.seq_len - 200:
                        img = f['img'][i]
                        if self.transform is not None:
                            img = self.transform(img)
                        seq_img.append(img)
                    seq_label.append([f['steer'][i], ])
        else:
            with h5py.File(self.data_h5s[pos], 'r') as f:
                for i in range(rel_pos, rel_pos + self.seq_len + 1):
                    if i != rel_pos + self.seq_len:
                        img = f['img'][i]
                        if self.transform is not None:
                            img = self.transform(img)
                        # print(img.shape)
                        seq_img.append(img)
                    if i != rel_pos:
                        seq_label.append([f['steer'][i], ])
        seq_imgs = torch.stack(seq_img, 0)
        seq_labels = torch.tensor(seq_label)
        return seq_imgs, seq_labels

    def __len__(self):
        return len(self.data_h5s) * 200 - self.seq_len


if __name__ == "__main__":
    train_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    dataset = ADHDataset("../../test_dataset", 4, train_transform, mode='train')
    # dataset = ADDataset("../../DataSet", "../../ADLabel.csv", 4, transform=train_transform)
    data_loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)
    for i, (x, y) in enumerate(data_loader):
        print(i, x.shape, y.shape)
        # break
