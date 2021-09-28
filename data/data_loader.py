import torch
import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image


class ADDataset(Dataset):

    def __init__(self, data_dir, label_dir, seq_len, transform=None, mode='train'):
        super(ADDataset, self).__init__()
        self.data = self.get_data(data_dir)
        self.labels = self.get_labels(label_dir)
        self.transform = transform
        self.seq_len = seq_len
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

        seq_label = torch.tensor(self.labels[index+1:index+self.seq_len+1])
        return seq_img, seq_label

    def __len__(self):
        return len(self.data) - self.seq_len

