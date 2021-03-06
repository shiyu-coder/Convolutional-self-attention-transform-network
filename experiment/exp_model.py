from data.data_loader import ADDataset, ADHDataset
from compares.cmp_model import NVIDIA_ORIGIN, CNN_LSTM, TDCNN_LSTM
from models.model import CSATNet, PSACNN, SACNN, FSACNN, CNN, CSATNet_v2
import torch
import models.lossFun as lossFun
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import numpy as np
import os
import time
import torchvision.transforms as transforms
import warnings
from tensorboardX import SummaryWriter
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric

warnings.filterwarnings('ignore')


class Exp_model:
    def __init__(self, args):
        self.args = args
        self.use_gpu = self._acquire_device()
        self.model = self._build_model()
        if self.use_gpu:
            self.model = self.model.cuda()

        if self.args.tensorboard:
            self.writer = SummaryWriter(comment='CSATNet_exp_comment', filename_suffix="CSATNet_exp_suffix")

    def _build_model(self):
        model_dict = {
            'CSATNet': CSATNet,
            'CSATNet_multitask': CSATNet_v2,
            'NVIDIA_ORIGIN': NVIDIA_ORIGIN,
            'PSACNN': PSACNN,
            'SACNN': SACNN,
            'FSACNN': FSACNN,
            'CNN': CNN,
            'CSATNet_v2': CSATNet_v2,
            'CNN_LSTM': CNN_LSTM,
            'TDCNN_LSTM': TDCNN_LSTM,
        }
        if self.args.model == 'CSATNet':
            model = model_dict[self.args.model](
                self.args.num_hiddens,
                self.args.num_heads,
                self.args.seq_len,
                self.args.cnn_layer1_num,
                self.args.cnn_layer2_num,
                self.args.enc_layer_num,
                self.args.dec_layer_num,
                self.args.label_size,
                self.args.drop_out,
                self.args.min_output_size,
            )
        if self.args.model == 'CSATNet_v2':
            model = model_dict[self.args.model](
                self.args.num_hiddens,
                self.args.num_heads,
                self.args.seq_len,
                self.args.cnn_layer1_num,
                self.args.cnn_layer2_num,
                self.args.enc_layer_num,
                self.args.dec_layer_num,
                self.args.vector_num,
                self.args.label_size,
                self.args.drop_out,
                self.args.min_output_size,
                self.args.attention,
                self.args.channel_expansion,
            )
        elif self.args.model == 'PSACNN':
            model = model_dict[self.args.model](
                self.args.num_hiddens,
                self.args.cnn_layer1_num,
                self.args.cnn_layer2_num,
                self.args.input_size,
                self.args.label_size,
                self.args.attention,
            )
        elif self.args.model == 'SACNN':
            model = model_dict[self.args.model](
                self.args.cnn_layer1_num,
                self.args.cnn_layer2_num,
                self.args.label_size,
            )
        elif self.args.model == 'FSACNN':
            model = model_dict[self.args.model](
                self.args.cnn_layer1_num,
                self.args.cnn_layer2_num,
                self.args.label_size,
            )
        elif self.args.model == 'CNN':
            model = model_dict[self.args.model](
                self.args.cnn_layer1_num,
                self.args.cnn_layer2_num,
                self.args.label_size,
            )
        elif self.args.model == 'NVIDIA_ORIGIN' or self.args.model == 'CNN_LSTM' or self.args.model == 'TDCNN_LSTM':
            model = model_dict[self.args.model]()

        return model.float()

    def _acquire_device(self):
        if self.args.use_gpu:
            # os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu)
            # device = torch.device('cuda:0')
            print('Use GPU: cuda:0')
            return True
        else:
            # device = torch.device('cpu')
            print('Use CPU')
            return False
        # return device

    def _get_data(self, mode):
        args = self.args

        data_dict = {
            'ADDataset': ADDataset,
            'ADHDataset': ADHDataset,
        }
        Data = data_dict[self.args.data]

        if mode == 'test' or mode == 'valid':
            shuffle_flag = False
            drop_last = True
            batch_size = args.batch_size
        else:
            shuffle_flag = True
            drop_last = True
            batch_size = args.batch_size

        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        if self.args.data == 'ADDataset':
            data_set = Data(
                data_dir=args.data_path,
                label_dir=args.label_path,
                mode=mode,
                seq_len=args.seq_len,
                transform=transform,
                multitask=args.multitask,
            )
        elif self.args.data == 'ADHDataset':
            data_set = Data(
                data_dir=args.data_path,
                mode=mode,
                seq_len=args.seq_len,
                transform=transform,
            )

        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)

        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self, valid=False):
        if valid:
            criterion = nn.MSELoss()
            return criterion
        if self.args.loss == 'mae':
            criterion = nn.L1Loss()
        elif self.args.loss == 'mse':
            criterion = nn.MSELoss()
        elif self.args.loss == 'smoothL1':
            criterion = nn.SmoothL1Loss()
        elif self.args.loss == 'steeringLoss':
            criterion = lossFun.SteeringLoss(1, 1, 1)
        elif self.args.loss == 'unbalancedLoss':
            criterion = lossFun.UnbalancedLoss(self.args.loss_arg[0], self.args.loss_arg[1], self.args.loss_arg[2])
        elif self.args.loss == 'SmartMSE':
            criterion = lossFun.SmartMSE(0.02)
        if self.use_gpu:
            criterion = criterion.cuda()
        return criterion

    def train(self, setting):
        train_data, train_loader = self._get_data('train')
        vali_data, vali_loader = self._get_data('valid')

        path = './checkpoints/' + setting
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        vali_criterion = self._select_criterion(valid=True)
        total_count = 0
        for epoch in range(self.args.epoch):
            iter_count = 0
            train_loss = []

            self.model.train()
            for i, (batch_x, batch_y) in enumerate(train_loader):

                iter_count += 1
                total_count += 1
                model_optim.zero_grad()

                if self.use_gpu:
                    batch_x, batch_y = batch_x.cuda(), batch_y.cuda()

                outputs = self.model(batch_x)

                batch_y = batch_y.float()

                if self.args.single_learning:
                    batch_y = batch_y[:, -1, 0]
                    outputs = outputs[:, -1, 0]

                if outputs.shape[1] < batch_y.shape[1]:
                    batch_y = batch_y[:, -outputs.shape[1]:, :]

                loss = criterion(outputs, batch_y)

                if self.use_gpu:
                    loss = loss.cpu()

                if self.args.tensorboard:
                    self.writer.add_scalar("Train loss", loss, global_step=epoch*train_steps+i)

                train_loss.append(loss.item())

                if (i + 1) % 200 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.epoch - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()

            criterion.plot_g()

            train_loss = np.average(train_loss)

            vali_loss = self.vali(vali_loader, vali_criterion)

            if self.args.tensorboard:
                self.writer.add_scalar("Valid loss", vali_loss, global_step=epoch)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss))

            # if epoch > 4:
            #     early_stopping(vali_loss, self.model, path)
            #     if early_stopping.early_stop:
            #         print("Early stopping")
            #         break
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def vali(self, vali_loader, criterion):
        self.model.eval()
        total_loss = []
        for i, (batch_x, batch_y) in enumerate(vali_loader):
            if self.use_gpu:
                batch_x, batch_y = batch_x.cuda(), batch_y.cuda()

            outputs = self.model(batch_x)

            batch_y = batch_y.float()

            if self.args.single_learning:
                batch_y = batch_y[:, -1, 0]
                outputs = outputs[:, -1, 0]

            if outputs.shape[1] < batch_y.shape[1]:
                batch_y = batch_y[:, -outputs.shape[1]:, :]

            loss = criterion(outputs, batch_y)

            if self.use_gpu:
                loss = loss.cpu().detach().numpy()
            else:
                loss = loss.detach().numpy()

            total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def test(self, setting):
        test_data, test_loader = self._get_data('test')

        self.model.eval()

        preds = []
        trues = []

        for i, (batch_x, batch_y) in enumerate(test_loader):
            if self.use_gpu:
                batch_x, batch_y = batch_x.cuda(), batch_y.cuda()

            outputs = self.model(batch_x)

            batch_y = batch_y.float()

            pred = outputs.detach().cpu().numpy()
            true = batch_y.detach().cpu().numpy()

            preds.append(pred)
            trues.append(true)

        preds = np.array(preds)
        trues = np.array(trues)
        # print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-1])
        # print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        # np.save(folder_path + 'pred.npy', preds)
        # np.save(folder_path + 'true.npy', trues)

        return





