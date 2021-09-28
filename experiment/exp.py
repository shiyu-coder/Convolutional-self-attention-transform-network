import argparse
import torch
from exp_model import Exp_model


parser = argparse.ArgumentParser(description='CSATNet: Convolutional Self-attention Transform Network')

parser.add_argument('--model', type=str, required=True, default='CSATNet', help='model for experiment, options: [CSATNet]')
parser.add_argument('--data', type=str, required=True, default='ADDataset', help='dataset for experiment, options: [ADDataset]')
parser.add_argument('--root_path', type=str, default='./data/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='DataSet', help='data file')
parser.add_argument('--label_path', type=str, default='ADLabel.csv', help='label data file')

parser.add_argument('--num_hiddens', type=int, default=128, help='the length of input of decoder')
parser.add_argument('--num_heads', type=int, default=4, help='the num of heads of multi-head self-attention')
parser.add_argument('--seq_len', type=int, default=4, help='the length of history sequence')
parser.add_argument('--cnn_layer1_num', type=int, default=2, help='the num of layer in the first part of CNN')
parser.add_argument('--cnn_layer2_num', type=int, default=1, help='the num of layer in the second part of CNN')
parser.add_argument('--enc_layer_num', type=int, default=2, help='the num of self-attention layer in encoder')
parser.add_argument('--dec_layer_num', type=int, default=2, help='the num of self-attention layer in decoder')
parser.add_argument('--input_size', type=tuple, default=(88, 200), help='input size(image size)')
parser.add_argument('--label_size', type=int, default=1, help='the num of label for one piece of input')
parser.add_argument('--drop_out', type=float, default=0.01, help='drop out probability')
parser.add_argument('--min_output_size', type=int, default=32, help='the minimize size of output size of encoder')

parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=6, help='train epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--loss', type=str, default='mse',help='loss function')
parser.add_argument('--lradj', type=str, default='type1',help='adjust learning rate')
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')

args = parser.parse_args()

args.use_gpu = True if torch.cuda.is_available() else False

print('Args in experiment:')
print(args)

for ii in range(args.itr):
    setting = '{}_{}_num_hiddens{}_num_heads{}_cnn_layer1_num{}_cnn_layer2_num{}_enc_layer_num{}_input_size{}_label_size{}_drop_out{}_min_output_size{}_{}'.format(
        args.model, args.data, args.num_hiddens, args.num_heads, args.seq_len, args.cnn_layer1_num, args.cnn_layer2_num,
        args.enc_layer_num, args.dec_layer_num, args.input_size, args.label_size, args.drop_out, args.min_output_size,
        ii)

    exp = Exp_model(args)
    print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
    exp.train(setting)

    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    exp.test(setting)

    torch.cuda.empty_cache()





