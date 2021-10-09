import argparse
import torch
from experiment.exp_model import Exp_model


parser = argparse.ArgumentParser(description='CSATNet: Convolutional Self-attention Transform Network')

parser.add_argument('--model', type=str, default='CSATNet_v2',
                    help='model for experiment, options: [CSATNet, CSATNet_v2, SACNN, FSACNN, PSACNN, CNN, NVIDIA_ORIGIN]')
parser.add_argument('--data', type=str, default='ADHDataset',
                    help='dataset for experiment, options: [ADDataset, ADHDataset]')
parser.add_argument('--data_path', type=str, default='../test_dataset', help='data file')
parser.add_argument('--label_path', type=str, default='../ADLabel.csv', help='label data file')
parser.add_argument('--multitask', type=bool, default=False, help='multi task learning')
parser.add_argument('--single_learning', type=bool, default=False,
                    help='Only the last bit predicted by the model is optimized')

parser.add_argument('--num_hiddens', type=int, default=128, help='the length of input of decoder')
parser.add_argument('--num_heads', type=int, default=4, help='the num of heads of multi-head self-attention')
parser.add_argument('--seq_len', type=int, default=6, help='the length of history sequence')
parser.add_argument('--cnn_layer1_num', type=int, default=3, help='the num of layer in the first part of CNN')
parser.add_argument('--cnn_layer2_num', type=int, default=2, help='the num of layer in the second part of CNN')
parser.add_argument('--enc_layer_num', type=int, default=3, help='the num of self-attention layer in encoder')
parser.add_argument('--dec_layer_num', type=int, default=3, help='the num of self-attention layer in decoder')
parser.add_argument('--vector_num', type=int, default=32, help='eigenvector length of convolution and timing output')
parser.add_argument('--input_size', type=tuple, default=(180, 320), help='input size(image size)')
parser.add_argument('--label_size', type=int, default=1, help='the num of label for one piece of input')
parser.add_argument('--drop_out', type=float, default=0.05, help='drop out probability')
parser.add_argument('--min_output_size', type=int, default=32, help='the minimize size of output size of encoder')
parser.add_argument('--attention', type=bool, default=False, help='use Parallel Self-attention Inception')
parser.add_argument('--channel_expansion', type=bool, default=True, help='Input channel number expansion')

parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--epoch', type=int, default=10, help='train epochs')
parser.add_argument('--batch_size', type=int, default=6, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0004, help='optimizer learning rate')
parser.add_argument('--loss', type=str, default='unbalancedLoss',
                    help='loss function. options: [mse, mae, smoothL1, steeringLoss, unbalancedLoss]')
parser.add_argument('--lr_adj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
parser.add_argument('--tensorboard', type=bool, default=True, help='use tensorboard')

args = parser.parse_args()

args.use_gpu = True if torch.cuda.is_available() else False

print('Args in experiment:')
print(args)

for ii in range(args.itr):
    setting = '{}-{}-{}-nhi{}-nhe{}-sl{}-cl1n{}-cl2n{}-eln{}-dln{}-vn{}-is{}-ls{}-do{}-mos{}-a{}-ce{}-{}'.format(
        args.model, args.data, args.loss, args.num_hiddens, args.num_heads, args.seq_len, args.cnn_layer1_num, args.cnn_layer2_num,
        args.enc_layer_num, args.dec_layer_num, args.vector_num, args.input_size, args.label_size, args.drop_out, args.min_output_size,
        args.attention, args.channel_expansion, ii)

    exp = Exp_model(args)
    print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
    exp.train(setting)

    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    exp.test(setting)

    torch.cuda.empty_cache()





