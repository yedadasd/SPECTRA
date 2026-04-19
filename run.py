import argparse
import os
import torch
from exp.exp_forecasting import Exp_Forecast
from utils.print_args import print_args
import random
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Karma')
    
    # random seed
    parser.add_argument('--random_seed', type=int, default=2024, help='random seed')
    
    
    # basic config
    parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast',
                        help='task name, options:[long_term_forecast, short_term_forecast]')
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='Mamba',
                        help='model name, options: [Mamba, Transformer')


    # data loader
    parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./dataset/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    
    parser.add_argument('--features', type=str, default='M', help='forecasting task, options:[M, S, MS]; '
            'M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, '
                             'b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--step', type=int, default=1, help='step size for data sampling')



    # forecasting
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    # TODO: for what?

    # models
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--num_channels', type=list, default=[25,25,25,25], help='number of channels for CNN')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--expand', type=int, default=2, help='expansion factor for Mamba')
    parser.add_argument('--d_state', type=int, default=32, help='feature dimension (N) of Mamba')
    parser.add_argument('--d_conv', type=int, default=4, help='1dConv kernel size for Mamba')
    parser.add_argument('--embed_dim', type=int, default=128, help='Karma decomposition embedding dimension')
    parser.add_argument('--wavelet_type', type=str, default='db4', help='htfd')
    parser.add_argument('--cross_attn_heads', type=int, default=4, help='number of heads for cross-attention in MTPD')
    # probabilistic forecasting
    import ast
    def str_list(x):
        return ast.literal_eval(x)
    parser.add_argument('--quantiles', type=str_list, default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], help='quantiles to predict')
    parser.add_argument('--top_k', type=int, default=5, help='for fft')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--channel_independence', type=int, default=1,
                        help='0: channel dependence 1: channel independence for FreTS model')
    parser.add_argument('--use_norm', type=int, default=1, help='whether to use normalize; True 1 False 0')
    parser.add_argument('--norm_method', type=str, default='NS',help='Norm method to use')
    parser.add_argument('--patch_len', type=int, default=16, help='patch length')
    parser.add_argument('--stride', type=int, default=8, help='stride for patching')
    
    # lossf
    parser.add_argument('--loss_type', type=str, default='quantileLoss', help='loss function 4 probabilistic forecasting')
    parser.add_argument('--loss_alpha', type=float, default=0.1, help='weight of frequency domain loss in SpectralQuantileLoss')
    parser.add_argument('--fft_type', type=str, default='rfft', help='type of FFT to use in SpectralQuantileLoss, options: [rfft, fft]')
    

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--pct_start', type=float, default=0.3, help='pct_start')
    parser.add_argument('--use_amp', action='store_true', default=False, help='use automatic mixed precision training')

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', default=True, help='use multiple gpus')
    parser.add_argument('--devices', type=str, default='0,1', help='device ids of multile gpus')


    args = parser.parse_args()
    
    # gpu
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        print(args.device_ids)
        args.gpu = args.device_ids[0]
        
    # random seed
    print('Set random seed to {}'.format(args.random_seed))
    random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    if torch.cuda.is_available():
        # For current GPU
        torch.cuda.manual_seed(args.random_seed)
        # For all GPUs, if you are using more than one
        torch.cuda.manual_seed_all(args.random_seed)
        
    print('Args in experiment:')
    # print_args(args)
    print(args)

    Exp = Exp_Forecast

    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            exp = Exp(args)  # set experiments
            setting = '{}_{}_{}_{}_{}_{}_{}_{}'.format(
                args.task_name,
                args.model_id,
                args.model,
                args.data,
                args.seq_len,
                args.pred_len,
                
                args.des, 
                ii)

            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)
            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = '{}_{}_{}_{}_{}_{}_{}_{}'.format(
                args.task_name,
                args.model_id,
                args.model,
                args.data,
                args.seq_len,
                args.pred_len,
                args.des, ii)

        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()
