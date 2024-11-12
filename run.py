import argparse
import os
import torch
from exp.exp_long_term_forecasting_ray_tune import Exp_Long_Term_Forecast
# from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from exp.exp_imputation import Exp_Imputation
from exp.exp_short_term_forecasting import Exp_Short_Term_Forecast
from exp.exp_anomaly_detection import Exp_Anomaly_Detection
from exp.exp_classification import Exp_Classification
from utils.print_args import print_args
import random
import numpy as np

from ray import tune
from ray import train
from ray.tune.search.optuna import OptunaSearch
from ray.tune.search.bayesopt import BayesOptSearch
from ray.tune.search.ax import AxSearch
from ray.tune.search.basic_variant import BasicVariantGenerator
import wandb

#wandb
from ray.tune.logger import DEFAULT_LOGGERS
from ray.air.integrations.wandb import WandbLoggerCallback, setup_wandb

if __name__ == '__main__':
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='TimesNet')

    # basic config
    parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast',
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='Autoformer',
                        help='model name, options: [Autoformer, Transformer, TimesNet]')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

    # inputation task
    parser.add_argument('--mask_rate', type=float, default=0.25, help='mask ratio')

    # anomaly detection task
    parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='prior anomaly ratio (%)')

    # model define
    parser.add_argument('--expand', type=int, default=2, help='expansion factor for Mamba')
    parser.add_argument('--d_conv', type=int, default=4, help='conv kernel size for Mamba')
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--channel_independence', type=int, default=1,
                        help='0: channel dependence 1: channel independence for FreTS model')
    parser.add_argument('--decomp_method', type=str, default='moving_avg',
                        help='method of series decompsition, only support moving_avg or dft_decomp')
    parser.add_argument('--use_norm', type=int, default=1, help='whether to use normalize; True 1 False 0')
    parser.add_argument('--down_sampling_layers', type=int, default=0, help='num of down sampling layers')
    parser.add_argument('--down_sampling_window', type=int, default=1, help='down sampling window size')
    parser.add_argument('--down_sampling_method', type=str, default=None,
                        help='down sampling method, only support avg, max, conv')
    parser.add_argument('--seg_len', type=int, default=48,
                        help='the length of segmen-wise iteration of SegRNN')

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
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    # de-stationary projector params
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                        help='hidden layer dimensions of projector (List)')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')

    # metrics (dtw)
    parser.add_argument('--use_dtw', type=bool, default=False, 
                        help='the controller of using dtw metric (dtw is time consuming, not suggested unless necessary)')
    
    # Augmentation
    parser.add_argument('--augmentation_ratio', type=int, default=0, help="How many times to augment")
    parser.add_argument('--seed', type=int, default=2, help="Randomization seed")
    parser.add_argument('--jitter', default=False, action="store_true", help="Jitter preset augmentation")
    parser.add_argument('--scaling', default=False, action="store_true", help="Scaling preset augmentation")
    parser.add_argument('--permutation', default=False, action="store_true", help="Equal Length Permutation preset augmentation")
    parser.add_argument('--randompermutation', default=False, action="store_true", help="Random Length Permutation preset augmentation")
    parser.add_argument('--magwarp', default=False, action="store_true", help="Magnitude warp preset augmentation")
    parser.add_argument('--timewarp', default=False, action="store_true", help="Time warp preset augmentation")
    parser.add_argument('--windowslice', default=False, action="store_true", help="Window slice preset augmentation")
    parser.add_argument('--windowwarp', default=False, action="store_true", help="Window warp preset augmentation")
    parser.add_argument('--rotation', default=False, action="store_true", help="Rotation preset augmentation")
    parser.add_argument('--spawner', default=False, action="store_true", help="SPAWNER preset augmentation")
    parser.add_argument('--dtwwarp', default=False, action="store_true", help="DTW warp preset augmentation")
    parser.add_argument('--shapedtwwarp', default=False, action="store_true", help="Shape DTW warp preset augmentation")
    parser.add_argument('--wdba', default=False, action="store_true", help="Weighted DBA preset augmentation")
    parser.add_argument('--discdtw', default=False, action="store_true", help="Discrimitive DTW warp preset augmentation")
    parser.add_argument('--discsdtw', default=False, action="store_true", help="Discrimitive shapeDTW warp preset augmentation")
    #parser.add_argument('--extra_tag', type=str, default="", help="Anything extra")
    parser.add_argument('--extra_tag', type=str, default="", help="Anything extra")

    args = parser.parse_args()
    # args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    args.use_gpu = True if torch.cuda.is_available() else False

    print(torch.cuda.is_available())

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print_args(args)
    # Define the hyperparameter search space
    def getDict(args):
        return args.__dict__

    ###search space
    param_space = getDict(args)
    param_space_update = {
        # 'seq_len': tune.choice([64, 128, 256]),
        # 'label_len': tune.choice([16, 32, 64]),
        # 'pred_len': tune.choice([24, 48, 96]),

        ## model define
        'd_ff': tune.choice([16,32,64,128,256,512,1024,2048,4096]),
        'd_layers': tune.choice([1, 2, 3]),
        'd_model': tune.choice([16,32,64,128,256,512,1024,2048,4096]),
        'e_layers': tune.choice([1, 2, 3]),
        'factor': tune.choice([1, 2, 3, 4]),
        # 'd_model': tune.grid_search(list(args.d_model)),
        # 'd_model': args.d_model,
        'n_heads': tune.choice([2, 4, 8, 16]),

        ## optimization
        'batch_size': tune.choice([4, 16, 32, 64, 128, 256]),
        # 'itr': tune.choice([1, 2, 3, 4]),
        'learning_rate': tune.choice([0.00001, 0.0001, 0.001]),
        #'num_workers': tune.choice([8, 9, 10, 11, 12]),
        #'patience': tune.choice([1, 2, 3, 5, 7, 9, 11]),
        'train_epochs': tune.choice([1,2,3,4,5,6,7,8,9,10,11]),
        # 'e_layers': tune.choice([2, 4, 6]),
        # 'd_layers': tune.choice([1, 2, 3]),
        # 'd_ff': tune.choice(args.d_ff),
        # 'factor': tune.choice([3, 5, 7]),
        # 'embed': tune.choice(['fixed', 'learnable']),
        # 'distil': tune.choice([True, False])
    }

    ### Search algo
    algo = OptunaSearch()
    if param_space['extra_tag']=='OptunaSearch':
        algo = OptunaSearch()
    elif param_space['extra_tag']=='BayesOptSearch':
        algo = BayesOptSearch()
    elif param_space['extra_tag']=='AxSearch':
        algo = AxSearch()
    elif param_space['extra_tag']=='BasicVariantGenerator':
        #algo = BasicVariantGenerator()
        ##update ssearch space
        param_space_update = {
            ## model define
            'd_ff': tune.grid_search([16,32,64,128,256,512,1024,2048,4096]),
            'd_layers': tune.grid_search([1, 2, 3]),
            'd_model': tune.grid_search([16,32,64,128,256,512,1024,2048,4096]),
            'e_layers': tune.grid_search([1, 2, 3]),
            'factor': tune.grid_search([1, 2, 3, 4]),
            'n_heads': tune.grid_search([2, 4, 8, 16]),
            ## optimization
            'batch_size': tune.grid_search([4, 16, 32, 64, 128, 256]),
            # 'itr': tune.choice([1, 2, 3, 4]),
            'learning_rate': tune.grid_search([0.00001, 0.0001, 0.001]),
            'train_epochs': tune.grid_search([1,2,3,4,5,6,7,8,9,10,11]),
        }
    ### config
    if param_space['extra_tag']=='BasicVariantGenerator':
        tune_config=tune.TuneConfig(
            metric="vali_loss",
            mode="min",
            num_samples=20,
        )
    else:
        tune_config=tune.TuneConfig(
            metric="vali_loss",
            mode="min",
            search_alg=algo,
            num_samples=20,
        )
    param_space.update(param_space_update)

    def train_fn(param_space):
        # wandb = setup_wandb(param_space)
        print('is cuda avaiable ', torch.cuda.is_available())

        if param_space['task_name'] == 'long_term_forecast':
            Exp = Exp_Long_Term_Forecast
        elif param_space['task_name'] == 'short_term_forecast':
            Exp = Exp_Short_Term_Forecast
        elif param_space['task_name'] == 'imputation':
            Exp = Exp_Imputation
        elif param_space['task_name'] == 'anomaly_detection':
            Exp = Exp_Anomaly_Detection
        elif param_space['task_name'] == 'classification':
            Exp = Exp_Classification
        else:
            Exp = Exp_Long_Term_Forecast
        # if args.task_name == 'long_term_forecast':
        #     Exp = Exp_Long_Term_Forecast
        # elif args.task_name == 'short_term_forecast':
        #     Exp = Exp_Short_Term_Forecast
        # elif args.task_name == 'imputation':
        #     Exp = Exp_Imputation
        # elif args.task_name == 'anomaly_detection':
        #     Exp = Exp_Anomaly_Detection
        # elif args.task_name == 'classification':
        #     Exp = Exp_Classification
        # else:
        #     Exp = Exp_Long_Term_Forecast

        if param_space['is_training']:
            for ii in range(param_space['itr']):
                # setting record of experiments
                setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_expand{}_dc{}_fc{}_eb{}_dt{}_{}_{}'.format(
                    param_space['task_name'],
                    param_space['model_id'],
                    param_space['model'],
                    param_space['data'],
                    param_space['features'],
                    param_space['seq_len'],
                    param_space['label_len'],
                    param_space['pred_len'],
                    param_space['d_model'],
                    param_space['n_heads'],
                    param_space['e_layers'],
                    param_space['d_layers'],
                    param_space['d_ff'],
                    param_space['expand'],
                    param_space['d_conv'],
                    param_space['factor'],
                    param_space['embed'],
                    param_space['distil'],
                    param_space['des'], ii)

                print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
                exp = Exp(param_space,setting)  # set
                exp.train(setting)

                print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.test(setting)
                torch.cuda.empty_cache()
        else:
            ii = 0
            setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_expand{}_dc{}_fc{}_eb{}_dt{}_{}_{}'.format(
                param_space['task_name'],
                param_space['model_id'],
                param_space['model'],
                param_space['data'],
                param_space['features'],
                param_space['seq_len'],
                param_space['label_len'],
                param_space['pred_len'],
                param_space['d_model'],
                param_space['n_heads'],
                param_space['e_layers'],
                param_space['d_layers'],
                param_space['d_ff'],
                param_space['expand'],
                param_space['d_conv'],
                param_space['factor'],
                param_space['embed'],
                param_space['distil'],
                param_space['des'], ii)

            exp = Exp(param_space,setting)  # set experiments
            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting, test=1)
            torch.cuda.empty_cache()


    trainable = tune.with_resources(
        tune.with_parameters(train_fn),
        {"gpu": 1},
    )
#The BasicVariantGenerator is used per default if no search algorithm is passed to Tuner.
    tuner = tune.Tuner(
        trainable,
        tune_config=tune_config,
        param_space=param_space,
    )
    results = tuner.fit()
    print("Best config is:", results.get_best_result().config)
    wandb.finish()