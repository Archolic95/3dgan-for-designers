'''
main.py

Welcome, this is the entrance to 3dgan
'''

import argparse
from trainer import trainer
import torch

from tester import tester
import params

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main():
    #add arguments
    parser = argparse.ArgumentParser()

    # loggings parameters
    parser.add_argument('--logs', type=str, default=None, help='logs by tensorboardX')
    parser.add_argument('--local_test', type=str2bool, default=False, help='local test verbose')
    parser.add_argument('--model_name', type=str, default="dcgan", help='model name for saving')
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
    parser.add_argument('--model_save_step', type=int, default=5, help='steps to save model')
    parser.add_argument('--cube_len', type=int, default=64, help='edge length for voxel cube')
    parser.add_argument('--channel', type=int, default=1, help='number of channels')
    parser.add_argument('--data_dir', type=str, default='../volumetric_data/', help='directory for 3d data')
    parser.add_argument('--output_dir', type=str, default='../outputs', help='sub-directory for output')
    parser.add_argument('--cloud_tpu', type=bool, default=False, help='whether using colab for training')
    parser.add_argument('--test', type=str2bool, default=False, help='call tester.py')
    parser.add_argument('--use_visdom', type=str2bool, default=False, help='visualization by visdom')
    args = parser.parse_args()

    # list params and print
    params.cloud_tpu=args.cloud_tpu
    params.epochs = args.epochs
    params.batch_size = args.batch_size
    params.cube_len = args.cube_len
    params.data_dir=args.data_dir
    params.output_dir = args.output_dir
    params.model_save_step=args.model_save_step
    params.channel = args.channel
    
    params.print_params()

    # run program
    if args.test == False:
        trainer(args)
    else:
        tester(args)


if __name__ == '__main__':
    main()

    
