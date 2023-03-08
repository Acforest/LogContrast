import os
import sys
import time
import torch
import random
import logging
import argparse
from datetime import datetime


def load_configs():
    parser = argparse.ArgumentParser()

    ''' Base '''
    parser.add_argument('--log_type', type=str, default='HDFS', choices=['HDFS', 'BGL', 'Thunderbird'],
                        help='The type of log dataset')
    parser.add_argument('--do_train', action='store_true',
                        help='Whether to perform training')
    parser.add_argument('--do_test', action='store_true',
                        help='Whether to perform testing')
    parser.add_argument('--feat_type', type=str, default='both', choices=['semantics', 'logkey', 'both'],
                        help='Feature dimension of log semantics and logkey')
    parser.add_argument('--feat_dim', type=int, default=512,
                        help='Feature dimension of log semantics and logkey')

    ''' Training '''
    parser.add_argument('--train_batch_size', type=int, default=32,
                        help='')
    parser.add_argument('--train_data_dir', type=str, default=r'E:\LogX\output\HDFS\HDFS_train_10000.csv',
                        help='')
    parser.add_argument('--semantic_model', type=str, default='bert', choices=['bert', 'roberta', 'albert'],
                        help='')
    parser.add_argument('--loss_fct', type=str, default='cl', choices=['ce', 'cl'],
                        help='Loss function used in training stage')
    parser.add_argument('--num_epoch', type=int, default=20,
                        help='Number of epochs in training stage')
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='Initial learning rate for optimizer in training stage')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Initial learning rate for optimizer in training stage')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='')
    parser.add_argument('--temperature', type=float, default=0.1,
                        help='')
    parser.add_argument('--save_dir', type=str, default=r'output/HDFS',
                        help='')

    ''' Testing '''
    parser.add_argument('--test_batch_size', type=int, default=32,
                        help='')
    parser.add_argument('--test_data_dir', type=str, default=r'E:\LogX\output\HDFS\HDFS_test_575061.csv',
                        help='')

    ''' Environment '''
    parser.add_argument('--seed', type=int, default=1234,
                        help='Random seed for reproducibility')
    parser.add_argument('--backend', default=False, action='store_true')
    parser.add_argument('--timestamp', type=int, default='{:.0f}{:03}'.format(time.time(), random.randint(0, 999)))
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='')

    args = parser.parse_args()

    args.device = torch.device(args.device)

    args.model_name = f'{args.dataset}_{args.loss_fct}_epoch{args.num_epoch}.pt'
    args.log_name = f'{args.model_name[:-3]}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")[2:]}.log'
    if not os.path.exists('logs'):
        os.mkdir('logs')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.addHandler(logging.FileHandler(os.path.join('logs', args.log_name)))

    return args, logger
