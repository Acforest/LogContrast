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
                        help='The type of log dataset ["HDFS", "BGL", "Thunderbird"] (default: HDFS)')
    parser.add_argument('--model_dir', type=str, default='./models/',
                        help='The directory where LogContrast model checkpoints will be loaded or saved (default: "./models/")')
    parser.add_argument('--model_name', type=str, default='',
                        help='The name of LogContrast model (default: "")')
    parser.add_argument('--semantic_model_name', type=str, default='albert',
                        help='The name of LogContrast semantic model ["bert", "roberta", "albert"] (default: "albert")')
    parser.add_argument('--feat_type', type=str, default='both', choices=['semantics', 'logkey', 'both'],
                        help='Feature dimension of log semantics and logkey ["semantics", "logkey", "both"] (default: "both")')
    parser.add_argument('--feat_dim', type=int, default=512,
                        help='Feature dimension of log semantics and logkey (default: 512)')
    parser.add_argument('--vocab_size', type=int, default=2000,
                        help='Vocaburary size of different kind of logkeys (default: 2000)')

    ''' Training '''
    parser.add_argument('--do_train', action='store_true', default=True,
                        help='Whether to perform training (default: True)')
    parser.add_argument('--train_batch_size', type=int, default=32,
                        help='Batch size for training (default: 32)')
    parser.add_argument('--train_data_dir', type=str, default='./output/HDFS/HDFS_train_10000.csv',
                        help='The directory of training data (default: "./output/HDFS/HDFS_train_10000.csv")')
    parser.add_argument('--loss_fct', type=str, default='cl', choices=['ce', 'cl'],
                        help='Loss function used in training stage ["cl", "ce"] (default: "cl")')
    parser.add_argument('--num_epoch', type=int, default=20,
                        help='Number of epochs in training stage (default: 20)')
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='Initial learning rate for optimizer in training stage (default: 1e-5)')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay for optimizer in training stage (default: 0.01)')
    parser.add_argument('--lambda_c', type=float, default=0.1,
                        help='Weight hyperparameter of contrastive loss (default: 0.1)')
    parser.add_argument('--temperature', type=float, default=0.5,
                        help='Temperature hyperparameter of contrastive loss (default: 0.5)')

    ''' Testing '''
    parser.add_argument('--do_test', action='store_true', default=True,
                        help='Whether to perform testing (default: True)')
    parser.add_argument('--test_batch_size', type=int, default=32,
                        help='Batch size for testing (default: 32)')
    parser.add_argument('--test_data_dir', type=str, default='./output/HDFS/HDFS_test_575061.csv',
                        help='The directory of testing data (default: "./output/HDFS/HDFS_test_575061.csv")')

    ''' Environment '''
    parser.add_argument('--seed', type=int, default=1234,
                        help='Random seed for reproducibility (default: 1234)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='The device for training or testing (default: "cuda" if available else "cpu")')

    args = parser.parse_args()

    args.device = torch.device(args.device)

    if args.model_name == '' and args.do_train:
        args.model_name = f'{args.log_type}_{args.loss_fct}_epoch{args.num_epoch}.pt'
    args.log_name = f'{args.model_name[:-3]}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")[2:]}.log'
    if not os.path.exists('logs'):
        os.mkdir('logs')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.addHandler(logging.FileHandler(os.path.join('logs', args.log_name)))

    return args, logger
