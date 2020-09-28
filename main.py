'''
@Author: ConghaoWong
@Date: 2019-12-20 09:38:24
LastEditors: Conghao Wong
LastEditTime: 2020-09-16 16:31:38
@Description: main of Erina
'''
import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'        # 去除TF输出
import time

import numpy as np
import tensorflow as tf
from matplotlib.axes._axes import _log as matplotlib_axes_logger

from helpmethods import dir_check
from models import BGM, Linear
from PrepareTrainData import DataManager

matplotlib_axes_logger.setLevel('ERROR')        # 画图警告
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"       # kNN问题
TIME = time.strftime('%Y%m%d-%H%M%S',time.localtime(time.time()))


def get_parser():
    parser = argparse.ArgumentParser(description='linear')

    # environment settrings and test options
    parser.add_argument('--gpu', type=int, default=2)
    parser.add_argument('--load', type=str, default='null')
    parser.add_argument('--draw_results', type=int, default=False)
    parser.add_argument('--save_base_dir', type=str, default='./logs')
    parser.add_argument('--log_dir', type=str, default='null')
    parser.add_argument('--sr_enable', type=int, default=False)

    # model basic settings
    parser.add_argument('--obs_frames', type=int, default=8)
    parser.add_argument('--pred_frames', type=int, default=12)
    parser.add_argument('--test_set', type=int, default=2)
    parser.add_argument('--save_best', type=int, default=True)

    # training data settings
    parser.add_argument('--train_type', type=str, default='all')        
    # 'one': 使用一个数据集按照分割训练集训练
    # 'all': 使用除测试外的所有数据集训练
    parser.add_argument('--train_base', type=str, default='agent')
    # parser.add_argument('--frame', type=str, default='01234567')
    parser.add_argument('--train_percent', type=float, default=[0.0], nargs='+')     # 用于训练数据的百分比, 0表示全部
    parser.add_argument('--step', type=int, default=4)                  # 数据集滑动窗步长
    parser.add_argument('--reverse', type=int, default=False)            # 按时间轴翻转训练数据
    parser.add_argument('--add_noise', type=int, default=False)         # 训练数据添加噪声
    parser.add_argument('--rotate', type=int, default=False)            # 旋转训练数据(起始点保持不变)
    parser.add_argument('--normalization', type=int, default=False)

    # test settings when training
    parser.add_argument('--test', type=int, default=True)
    parser.add_argument('--start_test_percent', type=float, default=0.0)    
    parser.add_argument('--test_step', type=int, default=3)     # 训练时每test_step个epoch，test一次
    
    # training settings
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=500)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=1e-3)
   
    # save/load settings
    parser.add_argument('--model_name', type=str, default='model')
    parser.add_argument('--save_model', type=int, default=True)
    parser.add_argument('--save_per_step', type=bool, default=True)

    # Linear args
    parser.add_argument('--diff_weights', type=float, default=0.95)

    # BGM args
    parser.add_argument('--model', type=str, default='bgm')

    # Social args
    # parser.add_argument('--max_neighbor', type=int, default=6)
    parser.add_argument('--init_position', type=float, default=20)
    # parser.add_argument('--future_interaction', type=int, default=True)
    parser.add_argument('--calculate_social', type=int, default=False)

    # SR args
    parser.add_argument('--grid_shape_x', type=int, default=700)
    parser.add_argument('--grid_shape_y', type=int, default=700)
    parser.add_argument('--grid_length', type=float, default=0.1)   # 网格的真实长度
    parser.add_argument('--avoid_size', type=int, default=15)   # 主动避让的半径网格尺寸
    parser.add_argument('--interest_size', type=int, default=20)   # 原本感兴趣的预测区域
    # parser.add_argument('--social_size', type=int, default=1)   # 互不侵犯的半径网格尺寸
    parser.add_argument('--max_refine', type=float, default=0.8)   # 最大修正尺寸

    # Guidance Map args
    parser.add_argument('--gridmapsize', type=int, default=32)

    return parser


def gpu_config(args):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


def load_args(save_args_path, current_args):
    save_args = np.load(current_args.load+'args.npy', allow_pickle=True).item()
    save_args.gpu = current_args.gpu
    save_args.load = current_args.load
    save_args.draw_results = current_args.draw_results
    save_args.sr_enable = current_args.sr_enable
    return save_args


def main():
    args = get_parser().parse_args()
    # args.frame = [int(i) for i in args.frame]
    
    gpu_config(args)
    if args.load == 'null':
        inputs = DataManager(args).train_info
        
    else:
        inputs = 0
        args = load_args(args.load+'args.npy', args)
    
    if args.log_dir == 'null':
        log_dir_current = TIME + args.model_name + args.model + str(args.test_set)
        args.log_dir = os.path.join(dir_check(args.save_base_dir), log_dir_current)
    else:
        args.log_dir = dir_check(args.log_dir)
    
    if args.model == 'bgm':
        model = BGM
    elif args.model == 'linear':
        model = Linear

    model(train_info=inputs, args=args).run_commands()


if __name__ == "__main__":
    main()
