'''
@Author: Conghao Wong
@Date: 2020-05-25 20:14:28
LastEditors: Conghao Wong
LastEditTime: 2020-09-15 16:33:22
@Description: file content
'''

import argparse

import cv2
import numpy as np

from scipy import signal

from tqdm import tqdm
import matplotlib.pyplot as plt

from PrepareTrainData import Agent_Part
from helpmethods import predict_linear_for_person


import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


class GridMap():
    def __init__(self, args, agent:Agent_Part, save=False, save_path='null'):
        self.args = args
        self.agent = agent
        self.pred_original = agent.get_pred_traj()
        self.pred_neighbor = agent.get_pred_traj_neighbor()
        
        self.add_mask = cv2.imread('./mask_circle.png')[:, :, 0]
        # self.add_mask = cv2.imread('./mask_square.png')[:, :, 0]

        self.grid_map = self.create_grid_map(save=save, save_path=save_path)      

    
    def __calculate_grid(self, input_coor):
        new_coor = [
            int(input_coor[0]//self.args.grid_length + self.args.grid_shape_x//2), 
            int(input_coor[1]//self.args.grid_length + self.args.grid_shape_y//2),
        ]
        return new_coor

    def real2grid(self, real_coor):
        return np.stack([self.__calculate_grid(coor) for coor in real_coor])

    def create_grid_map(self, save=False, save_path='null'):
        mmap = np.zeros([self.args.grid_shape_x, self.args.grid_shape_y])
        pred_current_grid = self.real2grid(self.pred_original)

        avoid_person_list = []
        avoid_person_cosine = []
        interest_person_list = []
        interest_person_cosine = []
        pred_grid = dict()
        for index, pred in enumerate(self.pred_neighbor):
            pred_grid[str(index)] = self.real2grid(pred)
            cosine = calculate_cosine(self.pred_original[-1] - self.pred_original[0], pred[-1] - pred[0])
            if cosine >= 0:
                interest_person_list.append(str(index))
                interest_person_cosine.append(cosine)
            else:
                avoid_person_list.append(str(index))
                avoid_person_cosine.append(cosine)
            
        
        # 帧的权重mask, 1～12帧依次从1递减至1/12 + 0.5
        mask = np.minimum(np.stack([(self.args.pred_frames-i)/self.args.pred_frames for i in range(self.args.pred_frames)]).reshape([-1, 1]) + 0.5, 1)
        
        # 将原始预测作为吸引力添加
        mmap = self.add_to_grid(
            pred_current_grid, 
            mmap,
            coe=-1 * np.ones_like(mask),
            add_size=self.args.interest_size,
        )
        
        # 防止碰撞
        for index, cosine in zip(avoid_person_list, avoid_person_cosine):
            mmap = self.add_to_grid(
                pred_grid[index], 
                mmap,
                coe=mask*np.abs(cosine),
                add_size=self.args.avoid_size,
            )

        # 同行者吸引
        for index, cosine in zip(interest_person_list, interest_person_cosine):
            mmap = self.add_to_grid(
                pred_grid[index], 
                mmap,
                coe=-0.2*mask*np.abs(cosine),
                add_size=self.args.avoid_size,
            )
        
        mmap_new = mmap # self.add_to_grid(pred_current_grid, mmap, coe=np.ones([self.args.pred_frames]))
        if save:
            cv2.imwrite(
                save_path,
                127*(mmap_new/mmap_new.max()+1)
            )
        return mmap
    
    def add_to_grid(self, coor_list, gridmap, coe=1, add_size=1, interp=True, replace=True):
        mask = cv2.resize(self.add_mask, (2*add_size, 2*add_size))
        gridmap_c = gridmap.copy()
        coor_list_new = []  # 删除重复项目
        for coor in coor_list:
            if not coor.tolist() in coor_list_new:
                coor_list_new.append(coor.tolist())
        
        # coor_list_new = np.stack(coor_list_new)
        if interp:
            coe_new = []
            for i in range(1, len(coor_list_new)):
                if abs(coor_list_new[i][0] - coor_list_new[i-1][0]) + abs(coor_list_new[i][1] - coor_list_new[i-1][1]) <= 1:
                    continue

                for inter_x in range(1, abs(coor_list_new[i][0] - coor_list_new[i-1][0])):
                    coor_list_new.append([coor_list_new[i-1][0]+inter_x, coor_list_new[i-1][1]])
                    coe_new.append(coe[i-1])

                for inter_y in range(1, abs(coor_list_new[i][1] - coor_list_new[i-1][1])):
                    coor_list_new.append([coor_list_new[i][0], coor_list_new[i-1][1]+inter_y])
                    coe_new.append(coe[i-1])

            if len(coe_new):
                coe = np.concatenate([coe, np.stack(coe_new)])
                    

        for coor, coe_c in zip(coor_list_new, coe):
            gridmap_c[coor[0]-add_size:coor[0]+add_size, coor[1]-add_size:coor[1]+add_size] = coe_c*mask + gridmap_c[coor[0]-add_size:coor[0]+add_size, coor[1]-add_size:coor[1]+add_size]
        return gridmap_c

    def find_linear_neighbor(self, index):
        index1 = np.floor(index)
        index2 = index1 + 1
        percent = index - index1
        return index1.astype(np.int), index2.astype(np.int), percent
        
    def linear_interp(self, value1, value2, percent):
        return value1 + (value2 - value1) * percent.reshape([-1, 1])
        
    def length_refine(self, input_traj, original_traj):
        original_length = np.linalg.norm(original_traj[-1, :] - original_traj[0, :])
        current_length = np.linalg.norm(input_traj[-1, :] - input_traj[0, :])
        
        if current_length >= original_length:
            fix_index = [i*original_length/current_length for i in range(self.args.pred_frames)]
            index1, index2, percent = self.find_linear_neighbor(fix_index)
            linear_fix = self.linear_interp(input_traj[index1], input_traj[index2], percent)
        
        else:
            # print('!')
            fix_index = [i*original_length/current_length for i in range(self.args.pred_frames)]
            max_index = np.ceil(fix_index[-1]).astype(np.int)
            input_traj_expand = np.concatenate([
                input_traj,
                predict_linear_for_person(input_traj, max_index+1)[len(input_traj):, :]
            ], axis=0)
            index1, index2, percent = self.find_linear_neighbor(fix_index)
            linear_fix = self.linear_interp(input_traj_expand[index1], input_traj_expand[index2], percent)
        
        return linear_fix 

    def refine_model(self, epochs=30):
        prev_result = self.pred_original
        grid_map = self.grid_map
        
        # 原预测静止不动的不需要微调
        if calculate_length(prev_result[-1] - prev_result[1]) <= 1.0:
            return prev_result
            
        for epoch in range(epochs):
            result = prev_result
            input_traj_grid = (self.real2grid(result) + 1.0).astype(np.int)

            diff_x = grid_map[1:, 1:] - grid_map[:-1, 1:]
            diff_y = grid_map[1:, 1:] - grid_map[1:, :-1]

            dx_current = diff_x[input_traj_grid.T[0], input_traj_grid.T[1]]
            dy_current = diff_y[input_traj_grid.T[0], input_traj_grid.T[1]]

            x_bias = dx_current * 0.001
            y_bias = dy_current * 0.001

            prev_result = np.stack([
                result.T[0] - x_bias,
                result.T[1] - y_bias,
            ]).T

        delta = np.minimum(prev_result - self.pred_original, self.args.max_refine)
        coe = 0.7 * np.stack([i/(self.args.pred_frames-1) for i in range(self.args.pred_frames)]).reshape([-1, 1])
    
        # print('!')
        # np.savetxt('res.txt', prev_result)
        # np.savetxt('inp.txt', input_traj)
        social_fix = self.pred_original + coe * delta
        length_fix = self.length_refine(social_fix, self.pred_original)
        return length_fix


def calculate_cosine(vec1, vec2):
    """
    两个输入均为表示方向的向量, shape=[2]
    """
    length1 = np.linalg.norm(vec1)
    length2 = np.linalg.norm(vec2)
    return np.sum(vec1 * vec2) / (length1 * length2)


def calculate_length(vec1):
    """
    表示方向的向量, shape=[2]
    """
    length1 = np.linalg.norm(vec1)
    return length1


def SocialRefine_one(agent:Agent_Part, args, epochs=10, save=False, save_path='null'):
    a = GridMap(args, agent, save=save, save_path=save_path)
    traj_refine = a.refine_model(epochs=epochs)
    return traj_refine
