'''
Author: Conghao Wong
Date: 2020-08-20 23:05:05
LastEditors: Conghao Wong
LastEditTime: 2020-09-15 09:36:07
Description: file content
'''

import os

import cv2
import numpy as np
from tqdm import tqdm

from helpmethods import dir_check
from PrepareTrainData import Agent_Part

OBS_IMAGE = './vis_pngs/obs.png'
GT_IMAGE = './vis_pngs/gt.png'
PRED_IMAGE = './vis_pngs/pred.png'


class TrajVisual():
    def __init__(self, save_base_path, verbose=False, draw_neighbors=False, social_refine=False):
        self.video_path = [
            './videos/eth.avi',
            './videos/hotel.avi',
            './videos/zara1.mp4',
            './videos/zara2.avi',
            'null',
        ]

        self.weights = [
            [np.array([
                [2.8128700e-02, 2.0091900e-03, -4.6693600e+00],
                [8.0625700e-04, 2.5195500e-02, -5.0608800e+00],
                [3.4555400e-04, 9.2512200e-05, 4.6255300e-01],
            ]), 0.65, 225, 0.6, 160],
            [np.array([
                [-1.5966000e-03, 1.1632400e-02, -5.3951400e+00],
                [1.1048200e-02, 6.6958900e-04, -3.3295300e+00],
                [1.1190700e-04, 1.3617400e-05, 5.4276600e-01],
            ]), 0.54, 470, 0.54, 300],
            [-42.54748107, 580.5664891, 47.29369894, 3.196071003],
            [-42.54748107, 635.5664891, 47.29369894, -16.196071003],
            [],
        ]   # [Wx, bx, Wy, by]

        self.paras = [
            [6, 25],
            [10, 25],
            [10, 25],
            [10, 25],
            [],
        ]  # sample_rate, frame_rate
        
        self.verbose = verbose
        self.draw_neighbors = draw_neighbors
        self.save_base_path = save_base_path
        self.social_refine = social_refine
        
    def visual(self, agents, dataset):
        vc = cv2.VideoCapture(self.video_path[dataset])
        paras = self.paras[dataset]
        weights = self.weights[dataset]

        if self.verbose:
            itera = enumerate(tqdm(agents, desc='Save prediction figs...'))
        else:
            itera = enumerate(agents)

        dir_check(self.save_base_path)
        save_format = os.path.join(dir_check(os.path.join(self.save_base_path, 'VisualTrajs')), '{}.{}')

        for index, agent in itera:            
            self.draw(agent, vc, paras, weights, save_format.format(index, 'jpg'), draw_neighbors=self.draw_neighbors)
    
    def real2pixel(self, real_pos, weights):
        if len(weights) == 4:
            return np.column_stack([
                weights[2] * real_pos.T[1] + weights[3],
                weights[0] * real_pos.T[0] + weights[1],
            ]).astype(np.int)
        else:
            H = weights[0]
            real = np.ones([real_pos.shape[0], 3])
            real[:, :2] = real_pos
            pixel = np.matmul(real, np.linalg.inv(H))
            pixel = pixel[:, :2].astype(np.int)
            return np.column_stack([
                weights[1] * pixel.T[0] + weights[2],
                weights[3] * pixel.T[1] + weights[4],
            ]).astype(np.int)

    def draw(self, agent: Agent_Part, video_file:cv2.VideoCapture, video_para, traj_weights, save_path, draw_neighbors=False):
        obs_frame = int(float(agent.frame_list[agent.obs_length]))
        time = 1000 * obs_frame / video_para[1]
        video_file.set(cv2.CAP_PROP_POS_MSEC, time - 1)
        _, f = video_file.read()

        obs = self.real2pixel(agent.get_train_traj(), traj_weights)
        if self.social_refine:
            pred = self.real2pixel(agent.get_pred_traj_sr(), traj_weights)
        else:
            pred = self.real2pixel(agent.get_pred_traj(), traj_weights)
            
        gt = self.real2pixel(agent.get_gt_traj(), traj_weights)

        obs_file = cv2.imread(OBS_IMAGE, -1)
        gt_file = cv2.imread(GT_IMAGE, -1)
        pred_file = cv2.imread(PRED_IMAGE, -1)

        for p in obs:
            f = add_png_to_source(f, obs_file, p)
        
        for p in gt:
            f = add_png_to_source(f, gt_file, p)

        for p in pred:
            f = add_png_to_source(f, pred_file, p)

        if draw_neighbors:
            for obs, pred in zip(agent.get_neighbor_traj(), agent.get_pred_traj_neighbor()):
                obs = self.real2pixel(obs, traj_weights)
                pred = self.real2pixel(pred, traj_weights)

                for p in obs:
                    cv2.circle(f, (p[0], p[1]), 8, (255, 80, 0), 0)  

                for p in pred:
                    cv2.circle(f, (p[0], p[1]), 8, (80, 255, 0), 0)

        cv2.imwrite(save_path, f)

    def draw_video(self, agent:Agent_Part, video_file:cv2.VideoCapture, video_para, traj_weights, save_path, interp=True, indexx=0):
        _, f = video_file.read()
        video_shape = (f.shape[1], f.shape[0])
        frame_list = (agent.frame_list.astype(np.float)).astype(np.int)
        frame_list_original = frame_list

        if interp:
            frame_list = np.array([index for index in range(frame_list[0], frame_list[-1]+1)])

        video_list = []
        times = 1000 * frame_list / video_para[1]

        obs = self.real2pixel(agent.get_train_traj(), traj_weights)
        if self.social_refine:
            pred = self.real2pixel(agent.get_pred_traj_sr(), traj_weights)
        else:
            pred = self.real2pixel(agent.get_pred_traj(), traj_weights)

        # # load from npy file
        # pred = np.load('./figures/hotel_{}_stgcnn.npy'.format(indexx)).reshape([-1, 2])
        # pred = self.real2pixel(np.column_stack([
        #     pred.T[0],  # sr: 0,1; sgan: 1,0; stgcnn: 1,0
        #     pred.T[1],
        # ]), traj_weights)
                    
        gt = self.real2pixel(agent.get_gt_traj(), traj_weights)

        obs_file = cv2.imread(OBS_IMAGE, -1)
        gt_file = cv2.imread(GT_IMAGE, -1)
        pred_file = cv2.imread(PRED_IMAGE, -1)

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        VideoWriter = cv2.VideoWriter(save_path, fourcc, 25.0, video_shape)
        
        for time, frame in zip(times, frame_list):
            video_file.set(cv2.CAP_PROP_POS_MSEC, time - 1)
            _, f = video_file.read()

            # draw observations
            for obs_step in range(agent.obs_length):
                if frame >= frame_list_original[obs_step]:
                    f = add_png_to_source(f, obs_file, obs[obs_step])

            # draw predictions
            if frame >= frame_list_original[agent.obs_length]:
                for p in pred:
                    f = add_png_to_source(f, pred_file, p, alpha=0.5)

            # draw GTs
            for gt_step in range(agent.obs_length, agent.total_frame):
                if frame >= frame_list_original[gt_step]:
                    f = add_png_to_source(f, obs_file, gt[gt_step - agent.obs_length])

            video_list.append(f)
            VideoWriter.write(f)


def add_png_to_source(source:np.ndarray, png:np.ndarray, position, alpha=1.0):
    original_png = png[:, :, :3]
    png_mask = alpha * png[:, :, -1]/255
    
    yc, xc = position 
    xp, yp = png_mask.shape
    xs, ys = source.shape[:2]
    x0, y0 = [xc-xp//2, yc-yp//2]
    if x0 > 0 and y0 > 0 and x0 + xp < xs and y0 + yp < ys:
        source[x0:x0+xp, y0:y0+yp] = (1.0 - png_mask).reshape([xp, yp, 1]) * source[x0:x0+xp, y0:y0+yp] + original_png * png_mask.reshape([xp, yp, 1])
    return source