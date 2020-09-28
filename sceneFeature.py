'''
Author: Conghao Wong
Date: 1970-01-01 08:00:00
LastEditors: ConghaoWong
LastEditTime: 2020-08-30 04:09:30
Description: file content
'''

import numpy as np

class TrajectoryMapManager():
    def __init__(self, agent_list:list):
        self.agent_list = agent_list
        
        self.window_size_expand_meter = 5.0
        self.window_size_map = 4

        self.traj_map = 'null'
        self.traj = self.get_all_obs_traj()
        self.traj_map, self.W, self.b = self.initialize_traj_map(self.traj)
        self.add_to_map()

    def get_all_obs_traj(self):
        traj = []
        for agent in self.agent_list:
            if agent.rotate == 0:
                traj.append(agent.get_train_traj())
        return np.stack(traj)

    def initialize_traj_map(self, traj):
        x_max = np.max(traj[:, :, 0])
        x_min = np.min(traj[:, :, 0])
        y_max = np.max(traj[:, :, 1])
        y_min = np.min(traj[:, :, 1])
        traj_map = np.zeros([
            int((x_max - x_min + 2*self.window_size_expand_meter)*self.window_size_map) + 1,
            int((y_max - y_min + 2*self.window_size_expand_meter)*self.window_size_map) + 1,
        ])
        
        W = np.array([self.window_size_map, self.window_size_map])
        b = np.array([x_min - self.window_size_expand_meter, y_min - self.window_size_expand_meter])

        self.mvalue = [x_max, x_min, y_max, y_min]
        return traj_map, W, b

    def add_to_map(self, val=1):
        for traj in self.traj:
            map_pos = self.real2map(traj)
            self.traj_map[map_pos.T[0], map_pos.T[1]] += val

    def real2map(self, traj:np.array):
        return ((traj - self.b) * self.W).astype(np.int)
