'''
Author: Conghao Wong
Date: 2020-09-28 20:05:17
LastEditors: Conghao Wong
LastEditTime: 2020-09-28 20:34:40
Description: file content
'''
import numpy as np

class Args():
    def __init__(self, ):
        self.window_size_expand_meter = 5.0
        self.args.window_size_guidancemap = 4
        

class MapManager():
    def __init__(self, agent_list:list, args:Args, target_agent_index=-1):
        self.args = args
        self.observed_agent_list = agent_list
        self.target_agent = agent_list[target_agent_index]

        self.traj_map, self.W, self.b = self.init_guidancemap(agent_list)
        
    def init_guidancemap(self, agent_list):
        traj = []
        for agent in agent_list:
            if agent.rotate == 0:
                traj.append(agent.get_train_traj())
        traj = np.array(traj)

        x_max = np.max(traj[:, :, 0])
        x_min = np.min(traj[:, :, 0])
        y_max = np.max(traj[:, :, 1])
        y_min = np.min(traj[:, :, 1])
        traj_map = np.zeros([
            int((x_max - x_min + 2*self.args.window_size_expand_meter)*self.args.window_size_guidancemap) + 1,
            int((y_max - y_min + 2*self.args.window_size_expand_meter)*self.args.window_size_guidancemap) + 1,
        ])
        
        W = np.array([self.args.window_size_guidancemap, self.args.window_size_guidancemap])
        b = np.array([x_min - self.args.window_size_expand_meter, y_min - self.args.window_size_expand_meter])
        self.map_coe = [x_max, x_min, y_max, y_min]
        return traj_map, W, b

    def add_to_map(self, target_map, trajs:np.array, map_function, amplitude=1, radius=0, add_mask=None):
        """
        `amplitude`: Value of each add point. Accept both `float` and `np.array` types.
        `radius`: Raduis of each add point. Accept both `float` and `np.array` types.
        """
        
        n_traj = trajs.shape[0]

        if type(add_mask) == None:
            add_mask = np.ones([1, 1])

        for traj in trajs:
            p = map_function(traj)
            target_map[p.T[0], p.T[1]] 
          
        

        