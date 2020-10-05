'''
Author: Conghao Wong
Date: 2020-09-28 20:05:17
LastEditors: Conghao Wong
LastEditTime: 2020-09-30 16:25:34
Description: file content
'''
import numpy as np
import cv2

from PrepareTrainData import Agent_Part
from visual import TrajVisual

class _ARGS_USED():
    """
    Args used in class `MapManager`.
    Actual NOT used.
    Please change them in `main.py`.
    """
    def __init__(self, ):
        self.pred_frames = 12
        
        self.window_size_expand_meter = 5.0
        self.window_size_guidance_map = 10 # 4
        self.window_size_social_map = 10

        self.interest_size = 20
        self.avoid_size = 15
        self.max_refine = 2.0

        self.refine_epochs = 30
        self.theta = 5 # 0.001


class MapManager():
    def __init__(self, agent_list:list, args:_ARGS_USED, target_agent_index=-1, init_manager=None, calculate_guidance=False, calculate_social=False):
        self.args = args
        self.agent_list = agent_list
        self.target_agent = agent_list[target_agent_index]

        if init_manager:
            self.void_map, self.W, self.b = [init_manager.void_map, init_manager.W, init_manager.b]
        else:
            self.void_map, self.W, self.b = self.init_guidance_map(agent_list)
        
        if calculate_guidance:
            self.build_guidance_map(source=self.void_map)
        
        if calculate_social:
            self.build_social_map(source=self.void_map)
        # 
        # self.context_map = 0.4 * self.social_map + 0.6 * self.guidance_map

    def init_guidance_map(self, agent_list):
        traj = []
        for agent in agent_list:
            traj.append(agent.get_train_traj())

        traj = np.array(traj)

        x_max = np.max(traj[:, :, 0])
        x_min = np.min(traj[:, :, 0])
        y_max = np.max(traj[:, :, 1])
        y_min = np.min(traj[:, :, 1])
        guidance_map = np.zeros([
            int((x_max - x_min + 2*self.args.window_size_expand_meter)*self.args.window_size_guidance_map) + 1,
            int((y_max - y_min + 2*self.args.window_size_expand_meter)*self.args.window_size_guidance_map) + 1,
        ])

        W = np.array([self.args.window_size_guidance_map, self.args.window_size_guidance_map])
        b = np.array([x_min - self.args.window_size_expand_meter, y_min - self.args.window_size_expand_meter])
        self.map_coe = [x_max, x_min, y_max, y_min]
        return guidance_map, W, b

    def build_guidance_map(self, source=None, regulation=True):
        agent_list = self.agent_list

        if type(source) == type(None):
            source = self.void_map

        source = source.copy()
        for agent in agent_list:
            source = self.add_to_map(
                source,
                agent.get_neighbor_traj(),
                self.real2grid,
                amplitude=1,
                radius=7,
                add_mask=(cv2.imread('./mask_circle.png')[:, :, 0])/50,
                decay=False,
                max_limit=False,
            )
        source = np.minimum(source, 100)
        
        if regulation:
            source = 1 - source / np.max(source)
        
        self.guidance_map = source
        return source

    def build_social_map(self, source=None, regulation=True):
        target_agent = self.target_agent
        
        if type(source) == type(None):
            source = self.void_map

        source = source.copy()
        add_mask = (cv2.imread('./mask_circle.png')[:, :, 0])

        # Draw Historical (Only for test)
        source = self.add_to_map(
            source,
            target_agent.get_train_traj(),
            self.real2grid,
            amplitude=-5,
            radius=2,
            add_mask=add_mask,
        )

        # Destination
        source = self.add_to_map(
            source,
            target_agent.get_pred_traj(),
            self.real2grid,
            amplitude=-2,
            radius=self.args.interest_size,
            add_mask=add_mask,
        )

        # Interplay
        vec_target = target_agent.get_pred_traj()[-1] - target_agent.get_pred_traj()[0]
        for pred in target_agent.get_pred_traj_neighbor():
            vec_neighbor = pred[-1] - pred[0]
            cosine = activation(
                calculate_cosine(vec_target, vec_neighbor),
                a = 1.0,
                b = 0.2,
            )
            velocity = calculate_length(vec_neighbor) / calculate_length(vec_target)

            # # Only for test
            # source = self.add_to_map(
            #     source,
            #     pred,
            #     self.real2grid,
            #     amplitude=-5,
            #     radius=2,
            #     add_mask=add_mask,
            # )

            source = self.add_to_map(
                source,
                pred,
                self.real2grid,
                amplitude=-cosine*velocity,
                radius=self.args.avoid_size,
                add_mask=add_mask,
            )
        
        if regulation:
            source = (source - np.min(source)) / (np.max(source) - np.min(source))
        
        self.social_map = source
        return source

    def add_to_map(self, target_map, trajs:np.array, map_function, amplitude=1, radius=0, add_mask=None, interp=True, max_limit=False, decay=True):
        """
        `amplitude`: Value of each add point. Accept both `float` and `np.array` types.
        `radius`: Raduis of each add point. Accept both `float` and `np.array` types.
        """
        if not type(trajs) == np.array:
            trajs = np.array(trajs)

        if len(trajs.shape) == 2:
            trajs = np.reshape(trajs, [1, trajs.shape[0], trajs.shape[1]])

        n_traj = trajs.shape[0]
        if not type(amplitude) == np.array:
            amplitude *= np.ones(n_traj, dtype=np.int)
            radius *= np.ones(n_traj, dtype=np.int)
        elif len(amplitude.shape) == 2:
            amplitude = np.reshape(amplitude, [1, trajs.shape[0], trajs.shape[1]])

        target_map = target_map.copy()

        if type(add_mask) == type(None):
            add_mask = np.ones([1, 1], dtype=np.int)

        if interp:
            trajs_grid = [interp_2d(map_function(traj), step=1) for traj in trajs]
        else:
            trajs_grid = [map_function(traj) for traj in trajs]

        for traj, a, r in zip(trajs_grid, amplitude, radius):
            add_mask = cv2.resize(add_mask, (r*2+1, r*2+1))
            target_map = self.add_one_traj(target_map, traj, a, r, add_mask, max_limit=max_limit, amplitude_decay=decay)

        return target_map

    def real2grid(self, traj:np.array):
        return ((traj - self.b) * self.W).astype(np.int)

    def add_one_traj(self, source_map, traj, amplitude, radius, add_mask, max_limit=True, amplitude_decay=False, amplitude_decay_p=np.array([[0.0, 0.7, 1.0], [1.0, 1.0, 0.5]])):
        if amplitude_decay:
            amplitude *= np.interp(
                np.arange(0, len(traj))/len(traj),
                amplitude_decay_p[0],
                amplitude_decay_p[1],
            )

        if not len(amplitude.shape) == 2:
            amplitude *= np.ones(len(traj))

        new_map = np.zeros_like(source_map)
        for pos, a in zip(traj, amplitude):
            if pos[0]-radius >= 0 and pos[1]-radius >=0 and pos[0]+radius+1 < new_map.shape[0] and pos[1]+radius+1 < new_map.shape[1]:
                new_map[pos[0]-radius:pos[0]+radius+1, pos[1]-radius:pos[1]+radius+1] = a * add_mask + new_map[pos[0]-radius:pos[0]+radius+1, pos[1]-radius:pos[1]+radius+1]

        if max_limit:
            new_map = np.sign(new_map)

        return new_map + source_map

    def length_refine(self, traj, original_traj):
        """
        预测长度修正，即认为优化前后行走速度一致
        """
        original_length = np.linalg.norm(original_traj[-1, :] - original_traj[0, :])
        current_length = np.linalg.norm(traj[-1, :] - traj[0, :])
        refine_coe = original_length / current_length
        
        # 总体长度修正
        if refine_coe < 1:
            x = [i*refine_coe for i in range(self.args.pred_frames)]
            x_original = [i for i in range(self.args.pred_frames)]
            traj_fix = np.column_stack([
                np.interp(x, x_original, traj.T[0]),
                np.interp(x, x_original, traj.T[1]),
            ])
            traj = traj_fix

        # 速度修正
        x = (traj.T[0] - np.min(traj.T[0]))/(np.max(traj.T[0]) - np.min(traj.T[0]))
        y = (traj.T[1] - np.min(traj.T[1]))/(np.max(traj.T[1]) - np.min(traj.T[1]))
        x.sort()
        y.sort()
        
        xp = np.arange(len(traj.T[0])) / (len(traj.T[0])-1)
        traj_fix = np.column_stack([
            np.interp(xp, x, traj.T[0]),
            np.interp(xp, y, traj.T[1]),
        ])

        return traj_fix

    def run_optimize(self, grid_map):
        prev_result = self.target_agent.get_pred_traj()
        pred_original = prev_result

        # 原预测静止不动的不需要微调
        if calculate_length(prev_result[-1] - prev_result[1]) <= 1.0:
            return prev_result

        for epoch in range(self.args.refine_epochs):
            result = prev_result
            input_traj_grid = self.real2grid(result) + 1

            diff_x = grid_map[1:, 1:] - grid_map[:-1, 1:]
            diff_y = grid_map[1:, 1:] - grid_map[1:, :-1]

            dx_current = diff_x[input_traj_grid.T[0], input_traj_grid.T[1]]
            dy_current = diff_y[input_traj_grid.T[0], input_traj_grid.T[1]]

            x_bias = dx_current * self.args.theta
            y_bias = dy_current * self.args.theta

            prev_result = np.stack([
                result.T[0] - x_bias,
                result.T[1] - y_bias,
            ]).T

        delta = np.minimum(prev_result - pred_original, self.args.max_refine)
        coe = 0.7 * np.stack([i/(self.args.pred_frames-1) for i in range(self.args.pred_frames)]).reshape([-1, 1])
    
        social_fix = pred_original + coe * delta
        length_fix = self.length_refine(social_fix, pred_original)
        return social_fix


def interp_2d(traj:np.array, step=1):
    """
    shape(traj) should be [m, 2].
    """
    x = traj
    if type(step) == int:
        step = step * np.ones(2).astype(np.int)

    x_p = []
    index = 0
    while True:
        if len(x_p):
            x_last = x_p[-1]
            if np.linalg.norm(x[index] - x_last, ord=1) >= np.min(step):
                coe = np.sign(x[index] - x_last)
                coe_mask = (abs(x[index] - x_last) == np.max(abs(x[index] - x_last)))
                x_p.append(x_last + coe * coe_mask * step)
                continue
        
        if len(x_p) and np.linalg.norm(x[index] - x_last, ord=1) > 0:
            x_p.append(x[index])
        elif len(x_p) == 0:
            x_p.append(x[index])

        index += 1
        if index >= len(x):
            break

    return np.array(x_p)


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


def activation(x:np.array, a=1, b=1):
    return (x <= 0) * a * x + (x > 0) * b * x


def context_refine(agent_list:list, args):
    mm = MapManager(agent_list, args)
    traj_refine = mm.run_optimize(mm.context_map)
    return traj_refine, mm.context_map
    

if __name__ == '__main__':
    args = _ARGS_USED()
    agents = np.load('./test_agents.npy', allow_pickle=True)
    
    index = 81
    mm = MapManager(agents, args, target_agent_index=index)
    social_map = mm.build_social_map()
    guidance_map = mm.build_guidance_map()
    
    traj_refine = mm.run_optimize(0.4 * social_map + 0.6 * guidance_map)
    agents[index].write_pred_sr(traj_refine)

    source = 0.4 * social_map + 0.6 * guidance_map
    cv2.imwrite('./test.png', 127*(1+source/np.max(source)))
    np.save('./map.npz', source)
    
    tv = TrajVisual(save_base_path='./', verbose=True, draw_neighbors=False, social_refine=True)
    tv.visual([agents[index]], dataset=2)
