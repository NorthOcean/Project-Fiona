'''
@Author: ConghaoWong
@Date: 2019-12-20 09:39:02
LastEditors: Conghao Wong
LastEditTime: 2020-09-16 16:46:14
@Description: file content
'''
import os
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from helpmethods import (calculate_ADE_FDE_numpy, dir_check,
                         predict_linear_for_person)
from sceneFeature import TrajectoryMapManager

USE_SEED = True
SEED = 10


def prepare_rotate_matrix(min_angel=1, save_path='./rotate_matrix.npy', load=False):
    need_to_re_calculate = True
    if os.path.exists(save_path):
        rotate_matrix = np.load(save_path)
        if rotate_matrix.shape[0] == 360//min_angel:
            need_to_re_calculate = False
    
    if need_to_re_calculate:
        angles = np.arange(0, 2 * np.pi, min_angel * np.pi / 180)
        sin = np.sin(angles)
        cos = np.cos(angles)

        rotate_matrix = np.empty((angles.shape[0], 2, 2))
        rotate_matrix[..., 0, 0] = cos
        rotate_matrix[..., 0, 1] = -sin
        rotate_matrix[..., 1, 0] = sin
        rotate_matrix[..., 1, 1] = cos
        np.save(save_path, rotate_matrix)
    
    if load:
        return rotate_matrix

rotate_matrix = prepare_rotate_matrix(min_angel=1, load=True)


class DataManager():
    """
        管理所有数据集的训练与测试数据
    """
    def __init__(self, args, save=True):
        self.args = args
        self.obs_frames = args.obs_frames
        self.pred_frames = args.pred_frames
        self.total_frames = self.pred_frames + self.obs_frames
        self.step = args.step

        self.init_position = np.array([args.init_position, args.init_position])
        self.god_past_traj = np.stack([self.init_position for _ in range(self.obs_frames)])
        self.god_future_traj = np.stack([self.init_position for _ in range(self.pred_frames)])

        self.log_dir = dir_check(args.log_dir)
        self.save_file_name = args.model_name + '_{}.npy'
        self.save_path = os.path.join(self.log_dir, self.save_file_name)
        self.train_info = self.get_train_and_test_agents()
        
    def get_train_and_test_agents(self):
        dir_check('./dataset_npz/')
        self.npy_file_base_path = './dataset_npz/{}/data.npz'

        if self.args.train_type == 'one':
            train_list = [self.args.test_set]
            test_list = [self.args.test_set]
        
        elif self.args.train_type == 'all':
            test_list = [self.args.test_set]
            train_list = [i for i in range(8) if not i == self.args.test_set]
            # train_list = [i for i in range(3) if not i == self.args.test_set]   # toy exp
        
        data_managers_train = []
        for dataset in train_list:
            data_managers_train.append(self.get_agents_from_dataset(dataset)) 

        data_managers_test = []
        for dataset in test_list:
            data_managers_test.append(self.get_agents_from_dataset(dataset)) 

        sample_number_original = 0
        sample_time = 1
        for dm in data_managers_train:
            sample_number_original += dm.person_number
            
        if self.args.train_type == 'one':
            index = set([i for i in range(sample_number_original)])
            if USE_SEED:
                random.seed(SEED)
            train_index = random.sample(index, int(sample_number_original * self.args.train_percent))
            test_index = list(index - set(train_index))
            
            test_agents = self.sample_data(data_managers_train[0], test_index)
            train_agents = self.sample_data(data_managers_train[0], train_index)
            if self.args.reverse:
                train_agents += self.sample_data(data_managers_train[0], train_index, reverse=True, desc='Preparing reverse data')
                sample_time += 1

            if self.args.add_noise:                
                for repeat in tqdm(range(self.args.add_noise), desc='Prepare noise data...'):
                    train_agents += self.sample_data(data_managers_train[0], train_index, add_noise=True, use_time_bar=False)
                    sample_time += 1
        
        elif self.args.train_type == 'all':
            train_agents = []
            trajmaps = []
            if len(self.args.train_percent) == 1:
                train_percent = self.args.train_percent * np.ones([len(train_list)])
            else:
                train_percent = [self.args.train_percent[index] for index in train_list]
                
            for index, dm in enumerate(data_managers_train):
                agents, trajmap = self.sample_data(
                    dm, 
                    person_index='auto', 
                    random_sample=train_percent[index], 
                    return_trajmap=True
                )
                train_agents += agents
                trajmaps.append(trajmap)

            if self.args.reverse:
                for index, dm in enumerate(data_managers_train):
                    train_agents += self.sample_data(
                        dm, 
                        person_index='auto', 
                        random_sample=train_percent[index], 
                        reverse=True, 
                        use_time_bar=False
                    )
                sample_time += 1

            if self.args.rotate:
                for angel in tqdm(range(360//self.args.rotate, 360, 360//self.args.rotate), desc='Prepare rotate data...'):
                    sample_time += 1
                    for index, [dm, gm] in enumerate(zip(data_managers_train, trajmaps)):
                        train_agents += self.sample_data(
                            dm, 
                            person_index='auto', 
                            random_sample=train_percent[index], 
                            rotate=angel, 
                            use_time_bar=False, 
                            given_trajmap=gm
                        )
                
            test_agents, test_trajmap = self.sample_data(
                data_managers_test[0], 
                person_index='auto', 
                return_trajmap=True, 
                random_sample=False,
            )
        
        train_info = dict()
        train_info['train_data'] = train_agents
        train_info['test_data'] = test_agents
        train_info['train_number'] = len(train_agents)
        train_info['sample_time'] = sample_time  

        return train_info

    def data_loader(self, dataset_index):
        """
        Read trajectory data from csv file.
        returns: `person_data`, `frame_data`
        """
        dataset_dir = [
            './data/eth/univ',
            './data/eth/hotel',
            './data/ucy/zara/zara01',
            './data/ucy/zara/zara02',
            './data/ucy/univ/students001',
            './data/ucy/zara/zara03',
            './data/ucy/univ/students003',
            './data/ucy/univ/uni_examples',
        ]

        dataset_xy_order = [
            [3, 2],
            [2, 3],
            [3, 2],
            [3, 2],
            [2, 3],
            [3, 2],
            [2, 3],
            [2, 3],
        ]

        # dataset_dir = [         # toy exp
        #     './data/toy/half_circle',
        #     './data/toy/line_circle',
        #     './data/toy',
        # ]

        # dataset_xy_order = [    # toy exp
        #     [2, 3],
        #     [2, 3],
        #     [2, 3],
        # ]

        dataset_dir_current = dataset_dir[dataset_index]
        order = dataset_xy_order[dataset_index]

        csv_file_path = os.path.join(dataset_dir_current, 'true_pos_.csv')
        data = np.genfromtxt(csv_file_path, delimiter=',').T 

        # 加载数据（使用帧排序）
        frame_data = {}
        frame_list = set(data.T[0])
        for frame in frame_list:
            index_current = np.where(data.T[0] == frame)[0]
            frame_data[str(frame)] = np.column_stack([
                data[index_current, 1],
                data[index_current, order[0]],
                data[index_current, order[1]],
            ])

        # 加载数据（使用行人编号排序）
        person_data = {}
        person_list = set(data.T[1])
        for person in person_list:
            index_current = np.where(data.T[1] == person)[0]
            person_data[str(person)] = np.column_stack([
                data[index_current, 0],
                data[index_current, order[0]],
                data[index_current, order[1]],
            ])
        
        print('Load dataset from csv file done.')
        return person_data, frame_data

    def get_agents_from_dataset(self, dataset):
        """
        使用数据计算social关系，并组织为`Agent_part`类或`Frame`类
            return: agents, original_sample_number
        """
        base_path = dir_check(os.path.join('./dataset_npz/', '{}'.format(dataset)))
        npy_path = self.npy_file_base_path.format(dataset)

        if os.path.exists(npy_path):
            # 从保存的npy数据集文件中读
            video_neighbor_list, video_matrix, frame_list = self.load_video_matrix(dataset)
        else:
            # 新建npy数据集文件
            person_data, frame_data = self.data_loader(dataset)
            video_neighbor_list, video_matrix, frame_list = self.create_video_matrix(
                person_data, 
                frame_data, 
                save_path=npy_path
            )

        if self.args.train_base == 'agent':
            data_manager = self.get_agents(video_neighbor_list, video_matrix, frame_list)
            print('\nPrepare agent data in dataset {} done.'.format(dataset))
            return data_manager
        
    def load_video_matrix(self, dataset):
        """
        从保存的文件中读取social matrix和social neighbor
        """
        print('Load data from "{}"...'.format(self.npy_file_base_path.format(dataset)))
        all_data = np.load(self.npy_file_base_path.format(dataset), allow_pickle=True)
        video_neighbor_list = all_data['video_neighbor_list']
        video_matrix = all_data['video_matrix']
        frame_list = all_data['frame_list']
        return video_neighbor_list, video_matrix, frame_list

    def create_video_matrix(self, person_data, frame_data, save_path='null'):
        """
        计算social neighbor
        `video_matrix`: shape = [frame_number, person_number, 2]
        """
        person_list = np.sort(np.stack([float(person) for person in person_data])).astype(np.str)
        frame_list = np.sort(np.stack([float(frame) for frame in frame_data])).astype(np.str)

        person_number = len(person_list)
        frame_number = len(frame_list)

        video_matrix = self.args.init_position * np.ones([frame_number, person_number, 2])
        for person in person_data:
            person_index = np.where(person_list == person)[0][0]
            frame_list_current = (person_data[person]).T[0].astype(np.str)
            frame_index_current = np.reshape(np.stack([np.where(frame_current == frame_list) for frame_current in frame_list_current]), [-1])
            traj_current = person_data[person][:, 1:]
            video_matrix[frame_index_current, person_index, :] = traj_current

        video_neighbor_list = []
        for frame_index, data in enumerate(tqdm(video_matrix, desc='Calculate social matrix...')):
            person_appear = np.where(np.not_equal(data.T[0], self.args.init_position))[0]
            video_neighbor_list.append(person_appear)

        if not save_path == 'null':
            np.savez(
                save_path, 
                video_neighbor_list=video_neighbor_list,
                video_matrix=video_matrix,
                frame_list=frame_list,
            )
        return video_neighbor_list, video_matrix, frame_list

    def sample_data(self, data_manager, person_index, add_noise=False, reverse=False, rotate=False, desc='Calculate agent data', use_time_bar=True, random_sample=False, sample_start=0.0, given_trajmap=False, return_trajmap=False):
        """
        Sample training data from data_manager.
        `random_sample`: 为0到1的正数时表示随机取样百分比，为-1到0的负数时表示按照数据集时间顺序百分比取样的终点，此时0～1正数`sample_start`表示起点
        return: a list of Agent_Part
        """
        agents = []
        if person_index == 'auto':
            if random_sample > 0 and random_sample < 1:
                if USE_SEED:
                    random.seed(SEED)
                person_index = random.sample(
                    [i for i in range(data_manager.person_number)], 
                    int(data_manager.person_number * random_sample),
                )
            elif random_sample == 0 or random_sample >= 1 or random_sample < -1:
                person_index = range(data_manager.person_number)

            elif random_sample < 0 and random_sample >= -1:
                person_index = [i for i in range(
                    (data_manager.person_number * np.abs(sample_start)).astype(int),   # start index
                    (data_manager.person_number * np.abs(random_sample)).astype(int),   # end index
                )]

        if use_time_bar:
            itera = tqdm(person_index, desc=desc)
        else:
            itera = person_index

        for person in itera:
            agent_current = data_manager.agent_data[person]
            start_frame = agent_current.start_frame
            end_frame = agent_current.end_frame

            for frame_point in range(start_frame, end_frame, self.args.step):
                if frame_point + self.total_frames > end_frame:
                    break
                
                # type: Agent_Part
                sample_agent = data_manager.get_trajectory(
                    person,
                    frame_point, 
                    frame_point+self.obs_frames, 
                    frame_point+self.total_frames,
                    calculate_social=self.args.calculate_social,
                    normalization=self.args.normalization,
                    add_noise=add_noise,
                    reverse=reverse,
                    rotate=rotate,
                )     
                agents.append(sample_agent)

        if not given_trajmap:
            traj_trajmap = TrajectoryMapManager(agents)
            for index in range(len(agents)):
                agents[index].write_traj_map(traj_trajmap)  

            if return_trajmap:
                return agents, traj_trajmap
            else:
                return agents

        else:
            for index in range(len(agents)):
                agents[index].write_traj_map(given_trajmap)   
            return agents
    
    def get_agents(self, video_neighbor_list, video_matrix, frame_list):
        """
        使用social matrix计算每个人的`Agent`类，并取样得到用于训练的`Agent_part`类数据
            return: agents(取样后, type=`Agent_part`), original_sample_number
        """
        data_manager = DatasetManager(
            video_neighbor_list, video_matrix, frame_list, self.args.init_position
        )
        return data_manager


class DatasetManager():
    """
        管理一个数据集内的所有轨迹数据
    """
    def __init__(self, video_neighbor_list, video_matrix, frame_list, init_position):
        self.video_neighbor_list = video_neighbor_list
        self.video_matrix = video_matrix
        self.frame_list = frame_list
        self.init_position = init_position
        self.agent_data = self.prepare_agent_data()

    def prepare_agent_data(self):
        self.frame_number, self.person_number, _ = self.video_matrix.shape
        agent_data = []
        for person in range(self.person_number):
            agent_data.append(Agent(
                person, 
                self.video_neighbor_list, 
                self.video_matrix, 
                self.frame_list,
                self.init_position,
            ))
        return agent_data

    def get_trajectory(self, agent_index, start_frame, obs_frame, end_frame, calculate_social=True, normalization=False, add_noise=False, reverse=False, rotate=False):
        target_agent = self.agent_data[agent_index]
        frame_list = target_agent.frame_list
        neighbor_list = target_agent.video_neighbor_list[obs_frame-1].tolist()
        neighbor_list = set(neighbor_list) - set([agent_index])
        neighbor_agents = [self.agent_data[nei] for nei in neighbor_list]

        return Agent_Part(
            target_agent, neighbor_agents, frame_list, start_frame, obs_frame, end_frame, calculate_social=calculate_social, normalization=normalization, add_noise=add_noise, reverse=reverse, rotate=rotate
        )
        

class Agent():
    def __init__(self, agent_index, video_neighbor_list, video_matrix, frame_list, init_position):
        self.agent_index = agent_index
        self.traj = video_matrix[:, agent_index, :]
        self.video_neighbor_list = video_neighbor_list
        self.frame_list = frame_list

        self.start_frame = np.where(np.not_equal(self.traj.T[0], init_position))[0][0]
        self.end_frame = np.where(np.not_equal(self.traj.T[0], init_position))[0][-1] + 1    # 取不到


class Agent_Part():
    def __init__(self, target_agent, neighbor_agents, frame_list, start_frame, obs_frame, end_frame, calculate_social=True, normalization=False, add_noise=False, reverse=False, rotate=False):        
        # Trajectory info
        self.start_frame = start_frame
        self.obs_frame = obs_frame
        self.end_frame = end_frame
        self.obs_length = obs_frame - start_frame
        self.total_frame = end_frame - start_frame
        self.frame_list = frame_list[start_frame:end_frame]
        self.vertual_agent = False
        self.rotate = rotate
        self.reverse = reverse
        self.traj_map = 'null'

        # Trajectory
        self.traj = target_agent.traj[start_frame:end_frame]
        if add_noise:
            noise_curr = np.random.normal(0, 0.1, size=self.traj.shape)
            self.traj += noise_curr
            self.vertual_agent = True

        elif reverse:
            self.traj = self.traj[::-1]
            self.vertual_agent = True

        elif rotate:    # rotate 为旋转角度
            rotate_matrix_current = rotate_matrix[rotate, :, :]
            self.traj_original = self.traj
            self.traj = self.traj[0] + np.matmul(self.traj - self.traj[0], rotate_matrix_current)
            self.vertual_agent = True
            
        self.pred = 0
        self.start_point = self.traj[0]

        # Options
        self.calculate_social = calculate_social  
        self.normalization = normalization 

        # Neighbor info
        if not self.vertual_agent:
            self.neighbor_traj = []
            for neighbor in neighbor_agents:
                neighbor_traj = neighbor.traj[start_frame:obs_frame]
                neighbor_traj[0:np.maximum(neighbor.start_frame, start_frame)-start_frame] = neighbor_traj[np.maximum(neighbor.start_frame, start_frame)-start_frame]
                neighbor_traj[np.minimum(neighbor.end_frame, obs_frame)-start_frame:obs_frame-start_frame] = neighbor_traj[np.minimum(neighbor.end_frame, obs_frame)-start_frame-1]
                self.neighbor_traj.append(neighbor_traj)
            
            self.neighbor_number = len(neighbor_agents)

        # Initialize
        self.need_to_fix = False
        self.need_to_fix_neighbor = False
        self.initialize()  
        if normalization:
            self.agent_normalization()   

    def initialize(self):
        self.traj_train = self.traj[:self.obs_length]
        self.traj_gt = self.traj[self.obs_length:]

    def agent_normalization(self):
        """Attention: This method will change the value inside the agent!"""
        self.start_point = np.array([0.0, 0.0])
        if np.linalg.norm(self.traj[0] - self.traj[7]) >= 0.2:
            self.start_point = self.traj[7]
            self.traj = self.traj - self.start_point

        for neighbor_index in range(self.neighbor_number):
            self.neighbor_traj[neighbor_index] -= self.start_point
        
        self.initialize()
        self.need_to_fix = True
        self.need_to_fix_neighbor = True

    def pred_fix(self):
        if not self.need_to_fix:
            return
        
        self.traj += self.start_point
        self.pred += self.start_point

        self.need_to_fix = False
        self.initialize()

    def pred_fix_neighbor(self, pred):
        if not self.need_to_fix_neighbor:
            return pred
            
        pred += self.start_point
        self.need_to_fix_neighbor = False
        return pred

    def get_train_traj(self):
        return self.traj_train

    def get_neighbor_traj(self):
        return self.neighbor_traj

    def get_gt_traj(self):
        return self.traj_gt

    def get_pred_traj(self):
        return self.pred

    def get_pred_traj_sr(self):
        return self.pred_sr

    def get_pred_traj_neighbor(self):
        return self.neighbor_pred

    def get_traj_map(self):
        return self.traj_map

    def get_traj_map_for_neighbors(self):
        return self.traj_map_neighbors

    def write_pred(self, pred):
        self.pred = pred
        self.pred_fix()

    def write_pred_sr(self, pred):
        self.pred_sr = pred
        self.sr = True

    def write_pred_neighbor(self, pred):
        self.neighbor_pred = self.pred_fix_neighbor(pred)

    def write_traj_map(self, trajmap:TrajectoryMapManager):
        full_map = trajmap.traj_map
        half_size = 16  # half of map size, in map size
        if not self.rotate:
            center_pos = trajmap.real2map(self.traj_train[-1])
        else:
            center_pos = trajmap.real2map(self.traj_original[self.obs_length])
            
        original_map = cv2.resize(full_map[
            np.maximum(center_pos[0]-2*half_size, 0):np.minimum(center_pos[0]+2*half_size, full_map.shape[0]), 
            np.maximum(center_pos[1]-2*half_size, 0):np.minimum(center_pos[1]+2*half_size, full_map.shape[1]),
        ], (4*half_size, 4*half_size))

        final_map = original_map[half_size:3*half_size, half_size:3*half_size]
        if self.reverse:
            final_map = np.flip(original_map[half_size:3*half_size, half_size:3*half_size])

        if self.rotate:
            final_map = cv2.warpAffine(
                original_map,
                cv2.getRotationMatrix2D(
                    (2*half_size, 2*half_size),
                    self.rotate,
                    1,
                ),
                (4*half_size, 4*half_size),
            )
            final_map = final_map[half_size:3*half_size, half_size:3*half_size]
        self.traj_map = final_map

    def write_traj_map_for_neighbors(self, trajmap:TrajectoryMapManager):
        self.traj_map_neighbors = []
        full_map = trajmap.traj_map
        half_size = 16

        for nei_traj in self.get_neighbor_traj():
            center_pos = trajmap.real2map(nei_traj[-1, :])
            original_map = cv2.resize(full_map[
                np.maximum(center_pos[0]-2*half_size, 0):np.minimum(center_pos[0]+2*half_size, full_map.shape[0]), 
                np.maximum(center_pos[1]-2*half_size, 0):np.minimum(center_pos[1]+2*half_size, full_map.shape[1]),
            ], (4*half_size, 4*half_size))
            final_map = original_map[half_size:3*half_size, half_size:3*half_size]
            self.traj_map_neighbors.append(final_map)


    def calculate_loss(self, loss_function=calculate_ADE_FDE_numpy, SR=False):
        if SR and self.sr:
            self.loss = loss_function(self.get_pred_traj_sr(), self.get_gt_traj())
        else:
            self.loss = loss_function(self.get_pred_traj(), self.get_gt_traj())
        return self.loss

    def clear_pred(self):
        self.pred = 0

    def draw_results(self, log_dir, file_name, draw_neighbors=False, draw_sr=False):
        """
        结果保存路径为`log_dir/test_figs/`
        """
        save_base_dir = dir_check(os.path.join(log_dir, 'test_figs/'))
        save_format = os.path.join(save_base_dir, file_name)

        obs = self.get_train_traj()
        gt = self.get_gt_traj()
        pred = self.get_pred_traj()

        plt.figure()
        plt.plot(pred.T[0], pred.T[1], '-b*')
        plt.plot(gt.T[0], gt.T[1], '-ro')
        plt.plot(obs.T[0], obs.T[1], '-go')
        if draw_sr:
            pred_sr = self.get_pred_traj_sr()
            plt.plot(pred_sr.T[0], pred.T[1], '--b*')

        if draw_neighbors:
            obs_nei = self.get_neighbor_traj()
            pred_nei = self.get_pred_traj_neighbor()
            
            for obs_c, pred_c in zip(obs_nei, pred_nei):
                plt.plot(pred_c.T[0], pred_c.T[1], '--b*')
                plt.plot(obs_c.T[0], obs_c.T[1], '--go')
        
        plt.axis('scaled')
        plt.title('frame=[{}, {}]'.format(
            self.start_frame,
            self.end_frame,
        ))
        plt.savefig(save_format)
        plt.close()
