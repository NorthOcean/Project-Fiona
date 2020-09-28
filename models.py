'''
@Author: ConghaoWong
@Date: 2019-12-20 09:39:34
LastEditors: Conghao Wong
LastEditTime: 2020-09-16 16:27:24
@Description: classes and methods of training model
'''
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm

from GridRefine import SocialRefine_one
from helpmethods import calculate_ADE_FDE_numpy, dir_check, list2array
from sceneFeature import TrajectoryMapManager
from visual import TrajVisual


class Base_Model():
    """
    Base model for prediction.

    Following items should be given when using this model:
    ```
    self.create_model(self), # create prediction model
    self.loss(self, model_output, gt, obs='null'),  # loss function when training model
    self.loss_eval(self, model_output, gt, obs='null'), # loss function when test model
    self.forward_train(self, mode_inputs), # model result in training steps
    self.forward_test(self, test_tensor:list). # model result in test steps
    ```
    """
    def __init__(self, train_info, args):
        self.args = args
        self.train_info = train_info
        
    def run_commands(self):
        self.get_data()     # 获取与训练数据有关的信息

        if self.args.load == 'null':
            self.model, self.optimizer = self.create_model()
            self.model.summary()
            self.train()
        else:
            self.model, self.agents_test = self.load_from_checkpoint()
            self.model.summary()
        
            if self.args.test:
                self.test_batch(
                    self.agents_test, 
                    test_on_neighbors=False,
                    batch_size=0.2,         # set 0.5 on toy exp
                    social_refine=self.args.sr_enable,
                    draw_results=self.args.draw_results,
                    save_agents=False,
                )

    def get_data(self):
        self.obs_frames = self.args.obs_frames
        self.pred_frames = self.args.pred_frames
        self.total_frames = self.obs_frames + self.pred_frames
        self.log_dir = dir_check(self.args.log_dir)

        if not self.args.load == 'null':
            return

        self.agents_train = self.train_info['train_data']
        self.agents_test = self.train_info['test_data']
        self.train_number = self.train_info['train_number']
        self.sample_time = self.train_info['sample_time'] 
    
    def load_from_checkpoint(self):
        base_path = self.args.load + '{}'
        if self.args.save_best:
            best_epoch = np.loadtxt(os.path.join(self.args.log_dir, 'best_ade_epoch.txt'))[1].astype(int)
            model = keras.models.load_model(base_path.format('_epoch{}.h5'.format(best_epoch)))
        else:
            model = keras.models.load_model(base_path.format('.h5'))

        agents_test = np.load(base_path.format('test.npy'), allow_pickle=True)
        return model, agents_test
    
    def create_model(self):
        raise 'MODEL is not defined!'
        return model, optimizer

    def loss(self, model_output, gt, obs='null'):
        """
        Train loss, using ADE by default
        """
        self.loss_namelist = ['ADE_t']
        loss_ADE = calculate_ADE(model_output[0], gt)
        loss_list = tf.stack([loss_ADE])
        return loss_ADE, loss_list

    def loss_eval(self, model_output, gt, obs='null'):
        """
        Eval metrics, using ADE and FDE by default.
        return: `np.array`
        """
        self.loss_eval_namelist = ['ADE', 'FDE']
        return calculate_ADE(model_output[0], gt).numpy(), calculate_FDE(model_output[0], gt).numpy()

    def prepare_model_inputs_all(self, input_agents):
        model_inputs = []
        gt = []
        agent_index = []
        for agent_index_current, agent in enumerate(tqdm(input_agents, desc='Prepare inputs...')):
            model_inputs.append(agent.get_train_traj())
            gt.append(agent.get_gt_traj())
            agent_index.append(agent_index_current)

        model_inputs = tf.cast(tf.stack(model_inputs), tf.float32)
        gt = tf.cast(tf.stack(gt), tf.float32)
        return [model_inputs, gt], agent_index

    def prepare_model_inputs_batch(self, train_tensor=0, batch_size=0, init=False):
        """
        Get batch data from all data
        """
        if init:
            self.batch_start = 0
            self.train_length = len(train_tensor[1])
            return self.train_length
        
        start = self.batch_start
        end = (self.batch_start + batch_size) % self.train_length
        # 每次最多取 1 epoch
        if end < start:
            if type(train_tensor[0]) == list:
                train_inputs = [
                    tf.concat([
                        train_input[start:],
                        train_input[:end],
                    ], axis=0) for train_input in train_tensor[0]
                ]
            else:
                train_inputs = tf.concat([
                    train_tensor[0][start:],
                    train_tensor[0][:end],
                ], axis=0)
                
            gt = tf.concat([
                train_tensor[1][start:],
                train_tensor[1][:end],
            ], axis=0)

        elif start + batch_size < self.train_length:
            if type(train_tensor[0]) == list:
                train_inputs = [train_input[start:end] for train_input in train_tensor[0]]
            else:
                train_inputs = train_tensor[0][start:end]
            gt = train_tensor[1][start:end]

        else:
            train_inputs = train_tensor[0]
            gt = train_tensor[1]

        self.batch_start = end
        return train_inputs, gt, len(gt)

    def forward_train(self, model_inputs):
        """
        Run a training implement
        """
        output = self.model(model_inputs)
        if not type(output) == list:
            output = [output]
        return output

    def forward_test(self, test_tensor:list):
        """
        Run test once.
        `test_tensor` is a `list`. `test_tensor[0]` is the inputs of model and `test_tensor[1]` are their grount truths.
        """
        model_inputs = test_tensor[0]
        gt = test_tensor[1]
        output = self.model(model_inputs)
        if not type(output) == list:
            output = [output]
        return output, gt, model_inputs

    def test_during_training(self, test_tensor, input_agents, test_index):
        """
        Run test during training.
        Results will NOT be written to inputs.
        """
        model_output, gt, obs = self.forward_test(test_tensor)
        loss_eval = self.loss_eval(model_output, gt, obs=obs)
        return model_output, loss_eval, gt, input_agents
    
    def train(self):
        """
        Train the built model `self.model`
        """
        batch_number = int(np.ceil(self.train_number / self.args.batch_size))
        summary_writer = tf.summary.create_file_writer(self.args.log_dir)

        print('\n-----------------dataset options-----------------')
        if self.args.train_percent[0] and self.args.train_type == 'all':
            print('Sampling data from training sets. ({}x)'.format(self.args.train_percent))
        if self.args.reverse:
            print('Using reverse data to train. (2x)')
        if self.args.add_noise:
            print('Using noise data to train. ({}x)'.format(self.args.add_noise))
        if self.args.rotate:
            print('Using rotate data to train. ({}x)'.format(self.args.rotate))
        print('train_number = {}, total {}x train samples.'.format(self.train_number, self.sample_time))

        print('-----------------training options-----------------')
        print('model_name = {}, \ndataset = {},\nbatch_number = {},\nbatch_size = {},\nlr={}'.format(
            self.args.model_name,
            self.args.test_set, 
            batch_number, 
            self.args.batch_size,
            self.args.lr,
        ))

        print('\nPrepare training data...')
        self.train_tensor, self.train_index = self.prepare_model_inputs_all(self.agents_train)
        self.test_tensor, self.test_index = self.prepare_model_inputs_all(self.agents_test)
        train_length = self.prepare_model_inputs_batch(self.train_tensor, init=True)

        if self.args.save_model:
            self.test_data_save_path = os.path.join(self.args.log_dir, '{}.npy'.format(self.args.model_name + '{}'))
            np.save(self.test_data_save_path.format('test'), self.agents_test)   
            np.save(self.test_data_save_path.format('args'), self.args)
            
        test_results = []
        test_loss_dict = dict()
        test_loss_dict['-'] = 0

        batch_number = 1 + (train_length * self.args.epochs)// self.args.batch_size
        print(batch_number, train_length, self.args.epochs, self.args.batch_size)
        
        time_bar = tqdm(range(batch_number), desc='Training...')
        best_ade = 100.0
        best_epoch = 0
        for batch in time_bar:
            ADE = 0
            ADE_move_average = tf.cast(0.0, dtype=tf.float32)    # 计算移动平均
            loss_list = []
            
            obs_current, gt_current, train_sample_number = self.prepare_model_inputs_batch(self.train_tensor, self.args.batch_size)

            if train_sample_number < 20:
                continue

            with tf.GradientTape() as tape:
                model_output_current = self.forward_train(obs_current)
                loss_ADE, loss_list_current = self.loss(model_output_current, gt_current, obs=obs_current)
                ADE_move_average = 0.7 * loss_ADE + 0.3 * ADE_move_average

            ADE += loss_ADE
            grads = tape.gradient(ADE_move_average, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

            loss_list.append(loss_list_current)
            loss_list = tf.reduce_mean(tf.stack(loss_list), axis=0).numpy()

            epoch = (batch * self.args.batch_size) // train_length

            if (epoch >= self.args.start_test_percent * self.args.epochs) and (epoch % self.args.test_step == 0):
                model_output, loss_eval, _, _ = self.test_during_training(self.test_tensor, self.agents_test, self.test_index)
                test_results.append(loss_eval)
                test_loss_dict = create_loss_dict(loss_eval, self.loss_eval_namelist)
                ade_current = loss_eval[0]
                if ade_current <= best_ade:
                    best_ade = ade_current
                    best_epoch = epoch
                    
                    if self.args.save_best:
                        self.model.save(os.path.join(self.args.log_dir, '{}_epoch{}.h5'.format(self.args.model_name, epoch)))
                        np.savetxt(os.path.join(self.args.log_dir, 'best_ade_epoch.txt'), np.array([best_ade, best_epoch]))

            
            if epoch % 2 == 0:
                train_loss_dict = create_loss_dict(loss_list, self.loss_namelist)
                loss_dict = dict(train_loss_dict, **test_loss_dict) # 拼接字典
                time_bar.set_postfix(loss_dict)
                
            with summary_writer.as_default():
                for loss_name in loss_dict:
                    value = loss_dict[loss_name]
                    tf.summary.scalar(loss_name, value, step=epoch)

        print('Training done.')
        print('Tensorboard training log file is saved at "{}"'.format(self.args.log_dir))
        print('To open this log file, please use "tensorboard --logdir {} --port 54393"'.format(self.args.log_dir))
        
        latest_epochs = 10
        test_results = list2array(test_results)
        latest_results = np.mean(test_results[-latest_epochs-1:-1, :], axis=0)
        print('In latest {} test epochs, average test loss = {}'.format(
            latest_epochs,
            latest_results
        ))
        np.savetxt(os.path.join(self.args.log_dir, 'train_log.txt'), list2array(test_results))

        if self.args.save_model:
            self.model_save_path = os.path.join(self.args.log_dir, '{}.h5'.format(self.args.model_name))
            self.model.save(self.model_save_path)
            
            print('Trained model is saved at "{}".'.format(self.model_save_path.split('.h5')[0]))
            print('To re-test this model, please use "python main.py --load {}".'.format(self.model_save_path.split('.h5')[0]))
            
            model_name = self.model_save_path.split('.h5')[0].split('/')[-1]
            np.savetxt('./results/result-{}{}.txt'.format(model_name, self.args.test_set), latest_results)
            with open('./results/path-{}{}.txt'.format(model_name, self.args.test_set), 'w+') as f:
                f.write(self.model_save_path.split('.h5')[0])

    def test_batch(self, agents_test, test_on_neighbors=False, draw_results=True, batch_size=0.2, save_agents=False, social_refine=False):
        """
        Eval model on test sets.
        Results WILL be written to inputs.
        测试可以分段进行，并使用`batch_size`以百分比形式调节时间段长短;
        `test_on_neighbors`将会被自动打开当`social_refine == True`
        """
        print('-----------------Test options-----------------')
        print('model_name = {},\ndataset = {},\ntest_length= {} * length of test video.\n'.format(
            self.args.model_name,
            self.args.test_set, 
            batch_size,
        ))
        
        start_frame = agents_test[0].obs_frame
        end_frame = agents_test[-1].obs_frame
        frame_length = end_frame - start_frame
        
        # sort by obs time
        agents_batch = dict()
        for agent in agents_test:
            batch_index = min(int((agent.obs_frame - start_frame)/(batch_size * frame_length)), int(1/batch_size)-1)
            if not batch_index in agents_batch:
                agents_batch[batch_index] = []
            else:
                agents_batch[batch_index].append(agent)
        
        if social_refine:
            test_on_neighbors = True

        agents_batch, test_index = self.prepare_test_agents_batch(agents_batch, test_on_neighbors)
        
        # run test
        all_loss = []
        all_loss_batch = []
        for batch_index in agents_batch:
            batch_loss = []
            [test_tensor, _], _ = self.prepare_model_inputs_all(agents_batch[batch_index], calculate_neighbor=test_on_neighbors)
            pred = self.forward_train(test_tensor)
            pred = pred[0].numpy()

            for agent_index, index in enumerate(test_index[batch_index]):
                current_pred = pred[index]
                agents_batch[batch_index][agent_index].write_pred(current_pred[0])
                if test_on_neighbors:
                    agents_batch[batch_index][agent_index].write_pred_neighbor(current_pred[1:])
                
                if social_refine:
                    agents_batch[batch_index][agent_index].write_pred_sr(SocialRefine_one(
                        agent=agents_batch[batch_index][agent_index],
                        args=self.args,
                        epochs=10,
                        save=False,
                    ))
                
                loss = agents_batch[batch_index][agent_index].calculate_loss(SR=social_refine)
                all_loss.append(loss)
                batch_loss.append(loss)
            
            all_loss_batch.append(np.mean(np.stack(batch_loss), axis=0))
        
        average_loss = np.mean(np.stack(all_loss), axis=0)
        print('test_loss={}\nTest done.'.format(create_loss_dict(average_loss, ['ADE', 'FDE'])))
        # print(all_loss_batch)

        if draw_results:
            result_agents = []
            for batch_index in agents_batch:
                result_agents += agents_batch[batch_index]

            # draw results only
            for index in range(len(result_agents)):
                result_agents[index].draw_results(self.log_dir, '{}.png'.format(index), draw_neighbors=False)
            
            # draw results on video frames
            # tv = TrajVisual(save_base_path=self.args.log_dir, verbose=True, draw_neighbors=False, social_refine=social_refine)
            # tv.visual(result_agents, dataset=self.args.test_set)

        if save_agents:
            result_agents = []
            for batch_index in agents_batch:
                result_agents += agents_batch[batch_index]
            np.save(os.path.join(self.log_dir, 'pred.npy'), result_agents)
            return result_agents
    
    def test(self, agents_test, test_on_neighbors=False, social_refine=True, draw_results=True, batch_size=0.2, save_agents=False):
        """
        Eval model on test sets.
        Results WILL be written to inputs.
        """
        all_loss = []
        loss_name_list = ['ADE', 'FDE']
        loss_function = calculate_ADE_FDE_numpy

        self.test_tensor, self.test_index = self.prepare_model_inputs_all(self.agents_test)
        pred = self.forward_train(self.test_tensor)

        for index in tqdm(range(len(agents_test)), desc='Testing...'):
            obs = agents_test[index].get_train_traj().reshape([1, agents_test[index].obs_length, 2])
            
            # if test_on_neighbors and agents_test[index].neighbor_number > 0:
            #     obs_neighbor = (np.stack(agents_test[index].get_neighbor_traj())).reshape([agents_test[index].neighbor_number, agents_test[index].obs_length, 2])
            #     obs = np.concatenate([obs, obs_neighbor], axis=0)

            
            agents_test[index].write_pred(pred[0].numpy()[index])
            # if test_on_neighbors:
            #     agents_test[index].write_pred_neighbor(pred[1:].numpy()[index])

            # if social_refine:
            #     agents_test[index].write_pred_sr(SocialRefine_one(agents_test[index], self.args_old))
            
            if draw_results:
                agents_test[index].draw_results(self.log_dir, '{}.png'.format(index), draw_neighbors=False # test_on_neighbors
                )

            all_loss.append(agents_test[index].calculate_loss())
            
        
        loss = np.mean(np.stack(all_loss), axis=0)
            
        print('test_loss={}'.format(create_loss_dict(loss, loss_name_list)))
        # for l in loss:
        #     print(loss, end='\t')
        print('\nTest done.')

        if save_agents:
            np.save(os.path.join(self.log_dir, 'pred.npy'), agents_test)
        return agents_test
    
    def prepare_test_agents_batch(self, agents_batch:dict, test_on_neighbors=False):
        """
        Prepare test agents and save test order. (When test on neighbors of current agent)
        returns: Test agents (in batch order) `agents_batch` and their order `test_index`.
        """
        # save test order
        test_index = dict()
        for batch_index in agents_batch:
            total_count = 0
            if not batch_index in test_index:
                test_index[batch_index] = []

            for agent_index, _ in enumerate(agents_batch[batch_index]):
                start_count = total_count
                total_count += 1
                if test_on_neighbors:
                    nei_len = agents_batch[batch_index][agent_index].neighbor_number
                    total_count += nei_len
                test_index[batch_index].append([i for i in range(start_count, total_count)])
        
        return agents_batch, test_index


class BGM(Base_Model):
    """
    `B`uilding a Dynamic `G`uidance `M`ap for Trajectory Prediction
    """
    def __init__(self, train_info, args):
        super().__init__(train_info, args)
        self.given_maps_when_test=False

    def create_model(self):
        positions = keras.layers.Input(shape=[self.obs_frames, 2])
        traj_maps = keras.layers.Input(shape=[self.args.gridmapsize, self.args.gridmapsize])
        start_point = tf.reshape(positions[:, -1, :], [-1, 1, 2])
        
        # sequence feature
        positions_n = positions - start_point
        positions_embadding_lstm = keras.layers.Dense(64)(positions_n)
        traj_feature = keras.layers.LSTM(64, return_sequences=True)(positions_embadding_lstm)
        feature_flatten = tf.reshape(traj_feature, [-1, self.obs_frames * 64])
        sequence_feature = keras.layers.Dense(self.obs_frames * 32, activation=tf.nn.tanh)(feature_flatten)
        
        # context feature
        traj_maps_r = tf.reshape(traj_maps, [-1, self.args.gridmapsize, self.args.gridmapsize, 1])
        average_pooling = keras.layers.AveragePooling2D([2, 2], padding='same')(traj_maps_r)
        cnn1 = keras.layers.Conv2D(32, [8, 8], activation=tf.nn.relu)(average_pooling)
        cnn2 = keras.layers.Conv2D(32, [5, 5], activation=tf.nn.relu)(cnn1)
        pooling2 = keras.layers.AveragePooling2D([2, 2])(cnn2)
        flatten = keras.layers.Flatten()(pooling2)
        context_feature = keras.layers.Dense(self.obs_frames * 32, activation=tf.nn.tanh)(flatten)
        
        # joint feature
        concat_feature = tf.concat([sequence_feature, context_feature], axis=-1)
        feature_fc = keras.layers.Dense(self.pred_frames * 64)(concat_feature)
        feature_reshape = tf.reshape(feature_fc, [-1, self.pred_frames, 64])
        output5 = keras.layers.Dense(2)(feature_reshape)
        output5 = output5 + start_point
        
        lstm = keras.Model(inputs=[positions, traj_maps], outputs=[output5])
        lstm.build(input_shape=[None, self.obs_frames, 2])
        lstm_optimizer = keras.optimizers.Adam(lr=self.args.lr)
        
        return lstm, lstm_optimizer

    def get_feature(self, inputs, layer_name):
        submodel = keras.Model(inputs=self.model.input, outputs=self.model.get_layer(layer_name).output)
        return submodel(inputs)

    def prepare_model_inputs_all(self, input_agents, calculate_neighbor=False):
        input_trajs = []
        input_maps = []
        gt = []
        agent_index = []
        for agent_index_current, agent in enumerate(tqdm(input_agents, desc='Prepare inputs...')):
            input_trajs.append(agent.get_train_traj())
            input_maps.append(agent.get_traj_map())
            gt.append(agent.get_gt_traj())
            agent_index.append(agent_index_current)

            if calculate_neighbor and agent.neighbor_number:
                for traj, traj_map in zip(agent.get_neighbor_traj(), agent.get_traj_map_for_neighbors()):
                    input_trajs.append(traj)
                    input_maps.append(agent.get_traj_map())
                    # No GT

        input_trajs = tf.cast(tf.stack(input_trajs), tf.float32)
        input_maps = tf.cast(tf.stack(input_maps), tf.float32)
        gt = tf.cast(tf.stack(gt), tf.float32)
        return [[input_trajs, input_maps], gt], agent_index

    def prepare_test_agents_batch(self, agents_batch:dict, test_on_neighbors=False):
        # create trajectory map for each batch
        if not type(self.given_maps_when_test) == np.ndarray:
            traj_maps = [TrajectoryMapManager(agents_batch[batch_index]) for batch_index in agents_batch]
        else:
            traj_maps = self.given_maps_when_test
            print('Using given maps')

        # write traj map and save batch order     
        test_index = dict()
        for batch_index, traj_map in zip(agents_batch, traj_maps):
            total_count = 0
            if not batch_index in test_index:
                test_index[batch_index] = []
                
            for agent_index, _ in enumerate(agents_batch[batch_index]):
                agents_batch[batch_index][agent_index].write_traj_map(traj_map)
                start_count = total_count
                total_count += 1
                if test_on_neighbors:
                    agents_batch[batch_index][agent_index].write_traj_map_for_neighbors(traj_map)
                    nei_len = agents_batch[batch_index][agent_index].neighbor_number
                    total_count += nei_len
                test_index[batch_index].append([i for i in range(start_count, total_count)])

        return agents_batch, test_index


class Linear(Base_Model):
    def __init__(self, train_info, args):
        super().__init__(train_info, args)
        self.args.batch_size = 1
        self.args.epochs = 1
        self.args.draw_results = False
        self.args.train_percent = 0.0
    
    def run_commands(self):
        self.get_data()
        self.model, self.optimizer = self.create_model()
        self.test_batch(
            self.agents_test,
            batch_size=1.0,
            test_on_neighbors=False,
            social_refine=False,
            draw_results=self.args.draw_results,
            save_agents=False
        )

    def predict_linear(self, x, y, x_p, diff_weights=0):
        if diff_weights == 0:
            P = np.diag(np.ones(shape=[x.shape[0]]))
        else:
            P = np.diag(softmax([(i+1)**diff_weights for i in range(x.shape[0])]))

        A = tf.transpose(tf.stack([np.ones_like(x), x]))
        A_p = tf.transpose(tf.stack([np.ones_like(x_p), x_p]))
        Y = tf.transpose(y)

        P = tf.cast(P, tf.float32)
        A = tf.cast(A, tf.float32)
        A_p = tf.cast(A_p, tf.float32)
        Y = tf.reshape(tf.cast(Y, tf.float32), [-1, 1])
        
        B = tf.matmul(tf.matmul(tf.matmul(tf.linalg.inv(tf.matmul(tf.matmul(tf.transpose(A), P), A)), tf.transpose(A)), P), Y)
        Y_p = np.matmul(A_p, B)
        return Y_p, B

    def predict_linear_for_person(self, positions, diff_weights):
        t = np.array([t for t in range(self.obs_frames)])
        t_p = np.array([t + self.obs_frames for t in range(self.pred_frames)])
        x = tf.transpose(positions)[0]
        y = tf.transpose(positions)[1]

        x_p, _ = self.predict_linear(t, x, t_p, diff_weights=diff_weights)
        y_p, _ = self.predict_linear(t, y, t_p, diff_weights=diff_weights)

        return tf.transpose(tf.reshape(tf.stack([x_p, y_p]), [2, self.pred_frames]))
    
    def create_model(self):
        return self.predict_linear_for_person, 0

    def forward_train(self, train_tensor, index):
        input_trajs = train_tensor[0][index[0]:index[1]]
        gt = train_tensor[1][index[0]:index[1]]

        results = []
        for inputs_current in input_trajs:
            results.append(self.model(inputs_current, diff_weights=self.args.diff_weights))
        
        return [tf.stack(results)], gt, input_trajs

    def forward_test(self, test_tensor):
        input_trajs = test_tensor[0]
        gt = test_tensor[1]
        
        results = []
        for inputs_current in input_trajs:
            results.append(self.model(inputs_current, diff_weights=self.args.diff_weights))     
        
        return output, gt, input_trajs


"""
helpmethods
"""

def create_loss_dict(loss, name_list):
    return dict(zip(name_list, loss))


def softmax(x):
    return np.exp(x)/np.sum(np.exp(x),axis=0)


def calculate_ADE(pred, GT):
    """input_shape = [batch, pred_frames, 2]"""
    pred = tf.cast(pred, tf.float32)
    GT = tf.cast(GT, tf.float32)
    return tf.reduce_mean(tf.linalg.norm(pred - GT, ord=2, axis=2))
    

def calculate_FDE(pred, GT):
    pred = tf.cast(pred, tf.float32)
    GT = tf.cast(GT, tf.float32)
    return tf.reduce_mean(tf.linalg.norm(pred[:, -1, :] - GT[:, -1, :], ord=2, axis=1))
