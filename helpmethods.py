'''
@Author: ConghaoWong
@Date: 2019-12-20 09:39:11
LastEditors: Conghao Wong
LastEditTime: 2020-08-16 00:06:03
@Description: helpmethods
'''

import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.manifold import TSNE

from tqdm import tqdm

def list2array(x):
    return np.array(x)


def dir_check(target_dir):
    """
    Used for check if the `target_dir` exists.
    """
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)

    return target_dir


def reduce_dim(x, out_dim, pca=True):
    if not pca:
        tsne = TSNE(n_components=out_dim)
        result = tsne.fit_transform(x)
        return result

    else:
        x = tf.constant(x)
        s, u, v = tf.linalg.svd(x)
        return tf.matmul(u[:, :out_dim], tf.linalg.diag(s[:out_dim])).numpy()


def calculate_feature_lower_dim(feature, reduction_dimension=2, pca=True, regulation=True):
    current_dimension = feature.shape[1]
    if reduction_dimension < current_dimension:
        feature_vector_low_dim = reduce_dim(
            feature,
            out_dim=reduction_dimension,
            pca=pca,
        )
    else:
        feature_vector_low_dim = feature
    
    if regulation:
        feature_vector_low_dim = (feature_vector_low_dim - np.min(feature_vector_low_dim))/(np.max(feature_vector_low_dim) - np.min(feature_vector_low_dim))
    
    return feature_vector_low_dim

# help methods for linear predict
def softmax(x):
    return np.exp(x)/np.sum(np.exp(x),axis=0)


def __predict_linear(x, y, x_p, diff_weights=0):
    if diff_weights == 0:
        P = np.diag(np.ones(shape=[x.shape[0]]))
    else:
        P = np.diag(softmax([(i+1)**diff_weights for i in range(x.shape[0])]))

    A = np.stack([np.ones_like(x), x]).T
    A_p = np.stack([np.ones_like(x_p), x_p]).T
    Y = y.T
    B = np.matmul(np.matmul(np.matmul(np.linalg.inv(np.matmul(np.matmul(A.T, P), A)), A.T), P), Y)
    Y_p = np.matmul(A_p, B)
    return Y_p, B


def predict_linear_for_person(position, time_pred, different_weights=0.95):
    """
    对二维坐标的最小二乘拟合
    注意：`time_pred`中应当包含现有的长度，如`len(position)=8`, `time_pred=20`时，输出长度为20
    """
    time_obv = position.shape[0]
    t = np.arange(time_obv)
    t_p = np.arange(time_pred)
    x = position.T[0]
    y = position.T[1]

    x_p, _ = __predict_linear(t, x, t_p, diff_weights=different_weights)
    y_p, _ = __predict_linear(t, y, t_p, diff_weights=different_weights)

    return np.stack([x_p, y_p]).T


def calculate_ADE_FDE_numpy(pred, GT):
    all_loss = np.linalg.norm(pred - GT, ord=2, axis=1)
    ade = np.mean(all_loss)
    fde = all_loss[-1]
    return ade, fde


def draw_results(feature, result, all_traj='null', detail_save_path='null', whole_save_path='null', only_features=False):
    """
    根据分类结果画图

    `feature`: 需要画图的2维特征, 
    `result`: 分类结果标签，值为0，1，2，...，n-1, 
    `all_traj`: 可选，输入所有特征对应的轨迹，有输入则画出, 
    `detail_save_path`: 特征点与所有类轨迹的存储位置，有输入则保存, 
    `whole_save_path`: 整体轨迹分布存储位置，有输入则保存, 
    `only_features`: 选择图中是否仅画出特征分布

    """
    cluster_number = (np.max(result) + 1).astype(np.int)
    line_number = max(int(cluster_number//5), 1)
    color = [
        plt.cm.Greys, plt.cm.BuGn, plt.cm.BuPu,
        plt.cm.Oranges, plt.cm.PuBu, plt.cm.PuRd,
        plt.cm.Reds, plt.cm.YlGn, plt.cm.YlGnBu,
        plt.cm.YlOrBr, plt.cm.YlOrRd,
        plt.cm.Greys, plt.cm.BuGn, plt.cm.BuPu,
        plt.cm.Oranges, plt.cm.PuBu, plt.cm.PuRd,
        plt.cm.Reds, plt.cm.YlGn, plt.cm.YlGnBu,
        plt.cm.YlOrBr, plt.cm.YlOrRd,
        plt.cm.Greys, plt.cm.BuGn, plt.cm.BuPu,
        plt.cm.Oranges, plt.cm.PuBu, plt.cm.PuRd,
        plt.cm.Reds, plt.cm.YlGn, plt.cm.YlGnBu,
        plt.cm.YlOrBr, plt.cm.YlOrRd,
        plt.cm.Greys, plt.cm.BuGn, plt.cm.BuPu,
        plt.cm.Oranges, plt.cm.PuBu, plt.cm.PuRd,
        plt.cm.Reds, plt.cm.YlGn, plt.cm.YlGnBu,
        plt.cm.YlOrBr, plt.cm.YlOrRd,
    ]

    if not all_traj == 'null':
        mean_traj = []
        for i in range(cluster_number):
            person_index = np.where(result == i)[0].astype(np.int)
            traj_current = np.array([all_traj[person]
                                     for person in person_index])
            mean = np.mean(traj_current, axis=0)
            mean_traj.append(mean)

    if not detail_save_path == 'null':
        plt.figure(figsize=(16, 9))
        for i in range(cluster_number):
            person_index = np.where(result == i)[0].astype(np.int)
            if not only_features:
                plt.subplot(line_number, (cluster_number+1) //
                            line_number + 1, 1)
            plt.scatter(
                feature[person_index, 0],
                feature[person_index, 1],
                c=color[i](200),
            )
            plt.axis('scaled')

        if (not only_features) and (not all_traj == 'null'):
            for i in range(cluster_number):
                person_index = np.where(result == i)[0].astype(np.int)
                plt.subplot(line_number, (cluster_number+1) //
                            line_number + 1, i+2)
                for person in person_index:
                    traj = all_traj[person]
                    plt.scatter(
                        traj[:, 0],
                        traj[:, 1],
                        c=0.5 + 0.5 *
                        (np.arange(0, traj.shape[0])/(traj.shape[0]-1)),
                        cmap=color[i],
                        alpha=0.5
                    )

                plt.plot(
                    mean_traj[i][:, 0],
                    mean_traj[i][:, 1],
                    '-*',
                    c='black',
                )
                plt.axis('scaled')

        plt.savefig(detail_save_path)
        plt.close()

    if (not whole_save_path == 'null') and (not all_traj == 'null'):
        plt.figure(figsize=(9, 9))
        plt.subplot(1, 2, 1)
        for i in range(1, cluster_number):
            #person_index = np.where(result==i)[0].astype(np.int)
            # for person in person_index:
            traj = mean_traj[i]
            plt.scatter(
                traj[:, 0],
                traj[:, 1],
                c=0.5 + 0.5*(np.arange(0, traj.shape[0])/(traj.shape[0]-1)),
                cmap=color[i],
                alpha=0.5
            )

        plt.subplot(1, 2, 2)
        for i in range(1, cluster_number):
            person_index = np.where(result == i)[0].astype(np.int)
            line_number = all_traj[0].shape[0]
            line_index = np.arange(0, line_number, step=4)
            for person in person_index:
                traj = all_traj[person]
                plt.scatter(
                    traj[line_index, 0],
                    traj[line_index, 1],
                    c=color[i](200),
                    alpha=0.1,
                    s=50,
                    marker='o',
                    linewidths=0,
                )

        plt.axis('scaled')
        plt.savefig(whole_save_path)
        plt.close()


def draw_test_results(agents_test, log_dir, loss_function, save=True, train_base='agent'):
    if save:
        save_base_dir = dir_check(os.path.join(log_dir, 'test_figs/'))
        save_format = os.path.join(save_base_dir, '{}-pic{}.png')
    
    loss_static = []
    loss_move = []

    for index, agent in enumerate(tqdm(agents_test)):
        obs = agent.get_train_traj()
        gt = agent.get_gt_traj()
        pred = agent.get_pred_traj()

        # if len(pred.shape) == 3:
        #     pred_mean = np.mean(agent.pred, axis=0)
        # else:
        #     pred_mean = pred

        # loss = loss_function(pred_mean, gt)
        # if np.linalg.norm(obs[0] - gt[-1]) <= 1.0:
        #     state = 's'
        #     loss_static.append(loss)
        # else:
        #     state = 'n'
        #     loss_move.append(loss)
        
        if save:
            # print('Saving fig {}...'.format(i), end='\r')
            # plt.figure(figsize=(20, 20))
            for i in range(len(obs)):
                plt.plot(pred[i].T[0], pred[i].T[1], '-*')
                plt.plot(gt[i].T[0], gt[i].T[1], '-o')
                plt.plot(obs[i].T[0], obs[i].T[1], '-o')
        
            plt.axis('scaled')

            if train_base == 'agent':
                ade = loss_function(pred[0], gt[0])
            elif train_base == 'frame':
                ade = loss_function(pred, gt)

            plt.title('ADE={}, frame=[{}, {}]'.format(
                ade,
                agent.start_frame,
                agent.end_frame,
            ))
            plt.savefig(save_format.format('f', index))
            plt.close()
    
    # loss_static = np.mean(np.stack(loss_static))
    # loss_move = np.mean(np.stack(loss_move))
    
    # if save:
    #     print('\nSaving done.')
    # print('loss_s = {}, loss_n = {}'.format(loss_static, loss_move))
    