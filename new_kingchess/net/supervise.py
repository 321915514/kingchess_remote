import os
import numpy as np
import torch
from torch.utils.data import Dataset
import train_pb2
from agent.expert_agent import Expert_agent
from fundamental.utils import print_board
import concurrent.futures




# -*- coding: utf-8 -*-
"""
An implementation of the training pipeline of AlphaZero for Gomoku

@author: Junxiao Song
"""

# from __future__ import print_function
import os

from torch.utils.data import Dataset, DataLoader

from a3c.test_net import test_model

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import pickle
import sys

cur_path = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(cur_path)
import random
import time
import torch.multiprocessing as mp
import numpy as np
from collections import defaultdict, deque

import torch
from torch import nn, optim

from fundamental.board import Board, GameState, Move
from fundamental.coordinate import Player
from fundamental.utils import print_board
import torch.nn.functional as F
from net.mcts_pure import MCTSPlayer as MCTS_Pure
from net.encoder import encoder_board, moves2flip, moves2horizontally
from net.mcts_alphazreo import MCTSPlayer
from net.config import CONFIG
# from policy_value_net import PolicyValueNet  # Theano and Lasagne
from net.policy_value_net_pytorch import PolicyValueNet, Net  # Pytorch
import os
import uuid
# import train_pb2
# from utils.util import LOGGER
from decode_proto_file import DecodeDataset
import logging

logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p',
                    level=logging.INFO)
logger = logging.getLogger(__file__)


# os.environ['CUDA_VISIBLE_DEVICES'] = '8'

# from transfer_trt import export_engine


# from policy_value_net_tensorflow import PolicyValueNet # Tensorflow
# from policy_value_net_keras import PolicyValueNet # Keras


# class DecodeDataset:
#     def __init__(self, dataset: str):
#         self.dataset = dataset
#         self.file_paths = []
#         # self.read_dir()
#
#     def read_dir(self):
#         # if os.path.isdir(self.dataset):
#         self.file_paths = [os.path.join(self.dataset, file) for file in os.listdir(self.dataset)]
#
#
#     # def get_data(self):
#     #     self.read_dir()
#     #     for file_path in self.file_paths:
#     #         try:
#     #             train_list = self.parse_file(file_path)
#     #             yield train_list
#     #             # 删除已经处理过的文件
#     #             os.remove(file_path)
#     #         except Exception as e:
#     #             # 处理异常，例如文件不存在或处理文件时出错
#     #             print(f"Error processing file {file_path}: {e}")
#     #             continue
#     #             # 根据需要决定是否继续迭代或中断生成器
#     #             # 如果选择中断，可以抛出异常或使用其他机制
#
#     def process_files_concurrently(self):
#         self.read_dir()
#         with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:  # 设置线程池大小
#             select_file = random.sample(self.file_paths, min(1000, len(self.file_paths)))
#             future_to_file = {executor.submit(self.parse_file, file_path): file_path for file_path in select_file}
#             result = []
#             for future in concurrent.futures.as_completed(future_to_file):
#                 file_path = future_to_file[future]
#                 try:
#                     train_list = future.result()  # 获取处理结果
#                     # 在这里可以使用train_list，比如将其加入一个列表或进行其他操作
#                     # print(f"Processed {file_path}")
#
#                     # 删除已经处理过的文件（请小心操作，确保不会误删重要文件）
#                     # if os.path.exists(file_path):
#                     #     os.remove(file_path)
#                     result.extend(train_list)
#                 except Exception as e:
#                     print(f"Error processing file {file_path}: {e}")
#                     continue
#             return result
#
#     def parse_file(self, file_path):
#         game_file = open(file_path, 'rb').read()
#
#         train_data_array = train_pb2.TripletArray()
#
#         train_data_array.ParseFromString(game_file)
#
#         train_list = []
#
#         for triplet in train_data_array.triplets:
#             array_list = []
#             move = [] # 2024/4/4 modify
#             for arr in triplet.array:
#                 array_list.append(arr)
#             # print(len(array_list))
#             board_np = np.array(array_list).reshape((5, 9, 9))
#
#             # transformer
#             # board_np = np.array(array_list[:243])
#             # board_real, player = decoder_board(board_np)
#             # print_board(board_real)
#             for d in triplet.dictionary:
#                 move.append(d)
#             moves = np.array(move, dtype=float)
#             value = triplet.value
#             train_list.append((board_np, moves, value))
#
#         return train_list




class TrainPipeline():
    def __init__(self, init_model=None):
        # params of the board and the game
        self.board_width = 9
        self.board_height = 5
        self.device = 'cuda:7'
        # self.n_in_row = 4
        # self.board = Board(num_rows=self.board_height, num_cols=self.board_width)
        # self.game = GameState.new_game(self.board_height, self.board_width)
        # training params
        self.learn_rate = 1e-3
        self.lr_multiplier = 1.0  # adaptively adjust the learning rate based on KL
        self.temp = 1.0  # the temperature param
        self.n_playout = 1200  # num of simulations for each move
        self.c_puct = 5

        self.buffer_size = 20000
        self.batch_size = 512  # mini-batch size for training
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.play_batch_size = 1
        self.epochs = 50  # num of train_steps for each update
        self.kl_targ = 0.02
        self.check_freq = 100
        self.game_batch_num = 10000
        self.data_parse = DecodeDataset('/home/dev/dataset_cpp')
        self.best_win_ratio = 0.0
        self.black_win = 0
        self.white_win = 0
        # self.export_engine_freq = 20
        # num of simulations used for the pure mcts, which is used as
        # the opponent to evaluate the trained policy

        self.pure_mcts_playout_num = 10
        self.iters = 0
        if init_model:
            try:
                # start training from an initial policy-value net
                self.policy_value_net = PolicyValueNet(model_file=init_model)
                logger.info('加载上次最终模型')
            except:
                # start training from a new policy-value net
                logger.info('从头训练')
                self.policy_value_net = PolicyValueNet(model_file=CONFIG['pytorch_model_path'])
            # self.mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
            #                               c_puct=self.c_puct,
            #                               n_playout=self.n_playout,
            #                               is_selfplay=1)
        else:
            logger.info('从头训练')
            self.policy_value_net = PolicyValueNet()

    # def load_model(self):
    #     model_path = CONFIG['pytorch_current_model_path']
    #     try:
    #         self.policy_value_net = PolicyValueNet(model_file=model_path)
    #         logger.info('已加载最新模型')
    #     except:
    #         # self.policy_value_net = PolicyValueNet()
    #
    #         self.policy_value_net = PolicyValueNet(CONFIG['pytorch_model_path'])
    #
    #         logger.info('已加载初始模型')

    def start_play(self, current_mcts_player, init_mcts_player, is_shown=0):
        row_size = 5
        col_size = 9
        game = GameState.new_game(row_size, col_size)
        bot1 = current_mcts_player
        bot2 = init_mcts_player
        while True:
            end, winner = game.game_over()
            if end:
                break
            if is_shown:
                print_board(game.board)
            if game.player == Player.black:
                a = bot1.select_move(game)
                if isinstance(a, int):
                    move = game.a_trans_move(a)
                else:
                    move = a
            else:
                a = bot2.select_move(game)
                if isinstance(a, int):
                    move = game.a_trans_move(a)
                else:
                    move = a
            game = game.apply_move(move)
        return winner

    def policy_update(self):
        """update the policy-value net"""
        mini_batch = random.sample(self.data_buffer, self.batch_size)
        state_batch = [data[0] for data in mini_batch]
        state_batch = np.array(state_batch).astype('float32')
        # mcts_probs_batch = []
        #
        # for data in mini_batch:
        #     prob = np.zeros([1125]).astype('float32')
        #     for item in data[1]:
        #         prob[int(item[0])] = item[1]
        #     mcts_probs_batch.append(prob)

        mcts_probs_batch = [data[1] for data in mini_batch] # 2024/4/4 modify data[1] is list !!!!!!!!!!!!
        # mcts_probs_batch = [np.pad(arr, (0, 64 - len(arr)), 'constant', constant_values=0) for arr in mcts_probs_batch] # modify 24/4/19
        # mcts_probs_batch =[arr[:64] if len(arr)>64 else np.pad(arr, (0, 64 - len(arr)), 'constant', constant_values=0) for arr in mcts_probs_batch] # modify 24/4/19

        mcts_probs_batch = np.array(mcts_probs_batch).astype('float32')
        winner_batch = [data[2] for data in mini_batch]
        winner_batch = np.array(winner_batch).astype('float32')
        # old_probs, old_v = self.policy_value_net.policy_value(state_batch)  # (512,64)
        for i in range(self.epochs):
            accuracy, loss, policy, value_loss, entropy = self.policy_value_net.train_step(
                state_batch,
                mcts_probs_batch,
                winner_batch,
                # self.learn_rate * self.lr_multiplier)
                self.learn_rate)
            # new_probs, new_v = self.policy_value_net.policy_value(state_batch)
            # kl = np.mean(np.sum(old_probs * (
            #         np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)),
            #                     axis=1)
            #              )
            # if kl > self.kl_targ * 4:  # early stopping if D_KL diverges badly
            #     break
        # adaptively adjust the learning rate
        # if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
        #     self.lr_multiplier /= 1.5
        # elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
        #     self.lr_multiplier *= 1.5

        # explained_var_old = (1 -
        #                      np.var(np.array(winner_batch) - old_v.flatten()) /
        #                      np.var(np.array(winner_batch)))
        # explained_var_new = (1 -
        #                      np.var(np.array(winner_batch) - new_v.flatten()) /
        #                      np.var(np.array(winner_batch)))
        logger.info(("loss:{},"
                     "policy_loss:{},"
                     "value_loss:{},"
                     "entropy:{},"
                     "accuracy:{},"
                     ).format(loss,
                              policy,
                              value_loss,
                              entropy,
                              accuracy,
                              ))
        # logger.info(("kl:{:.5f},"
        #              "lr_multiplier:{:.3f},"
        #              "loss:{},"
        #              "entropy:{},"
        #              "explained_var_old:{:.3f},"
        #              "explained_var_new:{:.3f}"
        #              ).format(kl,
        #                       self.lr_multiplier,
        #                       loss,
        #                       entropy,
        #                       explained_var_old,
        #                       explained_var_new))
        return loss, entropy

    def policy_evaluate(self, n_games=50):
        """
        Evaluate the trained policy by playing against the pure MCTS player
        Note: this is only for monitoring the progress of training
        """
        current_mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn, c_puct=self.c_puct,
                                         n_playout=10)

        expert = Expert_agent()

        # init_mcts_player = MCTSPlayer(PolicyValueNet(9, 9, model_file='./model/init.model').policy_value_fn,
        #                               c_puct=self.c_puct, n_playout=self.n_playout)
        # pure_mcts_player = MCTS_Pure(c_puct=5, n_playout=self.pure_mcts_playout_num)
        win_cnt = {'current_net_black': 0, 'current_net_white': 0}
        for i in range(n_games):
            winner = self.start_play(current_mcts_player, expert, is_shown=0)
            if winner == Player.black:
                win_cnt['current_net_black'] += 1
        for i in range(n_games):
            winner = self.start_play(expert, current_mcts_player, is_shown=0)
            if winner == Player.white:
                win_cnt["current_net_white"] += 1

        current_black_win_rate = win_cnt['current_net_black'] / 50
        current_white_win_rate = win_cnt['current_net_white'] / 50
        logger.info("num_playouts:{}, black win: {},white win:{}, n_games : {}".format(self.pure_mcts_playout_num, current_black_win_rate,current_white_win_rate,
                                                                  n_games))
        return current_black_win_rate, current_white_win_rate
    # class GameDataset(Dataset):
    #     def __init__(self, data):
    #         self.data = data
    #
    #     def __len__(self):
    #         return len(self.data)
    #
    #     def __getitem__(self, idx):
    #         state, action, score = self.data[idx]
    #         # result = self.data[idx]
    #         # print(result)
    #         # 将状态转换为 PyTorch 张量
    #         state_tensor = torch.tensor(state, dtype=torch.float32)
    #         # 将动作转换为整数索引（根据具体情况修改）
    #         # actions = np.zeros((1125))
    #         # actions[np.argmax(action)] = 1
    #         action_tensor = torch.tensor(action, dtype=torch.float32)  # 假设动作是一个整数，表示输出动作的位置
    #         score_tensor = torch.tensor(score, dtype=torch.float32)
    #         return state_tensor, action_tensor, score_tensor

    def train_pkl(self):
        os.makedirs('supervise_model_conv', exist_ok=True)
        # with open('./collect_expert_data/picture_data.pkl',  'rb') as data_dict:

        while True:
            while True:
                try:
                    get_data = self.data_parse.process_files_concurrently()
                    self.data_buffer.extend(get_data)
                    logger.info(f'已加载数据,{len(self.data_buffer)}')
                    break
                except Exception as e:
                    time.sleep(30)

            if len(self.data_buffer):
                mini_batch = random.sample(self.data_buffer, self.batch_size)
                state_batch = [data[0] for data in mini_batch]
                state_batch = np.array(state_batch).astype('float32')
                mcts_probs_batch = [data[1] for data in mini_batch]
                mcts_probs_batch = np.array(mcts_probs_batch).astype('float32')
                winner_batch = [data[2] for data in mini_batch]
                winner_batch = np.array(winner_batch).astype('float32')
                # criterion = nn.CrossEntropyLoss()
                # mse = nn.MSELoss()
                optimizer = optim.Adam(self.policy_value_net.policy_value_net.parameters(), lr=0.0001)

                num_epochs = 50
                for epoch in range(num_epochs):
                    state_batch = torch.tensor(state_batch, dtype=torch.float32).to(self.device)
                    mcts_probs = torch.tensor(mcts_probs_batch, dtype=torch.float32).to(self.device)
                    winner_batch = torch.tensor(winner_batch, dtype=torch.float32).to(self.device)

                    optimizer.zero_grad()

                    log_act_probs, value = self.policy_value_net.policy_value_net(state_batch)  # (512,5,9,9)

                    value = torch.reshape(value, shape=[-1])
                    value_loss = F.mse_loss(value, winner_batch)

                    policy_loss = -torch.mean(torch.sum(mcts_probs * torch.log(log_act_probs), 1), dim=-1)

                    loss = value_loss + policy_loss
                    loss.backward()
                    optimizer.step()

                    if (epoch + 1) % 200 == 0:
                        torch.save(self.policy_value_net.policy_value_net.state_dict(),
                                   './supervise_model_conv/current.pth')
                        # win_white, win_black = test_model()
                        torch.save(self.policy_value_net.policy_value_net.state_dict(),
                                   './supervise_model_conv/current_' + str(epoch + 1) + '.pth')
                        # result[epoch] = (win_white, win_black)
                        # print(
                        #     f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f},win_white:{win_white},win_black:{win_black}')
                    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    def run(self):
        """run the training pipeline"""
        try:
            while True:
                if not os.path.dirname("supervise_model_resnet5"):
                    os.makedirs('supervise_model_resnet5', exist_ok=True)

                # self.policy_value_net.save_model('./model/init.model')
                # self.collect_selfplay_data(self.play_batch_size)
                # LOGGER.info("batch i:{}, episode_len:{}".format(
                #     i + 1, self.episode_len))

                # load game to buffer

                # generator = self.data_parse.process_files_concurrently()  # generator
                #
                # for play_data in generator:  # 包含增强的数据其中play_data为list(state,pro,value)
                #     self.data_buffer.extend(play_data)

                while True:
                    try:
                        # with open(CONFIG['train_data_buffer_path'], 'rb') as data_dict:
                        #     data_file = pickle.load(data_dict)
                        #     self.data_buffer = data_file['data_buffer']
                        #     # self.iters = data_file['iters']
                        #     # self.iters += 1
                        #     del data_file
                        get_data = self.data_parse.process_files_concurrently()
                        self.data_buffer.extend(get_data)
                        logger.info(f'已加载数据,{len(self.data_buffer)}')
                        break
                    except Exception as e:
                        time.sleep(30)

                if len(self.data_buffer) > self.batch_size:
                    self.policy_update()
                    self.iters += 1
                    self.policy_value_net.save_model("./supervise_model_resnet5/current.pth")
                    logger.info("current self-play batch: {}".format(self.iters+1))

                if self.iters % 100 == 0:
                    black_win, white_win = self.policy_evaluate()
                    if black_win > self.black_win:
                        self.black_win = black_win
                        self.policy_value_net.save_model("./supervise_model_resnet5/current_best_black.pth")
                    if white_win > self.white_win:
                        self.white_win = white_win
                        self.policy_value_net.save_model("./supervise_model_resnet5/current_best_white.pth")
        except KeyboardInterrupt:
            print('\n\rquit')






if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '7'
    # s = time.time()
    training_pipeline = TrainPipeline('./supervise_model_resnet5/current_best_black.pth') # resnet3 black win: 0.8,white win:0.02  black win: 0.92,white win:0.0, n_games : 50
    #
    training_pipeline.run()
    #
    # data = []
    # dataset = DecodeDataset('/home/dev/dataset_cpp')
    # for i in dataset:
    #     data.extend(i)
    # dataset.save_to_pkl('dataset.pkl')  # 保存到pkl文件

    # end = time.time() - s

    # logger.info(end)

# 使用示例
# if __name__ == '__main__':
#     dataset = DecodeDataset('/home/dev/dataset_cpp')
#
#

# data_buffer.extend(data_file)

# print(len(dataset))  # 打印数据集大小
# sample = dataset[0]  # 获取第一个样本
# print(sample)
