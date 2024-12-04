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
# from decode_proto_file import DecodeDataset
import logging

logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p',
                    level=logging.INFO)
logger = logging.getLogger(__file__)

#os.environ['CUDA_VISIBLE_DEVICES'] = '8'

# from transfer_trt import export_engine


# from policy_value_net_tensorflow import PolicyValueNet # Tensorflow
# from policy_value_net_keras import PolicyValueNet # Keras


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

        self.buffer_size = 100000
        self.batch_size = 512  # mini-batch size for training
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.play_batch_size = 1
        self.epochs = 50  # num of train_steps for each update
        self.kl_targ = 0.02
        self.check_freq = 100
        self.game_batch_num = 10000
        # self.data_parse = DecodeDataset('./dataset')
        self.best_win_ratio = 0.0
        self.export_engine_freq = 20
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

    def start_play(self, current_mcts_player: MCTSPlayer, init_mcts_player: MCTSPlayer, is_shown=0):
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
                move = bot1.get_action(game)
            else:
                move = bot2.get_action(game)
            game = game.apply_move(move)
        return winner





    def policy_update(self):
        """update the policy-value net"""
        mini_batch = random.sample(self.data_buffer, self.batch_size)
        state_batch = [data[0] for data in mini_batch]
        state_batch = np.array(state_batch).astype('float32')
        mcts_probs_batch = []

        for data in mini_batch:
            prob = np.zeros([1125]).astype('float32')
            for item in data[1]:
                prob[int(item[0])] = item[1]
            mcts_probs_batch.append(prob)

        # mcts_probs_batch = [[item[1] for item in data[1]] for data in mini_batch] # 2024/4/4 modify data[1] is list !!!!!!!!!!!!
        # mcts_probs_batch = [np.pad(arr, (0, 64 - len(arr)), 'constant', constant_values=0) for arr in mcts_probs_batch] # modify 24/4/19
        #mcts_probs_batch =[arr[:64] if len(arr)>64 else np.pad(arr, (0, 64 - len(arr)), 'constant', constant_values=0) for arr in mcts_probs_batch] # modify 24/4/19

        mcts_probs_batch = np.array(mcts_probs_batch).astype('float32')
        winner_batch = [data[2] for data in mini_batch]
        winner_batch = np.array(winner_batch).astype('float32')
        old_probs, old_v = self.policy_value_net.policy_value(state_batch)  # (512,64)
        for i in range(self.epochs):
            loss, entropy = self.policy_value_net.train_step(
                state_batch,
                mcts_probs_batch,
                winner_batch,
                self.learn_rate * self.lr_multiplier)
            new_probs, new_v = self.policy_value_net.policy_value(state_batch)
            kl = np.mean(np.sum(old_probs * (
                    np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)),
                                axis=1)
                         )
            if kl > self.kl_targ * 4:  # early stopping if D_KL diverges badly
                break
        # adaptively adjust the learning rate
        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5

        explained_var_old = (1 -
                             np.var(np.array(winner_batch) - old_v.flatten()) /
                             np.var(np.array(winner_batch)))
        explained_var_new = (1 -
                             np.var(np.array(winner_batch) - new_v.flatten()) /
                             np.var(np.array(winner_batch)))
        logger.info(("kl:{:.5f},"
                     "lr_multiplier:{:.3f},"
                     "loss:{},"
                     "entropy:{},"
                     "explained_var_old:{:.3f},"
                     "explained_var_new:{:.3f}"
                     ).format(kl,
                              self.lr_multiplier,
                              loss,
                              entropy,
                              explained_var_old,
                              explained_var_new))
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

    # def policy_evaluate(self, n_games=2):
    #     """
    #     Evaluate the trained policy by playing against the pure MCTS player
    #     Note: this is only for monitoring the progress of training
    #     """
    #     current_mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn, c_puct=self.c_puct,
    #                                      n_playout=self.n_playout)
    #
    #     init_mcts_player = MCTSPlayer(PolicyValueNet(9, 9, model_file='./model/init.model').policy_value_fn,
    #                                   c_puct=self.c_puct, n_playout=self.n_playout)
    #     # pure_mcts_player = MCTS_Pure(c_puct=5, n_playout=self.pure_mcts_playout_num)
    #     win_cnt = {}
    #     for i in range(n_games):
    #         winner = self.start_play(current_mcts_player, init_mcts_player, is_shown=0)
    #
    #         win_cnt[winner] += 1
    #     win_rate = 1.0 * (win_cnt[Player.black]) / n_games
    #     LOGGER.info(
    #         "num_playouts:{}, black win: {}, n_games : {}".format(self.pure_mcts_playout_num, win_cnt[Player.black],
    #                                                               n_games))
    #
    #     return win_rate

    class GameDataset(Dataset):
        def __init__(self, data):
            self.data = data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            state, action, score = self.data[idx]
            # 将状态转换为 PyTorch 张量
            print(state)
            state_tensor = torch.tensor(state,dtype=torch.float32)
            # 将动作转换为整数索引（根据具体情况修改）
            # actions = np.zeros((1125))
            # actions[action] = 1
            action_tensor = torch.tensor(action, dtype=torch.float32)  # 假设动作是一个整数，表示输出动作的位置
            return state_tensor, action_tensor


    def train_pkl(self):
        os.makedirs('alpha_data_model', exist_ok=True)
        with open('./collect_alpha_data/picture_data.pkl',  'rb') as data_dict:
            data_file = pickle.load(data_dict)
            self.data_buffer.extend(data_file)

        if len(self.data_buffer):
            dataset = self.GameDataset(self.data_buffer)

            dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

            criterion = nn.CrossEntropyLoss()

            optimizer = optim.Adam(self.policy_value_net.policy_value_net.parameters(), lr=0.001)

            result = {}
            # 训练循环
            num_epochs = 1000
            for epoch in range(num_epochs):
                for states, actions in dataloader:
                    # 将批数据展平成一维向量
                    states = states.to(self.device)

                    # 前向传播

                    outputs, _ = self.policy_value_net.policy_value_net(states)
                    actions = actions.to(self.device)
                    loss = criterion(outputs, actions)

                    # 反向传播和优化
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                # if (epoch + 1) % 200 == 0:
                #     torch.save(self.policy_value_net.policy_value_net.state_dict(), './alpha_data_model/current.pth')
                #     win_white, win_black = test_model()
                #     torch.save(self.policy_value_net.policy_value_net.state_dict(), './alpha_data_model/current_' + str(epoch + 1) + '.pth')
                #     result[epoch] = (win_white, win_black)
                #     print(
                #         f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f},win_white:{win_white},win_black:{win_black}')
                print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')








    # def run(self):
    #     """run the training pipeline"""
    #     try:
    #         while True:
    #             # if not os.path.dirname("model"):
    #             #     os.makedirs('model', exist_ok=True)
    #             # self.policy_value_net.save_model('./model/init.model')
    #             # self.collect_selfplay_data(self.play_batch_size)
    #             # LOGGER.info("batch i:{}, episode_len:{}".format(
    #             #     i + 1, self.episode_len))
    #
    #             # load game to buffer
    #
    #             # generator = self.data_parse.process_files_concurrently()  # generator
    #             #
    #             # for play_data in generator:  # 包含增强的数据其中play_data为list(state,pro,value)
    #             #     self.data_buffer.extend(play_data)
    #
    #             while True:
    #                 try:
    #                     with open(CONFIG['train_data_buffer_path'], 'rb') as data_dict:
    #                         data_file = pickle.load(data_dict)
    #                         self.data_buffer = data_file['data_buffer']
    #                         # self.iters = data_file['iters']
    #                         # self.iters += 1
    #                     #     del data_file
    #                     get_data = self.data_parse.process_files_concurrently()
    #                     if len(get_data):
    #                         self.iters+=1
    #                     self.data_buffer.extend(get_data)
    #                     logger.info(f'已加载数据,{len(self.data_buffer)}')
    #                     if len(self.data_buffer) == 0:
    #                         time.sleep(2024)
    #                     break
    #                 except Exception as e:
    #                     time.sleep(30)
    #
    #
    #             logger.info('step iteration {}: '.format(self.iters))
    #
    #             if len(self.data_buffer) > self.batch_size:
    #                 loss, entropy = self.policy_update()
    #
    #                 # check the performance of the current model,
    #                 # and save the model params
    #                 # self.policy_value_net.save_model(CONFIG['pytorch_model_path'])
    #                 self.policy_value_net.save_model(CONFIG['pytorch_current_model_path'])
    #
    #             # time.sleep(CONFIG['train_update_interval'])  # 每10分钟更新一次模型
    #
    #             if (self.iters + 1) % self.export_engine_freq == 0:
    #                 if os.path.exists(CONFIG['pytorch_current_model_path']):
    #                     export_engine(CONFIG['pytorch_current_model_path'], CONFIG['pytorch_current_onnx_path'],
    #                                       CONFIG['pytorch_current_engine_path'])
    #
    #             if (self.iters+ 1) % self.check_freq == 0:
    #                 #     LOGGER.info("current self-play batch: {}".format(i + 1))
    #                 #     # self.policy_value_net.save_model('./model/current_policy.model')
    #                 #     win_ratio = self.policy_evaluate()
    #                 #     self.policy_value_net.save_model('./model/current_policy.model')
    #                 #     if win_ratio > self.best_win_ratio:
    #                 #         LOGGER.info("New best policy!!!!!!!!")
    #                 #         self.best_win_ratio = win_ratio
    #                 #         # update the best_policy
    #                 #         self.policy_value_net.save_model('./model/best_policy.model')
    #                 #         if (self.best_win_ratio == 1.0 and
    #                 #                 self.pure_mcts_playout_num < 5000):
    #                 #             self.pure_mcts_playout_num += 1000
    #                 #             self.best_win_ratio = 0.0
    #                 logger.info("current self-play batch: {}".format(self.iters+1))
    #                 self.policy_value_net.save_model('./model/current_policy_batch_{}.pt'.format(str(self.iters+1)))
    #     except KeyboardInterrupt:
    #         print('\n\rquit')




if __name__ == '__main__':
    #os.environ['CUDA_VISIBLE_DEVICES'] = '8'
    # s = time.time()
    training_pipeline = TrainPipeline()

    training_pipeline.train_pkl()



    # end = time.time() - s

    # logger.info(end)
