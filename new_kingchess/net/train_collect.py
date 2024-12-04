# -*- coding: utf-8 -*-
"""
An implementation of the training pipeline of AlphaZero for Gomoku

@author: Junxiao Song
"""

# from __future__ import print_function
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '7'
import pickle
import sys



cur_path = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(cur_path)
import random
import time
from agent.expert_agent import Expert_agent
import torch.multiprocessing as mp
import numpy as np
from collections import defaultdict, deque

import torch
from torch import nn

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
import train_pb2
# from utils.util import LOGGER
from decode_proto_file import DecodeDataset
import logging

logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p',
                    level=logging.INFO)
logger = logging.getLogger(__file__)


# os.environ['CUDA_VISIBLE_DEVICES'] = '8'

# from transfer_trt import export_engine
#

# from policy_value_net_tensorflow import PolicyValueNet # Tensorflow
# from policy_value_net_keras import PolicyValueNet # Keras


class TrainPipeline():
    def __init__(self, init_model=None):
        # params of the board and the game
        self.board_width = 9
        self.board_height = 5
        # self.n_in_row = 4
        # self.board = Board(num_rows=self.board_height, num_cols=self.board_width)
        # self.game = GameState.new_game(self.board_height, self.board_width)
        # training params
        self.learn_rate = 1e-3
        self.lr_multiplier = 1.0  # adaptively adjust the learning rate based on KL
        self.temp = 1.0  # the temperature param
        self.n_playout = 1200  # num of simulations for each move
        self.c_puct = 5

        self.buffer_size = 1000
        self.batch_size = 256  # mini-batch size for training
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.play_batch_size = 1
        self.epochs = 50  # num of train_steps for each update
        self.kl_targ = 0.02
        self.check_freq = 50
        self.game_batch_num = 5000
        # self.data_parse = DecodeDataset('./dataset')
        self.best_win_ratio_black = 0.0
        self.best_win_ratio_white = 0.0
        # self.export_engine_freq = 20
        # num of simulations used for the pure mcts, which is used as
        # the opponent to evaluate the trained policy
        self.pure_mcts_playout_num = 1000
        self.iters = 0
        if init_model:
            try:
                # start training from an initial policy-value net
                self.policy_value_net = PolicyValueNet(model_file=init_model, device='cuda:0')
                logger.info('加载上次最终模型')
            except FileNotFoundError as e:
                # start training from a new policy-value net
                logger.info('从头训练')
                self.policy_value_net = PolicyValueNet(model_file=CONFIG['pytorch_model_path'], device='cuda:0')
            # self.mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
            #                               c_puct=self.c_puct,
            #                               n_playout=self.n_playout,
            #                               is_selfplay=1)
        else:
            logger.info('从头训练')
            self.policy_value_net = PolicyValueNet(device='cuda:0')

    def get_equi_data(self, play_data):
        """augment the data set by rotation and flipping
        play_data: [(state, mcts_prob, winner_z), ..., ...]
        前后的数组的换位置。
        """
        extend_data = []
        for state, mcts_porb, winner in play_data:
            for i in [1, 2, 3, 4]:
                # rotate counterclockwise
                equi_state = np.array([np.rot90(s, i) for s in state])

                # equi_mcts_prob = moves2flip_list(mcts_porb, i)  # AttributeError: 'zip' object has no attribute 'values' # equi_mcts_prob is list

                # equi_mcts_prob = np.rot90(np.flipud(
                #     mcts_porb.reshape(self.board_height, self.board_width)), i)

                # modify 24/7/6
                # extend_data.append((equi_state,
                #                     equi_mcts_prob,
                #                     winner))
                extend_data.append((equi_state,
                                    mcts_porb,
                                    winner))

                # flip horizontally 水平
                equi_state = np.array([np.fliplr(s) for s in equi_state])
                # equi_mcts_prob = moves2horizontally_list(mcts_porb, i) # equi_mcts_prob is list
                extend_data.append((equi_state,
                                    mcts_porb,
                                    winner))
                # modify 24/7/6
        return extend_data

    # collect
    def start_self_play(self, iters, cpu, temp=1e-3):
        ### winner, play_data
        states, mcts_probs, current_players = [], [], []
        game = GameState.new_game(5, 9)
        mcts = MCTSPlayer(self.policy_value_net.policy_value_fn,
                          c_puct=self.c_puct,
                          n_playout=self.n_playout,
                          is_selfplay=1)
        while True:
            end, winner = game.game_over()
            if end:
                winners_z = np.zeros(len(current_players))
                if winner != -1:
                    winners_z[np.array(current_players) == winner] = 1.0
                    winners_z[np.array(current_players) != winner] = -1.0
                mcts.reset_player()
                # return winner, zip(states, mcts_probs, winners_z)
                # print(states)
                play_data = zip(states, mcts_probs, winners_z)
                play_data = list(play_data)[:]
                self.episode_len = len(play_data)
                # augment the data
                play_data = self.get_equi_data(play_data)
                # after augment
                # self.write_state_proto(play_data)

                self.data_buffer.extend(play_data)

                logger.info("game state length: {}".format(self.episode_len))
                break
            move, move_probs = mcts.get_action(game, temp=temp, return_prob=1)
            # store the data
            states.append(encoder_board(game))
            mcts_probs.append(move_probs)
            current_players.append(game.player)
            game = game.apply_move(game.a_trans_move(move))

    def collect_selfplay_data(self, n_games=1):
        """collect self-play data for training"""
        # self.load_model()
        mp.set_start_method("spawn", force=True)
        for i in range(n_games):
            # self.start_self_play(self.iters, i, self.temp)
            # num_processes = mp.cpu_count()
            processes = []
            for i in range(2):
                p = mp.Process(target=self.start_self_play, args=(self.iters, i, self.temp))
                p.start()
                processes.append(p)
            for p in processes:
                p.join()

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
        # mcts_probs_batch =[arr[:64] if len(arr)>64 else np.pad(arr, (0, 64 - len(arr)), 'constant', constant_values=0) for arr in mcts_probs_batch] # modify 24/4/19

        mcts_probs_batch = np.array(mcts_probs_batch).astype('float32')
        winner_batch = [data[2] for data in mini_batch]
        winner_batch = np.array(winner_batch).astype('float32')
        old_probs, old_v = self.policy_value_net.policy_value(state_batch)  # (512,64)
        for i in range(self.epochs):
            loss, policy_loss, value_loss, entropy = self.policy_value_net.train_step(
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
                     "policy_loss:{},"
                     "value_loss:{},"
                     "entropy:{},"
                     "explained_var_old:{:.3f},"
                     "explained_var_new:{:.3f}"
                     ).format(kl,
                              self.lr_multiplier,
                              loss,
                              policy_loss,
                              value_loss,
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

    def start_play(self, current_mcts_player, init_mcts_player):
        row_size = 5
        col_size = 9
        game = GameState.new_game(row_size, col_size)
        bot1 = current_mcts_player
        bot2 = init_mcts_player
        while True:
            end, winner = game.game_over()
            if end:
                break
            if game.player == Player.black:
                move = bot1.get_action(game)
            else:
                move = bot2.get_action(game)
            game = game.apply_move(move)
        return winner

    def policy_evaluate(self, n_games=10):
        """
        Evaluate the trained policy by playing against the pure MCTS player
        Note: this is only for monitoring the progress of training
        """
        current_mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn, c_puct=self.c_puct,
                                         n_playout=self.n_playout)

        # pure_mcts_player = MCTS_Pure(c_puct=5, n_playout=self.pure_mcts_playout_num)

        expert = Expert_agent()

        win_cnt = {}
        for i in range(n_games):
            winner = self.start_play(current_mcts_player, expert)
            win_cnt[winner] += 1
        for i in range(n_games):
            winner = self.start_play(expert, current_mcts_player)
            win_cnt[winner] += 1
        win_rate_player_black = 1.0 * (win_cnt[Player.black]) / n_games
        win_rate_player_white = 1.0 * (win_cnt[Player.white]) / n_games
        logger.info("num_playouts:{}, black win: {}, white win: {}, n_games : {}".format(self.pure_mcts_playout_num,
                                                                                         win_cnt[Player.black],
                                                                                         win_cnt[Player.white],
                                                                                         n_games * 2))

        return win_rate_player_black, win_rate_player_white

    def run(self):
        """run the training pipeline"""
        try:

            for i in range(self.game_batch_num):

                self.collect_selfplay_data()

                logger.info('step iteration {}: '.format(self.iters))

                if len(self.data_buffer) > self.batch_size:
                    self.policy_update()

                if (i + 1) % self.check_freq == 0:
                    logger.info("current self-play batch: {}".format(i + 1))
                    win_ratio_black, win_ratio_white = self.policy_evaluate()

                    if win_ratio_black > self.best_win_ratio_black:
                        logger.info("New best policy!!!!!!!!")
                        self.best_win_ratio_black = win_ratio_black
                        self.policy_value_net.save_model('./model_new/best_policy_black.model')
                    if win_ratio_white > self.best_win_ratio_white:
                        logger.info("New best policy!!!!!!!!")
                        self.best_win_ratio_white = win_ratio_white
                        self.policy_value_net.save_model('./model_new/best_policy_white.model')

                    if win_ratio_white > self.best_win_ratio_white or win_ratio_black > self.best_win_ratio_black:
                        self.policy_value_net.save_model('./model_new/current.model')
                        self.policy_value_net.policy_value_net.load_state_dict(torch.load('./model_new/current.model'))
                        if (self.best_win_ratio_black == 1.0 and
                                self.pure_mcts_playout_num < 5000):
                            self.pure_mcts_playout_num += 1000
                            self.best_win_ratio_black = 0.0
                # logger.info("current self-play batch: {}".format(self.iters+1))
                # self.policy_value_net.save_model('./model/current_policy_batch_{}.pt'.format(str(self.iters+1)))
        except KeyboardInterrupt:
            print('\n\rquit')


if __name__ == '__main__':
    # s = time.time()
    os.makedirs('model_new', exist_ok=True)
    training_pipeline = TrainPipeline()
    training_pipeline.run()

    # end = time.time() - s

    # logger.info(end)
