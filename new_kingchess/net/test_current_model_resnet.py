# -*- coding: utf-8 -*-
"""
An implementation of the training pipeline of AlphaZero for Gomoku

@author: Junxiao Song
"""

# from __future__ import print_function
import os
import sys

import torch

cur_path = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(cur_path)


from socket_transfer_data.json_gamestate import json_state
from socket_transfer_data.server import send_request
from agent.random_agent import Random_agent

import random
import time
import tqdm
import numpy as np
from collections import defaultdict, deque
from agent.expert_agent import Expert_agent
from agent.alpha_beta import Alpha_beta


from fundamental.board import Board, GameState, Move
from fundamental.coordinate import Player
from fundamental.utils import print_board
from net.mcts_pure import MCTSPlayer as MCTS_Pure
from net.encoder import encoder_board, moves2flip, moves2horizontally
from net.mcts_alphazreo import MCTSPlayer
from net.config import CONFIG

from net.policy_value_net_pytorch import PolicyValueNet  # Pytorch
import os
import uuid


from decode_proto_file import DecodeDataset
import logging
from transfer_trt import export_engine
logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p',
                    level=logging.INFO)
logger = logging.getLogger(__file__)

#os.environ['CUDA_VISIBLE_DEVICES'] = '8'

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
        self.learn_rate = 1e-6
        self.lr_multiplier = 1.0  # adaptively adjust the learning rate based on KL
        self.temp = 1.0  # the temperature param
        self.n_playout = 1200  # num of simulations for each move
        self.c_puct = 5

        self.buffer_size = 20000
        self.batch_size = 512  # mini-batch size for training
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.play_batch_size = 1
        self.epochs = 5  # num of train_steps for each update
        self.kl_targ = 0.03
        self.check_freq = 200
        self.game_batch_num = 10000
        self.data_parse = DecodeDataset('./dataset_cpp')
        self.best_win_ratio_black = 0.0
        self.best_win_ratio_white = 0.0
        self.export_engine_freq = 10000
        # num of simulations used for the pure mcts, which is used as
        # the opponent to evaluate the trained policy
        self.pure_mcts_playout_num = 1200
        self.iters = 90000
        if init_model:
            try:
                # start training from an initial policy-value net
                self.policy_value_net = PolicyValueNet(model_file=init_model, device='cuda')
                logger.info('加载上次最终模型')
            except FileNotFoundError as e:
                # start training from a new policy-value net
                logger.info('加载初始模型训练')
                self.policy_value_net = PolicyValueNet(model_file=CONFIG['pytorch_model_path'], device='cuda')
            # self.mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
            #                               c_puct=self.c_puct,
            #                               n_playout=self.n_playout,
            #                               is_selfplay=1)
        else:
            #logger.info('从头训练')
#            self.policy_value_net = PolicyValueNet(device='cuda')
            pass

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

    def start_play_black_is_mcts(self, init_mcts_player, is_shown=0):
        row_size = 5
        col_size = 9
        game = GameState.new_game(row_size, col_size)

        bot2 = init_mcts_player
        while True:
            end, winner = game.game_over()
            if end:
                return winner
            if is_shown:
                print_board(game.board)
            if game.player == Player.black:
                json_game = json_state(game)
                action = send_request('10.122.7.125', 8999, json_game)
            else:
                action = bot2.select_move(game)

            #print(action)
            if isinstance(action, np.int64) or isinstance(action, int):
                game = game.apply_move(game.a_trans_move(action))
            else:
                game = game.apply_move(action)

        return winner

    def start_play_white_is_mcts(self, init_mcts_player, is_shown=0):
        row_size = 5
        col_size = 9
        game = GameState.new_game(row_size, col_size)

        bot2 = init_mcts_player
        while True:
            end, winner = game.game_over()
            if end:
                return winner
            if is_shown:
                print_board(game.board)
            if game.player == Player.white:
                json_game = json_state(game)
                action = send_request('10.122.7.125', 8999, json_game)
            else:
                action = bot2.select_move(game)

            if isinstance(action, np.int64) or isinstance(action, int):
                game = game.apply_move(game.a_trans_move(action))
            else:
                game = game.apply_move(action)

        return winner



    def softmax(self,x):
        probs = np.exp(x - np.max(x))
        probs /= np.sum(probs)
        return probs

 



    def policy_update(self):
        """update the policy-value net"""
        random.shuffle(self.data_buffer)
        mini_batch = random.sample(self.data_buffer, self.batch_size)
        state_batch = [data[0] for data in mini_batch]
        state_batch = np.array(state_batch).astype('float32')
        mcts_probs_batch = [data[1] for data in mini_batch]

        mcts_probs_batch = np.array(mcts_probs_batch).astype('float32')
        winner_batch = [data[2] for data in mini_batch]
        winner_batch = np.array(winner_batch).astype('float32')
        #old_probs, old_v = self.policy_value_net.policy_value(state_batch)  # (512,64)
        for i in range(self.epochs):
            accuracy, loss, policy_loss, value_loss, entropy,_ = self.policy_value_net.train_step(
                state_batch,
                mcts_probs_batch,
                winner_batch,
                self.learn_rate) # self.lr_multiplier
            #new_probs, new_v = self.policy_value_net.policy_value(state_batch)
            #kl = np.mean(np.sum(self.softmax(old_probs) * np.log(
            #    (self.softmax(old_probs) + 1e-10) / (self.softmax(new_probs) + 1e-10)),
            #                    axis=1)
            #             )
        #    if kl > self.kl_targ * 4:  # early stopping if D_KL diverges badly
        #        break
        # adaptively adjust the learning rate
        #if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
        #    self.lr_multiplier /= 1.5
        #elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
        #    self.lr_multiplier *= 1.5

        #logger.info("train end")

        #explained_var_old = (1 -
        #                     np.var(np.array(winner_batch) - old_v.flatten()) /
        #                     np.var(np.array(winner_batch)))
        #explained_var_new = (1 -
        #                     np.var(np.array(winner_batch) - new_v.flatten()) /
        #                     np.var(np.array(winner_batch)))
        logger.info((#"kl:{:.5f},"
                     "accuracy:{},"
                     "loss:{},"
                     "policy_loss:{},"
                     "value_loss:{},"
                     "entropy:{},"
                     ).format(#kl,
                              accuracy,
                              loss,
                              policy_loss,
                              value_loss,
                              entropy))
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


    def start_play(self, current, init_mcts_player, is_shown=0):
         row_size = 5
         col_size = 9
         game = GameState.new_game(row_size, col_size)
         bot1 = current
         bot2 = init_mcts_player
         while True:
             end, winner = game.game_over()
             if end:
                 break
             if is_shown:
                 print_board(game.board)
             if game.player == Player.black:
                 action = bot1.get_action(game)
    
             else:
                 action = bot2.get_action(game)
    
             if isinstance(action, np.int64):
                 game = game.apply_move(game.a_trans_move(action))
             else:
                 game = game.apply_move(action)
    
         return winner


    def policy_evaluate(self, n_games=100):
        """
        Evaluate the trained policy by playing against the pure MCTS player
        Note: this is only for monitoring the progress of training
        """
        #last_mcts_player = MCTSPlayer(PolicyValueNet(model_file=CONFIG["pytorch_model_path"], device='cuda').policy_value_fn, c_puct=self.c_puct, n_playout=2)

        # pure_mcts_player = MCTS_Pure(c_puct=5, n_playout=400)
        expert = Expert_agent()
        #alpha = Alpha_beta()
        #random = Random_agent()
        win_black_random = 0
        win_black_expert = 0
        win_white_expert = 0
        win_white_random = 0
#        for i in tqdm.tqdm(range(n_games)):
#            winner = self.start_play_black_is_mcts(last_mcts_player)
#            if winner == Player.black:
#                win_black_expert += 1
#        for i in tqdm.tqdm(range(n_games)):
#            winner = self.start_play_white_is_mcts(last_mcts_player)
#            if winner == Player.white:
#                win_white_expert += 1
        for i in tqdm.tqdm(range(n_games)):
            winner = self.start_play_black_is_mcts(expert)
            if winner == Player.black:
                 win_black_random += 1
        for i in tqdm.tqdm(range(n_games)):
            winner = self.start_play_white_is_mcts(expert)
            if winner == Player.white:
                 win_white_random += 1
        win_rate_player_black = 1.0 * (win_black_random) / n_games
        win_rate_player_white = 1.0 * (win_white_random) / n_games
        logger.info("num_playouts:{}, black win: {}, white win: {}, n_games : {}".format(300,
                                                                                         win_rate_player_black,
                                                                                         win_rate_player_white,
                                                                                         n_games))


        print("num_playouts:{}, black win: {}, white win: {}, n_games : {}".format(300,
                                                                                         win_rate_player_black,
                                                                                         win_rate_player_white,
                                                                                         n_games))



        return win_rate_player_black, win_rate_player_white

    def run(self):
        """run the training pipeline"""
        try:
            while True:
                # if not os.path.dirname("model"):
                #     os.makedirs('model', exist_ok=True)
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
                        if len(self.data_buffer) == 0:
                            time.sleep(30)
                        break
                    except Exception as e:
                        time.sleep(30)


                logger.info('step iteration {}: '.format(self.iters))
                
                #self.iters += 1
                if len(self.data_buffer) > self.batch_size:
                    #logger.info("train")
                    #self.iters += 1
                    self.policy_update()
                    self.iters += 1

                    # check the performance of the current model,
                    # and save the model params
                    # self.policy_value_net.save_model(CONFIG['pytorch_model_path'])
                    # self.policy_value_net.save_model(CONFIG['pytorch_current_model_path'])

                # time.sleep(CONFIG['train_update_interval'])  # 每10分钟更新一次模型

                #if (self.iters + 1) % 500 == 0:
                #    if os.path.exists(CONFIG['pytorch_current_model_path']):
                #        export_engine(CONFIG['pytorch_current_model_path'], CONFIG['pytorch_current_onnx_path'],
                #                          CONFIG['pytorch_current_engine_path'])

                    if (self.iters+ 1) % 2000 == 0:
                        logger.info("current self-play batch: {}".format(self.iters + 1))

                        self.policy_value_net.save_model(f'./model/{self.iters+1}.pth')

                        if os.path.exists(CONFIG['pytorch_current_model_path']):
                            export_engine(CONFIG['pytorch_current_model_path'], CONFIG['pytorch_current_onnx_path'],
                                          CONFIG['pytorch_current_engine_path'])
                        
                        self.data_buffer = deque(maxlen=self.buffer_size)

                        
                        # self.policy_evaluate()
                        # self.policy_value_net.save_model('./model/current_policy.model')


                        folder_path = "/home/test4/new_kingchess/net/dataset_cpp"

                        for filename in os.listdir(folder_path):
                            file_path = os.path.join(folder_path, filename)
                            if os.path.isfile(file_path):
                                os.remove(file_path)


                        # if win_ratio_white > self.best_win_ratio_white or win_ratio_black > self.best_win_ratio_black:


                        #     self.policy_value_net.save_model(CONFIG['pytorch_current_model_path'])
                        #     self.policy_value_net.save_model(CONFIG['pytorch_best_model_path'])



                        #     self.policy_value_net.save_model(CONFIG['pytorch_model_path'])



                        #     if win_ratio_black > self.best_win_ratio_black:
                        #         logger.info("New best policy!!!!!!!!")
                        #         self.best_win_ratio_black = win_ratio_black
                        #         self.policy_value_net.save_model(f'./model/best_policy_black_{win_ratio_black}.model')
                        #     if win_ratio_white > self.best_win_ratio_white:
                        #         logger.info("New best policy!!!!!!!!")
                        #         self.best_win_ratio_white = win_ratio_white
                        #         self.policy_value_net.save_model(f'./model/best_policy_white_{win_ratio_white}.model')

                        #     if (self.best_win_ratio_black == 1.0 or self.best_win_ratio_white == 1.0) and self.pure_mcts_playout_num < 5000:
                        #         self.pure_mcts_playout_num += 1000
                        #         self.best_win_ratio_black = 0.0
                        #         self.best_win_ratio_white = 0.0

                        # else:
                        #     if os.path.exists(CONFIG['pytorch_best_model_path']):
                        #         self.policy_value_net.policy_value_net.load_state_dict(torch.load(CONFIG['pytorch_best_model_path'], map_location="cuda"))
                        #         self.policy_value_net.save_model(CONFIG['pytorch_current_model_path'])
                        #         export_engine(CONFIG['pytorch_current_model_path'], CONFIG['pytorch_current_onnx_path'],
                        #                       CONFIG['pytorch_current_engine_path'])
                            #else:
                            #    self.policy_value_net.policy_value_net.load_state_dict(torch.load(CONFIG['pytorch_model_path'],map_location='cuda'))
                            #    self.policy_value_net.save_model(CONFIG['pytorch_current_model_path'])
                            #    export_engine(CONFIG['pytorch_current_model_path'], CONFIG['pytorch_current_onnx_path'],
                            #                  CONFIG['pytorch_current_engine_path'])


                    # logger.info("current self-play batch: {}".format(self.iters+1))
                    # self.policy_value_net.save_model('./model/current_policy_batch_{}.pt'.format(str(self.iters+1)))
        except KeyboardInterrupt:
            print('\n\rquit')




if __name__ == '__main__':

    # s = time.time()
    training_pipeline = TrainPipeline()
    #training_pipeline.run()

    training_pipeline.policy_evaluate()

    # end = time.time() - s

    # logger.info(end)
