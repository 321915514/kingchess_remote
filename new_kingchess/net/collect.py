import os
import os
import pickle
import sys
import time

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
import threading
import uuid
from collections import deque
from multiprocessing import Value

import numpy as np

from fundamental.board import GameState
from net import train_pb2
from net.config import CONFIG, CONFIG_TRAIN
from net.encoder import moves2flip, moves2horizontally, encoder_board, move2str
from net.mcts_alphazreo import MCTSPlayer
from net.policy_value_net_pytorch import PolicyValueNet
from concurrent.futures import ThreadPoolExecutor
# from utils.util import LOGGER
import torch.multiprocessing as mp
import logging
logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p',
                    level=logging.INFO)
logger = logging.getLogger(__file__)


class CollectPipeline:
    def __init__(self):
        # 逻辑和棋盘
        self.episode_len = 0
        self.game = GameState.new_game(5, 9)
        # 对弈参数
        self.temp = 1  # 温度
        self.n_playout = CONFIG_TRAIN['play_out']  # 每次移动的模拟次数
        self.c_puct = CONFIG_TRAIN['c_puct']  # u的权重
        self.buffer_size = CONFIG_TRAIN['buffer_size']  # 经验池大小
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.iters = 0  # 'i' 代表整型

        self.policy_value_net = PolicyValueNet(model_file=CONFIG_TRAIN['pytorch_current_model_path'])



        # self.mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
        #                                   c_puct=self.c_puct,
        #                                   n_playout=self.n_playout,
        #                                   is_selfplay=1)

    # # 从主体加载模型
    # def load_model(self):
    #     model_path = CONFIG['pytorch_current_model_path']
    #     try:
    #         self.policy_value_net = PolicyValueNet(model_file=model_path)
    #         logger.info('已加载最新模型')
    #     except:
    #         self.policy_value_net = PolicyValueNet(CONFIG['pytorch_model_path'])
    #         logger.info('已加载初始模型')
    #
    #     self.mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
    #                                   c_puct=self.c_puct,
    #                                   n_playout=self.n_playout,
    #                                   is_selfplay=1)

    def get_equi_data(self, play_data):
        """augment the data set by rotation and flipping
        play_data: [(state, mcts_prob, winner_z), ..., ...]
        前后的数组的换位置。
        """
        extend_data = []
        for state, mcts_prob, winner in play_data:
            for i in [1, 2, 3, 4]:
                # rotate counterclockwise
                equi_state = np.array([np.rot90(s, i) for s in state])

                # equi_mcts_prob = moves2flip(mcts_porb, i)  # AttributeError: 'zip' object has no attribute 'values'

                # equi_mcts_prob = np.rot90(np.flipud(
                #     mcts_porb.reshape(self.board_height, self.board_width)), i)
                extend_data.append((equi_state,
                                    mcts_prob,
                                    winner))
                # flip horizontally 水平
                equi_state = np.array([np.fliplr(s) for s in equi_state])
                # equi_mcts_prob = moves2horizontally(mcts_porb, i)
                extend_data.append((equi_state,
                                    mcts_prob,
                                    winner))
        return extend_data



    def save_to_pickle(self, play_data):
        if os.path.exists(CONFIG['train_data_buffer_path']):
            while True:
                try:
                    with open(CONFIG['train_data_buffer_path'], 'rb') as data_dict:
                        data_file = pickle.load(data_dict)
                        # self.data_buffer = deque(maxlen=self.buffer_size)
                        self.data_buffer.extend(data_file['data_buffer'])
                        # self.iters = data_file['iters']
                        del data_file
                        # self.iters += 1
                        self.data_buffer.extend(play_data)
                    print('成功载入数据')
                    break
                except RuntimeError:
                    time.sleep(30)
        else:
            self.data_buffer.extend(play_data)
            # self.iters += 1
        data_dict = {'data_buffer': self.data_buffer}
        if not os.path.exists('pickle'):
            os.makedirs('pickle', exist_ok=True)
        with open(CONFIG['train_data_buffer_path'], 'wb') as data_file:
            pickle.dump(data_dict, data_file)
        # return self.iters



    def write_state_proto(self, play_data):
        '''
        write state,prob,value to proto
        note: write data not augment, you should run self.get_equi_data(play_data),next to train
        :param play_data:
        :return:
        '''
        triplet_array = train_pb2.TripletArray()
        for array, dictionary, value in play_data:
            # 创建 Triplet 消息对象
            triplet = train_pb2.Triplet()
            # (3, 9, 9)
            array = array.astype(int)
            array = array.flatten().tolist()
            triplet.array.extend(array)  # 添加数组元素
            for key, val in dictionary.items():
                move = train_pb2.Move()
                move.point = move2str(key)
                move.prob = val
                triplet.dictionary.append(move)
            triplet.value = value  # 设置值
            # 将 Triplet 添加到 TripletArray 中
            triplet_array.triplets.append(triplet)
            # 将 TripletArray 序列化为字节串
        serialized_data = triplet_array.SerializeToString()
        # 将字节串写入文件
        if not os.path.dirname("dataset"):
            os.makedirs('dataset', exist_ok=True)
        with open("./dataset/" + str(uuid.uuid4()) + '.game', 'wb') as f:
            f.write(serialized_data)

    def start_self_play(self, iters, cpu, temp=1e-3):
        ### winner, play_data
        states, mcts_probs, current_players = [], [], []
        game = GameState.new_game(5, 9)
        mcts = MCTSPlayer(PolicyValueNet(model_file=CONFIG_TRAIN['pytorch_current_model_path']).policy_value_fn,
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

                self.save_to_pickle(play_data)

                logger.info("Iters:{}, cpu:{}, game state length: {}".format(iters, cpu, self.episode_len))
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
            num_processes = mp.cpu_count()
            processes = []
            for i in range(2):
                p = mp.Process(target=self.start_self_play, args=(self.iters, i, self.temp))
                p.start()
                processes.append(p)
            for p in processes:
                p.join()
        # self.load_model()
        # with ThreadPoolExecutor(max_workers=4) as executor:
        #     for _ in range(n_games):
        #         executor.submit(self.start_self_play)

    def run(self):
        try:
            while True:
                self.collect_selfplay_data()
                self.iters += 1
        except KeyboardInterrupt:
            logger.info('\n\rquit')


if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '6'
    if not os.path.dirname("model_torch"):
        os.makedirs('model_torch', exist_ok=True)

    collectPipeline = CollectPipeline()
    #
    collectPipeline.run()
