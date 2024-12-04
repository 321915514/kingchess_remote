import os
import sys

cur_path = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(cur_path)
import threading
from functools import partial

import numpy as np
import pickle

import tqdm

from agent.random_agent import Random_agent
from agent.expert_agent import Expert_agent
from agent.alpha_beta import Alpha_beta
from fundamental.board import GameState, Move
from fundamental.coordinate import Player
from fundamental.utils import print_board, print_move, print_move_go
import time
from net.collect import move2str
import torch.multiprocessing as mp

import concurrent.futures

from net.encoder import encoder_board

# import tqdm

lock = threading.Lock()


def main():
    row_size = 5
    col_size = 9
    game = GameState.new_game(row_size, col_size)
    random_agent = Random_agent()
    expert_agent = Expert_agent()

    while True:
        print_board(game.board)
        print(game)
        end, winner = game.game_over()
        if end:
            break
        if game.player == Player.black:
            # for i in game.legal_moves():
            #     print(i.__str__())
            move = random_agent.select_move(game)
            print_move_go(game.player, move)
            # play_out += 1
        else:
            # for i in game.legal_moves():
            #     print(i.__str__())
            move = expert_agent.select_move(game)

            if move.is_down:
                print_move(game.player, move)
            else:
                print_move_go(game.player, move)
            # play_out += 1
        game = game.apply_move(move)

        # if game.record:
        #     black_eat_chess += 1
        #     print("国王吃了{}个兵。".format(black_eat_chess))
        #     game.record = False

    print(winner)


def main_simple(data_white, data_black, data, picture_data):
    row_size = 5
    col_size = 9
    game = GameState.new_game(row_size, col_size)
    # bot1 = Random_agent()
    bot1 = Expert_agent()
    bot2 = Alpha_beta()
    while True:
        # print_board(game.board)

        end, winner = game.game_over()
        if end:
            break
        if game.player == Player.black:
            move = bot2.select_move(game)
            scores = bot1.score_moves(game)

            # if scores[move] > 10:
            data_black.append((game.encoder_board_137(), game.move_2_action(move), scores[move]))
            data.append((game.encoder_board_137(), game.move_2_action(move), scores[move]))

            picture_data.append((encoder_board(game), game.move_2_action(move), 0))

            # if move is None:
            #     return Player.white
            # print('dachen win')
            # break
            # if move not in scores:
            #     print('select move 有错误')
            # # print(scores)
            # print(move)
            # print_move_go(game.player, move, game.play_out)
        else:

            scores = bot1.score_moves(game)
            move = bot2.select_move(game)
            # print(scores)
            # if move not in scores:
            #     print('select move 有错误')
            # print(move)
            # if scores[move] > 10:
            data_white.append((game.encoder_board_137(), game.move_2_action(move), scores[move]))
            data.append((game.encoder_board_137(), game.move_2_action(move), scores[move]))
            picture_data.append((encoder_board(game), game.move_2_action(move), 0))
            # if move.is_down:
            #     # print_move(game.player, move, game.play_out)
            #     pass
            # else:
            #     pass
            # print_move_go(game.player, move,game.play_out)
            # play_out += 1
        game = game.apply_move(move)

        # if game.record:
        #     black_eat_chess += 1
        #     # print("国王吃了{}个兵。".format(black_eat_chess))
        #     game.record = False
        # if game.eat_chess() >= 11:
        #     return Player.black
        # print("king win")

    return winner


def worker(task_id, data_white, data_black, data, picture_data):
    return main_simple(data_white, data_black, data, picture_data)


def get_equi_data(play_data):
    """augment the data set by rotation and flipping
    play_data: [(state, mcts_prob, winner_z), ..., ...]
    前后的数组的换位置。
    """
    extend_data = []
    for state, mcts_porb, winner in play_data:
        # for i in [1, 2, 3, 4]:
        # rotate counterclockwise
        # equi_state = np.array([np.rot90(s, 2) for s in state])
        #
        #     # equi_mcts_prob = moves2flip_list(mcts_porb, i)  # AttributeError: 'zip' object has no attribute 'values' # equi_mcts_prob is list
        #
        #     # equi_mcts_prob = np.rot90(np.flipud(
        #     #     mcts_porb.reshape(self.board_height, self.board_width)), i)
        #
        #     # modify 24/7/6
        #     # extend_data.append((equi_state,
        #     #                     equi_mcts_prob,
        #     #                     winner))
        # extend_data.append((equi_state,
        #                         mcts_porb,
        #                         winner))

        # flip horizontally 水平
        equi_state = np.array([np.fliplr(s) for s in state])
        # equi_mcts_prob = moves2horizontally_list(mcts_porb, i) # equi_mcts_prob is list
        extend_data.append((equi_state,
                            mcts_porb,
                            winner))
        # modify 24/7/6
    return extend_data


if __name__ == '__main__':

    os.makedirs('collect_alpha_data', exist_ok=True)

    data_white = []
    data_black = []
    data = []
    picture_data = []

    num_tasks = 1000  # 总共要执行的任务数量
    num_threads = min(num_tasks, 24)  # 设置线程数量，可以根据需要调

    # with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
    #     futures = [executor.submit(worker, task_id, data_white, data_black, data, picture_data) for task_id in
    #                range(num_tasks)]
    #     for future in tqdm.tqdm(concurrent.futures.as_completed(futures), total=num_tasks):
    #         future.result()

    for _ in tqdm.tqdm(range(num_tasks)):
        row_size = 5
        col_size = 9
        game = GameState.new_game(row_size, col_size)
        # bot1 = Random_agent()
        # bot1 = Expert_agent()
        bot2 = Alpha_beta()
        while True:
            # print_board(game.board)

            end, winner = game.game_over()
            if end:
                break
            if game.player == Player.black:
                move = bot2.select_move(game)
                # scores = bot1.score_moves(game)

                # if scores[move] > 10:
                # data_black.append((game.encoder_board_137(), game.move_2_action(move), scores[move]))
                # data.append((game.encoder_board_137(), game.move_2_action(move), scores[move]))

                encoder_game = encoder_board(game)

                picture_data.append((encoder_game, game.move_2_action(move), 0))

            else:

                # scores = bot1.score_moves(game)
                move = bot2.select_move(game)
                #
                # data_white.append((game.encoder_board_137(), game.move_2_action(move), scores[move]))
                # data.append((game.encoder_board_137(), game.move_2_action(move), scores[move]))
                picture_data.append((encoder_board(game), game.move_2_action(move), 0))

            game = game.apply_move(move)

    # main()

    # for i in tqdm.tqdm(range(100000)):
    #     # result = main_simple()
    #     num_processes = mp.cpu_count()
    #     processes = []
    #     for i in range(num_processes):
    #         p = mp.Process(target=main_simple)
    #         p.start()
    #         processes.append(p)
    #     for p in processes:
    #         p.join()

    # states = np.array(item[0] for item in data)
    # actions = np.array(item[1] for item in data)

    # with open('./collect_alpha_data/game_data_add_score_white_137.pkl', 'ab') as f:
    #     pickle.dump(data_white, f)
    # with open('./collect_alpha_data/game_data_add_score_black_137.pkl', 'ab') as f:
    #     pickle.dump(data_black, f)
    # with open('./collect_alpha_data/game_data_add_score_137.pkl', 'ab') as f:
    #     pickle.dump(data, f)

    equi_data = get_equi_data(picture_data)
    #
    # with open('./collect_expert_data/picture_data.pkl', 'rb') as file:
    #     p_data = pickle.load(file)

    # 向数据中添加新数据
    # p_data.append(equi_data)

    # 将更新后的数据写回 .pkl 文件
    # with open('./collect_expert_data/picture_data.pkl', 'wb') as file:
    #     pickle.dump(picture_data, file)

    with open('./collect_alpha_data/picture_data.pkl', 'ab') as f:
        pickle.dump(equi_data, f)

    print('数据保存')
