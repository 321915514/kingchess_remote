import os
import sys

#import torch

cur_path = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(cur_path)



import socket
import json
import time
import tqdm
import numpy as np

from agent.expert_agent import Expert_agent
from agent.alpha_beta import Alpha_beta
from agent.random_agent import Random_agent
from fundamental.board import GameState
from fundamental.coordinate import Player, Move
from fundamental.utils import print_board
from socket_transfer_data.json_gamestate import json_state


def start_play_black_is_mcts(init_mcts_player, is_shown=0):
    row_size = 5
    col_size = 9
    game = GameState.new_game(row_size, col_size)

    bot2 = init_mcts_player
    while True:
        end, winner = game.game_over()
        if end:
            break
        if is_shown:
            print_board(game.board)
        if game.player == Player.black:
            json_game = json_state(game)
            action = send_request('10.122.7.125', 8900, json_game)
        else:
            action = bot2.get_action(game)
        if is_shown:
            game.print_game()

        if isinstance(action, np.int64) or isinstance(action, int):
#            print(game.a_trans_move(action))
            game = game.apply_move(game.a_trans_move(action))
        if isinstance(action, Move):
#            print(action)
            game = game.apply_move(action)

    return winner


def start_play_white_is_mcts(init_mcts_player, is_shown=0):
    row_size = 5
    col_size = 9
    game = GameState.new_game(row_size, col_size)

    bot2 = init_mcts_player
    while True:
        end, winner = game.game_over()
        if end:
            break
        if is_shown:
            print_board(game.board)
        if game.player == Player.white:
            json_game = json_state(game)
            action = send_request('10.122.7.125', 8900, json_game)
        else:
            action = bot2.get_action(game)
        if is_shown:
            game.print_game()
        if isinstance(action, np.int64) or isinstance(action, int):
            game = game.apply_move(game.a_trans_move(action))
        else:
            game = game.apply_move(action)

    return winner





def policy_evaluate(n_games=100):
    """
    Evaluate the trained policy by playing against the pure MCTS player
    Note: this is only for monitoring the progress of training
    """

    random = Random_agent()
    expert = Expert_agent()
    alpha = Alpha_beta()
    win_black_random = 0
    win_black_expert = 0
    win_white_expert = 0
    win_white_random = 0

    for i in tqdm.tqdm(range(n_games)):
        winner = start_play_black_is_mcts(expert)
        if winner == Player.black:
            win_black_random += 1
    for i in tqdm.tqdm(range(n_games)):
        winner = start_play_white_is_mcts(expert)
        if winner == Player.white:
            win_white_random += 1
    win_rate_player_black = 1.0 * (win_black_random) / n_games
    win_rate_player_white = 1.0 * (win_white_random) / n_games
    print("num_playouts:{}, black win: {}, white win: {}, n_games : {}".format(1200,
                                                                                     win_rate_player_black,
                                                                                     win_rate_player_white,
                                                                                     n_games))


def send_request(host, port, request):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((host, port))
    client_socket.sendall(request.encode())
    response = client_socket.recv(1024).decode()
    res = json.loads(response)
    return res['action']


if __name__ == "__main__":
    # game = GameState.new_game(5, 9)
    #
    # json_game = json_state(game)
    #
    # send_request('10.122.7.125', 65432, json_game)
    start = time.time()
    policy_evaluate()
    print(time.time() - start)
