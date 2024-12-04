import os
import sys

import torch

cur_path = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(cur_path)


import numpy as np
import tqdm
from fundamental.board import GameState
from fundamental.coordinate import Player
from fundamental.utils import print_board, print_move, print_move_go
from net.config import CONFIG
from net.encoder import encoder_board
from net.mcts_alphazreo import MCTSPlayer
from net.policy_value_net_pytorch import PolicyValueNet
from net.transfer_trt import export_engine
from net.trt_use_8_5_3 import TRTEngine
from agent.random_agent import Random_agent
from agent.expert_agent import Expert_agent
from collections import deque
def start_self_play_init_black_vs_current_white(model_file_init,model_file_current,temp=1e-3):
    ### winner, play_data
    # states, mcts_probs, current_players = [], [], []
    game = GameState.new_game(5, 9)
    # mcts = MCTSPlayer(TRTEngine('./pytorch_muc.trt').policy_value_fn,
    queue = []
    policy_value_net_init = PolicyValueNet(model_file=model_file_init,device='cuda')
    policy_value_net_current = PolicyValueNet(model_file=model_file_current,device='cuda')
    mcts_init = MCTSPlayer(policy_value_net_init.policy_value_fn,
                          c_puct=1,
                          n_playout=2,
                          is_selfplay=0)
    mcts_current = MCTSPlayer(policy_value_net_current.policy_value_fn,
                          c_puct=1,
                          n_playout=2,
                          is_selfplay=0)


    random = Random_agent()

    expert = Expert_agent()


    while True:
        #print_board(game.board)
        end, winner = game.game_over()
        if end:
            print(game.play_out)
            print(winner)
            return winner
        if game.player == Player.black:
            move = expert.select_move(game)
            #move = mcts_init.select_move(game, temp=temp, return_prob=0) # modify move_probs list
        else:
            move = mcts_current.select_move(game, temp=temp, return_prob=0)
        if isinstance(move, int):
            move = game.a_trans_move(move)


        #if move.is_down:
        #    print_move(game.player, move)
        #else:
        #    print_move_go(game.player, move)

        game = game.apply_move(move)

def start_self_play_init_white_vs_current_black(model_file_init,model_file_current,temp=1e-3):
    ### winner, play_data
    # states, mcts_probs, current_players = [], [], []
    game = GameState.new_game(5, 9)
    # mcts = MCTSPlayer(TRTEngine('./pytorch_muc.trt').policy_value_fn,
    policy_value_net_init = PolicyValueNet(model_file_init,device='cuda')
    policy_value_net_current = PolicyValueNet(model_file_current,device='cuda')
    mcts_init = MCTSPlayer(policy_value_net_init.policy_value_fn,
                          c_puct=1,
                          n_playout=2,
                          is_selfplay=0)
    mcts_current = MCTSPlayer(policy_value_net_current.policy_value_fn,
                          c_puct=1,
                          n_playout=2,
                          is_selfplay=0)
    random = Random_agent()
    expert = Expert_agent()
    while True:
        #print_board(game.board)
        end, winner = game.game_over()
        if end:
            print(game.play_out)
            print(winner)
            return winner
        if game.player == Player.black:
            move = mcts_current.select_move(game, temp=temp, return_prob=0) # modify move_probs list
        else:
            move = expert.select_move(game)
            #move = mcts_init.select_move(game, temp=temp, return_prob=0)
        if isinstance(move, int):
            move = game.a_trans_move(move)


        #if move.is_down:
        #    print_move(game.player, move)
        #else:
        #    print_move_go(game.player, move)

        game = game.apply_move(move)
        


if __name__ == '__main__':
    black_current = 0
    white_current = 0
    for _ in tqdm.tqdm(range(100)):
        result1 = start_self_play_init_black_vs_current_white("./model/46000.pth","./model/46000.pth")
        result2 = start_self_play_init_white_vs_current_black("./model/46000.pth","./model/46000.pth")
        if result1 == Player.white:
            white_current += 1
        if result2 == Player.black:
            black_current +=1
        #if result2 == Player.black:
        #    black_current += 1
    print("current black win rate:{}".format(black_current/100))
    print("current white win rate:{}".format(white_current/100))
    ### black is init_net white is current_net

#current black win rate vs random :0.78 
#current white win rate vs random :0.01

