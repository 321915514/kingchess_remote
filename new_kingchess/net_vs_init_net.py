import numpy as np

from fundamental.board import GameState
from fundamental.coordinate import Player
from fundamental.utils import print_board, print_move_go, print_move
from net.mcts_alphazreo import MCTSPlayer
from net.policy_value_net_pytorch import PolicyValueNet
from agent.random_agent import Random_agent
from agent.expert_agent import Expert_agent

def start_play(model_path, model_path_current):
    game = GameState.new_game(5, 9)

    # policy_value_net = PolicyValueNet(model_file=model_path)
    policy_value_net_current = PolicyValueNet(model_file=model_path_current)

    # mcts = MCTSPlayer(policy_value_net.policy_value_fn, c_puct=5, n_playout=400)
    random = Random_agent()
    expert = Expert_agent()
    mcts_current = MCTSPlayer(policy_value_net_current.policy_value_fn, c_puct=1, n_playout=40, is_selfplay=False)

    while True:
        end, winner = game.game_over()
        if end:
            # print("winner:{}".format(winner))
            return winner
        # print_board(game.board)

        if game.player == Player.black:

            # move = random.select_move(game)
            a = mcts_current.get_action(game)
            move = game.a_trans_move(a)

            # move = expert.select_move(game)

            # print_move_go(game.player, move)
        else:

            # a = mcts_current.get_action(game, full_search=True)
            # move = expert.select_move(game)

            move = random.select_move(game)
            # a = mcts_current.get_action(game)
            # move = game.a_trans_move(a)



            # move = game.a_trans_move(a)

            # if move.is_down:
            #     print_move(game.player, move)
            # else:
            #     print_move_go(game.player, move)
        game = game.apply_move(move)


if __name__ == '__main__':
    black = 0
    white = 0
    for i in range(100):
        winner = start_play('_', './net/model/best_policy_black_1.0.model')
        if winner == Player.black:
            black += 1
        if winner == Player.white:
            white += 1

    print('black:{}, white:{}'.format(black, white)) # net is black:78, random is white:0   black:96, white:2    random black:95, net is white:5
    #best_policy_black_1.0.model is black:64, random is white:0

