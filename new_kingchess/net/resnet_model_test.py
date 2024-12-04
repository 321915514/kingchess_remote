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
    mcts_current = MCTSPlayer(policy_value_net_current.policy_value_fn, c_puct=1, n_playout=100, is_selfplay=False)

    while True:
        end, winner = game.game_over()
        if end:
            # print("winner:{}".format(winner))
            break
        # print_board(game.board)

        if game.player == Player.black:

            # move = random.select_move(game)
            # a = mcts_current.get_action(game)
            # move = game.a_trans_move(a)

            move = expert.select_move(game)

            # print_move_go(game.player, move)
        else:

            a = mcts_current.get_action(game)
            # move = expert.select_move(game)

            # move = random.select_move(game)

            move = game.a_trans_move(a)

            # if move.is_down:
            #     print_move(game.player, move)
            # else:
            #     print_move_go(game.player, move)
        game = game.apply_move(move)

    return winner


if __name__ == '__main__':
    net_black = 0
    expert = 0
    for i in range(100):
        winner = start_play('_', './train_model/best_policy_white_0.6.model')
        if winner == Player.black:
            net_black += 1
        if winner == Player.white:
            expert += 1
    # // net_black:0.11,expert:0.86
    print("net_black:{},expert:{}".format(net_black / 100, expert / 100))
