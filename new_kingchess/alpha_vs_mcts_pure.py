import numpy as np

from fundamental.board import GameState
from fundamental.coordinate import Player
from fundamental.utils import print_board
from socket_transfer_data.json_gamestate import json_state
from socket_transfer_data.server import send_request
from agent.alpha_beta import Alpha_beta


def play(init_mcts_player, is_shown=0):
    row_size = 5
    col_size = 9
    game = GameState.new_game(row_size, col_size)

    bot2 = init_mcts_player
    while True:
        if is_shown:
            print_board(game.board)
        end, winner = game.game_over()
        if end:
            break
        if game.player == Player.black:
            json_game = json_state(game)
            action = send_request('10.122.7.125', 8888, json_game)
        else:
            action = bot2.select_move(game)

        if isinstance(action, np.int64) or isinstance(action, int):
            game = game.apply_move(game.a_trans_move(action))
        else:
            game = game.apply_move(action)

    return winner


if __name__ == '__main__':
    alpha = Alpha_beta()
    mcts_pure = 0
    alpha_win = 0
    for _ in range(100):
        winner = play(alpha, is_shown=0)
        if winner == Player.black:
            mcts_pure+=1
        if winner == Player.white:
            alpha_win+=1

    print('alpha_win:', alpha_win/100)
    print('mcts_pure:', mcts_pure/100)
    # alpha_win: 0.95
    # mcts_pure: 0.0
