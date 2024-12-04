import numpy as np

from fundamental.board import GameState
import json

from fundamental.coordinate import Player, Point
from fundamental.utils import print_board


# def return_move()


# def state_json(state, agent):
#     global count_win_black, count_win_white
#     game = GameState.new_game(5, 9)
#     game.board.get_grid().clear()
#     for i in range(state['board']):
#         if state['board'][i] != 0:
#             game.board.get_grid()[Point(row=i // 9, col=i % 9)] = Player.black if state['board'][i] == 1 else Player.white
#
#     game.player = Player.white if state['player'] == 1 else Player.black
#     game.play_out = state['play_out']
#     game.eat_point = state['eat_point']
#     game.move = game.a_trans_move(state['move'])
#
#     end, winner = game.game_over()
#     if end:
#         if winner == Player.black:
#             count_win_black += 1
#         if winner == Player.white:
#             count_win_white += 1
#         return
#     move = agent.select_move(game)
#     game = game.apply_move(move)
#     return json_state(game)

def json_state(game):
    state = {}
    board = np.zeros((5 * 9), dtype=int)
    for key, val in game.board.get_grid().items():
        if isinstance(key, Point):
            board[key.point_2_index()] = -1 if val == Player.white else 1
    state['board'] = board.tolist()
    state['player'] = -1 if game.player == Player.white else 1
    state['play_out'] = game.play_out
    state['eat_point'] = game.eat_point
    state['move'] = game.move_2_action(game.move)
    json_game = json.dumps(state)
    return json_game


if __name__ == '__main__':
    pass
