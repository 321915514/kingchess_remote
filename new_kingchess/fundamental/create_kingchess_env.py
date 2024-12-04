
from fundamental.board import GameState


import numpy as np

from fundamental.coordinate import Player


class Create_kingchess_env:
    def __init__(self):
        self.game = GameState.new_game(5, 9)
        self.observation_space = np.zeros((5, 9), dtype=np.float32)

        for point, player in self.game.board.get_grid().items():
            self.observation_space[point.row, point.col] = player

        self.action_space = 1125



    def reset(self):
        self.encode_board = np.zeros((5, 9), dtype=np.float32)
        self.game = GameState.new_game(5, 9)
        for point, player in self.game.board.get_grid().items():
            if player == Player.black:
                self.encode_board[point.row, point.col] = player



