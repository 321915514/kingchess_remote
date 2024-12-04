import copy
from collections import defaultdict

from math import sqrt

import numpy as np
import torch

from a3c.reward_model import RewardModelWithEmbedding
from agent.expert_agent import Expert_agent
from fundamental.coordinate import Point, Player, Move
from fundamental.utils import print_board, print_move, move_str2_move, list_from_board, move_2_move_str


# from utils import print_board, print_move, print_move_go


# class Move:
#     def __init__(self, point=None, point_=None):
#         assert (point is not None)
#         self.point = point
#         self.point_ = point_
#         self.is_down = (self.point is not None and self.point_ is None)
#         self.is_go = (self.point is not None and self.point_ is not None)
#
#     @classmethod
#     def play_down(cls, point):
#         return Move(point=point)
#
#     @classmethod
#     def play_go(cls, point, point_):
#         return Move(point=point, point_=point_)
#
#     def __str__(self):
#         if self.is_down:
#             return str(self.point.row) + str(self.point.col)
#         else:
#             return str(self.point.row) + str(self.point.col) + str(self.point_.row) + str(self.point_.col)


class Board:
    def __init__(self, num_rows, num_cols):
        self.num_rows = num_rows
        self.num_cols = num_cols
        self._grid = {Point(2, 2): Player.black, Point(1, 3): Player.white,
                      Point(1, 4): Player.white, Point(1, 5): Player.white, Point(2, 3): Player.white,
                      Point(2, 5): Player.white, Point(3, 3): Player.white, Point(3, 4): Player.white,
                      Point(3, 5): Player.white, Point(2, 6): Player.black}
        self.legal_grid = np.ones((num_rows, num_cols), dtype=np.bool_)
        self.init()
        # print(self.legal_grid)

    def init(self):
        illegal_grid = [Point(0, 1), Point(0, 7), Point(1, 0), Point(1, 8), Point(3, 0), Point(3, 8), Point(4, 1),
                        Point(4, 7)]
        for point in illegal_grid:
            self.legal_grid[point.row][[point.col]] = False

    # 判断point是否在棋盘上
    def is_on_grid(self, point):
        return 0 <= int(point.row) < self.num_rows and 0 <= int(point.col) < self.num_cols

    def get(self, point):  # 返回point
        return self._grid.get(point)

    def get_grid(self):
        return self._grid

    def stone_down(self, player, point):
        if self.is_on_grid(point) and self.legal_grid[point.row][point.col]:
            self.get_grid()[point] = player
            return True

    def stone_go(self, player, point, point_):
        # print('start place')
        # print(point)
        # print(point_)
        # print('end place')
        if self.legal_grid[point.row][point.col] and self.legal_grid[point_.row][point_.col]:
            if self.is_jump(point, point_):
                if abs(point.row - point_.row) == 2 and point_.col == point.col:
                    # 上下
                    del self._grid[point]
                    del self._grid[Point((point.row + point_.row) // 2, point.col)]
                    self._grid[point_] = player

                    return Point((point.row + point_.row) // 2, point.col)
                elif abs(point.col - point_.col) == 2 and point_.row == point.row:
                    # 左右
                    del self._grid[point]
                    del self._grid[Point(point.row, (point.col + point_.col) // 2)]
                    self._grid[point_] = player

                    return Point(point.row, (point.col + point_.col) // 2)
                elif abs(point.col - point_.col) == 2 and abs(point.row - point_.row) == 2:
                    # 斜着
                    # print('--------start------')
                    # print(point)
                    # print(point_)
                    # print('--------end----------')
                    del self._grid[point]
                    del self._grid[Point((point.row + point_.row) // 2, (point.col + point_.col) // 2)]
                    self._grid[point_] = player

                    return Point((point.row + point_.row) // 2, (point.col + point_.col) // 2)
                elif point.col == point_.col and abs(point.row - point_.row) == 4:
                    del self._grid[point]
                    del self._grid[Point((point.row + point_.row) // 2, (point.col + point_.col) // 2)]
                    self._grid[point_] = player

                    return Point((point.row + point_.row) // 2, (point.col + point_.col) // 2)
                # return True
            else:
                # print(point)
                # print(point_)
                del self._grid[point]
                self._grid[point_] = player
                # return False

    def is_jump(self, point, point_):
        distance = sqrt((point.row - point_.row) ** 2 + (point_.col - point.col) ** 2)
        coord = point.col + point.row * 9
        coord_ = point_.col + point_.row * 9
        if (coord == 0 and coord_ == 18) or (coord == 18 and coord_ == 0) or (coord == 36 and coord_ == 18) or (
                coord == 18 and coord_ == 36) or (coord == 8 and coord_ == 26) or (coord == 26 and coord_ == 8) or (
                coord == 26 and coord_ == 44) or (coord == 44 and coord_ == 26):
            return False
        elif distance == 1 or distance == sqrt(2):
            return False
        elif distance == 2 * sqrt(2) or distance == 4.0:
            return True
        else:
            return True

    def is_corner(self, point):
        '''
        isborder
        :param point:
        :return: true
        '''
        coord = point.col + point.row * 9
        if coord == 0 or coord == 8 or coord == 36 or coord == 44:
            return True
        return False

    def check_move1(self, point, point_):
        '''
        检查是否可以单走
        :param point:
        :param point_:
        :return:
        '''
        coord1 = point.col + point.row * 9
        coord2 = point_.col + point_.row * 9
        si = point.col
        sj = point.row
        ei = point_.col
        ej = point_.row
        if not self.is_on_grid(point_):
            return False
        if coord1 == coord2:
            return False
        if ((si == 3 and sj == 0 and ei == 2 and ej == 1) or (si == 2 and sj == 1 and ei == 3 and ej == 0)
                or (si == 3 and sj == 2 and ei == 2 and ej == 1) or (si == 2 and sj == 1 and ei == 3 and ej == 2)
                or (si == 3 and sj == 2 and ei == 2 and ej == 3) or (si == 2 and sj == 3 and ei == 3 and ej == 2)
                or (si == 3 and sj == 4 and ei == 2 and ej == 3) or (si == 2 and sj == 3 and ei == 3 and ej == 4)

                or (si == 3 and sj == 0 and ei == 4 and ej == 1) or (si == 4 and sj == 1 and ei == 3 and ej == 0)
                or (si == 3 and sj == 2 and ei == 4 and ej == 1) or (si == 4 and sj == 1 and ei == 3 and ej == 2)
                or (si == 3 and sj == 2 and ei == 4 and ej == 3) or (si == 4 and sj == 3 and ei == 3 and ej == 2)
                or (si == 3 and sj == 4 and ei == 4 and ej == 3) or (si == 4 and sj == 3 and ei == 3 and ej == 4)

                or (si == 5 and sj == 0 and ei == 4 and ej == 1) or (si == 4 and sj == 1 and ei == 5 and ej == 0)
                or (si == 5 and sj == 2 and ei == 4 and ej == 1) or (si == 4 and sj == 1 and ei == 5 and ej == 2)
                or (si == 5 and sj == 2 and ei == 4 and ej == 3) or (si == 4 and sj == 3 and ei == 5 and ej == 2)
                or (si == 5 and sj == 4 and ei == 4 and ej == 3) or (si == 4 and sj == 3 and ei == 5 and ej == 4)

                or (si == 5 and sj == 0 and ei == 6 and ej == 1) or (si == 6 and sj == 1 and ei == 5 and ej == 0)
                or (si == 5 and sj == 2 and ei == 6 and ej == 1) or (si == 6 and sj == 1 and ei == 5 and ej == 2)
                or (si == 5 and sj == 2 and ei == 6 and ej == 3) or (si == 6 and sj == 3 and ei == 5 and ej == 2)
                or (si == 5 and sj == 4 and ei == 6 and ej == 3) or (si == 6 and sj == 3 and ei == 5 and ej == 4)

                or (si == 3 and sj == 0 and ei == 5 and ej == 2) or (si == 5 and sj == 2 and ei == 3 and ej == 0)
                or (si == 4 and sj == 1 and ei == 6 and ej == 3) or (si == 6 and sj == 3 and ei == 4 and ej == 1)

                or (si == 2 and sj == 1 and ei == 4 and ej == 3) or (si == 4 and sj == 3 and ei == 2 and ej == 1)
                or (si == 3 and sj == 2 and ei == 5 and ej == 4) or (si == 5 and sj == 4 and ei == 3 and ej == 2)

                or (si == 5 and sj == 0 and ei == 3 and ej == 2) or (si == 3 and sj == 2 and ei == 5 and ej == 0)
                or (si == 4 and sj == 1 and ei == 2 and ej == 3) or (si == 2 and sj == 3 and ei == 4 and ej == 1)

                or (si == 6 and sj == 1 and ei == 4 and ej == 3) or (si == 4 and sj == 3 and ei == 6 and ej == 1)
                or (si == 5 and sj == 2 and ei == 3 and ej == 4) or (si == 3 and sj == 4 and ei == 5 and ej == 2)):
            return False

        if ((si == 0 and sj == 2 and ei == 1 and ej == 1) or (si == 1 and sj == 1 and ei == 0 and ej == 2)
                or (si == 7 and sj == 1 and ei == 8 and ej == 2) or (si == 8 and sj == 2 and ei == 7 and ej == 1)
                or (si == 0 and sj == 2 and ei == 1 and ej == 3) or (si == 1 and sj == 3 and ei == 0 and ej == 2)
                or (si == 7 and sj == 3 and ei == 8 and ej == 2) or (si == 8 and sj == 2 and ei == 7 and ej == 3)
                or (si == 2 and sj == 0 and ei == 1 and ej == 1) or (si == 1 and sj == 1 and ei == 2 and ej == 0)
                or (si == 1 and sj == 1 and ei == 3 and ej == 1) or (si == 3 and sj == 1 and ei == 1 and ej == 1)
                or (si == 2 and sj == 1 and ei == 1 and ej == 2) or (si == 1 and sj == 2 and ei == 2 and ej == 1)
                or (si == 2 and sj == 1 and ei == 1 and ej == 1) or (si == 1 and sj == 1 and ei == 2 and ej == 1)
                or (si == 2 and sj == 1 and ei == 1 and ej == 1) or (si == 1 and sj == 1 and ei == 2 and ej == 1)
                or (si == 1 and sj == 3 and ei == 2 and ej == 3) or (si == 2 and sj == 3 and ei == 1 and ej == 3)
                or (si == 2 and sj == 3 and ei == 1 and ej == 2) or (si == 1 and sj == 2 and ei == 2 and ej == 3)
                or (si == 1 and sj == 3 and ei == 2 and ej == 4) or (si == 2 and sj == 4 and ei == 1 and ej == 3)
                or (si == 1 and sj == 3 and ei == 3 and ej == 3) or (si == 3 and sj == 3 and ei == 1 and ej == 3)
                or (si == 6 and sj == 0 and ei == 7 and ej == 1) or (si == 7 and sj == 1 and ei == 6 and ej == 0)
                or (si == 6 and sj == 1 and ei == 7 and ej == 1) or (si == 7 and sj == 1 and ei == 6 and ej == 1)
                or (si == 5 and sj == 1 and ei == 7 and ej == 1) or (si == 7 and sj == 1 and ei == 5 and ej == 1)
                or (si == 6 and sj == 1 and ei == 7 and ej == 2) or (si == 7 and sj == 2 and ei == 6 and ej == 1)
                or (si == 6 and sj == 3 and ei == 7 and ej == 2) or (si == 7 and sj == 2 and ei == 6 and ej == 3)
                or (si == 7 and sj == 3 and ei == 5 and ej == 3) or (si == 5 and sj == 3 and ei == 7 and ej == 3)
                or (si == 6 and sj == 3 and ei == 7 and ej == 3) or (si == 7 and sj == 3 and ei == 6 and ej == 3)
                or (si == 6 and sj == 4 and ei == 7 and ej == 3) or (si == 7 and sj == 3 and ei == 6 and ej == 4)
                or (si == 3 and sj == 0 and ei == 1 and ej == 2) or (si == 1 and sj == 2 and ei == 3 and ej == 0)
                or (si == 3 and sj == 4 and ei == 1 and ej == 2) or (si == 1 and sj == 2 and ei == 3 and ej == 4)
                or (si == 5 and sj == 0 and ei == 7 and ej == 2) or (si == 7 and sj == 2 and ei == 5 and ej == 0)
                or (si == 5 and sj == 4 and ei == 7 and ej == 2) or (si == 7 and sj == 2 and ei == 5 and ej == 4)
                or (ei == 1 and ej == 0) or (ei == 1 and ej == 4) or (ei == 7 and ej == 0) or (ei == 7 and ej == 4)
                or (ei == 0 and ej == 1) or (ei == 0 and ej == 3) or (ei == 8 and ej == 1) or (ei == 8 and ej == 3)):
            return False

        return True


def dis(point, point_):
    distance = sqrt((point.row - point_.row) ** 2 + (point_.col - point.col) ** 2)
    return distance


# def encoder_game(game):
#     s = np.zeros([153])
#     for point, player in game.board.get_grid().items():
#         coord = point.col + point.row * 9
#         if player == Player.black:
#             s[coord] = 1  # 1此位置黑棋
#         elif player == Player.white:
#             s[coord + 45] = 1
#     if game.player == Player.white:
#         s[-1] = 1
#     if game.player == Player.black:
#         s[-2] = 1
#     for i in range(90, 90 + 45):
#         if s[i - 45] == 1 or s[i - 45 * 2] == 1:
#             s[i] = 0  # 不为空
#         else:
#             s[i] = 1
#     s[135 + game.play_out // 2] = 1
#     return s


def predict_reward(game, action):
    # mean_reward = 81.63694672371473
    # std_reward = 102.00450246332814 # 53.591406517009425
    embedding_dim = 200  # 嵌入维度
    state_size = 137
    action_size = 1125  # 动作空间大小
    hidden_size = 128
    reward_model = RewardModelWithEmbedding(state_size, action_size, hidden_size, embedding_dim)

    reward_model.load_state_dict(
        torch.load('/home/test4/new_kingchess/a3c/reward_model/reward_model.pth', map_location='cuda:0'))
    state = game.encoder_board_137()
    result = reward_model(torch.unsqueeze(torch.tensor(state, dtype=torch.float32), dim=0),
                          torch.unsqueeze(torch.tensor(action, dtype=torch.long), dim=0))
    # 逆标准化
    # original_reward = result * std_reward + mean_reward
    return result.item()


class GameState():
    def __init__(self, board, player, move, play_out, eat_point, moves):
        self.board = board
        self.player = player
        self.move = move
        self.play_out = play_out
        self.eat_point = eat_point
        self.moves = moves



    def __str__(self):
        return "{},{},{},{}".format(
            self.player, self.move, self.play_out, self.eat_point
        )

    def print_game(self):
        print("{},{},{},{},{}".format(
            self.player, self.move, self.play_out, self.eat_point, self.eat_chess()
        ))

    def a_trans_move(self, a):
        if a < 720:
            coord = a // 16
            coord_ = self.transfer_coord(a)
            move = Move.play_go(Point(coord // 9, coord % 9), Point(coord_ // 9, coord_ % 9))
            return move
        elif 720 <= a < 765:
            # print(a)
            coord = (a - 720)
            move = Move.play_down(Point(coord // 9, coord % 9))
            return move
        elif a >= 765:
            coord = (a - 765) // 8
            coord_ = self.transfer_coord(a)
            move = Move.play_go(Point(coord // 9, coord % 9), Point(coord_ // 9, coord_ % 9))
            return move

    def step_black(self, a):
        # 计算位置
        # s_, r, done
        # print(a)
        expert_agent = Expert_agent()
        if self.game_over()[0]:
            # print('game over')
            done = True
            r = -1000
            return self.encoder_board(), r, done

        if a < 720:
            coord = a // 16
            coord_ = self.transfer_coord(a)
            move = Move.play_go(Point(coord // 9, coord % 9), Point(coord_ // 9, coord_ % 9))
            score_moves = expert_agent.score_moves(self)
            self.apply_step(move)

            # print_board(self.board)
            # print(move)
            # self.print_game()
            if self.game_over()[0]:
                # print('game over')
                done = True
                r = 1000
                return self.encoder_board(), r, done
            else:
                done = False
                # print(move.__str__())
                # for i in score_moves:
                #     print(i.__str__())
                if self.eat_point.row != -1 and self.eat_point.col != -1 and move in score_moves:
                    r = score_moves[move] + self.eat_chess()
                else:
                    r = -1000
                return self.encoder_board(), r, done
        elif 720 <= a < 765:
            # print(a)
            coord = (a - 720)

            move = Move.play_down(Point(coord // 9, coord % 9))
            score_moves = expert_agent.score_moves(self)
            self.player = self.player.other
            black_moves = self.legal_moves()
            self.player = self.player.other
            self.apply_step(move)
            black_moves_after = self.legal_moves()
            # print_board(self.board)
            # print(move)
            # self.print_game()
            if self.game_over()[0]:
                # print('game over')
                done = True
                r = -1000
                return self.encoder_board(), r, done
            else:
                done = False

                # print(move.__str__())
                # for i in score_moves:
                #     print(i.__str__())

                r = 1000
                # elif len(black_moves_after) < len(black_moves): # modify
                #     r = score_moves[move] + 200

                return self.encoder_board(), r, done

        elif a >= 765:
            coord = (a - 765) // 8

            coord_ = self.transfer_coord(a)

            move = Move.play_go(Point(coord // 9, coord % 9), Point(coord_ // 9, coord_ % 9))
            # print(move)
            # score_moves = expert_agent.score_moves(self)

            # self.player = self.player.other
            # black_moves = self.legal_moves()
            # self.player = self.player.other

            self.apply_step(move)

            # black_moves_after = self.legal_moves()

            # print_board(self.board)
            # print(move)
            # self.print_game()
            if self.game_over()[0]:
                done = True
                r = -1000
                return self.encoder_board(), r, done
            else:
                done = False

                # print(move.__str__())
                # for i in score_moves:
                #     print(i.__str__())
                r = 1000
                return self.encoder_board(), r, done

    def step_white(self, a):
        # 计算位置
        # s_, r, done
        # print(a)
        expert_agent = Expert_agent()
        if self.game_over()[0]:
            # print('game over')
            done = True
            r = 100
            return self.encoder_board(), r, done

        if a < 720:
            coord = a // 16
            coord_ = self.transfer_coord(a)
            move = Move.play_go(Point(coord // 9, coord % 9), Point(coord_ // 9, coord_ % 9))
            score_moves = expert_agent.score_moves(self)
            self.apply_step(move)

            # print_board(self.board)
            # print(move)
            # self.print_game()
            if self.game_over()[0]:
                # print('game over')
                done = True
                r = 0
                return self.encoder_board(), r, done
            else:
                done = False
                # print(move.__str__())
                # for i in score_moves:
                #     print(i.__str__())
                if self.eat_point.row != -1 and self.eat_point.col != -1 and move in score_moves:
                    r = 0
                else:
                    r = 0
                return self.encoder_board(), r, done
        elif 720 <= a < 765:
            # print(a)
            coord = (a - 720)

            move = Move.play_down(Point(coord // 9, coord % 9))
            score_moves = expert_agent.score_moves(self)
            # self.player = self.player.other
            # black_moves = self.legal_moves()
            # self.player = self.player.other
            self.apply_step(move)
            # black_moves_after = self.legal_moves()
            # print_board(self.board)
            # print(move)
            # self.print_game()
            if self.game_over()[0]:
                # print('game over')
                done = True
                r = 100
                return self.encoder_board(), r, done
            else:
                done = False

                # print(move.__str__())
                # for i in score_moves:
                #     print(i.__str__())
                if move in score_moves:
                    if score_moves[move] < 0:
                        r = score_moves[move] - 90
                    # elif len(black_moves_after) < len(black_moves): # modify
                    #     r = score_moves[move] + 200
                    else:
                        r = score_moves[move]
                else:
                    r = -100
                return self.encoder_board(), r, done

        elif a >= 765:
            coord = (a - 765) // 8

            coord_ = self.transfer_coord(a)

            move = Move.play_go(Point(coord // 9, coord % 9), Point(coord_ // 9, coord_ % 9))
            # print(move)
            score_moves = expert_agent.score_moves(self)

            self.player = self.player.other
            black_moves = self.legal_moves()
            self.player = self.player.other

            self.apply_step(move)

            black_moves_after = self.legal_moves()

            # print_board(self.board)
            # print(move)
            # self.print_game()
            if self.game_over()[0]:
                done = True
                r = 100
                return self.encoder_board(), r, done
            else:
                done = False

                # print(move.__str__())
                # for i in score_moves:
                #     print(i.__str__())

                if move in score_moves:
                    if len(black_moves_after) < len(black_moves):  # modify
                        r = score_moves[move] + 100
                    # elif score_moves[move] >0 :
                    #     r = score_moves[move] + 1000
                    else:
                        r = -100
                return self.encoder_board(), r, done

    def step(self, a):
        # 计算位置
        # s_, r, done
        # print(a)
        expert_agent = Expert_agent()
        if self.game_over()[0]:
            # print('game over')
            done = True
            r = 1000
            return self.encoder_board(), r, done

        if a < 720:
            coord = a // 16
            coord_ = self.transfer_coord(a)
            move = Move.play_go(Point(coord // 9, coord % 9), Point(coord_ // 9, coord_ % 9))
            score_moves = expert_agent.score_moves(self)
            self.apply_step(move)

            # print_board(self.board)
            # print(move)
            # self.print_game()
            if self.game_over()[0]:
                # print('game over')
                done = True
                r = 1000
                return self.encoder_board(), r, done
            else:
                done = False
                # print(move.__str__())
                # for i in score_moves:
                #     print(i.__str__())
                if self.eat_point.row != -1 and self.eat_point.col != -1 and move in score_moves:
                    r = score_moves[move] + self.eat_chess()
                else:
                    r = -100
                return self.encoder_board(), r, done
        elif 720 <= a < 765:
            # print(a)
            coord = (a - 720)

            move = Move.play_down(Point(coord // 9, coord % 9))
            score_moves = expert_agent.score_moves(self)
            # self.player = self.player.other
            # black_moves = self.legal_moves()
            # self.player = self.player.other
            self.apply_step(move)
            # black_moves_after = self.legal_moves()
            # print_board(self.board)
            # print(move)
            # self.print_game()
            if self.game_over()[0]:
                # print('game over')
                done = True
                r = 1000
                return self.encoder_board(), r, done
            else:
                done = False

                # print(move.__str__())
                # for i in score_moves:
                #     print(i.__str__())
                if move in score_moves:
                    if score_moves[move] < 0:
                        r = score_moves[move] - 90
                    # elif len(black_moves_after) < len(black_moves): # modify
                    #     r = score_moves[move] + 200
                    else:
                        r = score_moves[move]
                else:
                    r = -100
                return self.encoder_board(), r, done

        elif a >= 765:
            coord = (a - 765) // 8

            coord_ = self.transfer_coord(a)

            move = Move.play_go(Point(coord // 9, coord % 9), Point(coord_ // 9, coord_ % 9))
            # print(move)
            score_moves = expert_agent.score_moves(self)

            self.player = self.player.other
            black_moves = self.legal_moves()
            self.player = self.player.other

            self.apply_step(move)

            black_moves_after = self.legal_moves()

            # print_board(self.board)
            # print(move)
            # self.print_game()
            if self.game_over()[0]:
                done = True
                r = 1000
                return self.encoder_board(), r, done
            else:
                done = False

                # print(move.__str__())
                # for i in score_moves:
                #     print(i.__str__())

                if move in score_moves:
                    if len(black_moves_after) < len(black_moves):  # modify
                        r = score_moves[move] + 100
                    # elif score_moves[move] >0 :
                    #     r = score_moves[move] + 1000
                    else:
                        r = -100
                return self.encoder_board(), r, done

    def step_137(self, a):
        # 计算位置
        # s_, r, done
        # print(a)
        expert_agent = Expert_agent()
        if self.game_over()[0]:
            # print('game over')
            done = True
            r = 1000
            return self.encoder_board_137(), r, done

        if a < 720:
            coord = a // 16
            coord_ = self.transfer_coord(a)
            move = Move.play_go(Point(coord // 9, coord % 9), Point(coord_ // 9, coord_ % 9))
            score_moves = expert_agent.score_moves(self)
            self.apply_step(move)

            # print_board(self.board)
            # print(move)
            # self.print_game()
            if self.game_over()[0]:
                # print('game over')
                done = True
                r = 1000
                return self.encoder_board_137(), r, done
            else:
                done = False
                # print(move.__str__())
                # for i in score_moves:
                #     print(i.__str__())
                if self.eat_point.row != -1 and self.eat_point.col != -1 and move in score_moves:
                    r = score_moves[move] + self.eat_chess()
                else:
                    r = -100
                return self.encoder_board_137(), r, done
        elif 720 <= a < 765:
            # print(a)
            coord = (a - 720)

            move = Move.play_down(Point(coord // 9, coord % 9))
            score_moves = expert_agent.score_moves(self)
            # self.player = self.player.other
            # black_moves = self.legal_moves()
            # self.player = self.player.other
            self.apply_step(move)
            # black_moves_after = self.legal_moves()
            # print_board(self.board)
            # print(move)
            # self.print_game()
            if self.game_over()[0]:
                # print('game over')
                done = True
                r = 1000
                return self.encoder_board_137(), r, done
            else:
                done = False

                # print(move.__str__())
                # for i in score_moves:
                #     print(i.__str__())
                if move in score_moves:
                    if score_moves[move] < 0:
                        r = score_moves[move] - 90
                    # elif len(black_moves_after) < len(black_moves): # modify
                    #     r = score_moves[move] + 200
                    else:
                        r = score_moves[move]
                else:
                    r = -100
                return self.encoder_board_137(), r, done

        elif a >= 765:
            coord = (a - 765) // 8

            coord_ = self.transfer_coord(a)

            move = Move.play_go(Point(coord // 9, coord % 9), Point(coord_ // 9, coord_ % 9))
            # print(move)
            score_moves = expert_agent.score_moves(self)

            self.player = self.player.other
            black_moves = self.legal_moves()
            self.player = self.player.other

            self.apply_step(move)

            black_moves_after = self.legal_moves()

            # print_board(self.board)
            # print(move)
            # self.print_game()
            if self.game_over()[0]:
                done = True
                r = 1000
                return self.encoder_board_137(), r, done
            else:
                done = False

                # print(move.__str__())
                # for i in score_moves:
                #     print(i.__str__())

                if move in score_moves:
                    if len(black_moves_after) < len(black_moves):  # modify
                        r = score_moves[move] + 100
                    # elif score_moves[move] >0 :
                    #     r = score_moves[move] + 1000
                    else:
                        r = -100
                return self.encoder_board_137(), r, done

    def step_reward_model(self, a):
        # 计算位置
        # s_, r, done
        # print(a)
        # expert_agent = Expert_agent()

        if a < 720:
            coord = a // 16
            coord_ = self.transfer_coord(a)
            move = Move.play_go(Point(coord // 9, coord % 9), Point(coord_ // 9, coord_ % 9))
            # score_moves = expert_agent.score_moves(self)

            # 应用之前进行评分
            r = predict_reward(self, a)
            # 评分结束

            self.apply_step(move)

            # print_board(self.board)
            # print(move)
            # self.print_game()
            if self.game_over()[0]:
                # print('game over')
                done = True
                r = 1000 + r
                return self.encoder_board(), r, done
            else:
                done = False
                return self.encoder_board(), r, done
        elif 720 <= a < 765:

            coord = (a - 720)

            move = Move.play_down(Point(coord // 9, coord % 9))

            # 应用之前进行评分
            r = predict_reward(self, a)
            # 评分结束

            self.apply_step(move)

            if self.game_over()[0]:
                # print('game over')
                done = True
                r = 1000 + r
                return self.encoder_board(), r, done
            else:
                done = False
                return self.encoder_board(), r, done

        elif a >= 765:
            coord = (a - 765) // 8

            coord_ = self.transfer_coord(a)

            move = Move.play_go(Point(coord // 9, coord % 9), Point(coord_ // 9, coord_ % 9))

            # 应用之前进行评分
            r = predict_reward(self, a)
            # 评分结束
            self.apply_step(move)

            if self.game_over()[0]:
                done = True
                r = 1000 + r
                return self.encoder_board(), r, done
            else:
                done = False

                return self.encoder_board(), r, done

    def decoder_board(self, s_):
        self.board.get_grid().clear()
        for i in range(len(s_)):
            if i < 45 * 2:
                if i < 45:
                    if s_[i] == 1:
                        self.board.get_grid()[Point(row=i // 9, col=i % 9)] = Player.black
                elif 45 <= i < 90:
                    if s_[i] == 1:
                        self.board.get_grid()[Point(row=(i - 45) // 9, col=(i - 45) % 9)] = Player.white

    def move_2_action(self, move):
        if move is None:
            return None
        if self.player == Player.black:
            if move.is_go:
                coord = move.point.col + move.point.row * 9
                a = coord * 16 + self.transfer_pos(move)
                return a

        elif self.player == Player.white:
            if move.is_down:
                coord = move.point.col + move.point.row * 9
                a = 720 + coord
                return a
            elif move.is_go:
                coord = move.point.col + move.point.row * 9
                a = 765 + coord * 8 + self.transfer_pos(move)
                return a

    def encoder_board(self):
        s = np.zeros([153])
        for point, player in self.board.get_grid().items():
            coord = point.col + point.row * 9
            if player == Player.black:
                s[coord] = 1  # 1此位置黑棋
            elif player == Player.white:
                s[coord + 45] = 1
        if self.player == Player.white:
            s[-1] = 1
        if self.player == Player.black:
            s[-2] = 1
        for i in range(90, 90 + 45):
            if s[i - 45] == 1 or s[i - 45 * 2] == 1:
                s[i] = 0  # 不为空
            else:
                s[i] = 1
        if self.play_out <= 32:
            s[135 + self.play_out // 2] = 1
        return s

    def encoder_board_137(self):
        s = np.zeros([137])
        for point, player in self.board.get_grid().items():
            coord = point.col + point.row * 9
            if player == Player.black:
                s[coord] = 1  # 1此位置黑棋
            elif player == Player.white:
                s[coord + 45] = 1
        if self.player == Player.white:
            s[-1] = 1
        if self.player == Player.black:
            s[-2] = 1
        for i in range(90, 90 + 45):
            if s[i - 45] == 1 or s[i - 45 * 2] == 1:
                s[i] = 0  # 不为空
            else:
                s[i] = 1
        # if self.play_out <= 32:
        #     s[135 + self.play_out // 2] = 1
        return s

    def reset(self):
        self.board = Board(5, 9)
        self.play_out = 0
        self.player = Player.black
        self.move = None
        self.eat_point = None
        s = np.zeros([153])
        for point, player in self.board.get_grid().items():
            coord = point.col + point.row * 9
            if player == Player.black:
                s[coord] = 1  # 1此位置黑棋
            elif player == Player.white:
                s[coord + 45] = 1
        if self.player == Player.white:
            s[-1] = 1
        if self.player == Player.black:
            s[-2] = 1
        for i in range(90, 90 + 45):
            if s[i - 45] == 1 or s[i - 45 * 2] == 1:
                s[i] = 0  # 不为空
            else:
                s[i] = 1
        if self.play_out <= 32:
            s[135 + self.play_out // 2] = 1
        return s

    def transfer_pos(self, move):
        if move.is_go:
            coord = move.point.col + move.point.row * 9
            coord_ = move.point_.col + move.point_.row * 9
            if coord == 0 and coord_ == 18:
                return 0
            if coord == 18 and coord_ == 0:
                return 4
            if coord == 0 and coord_ == 36:
                return 8
            if coord == 36 and coord_ == 0:
                return 12

            elif coord == 8 and coord_ == 26:
                return 0
            elif coord == 8 and coord_ == 44:
                return 8
            elif coord == 26 and coord_ == 8:
                return 4
            elif coord == 44 and coord_ == 8:
                return 12

            elif coord == 18 and coord_ == 36:
                return 0
            elif coord == 36 and coord_ == 18:
                return 4
            elif coord == 26 and coord_ == 44:
                return 0
            elif coord == 44 and coord_ == 26:
                return 4

            elif coord - coord_ == 1:  # 左
                return 6
            elif coord - coord_ == -1:  # 右
                return 2
            elif coord - coord_ == -9:  # 下
                return 0
            elif coord - coord_ == 9:  # 上
                return 4
            elif coord - coord_ == -8:  # 左下
                return 7
            elif coord - coord_ == -10:  # 右下
                return 1
            elif coord - coord_ == 8:  # 右上
                return 3
            elif coord - coord_ == 10:  # 左上
                return 5
            elif coord - coord_ == 2:  # 左左
                return 14
            elif coord - coord_ == -2:  # 右
                return 10
            elif coord - coord_ == -18:  # 下下
                return 8
            elif coord - coord_ == 18:  # 上上
                return 12
            elif coord - coord_ == -16:  # 左下
                return 15
            elif coord - coord_ == -20:  # 右下
                return 9
            elif coord - coord_ == 16:  # 右上
                return 11
            elif coord - coord_ == 20:  # 左上
                return 13

    def transfer_coord(self, action):
        if action >= 765:
            coord = (action - 765) // 8
            action_ = (action - 765) % 8
        else:
            coord = action // 16
            action_ = action % 16

        if (coord == 0 or coord == 8) and action_ == 0: return coord + 18
        if (coord == 0 or coord == 8) and action_ == 8: return coord + 36

        if (coord == 36 or coord == 44) and action_ == 4: return coord - 18
        if (coord == 36 or coord == 44) and action_ == 12: return coord - 36

        if (coord == 18 or coord == 26) and action_ == 4: return coord - 18
        if (coord == 18 or coord == 26) and action_ == 0: return coord + 18

        if action_ == 0: return coord + 9
        if action_ == 1: return coord + 10
        if action_ == 2: return coord + 1
        if action_ == 3: return coord - 8
        if action_ == 4: return coord - 9
        if action_ == 5: return coord - 10
        if action_ == 6: return coord - 1
        if action_ == 7: return coord + 8

        if action_ == 8: return coord + 18
        if action_ == 9: return coord + 20
        if action_ == 10: return coord + 2
        if action_ == 11: return coord - 16
        if action_ == 12: return coord - 18
        if action_ == 13: return coord - 20
        if action_ == 14: return coord - 2
        if action_ == 15: return coord + 16

    def legal_position(self):
        pos = np.zeros([1125], dtype=int)
        moves = self.legal_moves()
        pos_moves = []
        if len(moves) > 0:
            if self.player == Player.black:
                for move in moves:
                    if move.is_go:
                        coord = move.point.col + move.point.row * 9
                        # print(move)
                        # print(f"{move.__str__()} -- {coord * 16 + self.transfer_pos(move)}")
                        pos_moves.append(coord * 16 + self.transfer_pos(move))
                        pos[coord * 16 + self.transfer_pos(move)] = 1
                return pos, pos_moves
            elif self.player == Player.white and self.play_out <= 32:
                for move in moves:
                    if move.is_down:
                        coord = move.point.col + move.point.row * 9
                        # print(f"{move.__str__()} -- {720 + coord}")
                        pos_moves.append(720 + coord)
                        pos[720 + coord] = 1
                return pos, pos_moves
            elif self.player == Player.white and self.play_out > 32:
                for move in moves:
                    if move.is_go:
                        # print('move')
                        # print(move)
                        coord = move.point.col + move.point.row * 9
                        # print(coord)
                        # print('---------')
                        # print('coord')
                        # print(coord)
                        # print(self.transfer_pos(move))
                        # print(f"{move.__str__()} -- {765 + (coord * 8 + self.transfer_pos(move))}")
                        pos_moves.append(765 + (coord * 8 + self.transfer_pos(move)))
                        pos[765 + (coord * 8 + self.transfer_pos(move))] = 1
                return pos, pos_moves
        else:
            self.move = None
            # print_board(self.board)
            return None, None

    @classmethod
    def new_game(cls, row_size, col_size):
        if isinstance(row_size, int) and isinstance(col_size, int):
            board = Board(row_size, col_size)
            game = GameState(board, Player.black, None, 0, None, [])
            #game.add_board_state()
            return game

            # def get_left_chess(self):

    #     return self.left_chess
    #
    # def get_sum_chess(self):
    #     return self.sum_chess

    def apply_move(self, move: Move):

        # print("Before move: sum_chess =", self.sum_chess)

        next_board = copy.deepcopy(self.board)
        if move is None:
            return GameState(next_board, self.player.other, move, self.play_out + 1, None)
        elif move.is_down:
            next_board.stone_down(self.player, move.point)
            #self.add_board_state()
            self.moves.append(move)
            self.eat_point = Point(-1, -1)
        elif move.is_go:
            eat_point = next_board.stone_go(self.player, move.point, move.point_)
            #self.add_board_state()
            self.moves.append(move)
            if eat_point is not None:
                self.eat_point = eat_point
            else:
                self.eat_point = Point(-1, -1)
        # else:
        #     next_board = self.board
        # print("After move: sum_chess =", self.sum_chess)
        return GameState(next_board, self.player.other, move, self.play_out + 1, self.eat_point, self.moves)

    def apply_step(self, move: Move):

        # print("Before move: sum_chess =", self.sum_chess)

        next_board = copy.deepcopy(self.board)
        if move is None:

            self.board = next_board
            self.player = self.player.other
            self.move = move
            self.play_out = self.play_out + 1
            self.eat_point = None
            self.moves = []

            # return GameState(next_board, self.player.other, move, self.play_out + 1, None)
        elif move.is_down:
            next_board.stone_down(self.player, move.point)
            #self.add_board_state()
            self.moves.append(move)
            self.eat_point = Point(-1, -1)
        elif move.is_go:
            eat_point = next_board.stone_go(self.player, move.point, move.point_)
            #self.add_board_state()
            self.moves.append(move)
            if eat_point is not None:
                self.eat_point = eat_point
            else:
                self.eat_point = Point(-1, -1)
        # else:
        #     next_board = self.board
        # print("After move: sum_chess =", self.sum_chess)
        self.board = next_board
        self.player = self.player.other
        self.move = move
        self.play_out = self.play_out + 1
        self.eat_point = self.eat_point
        self.moves = self.moves

        # return GameState(next_board, self.player.other, move, self.play_out + 1, self.eat_point)

    def is_None_in_grid(self, move):
        return self.board.is_on_grid(move.point) and self.board.legal_grid[move.point.row][
            move.point.col] and move.point not in self.board.get_grid()

    # def king_target_is_None_in_grid(self, move):
    #     # print(move.point)
    #     return self.board.is_on_grid(move.point) and self.board.legal_grid[move.point.row][
    #         move.point.col] and move.point not in self.board.get_grid()

    def is_valid_move_down_after_16(self, move):  # first point from grid
        for i in move.point.border_neighbors()[:move.point.border_neighbors()[-1]]:
            if i not in self.board.get_grid() and self.board.check_move1(move.point, i):
                return True

    def count_chess(self):
        count_white = 0
        for r in range(0, 45):
            candidate = Point(row=r // 9, col=r % 9)
            if self.board.get(candidate) == Player.white:
                count_white += 1
        return count_white

    def eat_chess(self):
        count_chess = self.count_chess()

        end_chess = 0 if self.play_out > 32 else 16 - self.play_out // 2

        return 24 - count_chess - end_chess

    def game_over(self):
        # for point,player in self.board.get_grid().items():
        # if player == Player.black and legal_moves_black():
        #print(self.moves)
        if self.play_out > 0 and self.move is None:
            return True, Player.white
        # if len(self.legal_moves()) == 1 and (self.legal_moves()[0].point.col<=1 or self.legal_moves()[0].point.col>=7):
        #     return True, Player.white
        if len(self.legal_moves()) == 0:
            return True, Player.white
        if self.eat_chess() >= 11:
            return True, Player.black
       # if self.has_repeated_three_times():
        #    return True, Player.draw
        if len(self.moves)>32 and len(self.moves)%2 == 0  and self.moves[-1] == self.moves[-5]:
            #`print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 重复走")
            return True,Player.white
        if self.play_out > 300:
            return True,Player.draw    
        return False, None

    def legal_actions(self):
        '''
        legal moves
        :return:
        '''
        if self.player == Player.black:
            move_routes = []
            '''
            找到两个国王的走子move
            '''
            kings = []
            for point, player in self.board.get_grid().items():
                if player == Player.black:
                    kings.append(point)
            kings = sorted(kings, key=lambda x: x.row * 9 + x.col)
            for king in kings:
                target_king_stone = self.king_legal_moves(king)
                if len(target_king_stone) > 0:
                    for point in target_king_stone:
                        move_routes.append(self.move_2_action(Move.play_go(king, point)))
            return move_routes
        if self.player == Player.white:
            moves = []
            if self.play_out <= 32:
                for r in range(0, 45):
                    candidate = Point(row=r // 9, col=r % 9)

                    if self.is_None_in_grid(
                            Move.play_down(candidate)):
                        moves.append(self.move_2_action(Move.play_down(candidate)))
                return moves
            else:
                for i in range(0, 45):
                    candidate = Point(row=i // 9, col=i % 9)
                    # 找到点
                    if self.board.get(candidate) == self.player and self.is_valid_move_down_after_16(
                            Move.play_down(candidate)):
                        for neighbor in candidate.border_neighbors()[:candidate.border_neighbors()[-1]]:
                            if self.is_None_in_grid(Move.play_down(neighbor)) and self.board.check_move1(candidate,
                                                                                                         neighbor):
                                moves.append(self.move_2_action(Move.play_go(candidate, neighbor)))
                # moves = sorted(moves, key=lambda x: x.point.row * 9 + x.point.col)
                return moves

    def legal_moves(self):
        '''
        legal moves
        :return:
        '''
        if self.player == Player.black:
            move_routes = []
            '''
            找到两个国王的走子move
            '''
            kings = []
            for point, player in self.board.get_grid().items():
                if player == Player.black:
                    kings.append(point)
            kings = sorted(kings, key=lambda x: x.row * 9 + x.col)
            for king in kings:
                target_king_stone = self.king_legal_moves(king)
                if len(target_king_stone) > 0:
                    for point in target_king_stone:
                        move_routes.append(Move.play_go(king, point))
            return move_routes
        if self.player == Player.white:
            moves = []
            if self.play_out <= 32:
                for r in range(0, 45):
                    candidate = Point(row=r // 9, col=r % 9)

                    if self.is_None_in_grid(
                            Move.play_down(candidate)):
                        moves.append(Move.play_down(candidate))
                return moves
            else:
                for i in range(0, 45):
                    candidate = Point(row=i // 9, col=i % 9)
                    # 找到点
                    if self.board.get(candidate) == self.player and self.is_valid_move_down_after_16(
                            Move.play_down(candidate)):
                        for neighbor in candidate.border_neighbors()[:candidate.border_neighbors()[-1]]:
                            if self.is_None_in_grid(Move.play_down(neighbor)) and self.board.check_move1(candidate,
                                                                                                         neighbor):
                                moves.append(Move.play_go(candidate, neighbor))
                moves = sorted(moves, key=lambda x: x.point.row * 9 + x.point.col)
                return moves

    def king_legal_moves(self, king):
        point_s = []
        # print(king.border_neighbors())
        for index, neighbor in enumerate(king.border_neighbors()[:king.border_neighbors()[-1]]):  #
            if self.board.check_move1(king, neighbor):
                if neighbor not in self.board.get_grid():
                    point_s.append(neighbor)
                else:
                    # 邻居有棋子
                    if len(king.border_neighbors()) - 1 == 16:
                        if self.board.get(neighbor) == Player.white and self.is_None_in_grid(
                                Move.play_down(king.border_neighbors()[index + 8][1])) and self.board.check_move1(
                            neighbor, king.border_neighbors()[index + 8][1]):
                            point_s.append(king.border_neighbors()[index + 8][1])
                    else:
                        # 走到边界
                        if self.board.get(neighbor) == Player.white:
                            # print(king.border_neighbors()[king.border_neighbors()[-1]:])
                            for i in king.border_neighbors()[king.border_neighbors()[-1]:-1]:
                                if neighbor in i:
                                    if self.is_None_in_grid(Move.play_down(i[1])):
                                        point_s.append(i[1])

        return point_s

    # def start_play(self,mcts):
    #     game = self.new_game(5,9)
    #     while True:
    #         end, winner = game.game_over()
    #         if end:
    #             break
    #         if self.player == Player.black:
    #             mcts.


def game_model(content):
    board_front = content['board']
    print(content['board'])
    print(content['eat'])
    print(content['player'])
    print(content['play_out'])
    game = GameState.new_game(5, 9)
    game.board.get_grid().clear()
    for row in range(len(board_front)):
        for col in range(len(board_front[0])):
            if board_front[row][col] == 1:
                game.board.get_grid()[Point(4 - col, row)] = Player.black
            if board_front[row][col] == -1:
                game.board.get_grid()[Point(4 - col, row)] = Player.white
    game.move = move_str2_move(content['move'])  #
    game.player = Player.white if content['player'] == 1 else Player.black
    game.play_out = content['play_out']
    game.eat_point = -1 if content['eat'] == '-1' else move_str2_move(content['eat'])
    return game


def game_model_to_dict(game):

    return {
        'board': list_from_board(game.board),
        'move': move_2_move_str(game.move),
        'eat': -1 if game.eat_point == -1 or game.eat_point is None else str(game.eat_point.col)+str(4-game.eat_point.row),
        'eat_c': game.eat_chess(),
        'player': 1 if game.player == Player.black else -1,
        'play_out': game.play_out
    }






if __name__ == '__main__':
    game = GameState.new_game(5, 9)
    #     print_board(board)
    #     board.get_grid()[Point(0,8)] = Player.white
    #     board.stone_go(Player.white,Point(0,8),Point(2,8))
    #     print_board(board)
    # game.reset()
    # print(game.legal_position())
    # expert = Expert_agent()

    # scores = expert.score_moves(game)

    # print_board(game.board)
    # s_ = game.encoder_board()
    #
    # print(s_)
    #
    # game.decoder_board(s_)
    #
    # print_board(game.board)
    #
    # print(game)
