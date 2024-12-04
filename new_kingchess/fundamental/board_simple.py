import copy
from math import sqrt

import numpy as np

from fundamental import coordinate
from fundamental.coordinate import Point, Player, Move


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
        # print([point_])
        # print('end place')
        if self.legal_grid[point.row][point.col] and self.legal_grid[point_.row][point_.col]:
            if self.is_jump(point, point_):
                if abs(point.row - point_.row) == 2 and point_.col == point.col:
                    # 上下
                    del self._grid[point]
                    del self._grid[Point((point.row + point_.row) // 2, point.col)]
                    self._grid[point_] = player
                elif abs(point.col - point_.col) == 2 and point_.row == point.row:
                    # 左右
                    del self._grid[point]
                    del self._grid[Point(point.row, (point.col + point_.col) // 2)]
                    self._grid[point_] = player
                elif abs(point.col - point_.col) == 2 and abs(point.row - point_.row) == 2:
                    # 斜着
                    # print('--------start------')
                    # print(point)
                    # print(point_)
                    # print('--------end----------')
                    del self._grid[point]
                    del self._grid[Point((point.row + point_.row) // 2, (point.col + point_.col) // 2)]
                    self._grid[point_] = player
                elif point.col == point_.col and abs(point.row - point_.row) == 4:
                    del self._grid[point]
                    del self._grid[Point((point.row + point_.row) // 2, (point.col + point_.col) // 2)]
                    self._grid[point_] = player
                return True
            else:
                del self._grid[point]
                self._grid[point_] = player
                return False

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


class GameState():
    def __init__(self, board, player, move, play_out):
        self.board = board
        self.player = player
        self.move = move
        self.play_out = play_out

    @classmethod
    def new_game(cls, row_size, col_size):
        if isinstance(row_size, int) and isinstance(col_size, int):
            board = Board(row_size, col_size)

            return GameState(board, Player.black, None, 0)

    # def get_left_chess(self):
    #     return self.left_chess
    #
    # def get_sum_chess(self):
    #     return self.sum_chess

    def apply_move(self, move: Move):

        # print("Before move: sum_chess =", self.sum_chess)
        next_board = copy.deepcopy(self.board)
        if move is None:
            return GameState(next_board, self.player.other, move, self.play_out + 1)
        elif move.is_down:
            next_board.stone_down(self.player, move.point)
        elif move.is_go:
            next_board.stone_go(self.player, move.point, move.point_)
        # else:
        #     next_board = self.board
        # print("After move: sum_chess =", self.sum_chess)
        return GameState(next_board, self.player.other, move, self.play_out + 1)

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
        if self.move is None and self.play_out>0:
            return True, Player.white
        if self.eat_chess() >= 11:
            return True, Player.black
        return False, None

    def legal_moves(self):

        if self.player == Player.black:
            move_routes = []
            '''
            找到两个国王的走子move
            '''
            kings = []
            for point, player in self.board.get_grid().items():
                if player == Player.black:
                    kings.append(point)
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


def print_board(board: Board):
    COLS = 'ABCDEFGHIJKLMNOPQRST'

    STONE_TO_CHAR = {
        None: '.',
        coordinate.Player.black.value: 'X',
        coordinate.Player.white.value: 'O',
    }
    for row in range(0, board.num_rows):
        bump = ' ' if row <= 9 else ''
        line = []
        for col in range(0, board.num_cols):
            stone = board.get(coordinate.Point(row=row, col=col))
            if stone is None:
                if board.legal_grid[row][col]:
                    line.append(STONE_TO_CHAR[stone] + '   ')
                else:
                    line.append(" " + '   ')
            else:
                line.append(STONE_TO_CHAR[stone.value] + '   ')
        print('%s%d %s' % (bump, row, ''.join(line)))
    print('   ' + '   '.join(COLS[:board.num_cols]))

# if __name__ == '__main__':
#     board = GameState.new_game(5, 9).board
#     print_board(board)
#     board.get_grid()[Point(0,8)] = Player.white
#     board.stone_go(Player.white,Point(0,8),Point(2,8))
#     print_board(board)
