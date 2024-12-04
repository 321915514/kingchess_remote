import enum

import numpy as np

from fundamental import coordinate
# from fundamental.board import GameState
from fundamental.coordinate import Player, Point, Move

COLS = 'ABCDEFGHIJKLMNOPQRST'

STONE_TO_CHAR = {
    None: '.',
    coordinate.Player.black.value: 'X',
    coordinate.Player.white.value: 'O',
}

CHAR_TO_STONE={
    None: '.',
    'X': coordinate.Player.black,
    'O': coordinate.Player.white,
}

def print_move(player,move):
    # if move.is_pass:
    #     move_str = 'passes'
    # elif move.is_resign:
    #     move_str = 'resigns'
    # else:
    if move.point_ is None:
        move_str = '%d%d' % (move.point.row, move.point.col)
    else:
        move_str = '%d%d%d%d' % (move.point.col, move.point.row, move.point_.row, move.point_.col)
    print('%s %s' % (player, move_str))


def print_move_go(player, move):
    # if move.is_pass:
    #     move_str = 'passes'
    # elif move.is_resign:
    #     move_str = 'resigns'
    # else:

    if move is None:
        return
    else:
        if move.point_ is None:
            move_str = '%s%d' % (COLS[move.point.col], move.point.row)
        else:
            move_str = '%s%d' % (COLS[move.point.col], move.point.row)
            # if isinstance(move.point_,list):
            #     for i, p in enumerate(move.point_):
            #         # if i == len(move.point_) - 1:
            #         #     move_str = move_str + '%s%d' % (COLS[p.col - 1], p.row)
            #         if i > 0:
            #             move_str = move_str + '--->%s%d' % (COLS[p.col - 1], p.row)
            # else:
            move_str = move_str + '--->%s%d' % (COLS[move.point_.col], move.point_.row)

        print('%s %s' % (player, move_str))


def print_board(board):
    for row in range(0, board.num_rows):
        bump = ' ' if row <=9 else ''
        line = []
        for col in range(0,board.num_cols):
            stone = board.get(coordinate.Point(row=row,col=col))
            if stone is None:
                if board.legal_grid[row][col]:
                    line.append(STONE_TO_CHAR[stone]+'   ')
                else:
                    line.append(" "+'   ')
            else:
                line.append(STONE_TO_CHAR[stone.value] + '   ')
        print('%s%d %s' % (bump,row,''.join(line)))
    print('   '+'   '.join(COLS[:board.num_cols]))






def point_from_coords(coords:str):
    if len(coords.split(' ')) == 2:
        points = []
        for coord in coords.split(' '):
            # A0
            col = COLS.index(coord[0])
            row = int(coord[1:])
            points.append(Point(row, col))
        return points
    else:
        col = COLS.index(coords[0])
        row = int(coords[1:])
        return Point(row, col)


def board_from_list(board_front, game_state,board_size):
    board_front = np.array(board_front).T.tolist()  # 列表
    count = 0
    # print(board_front)
    for i, key in enumerate(board_front):
        for j, val in enumerate(board_front[i]):
            if board_front[i][j] == 1:
                count += 1
                game_state.board.get_grid[coordinate.Point(row=board_size - i, col=j + 1)] = coordinate.Player.black
            if board_front[i][j] == 2:
                count += 1
                game_state.board.get_grid[coordinate.Point(row=board_size - i, col=j + 1)] = coordinate.Player.white
    return board_front, count


def list_from_board(board):

    board_front = np.zeros((5, 9), dtype=int)

    board_dict = board.get_grid()
    for key, val in board_dict.items():
        board_front[key.row][key.col] = (1 if val == coordinate.Player.black else -1)

    # print(np.rot90(board_front, -1))

    return np.rot90(board_front, -1).tolist()

def coords_from_point(point):
    return '%s%d'%(COLS[point.col-1],point.row)


def move_str2_move(move: str or int):
    # print(move)

    if move is None:
        return None
    if move == -1:
        return -1
    if len(move) == 4:
        row, col, row_, col_ = [i for i in move]
        row = int(row)
        col = int(col)
        row_ = int(row_)
        col_ = int(col_)
        return Move(Point(row=4 - col, col=row), point_=Point(row=4 - col_, col=row_))
    if len(move) == 2:
        print(move)
        row, col = [i for i in move]
        row = int(row)
        col = int(col)
        return Move(Point(row=4 - col, col=row))


def move_2_move_str(move:Move):
    if move is None:
        return None
    if move.is_down:
        return str(move.point.col)+str(4-move.point.row)
    if move.is_go:
        return str(move.point.col)+str(4-move.point.row) + str(move.point_.col)+str(4-move.point_.row)



