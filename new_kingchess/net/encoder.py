import numpy as np
from fundamental.coordinate import Player, Point
from fundamental.board import GameState, Move, Board
from fundamental.utils import print_board


def moves2flip(moves: dict, i: int):
    '''
    将moves编码
    :param movs:
    :return:
    '''
    moves_key_list = []
    move_val_list = list(moves.values())
    for move in moves.keys():
        if i == 1:
            if move.is_down:
                moves_key_list.append(Move(Point(8 - move.point.col, move.point.row)))
                # move.point.row, move.point.col = (8 - move.point.col, move.point.row)
            else:
                moves_key_list.append(
                    Move(Point(8 - move.point.col, move.point.row), Point(8 - move.point_.col, move.point_.row)))
                # move.point.row, move.point.col = (8 - move.point.col, move.point.row)
                # move.point_.row, move.point_.col = (8 - move.point_.col, move.point_.row)
        if i == 2:
            if move.is_down:
                moves_key_list.append(Move(Point(4 - move.point.row, 8 - move.point.col)))
                # move.point.row, move.point.col = (4-move.point.row, 8-move.point.col)
            else:
                moves_key_list.append(Move(Point(4 - move.point.row, 8 - move.point.col),
                                           Point(4 - move.point_.row, 8 - move.point_.col)))
                # move.point.row, move.point.col = (4 - move.point.row, 8-move.point.col)
                # move.point_.row, move.point_.col = (4 - move.point_.row, 8-move.point_.col)
        if i == 3:
            if move.is_down:

                moves_key_list.append(Move(Point(move.point.col, 4 - move.point.row)))

                # move.point.row, move.point.col = (move.point.col, 4-move.point.row)
            else:
                moves_key_list.append(
                    Move(Point(move.point.col, 4 - move.point.row), Point(move.point_.col, 4 - move.point_.row)))

                # move.point.row, move.point.col = (move.point.col, 4-move.point.row)
                # move.point_.row, move.point_.col = (move.point_.col, 4-move.point_.row)
        if i == 4:
            return moves
    return dict(zip(moves_key_list, move_val_list))

def moves2flip_list(moves: list, i: int):
    '''
    将moves编码
    :param movs:
    :return:
    '''
    moves_key_list = []
    move_val_list = [move[1] for move in moves]  # move is tuple
    for move in [move[0] for move in moves]:
        if i == 1:
            if move.is_down:
                moves_key_list.append(Move(Point(8 - move.point.col, move.point.row)))
                # move.point.row, move.point.col = (8 - move.point.col, move.point.row)
            else:
                moves_key_list.append(
                    Move(Point(8 - move.point.col, move.point.row), Point(8 - move.point_.col, move.point_.row)))
                # move.point.row, move.point.col = (8 - move.point.col, move.point.row)
                # move.point_.row, move.point_.col = (8 - move.point_.col, move.point_.row)
        if i == 2:
            if move.is_down:
                moves_key_list.append(Move(Point(4 - move.point.row, 8 - move.point.col)))
                # move.point.row, move.point.col = (4-move.point.row, 8-move.point.col)
            else:
                moves_key_list.append(Move(Point(4 - move.point.row, 8 - move.point.col),
                                           Point(4 - move.point_.row, 8 - move.point_.col)))
                # move.point.row, move.point.col = (4 - move.point.row, 8-move.point.col)
                # move.point_.row, move.point_.col = (4 - move.point_.row, 8-move.point_.col)
        if i == 3:
            if move.is_down:

                moves_key_list.append(Move(Point(move.point.col, 4 - move.point.row)))

                # move.point.row, move.point.col = (move.point.col, 4-move.point.row)
            else:
                moves_key_list.append(
                    Move(Point(move.point.col, 4 - move.point.row), Point(move.point_.col, 4 - move.point_.row)))

                # move.point.row, move.point.col = (move.point.col, 4-move.point.row)
                # move.point_.row, move.point_.col = (move.point_.col, 4-move.point_.row)
        if i == 4:
            return moves
    return list(zip(moves_key_list, move_val_list)) # should return list 2024/4/4 modify


def moves2horizontally(moves: dict, i):
    '''
    :param moves:
    :return:
    '''
    moves_key_list = []
    move_val_list = list(moves.values())
    for move in moves.keys():
        if i == 1:
            if move.is_down:

                moves_key_list.append(Move(Point(8 - move.point.col, 4 - move.point.row)))

                # move.point.row, move.point.col = (8-move.point.col, 4-move.point.row)
            else:
                moves_key_list.append(Move(Point(8 - move.point.col, 4 - move.point.row),
                                           Point(8 - move.point_.col, 4 - move.point_.row)))

                # move.point.row, move.point.col = (8-move.point.col, 4-move.point.row)
                # move.point_.row, move.point_.col = (8-move.point_.col, 4-move.point_.row)
        if i == 2:
            if move.is_down:

                moves_key_list.append(Move(Point(4 - move.point.row, move.point.col)))

                # move.point.row, move.point.col = (4-move.point.row, move.point.col)
            else:

                moves_key_list.append(
                    Move(Point(4 - move.point.row, move.point.col), Point(4 - move.point_.row, move.point_.col)))

                # move.point.row, move.point.col = (4-move.point.row, move.point.col)
                # move.point_.row, move.point_.col = (4-move.point_.row, move.point_.col)
        if i == 3:
            if move.is_down:

                moves_key_list.append(Move(Point(move.point.col, move.point.row)))

                # move.point.row, move.point.col = (move.point.col, move.point.row)
            else:

                moves_key_list.append(
                    Move(Point(move.point.col, move.point.row), Point(move.point_.col, move.point_.row)))

                # move.point.row, move.point.col = (move.point.col, move.point.row)
                # move.point_.row, move.point_.col = (move.point_.col, move.point_.row)
        if i == 4:
            if move.is_down:

                moves_key_list.append(Move(Point(move.point.row, 8 - move.point.col)))

                # move.point.row, move.point.col = (move.point.row, 8-move.point.col)
            else:

                moves_key_list.append(
                    Move(Point(move.point.row, 8 - move.point.col), Point(move.point_.row, 8 - move.point_.col)))
                # move.point.row, move.point.col = (move.point.row, 8-move.point.col)
                # move.point_.row, move.point_.col = (move.point_.row, 8-move.point_.col)

    return dict(zip(moves_key_list, move_val_list))

def moves2horizontally_list(moves: list, i):
    '''
    :param moves:
    :return:
    '''
    moves_key_list = []
    move_val_list = [move[1] for move in moves]
    for move in [move[0] for move in moves]:
        if i == 1:
            if move.is_down:

                moves_key_list.append(Move(Point(8 - move.point.col, 4 - move.point.row)))

                # move.point.row, move.point.col = (8-move.point.col, 4-move.point.row)
            else:
                moves_key_list.append(Move(Point(8 - move.point.col, 4 - move.point.row),
                                           Point(8 - move.point_.col, 4 - move.point_.row)))

                # move.point.row, move.point.col = (8-move.point.col, 4-move.point.row)
                # move.point_.row, move.point_.col = (8-move.point_.col, 4-move.point_.row)
        if i == 2:
            if move.is_down:

                moves_key_list.append(Move(Point(4 - move.point.row, move.point.col)))

                # move.point.row, move.point.col = (4-move.point.row, move.point.col)
            else:

                moves_key_list.append(
                    Move(Point(4 - move.point.row, move.point.col), Point(4 - move.point_.row, move.point_.col)))

                # move.point.row, move.point.col = (4-move.point.row, move.point.col)
                # move.point_.row, move.point_.col = (4-move.point_.row, move.point_.col)
        if i == 3:
            if move.is_down:

                moves_key_list.append(Move(Point(move.point.col, move.point.row)))

                # move.point.row, move.point.col = (move.point.col, move.point.row)
            else:

                moves_key_list.append(
                    Move(Point(move.point.col, move.point.row), Point(move.point_.col, move.point_.row)))

                # move.point.row, move.point.col = (move.point.col, move.point.row)
                # move.point_.row, move.point_.col = (move.point_.col, move.point_.row)
        if i == 4:
            if move.is_down:

                moves_key_list.append(Move(Point(move.point.row, 8 - move.point.col)))

                # move.point.row, move.point.col = (move.point.row, 8-move.point.col)
            else:

                moves_key_list.append(
                    Move(Point(move.point.row, 8 - move.point.col), Point(move.point_.row, 8 - move.point_.col)))
                # move.point.row, move.point.col = (move.point.row, 8-move.point.col)
                # move.point_.row, move.point_.col = (move.point_.row, 8-move.point_.col)

    return list(zip(moves_key_list, move_val_list))


def flip_board(board):
    board_np = np.zeros((board.num_rows, board.num_cols))
    for point, player in board.get_grid().items():
        board_np[point.row][point.col] = player.value

    board_height = np.rot90(board_np)

    return [board_np, board_height, np.rot90(board_height), np.rot90(np.rot90(board_height))]


def encoder_board(game: GameState):
    board_np = np.zeros((32, 5, game.board.num_cols))  # 5 9
    for point, player in game.board.get_grid().items():
        if player == player.black:
            board_np[0][point.row][point.col] = 1
        elif player == player.white:
            board_np[1][point.row][point.col] = 1
    # board_np[2][game.move.point][game.move.point_] = 1
    board_np[2] = 1 if game.player == Player.black else -1
    #if game.eat_point is not None and game.eat_point is not Point(-1, -1) and game.eat_point != -1:
    #    if isinstance(game.eat_point, Move):
    #        board_np[4][game.eat_point.point.row][game.eat_point.point.col] = 1
    #    else:
    #        board_np[4][game.eat_point.row][game.eat_point.col] = 1
    #if game.move:
    #    if game.move.is_down:
    #        board_np[3][game.move.point.row][game.move.point.col] = 1
    #    elif game.move.is_go:
    #        board_np[3][game.move.point.row][game.move.point.col] = -1
    #        board_np[3][game.move.point_.row][game.move.point_.col] = 1
    if game.play_out<33:
        #print(game.play_out)
        layer = int((game.play_out)/2)
        #print(layer)
        #print(layer+5)
        board_np[layer+3] = 1
    board_np[20+game.eat_chess()] = 1
    return board_np


def decoder_board(board_np: np.ndarray):
    '''
        :return board,player
        :param encoded:
        :return:
        '''
    char_2_player = {
        'X': Player.black,
        "O": Player.white
    }
    board = Board(9, 9)
    board.get_grid().clear()
    decoder_dict = {1: "O", 0: "X"}
    for row in range(board_np.shape[1]):
        for col in range(board_np.shape[1]):
            for k in range(2):
                if board_np[k, row, col]:
                    board.get_grid()[Point(row, col)] = char_2_player[decoder_dict[k]]

    # print_board(board)
    player = board_np[:, :, 2]
    return board, player


def move2str(move: Move):
    if move.is_down:
        return str(move.point.row) + str(move.point.col)
    elif move.is_go:
        return str(move.point.row) + str(move.point.col) + str(move.point_.row) + str(move.point_.col)


def str2move(move: str):
    if len(move) == 4:
        return Move.play_go(point=Point(row=int(move[0]), col=int(move[1])),
                            point_=Point(row=int(move[2]), col=int(move[3])))
    elif len(move) == 2:
        return Move.play_down(point=Point(row=int(move[0]), col=int(move[1])))


if __name__ == '__main__':
    # test
    # print(flip_board(Board(5, 9)))

    game = GameState.new_game(5, 9)

    board_np = encoder_board(game)

    print(board_np)
