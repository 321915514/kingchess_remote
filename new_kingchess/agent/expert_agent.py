import random
from math import sqrt

from agent.base import Agent
from fundamental.coordinate import Move
from fundamental.coordinate import Player, Point
from fundamental.utils import print_board


def distance(point, point_):
    distance = sqrt((point.row - point_.row) ** 2 + (point_.col - point.col) ** 2)

    return distance


def coord_point(coord):
    return Point(row=coord // 9, col=coord % 9)


def point_coord(point):
    return point.col + point.row * 9


def is_white_chess(game, coord):
    assert legal_pos(coord)
    return game.board.get_grid()[coord_point(coord)] == Player.white


def legal_pos(coord):
    # print(coord)
    if coord >= 0 and coord != 1 and coord != 7 and coord != 9 and coord != 17 and coord != 27 and coord != 35 and coord != 37 and coord != 43 and coord < 45:
        return True
    else:
        return False


def is_border(coord):
    return coord == 8 or coord == 18 or coord == 26 or coord == 36 or coord == 44 or coord == 0


class Expert_agent(Agent):
    def __init__(self):
        super(Expert_agent, self).__init__()
    
    def get_action(self, game_state):
        if game_state.player == Player.white:
            if game_state.play_out <= 32:
                candidates = []
                for r in range(0, game_state.board.num_rows):
                    for c in range(0, game_state.board.num_cols):
                        candidate = Point(row=r, col=c)
                        # 找到点
                        if game_state.is_None_in_grid(
                                Move.play_down(candidate)):
                            candidates.append(candidate)
                black_candidates = []
                for point, player in game_state.board.get_grid().items():
                    if player == Player.black:
                        black_candidates.append(point)

                scores_moves = dict()
                for black in black_candidates:  # point
                    coord = black.col + black.row * 9
                    point_s = game_state.king_legal_moves(black)
                    # modify 7/17
                    for possible_point in candidates:
                        if (distance(black, possible_point) == 2 or distance(black, possible_point) == 2 * sqrt(
                                2)) and not is_border(possible_point):
                            point_s.append(possible_point)
                    # end add possible distance == 2

                    for score_move in point_s:
                        coord_ = score_move.col + score_move.row * 9

                        # 添加目标点的周围的距离为2*sqrt(2)的位置
                        count = 0
                        for possible_point in candidates:
                            if (distance(score_move, possible_point) == 2 or distance(score_move,
                                                                                      possible_point) == 2 * sqrt(
                                2)) and not is_border(
                                possible_point) and possible_point in game_state.board.get_grid().keys() and \
                                    game_state.board.get_grid()[mid_point] == Player.black:
                                count += 1
                        if count == 2:
                            scores_moves[score_move] = scores_moves[score_move] + 100 + random.random()

                        # end

                        if game_state.board.is_jump(black, score_move):
                            mid_coord = max(coord, coord_) - abs(coord_ - coord) // 2
                            mid_point = Point(row=mid_coord // 9, col=mid_coord % 9)
                            if mid_point in game_state.board.get_grid().keys() and game_state.board.get_grid()[
                                mid_point] == Player.white and game_state.board.check_move1(score_move, mid_point):
                                scores_moves[score_move] = scores_moves[
                                                               score_move] + 150 + random.random() if score_move in scores_moves else 150 + random.random()
                            elif mid_point not in game_state.board.get_grid().keys() and game_state.board.check_move1(
                                    score_move, mid_point):
                                scores_moves[score_move] = scores_moves[
                                                               score_move] + 100 + random.random() if score_move in scores_moves else 100 + random.random()

                        elif (distance(black, score_move) == 2 or distance(black, score_move) == 2 * sqrt(
                                2)) and not is_border(coord):
                            mid_coord = max(coord, coord_) - abs(coord_ - coord) // 2
                            mid_point = Point(row=mid_coord // 9, col=mid_coord % 9)
                            if mid_point not in game_state.board.get_grid().keys() and game_state.board.check_move1(
                                    score_move, mid_point) and game_state.board.check_move1(mid_point, black):
                                scores_moves[score_move] = 100 + random.random()

                        if distance(black, score_move) == sqrt(2) or distance(black, score_move) == 1:
                            if coord_ > coord:
                                max_coord = (max(coord, coord_) + abs(coord - coord_))
                                if legal_pos(max_coord) and Point(row=max_coord // 9,
                                                                  col=max_coord % 9) in game_state.board.get_grid() and \
                                        game_state.board.get_grid()[
                                            Point(row=max_coord // 9,
                                                  col=max_coord % 9)] == Player.white and game_state.board.check_move1(
                                    score_move,
                                    Point(row=max_coord // 9, col=max_coord % 9)) and game_state.board.check_move1(
                                    score_move, black):
                                    scores_moves[score_move] = 100 + random.random()
                                else:
                                    scores_moves[score_move] = -100 + random.random()
                            elif coord_ < coord:
                                min_coord = (min(coord, coord_) - abs(coord - coord_))
                                if legal_pos(min_coord) and Point(row=min_coord // 9,
                                                                  col=min_coord % 9) in game_state.board.get_grid() and \
                                        game_state.board.get_grid()[
                                            Point(row=min_coord // 9,
                                                  col=min_coord % 9)] == Player.white and game_state.board.check_move1(
                                    score_move,
                                    Point(row=min_coord // 9, col=min_coord % 9)) and game_state.board.check_move1(
                                    score_move, black):
                                    scores_moves[score_move] = 100 + random.random()
                                else:
                                    scores_moves[score_move] = -100 + random.random()

                        if (coord == 0 or coord == 8) and coord_ == coord + 36:
                            scores_moves[score_move] = 70 + random.random()
                        if (coord == 44 or coord == 36) and coord_ == coord - 36:
                            scores_moves[score_move] = 70 + random.random()
                        if (coord == 18 or coord == 26) and coord_ == coord - 18:
                            scores_moves[score_move] = 70 + random.random()
                    # 中间的棋子
                    if coord == 19:
                        if Point(row=(coord - 1) // 9, col=(coord - 1) % 9) not in game_state.board.get_grid():
                            scores_moves[Point(row=(coord - 1) // 9, col=(coord - 1) % 9)] = 70 + random.random()
                        if Point(row=(coord - 9) // 9, col=(coord - 9) % 9) not in game_state.board.get_grid():
                            scores_moves[Point(row=(coord - 9) // 9, col=(coord - 9) % 9)] = 70 + random.random()
                        if Point(row=(coord + 9) // 9, col=(coord + 9) % 9) not in game_state.board.get_grid():
                            scores_moves[Point(row=(coord + 9) // 9, col=(coord + 9) % 9)] = 70 + random.random()
                    if coord == 25:
                        if Point(row=(coord + 1) // 9, col=(coord + 1) % 9) not in game_state.board.get_grid():
                            scores_moves[Point(row=(coord + 1) // 9, col=(coord + 1) % 9)] = 70 + random.random()
                        if Point(row=(coord - 9) // 9, col=(coord - 9) % 9) not in game_state.board.get_grid():
                            scores_moves[Point(row=(coord - 9) // 9, col=(coord - 9) % 9)] = 70 + random.random()
                        if Point(row=(coord + 9) // 9, col=(coord + 9) % 9) not in game_state.board.get_grid():
                            scores_moves[Point(row=(coord + 9) // 9, col=(coord + 9) % 9)] = 70 + random.random()

                # print(scores_moves)
                for i in candidates:
                    if i not in scores_moves:
                        scores_moves[i] = random.random()
                # print(scores_moves)

                # 添加(2,2)的分数

                for black in black_candidates:
                    if black.col < 2 and Point(2, 2) in scores_moves and scores_moves[Point(2, 2)] > 1:
                        scores_moves[Point(2, 2)] += 100
                    if black.col > 6 and Point(2, 6) in scores_moves and scores_moves[Point(2, 6)] > 1:
                        scores_moves[Point(2, 6)] += 100

                # end


                if len(scores_moves) != len(game_state.legal_moves()):
                    print_board(game_state.board)
                    print(scores_moves)
                    print(game_state.legal_moves())
                    exit(0)


                scores_move = max(scores_moves, key=scores_moves.get)
                # print(scores_moves[scores_move])



                return Move.play_down(scores_move)
            else:
                candidates = []
                for i in range(0, 45):
                    candidate = Point(row=i // 9, col=i % 9)
                    # 找到点
                    if game_state.board.get(
                            candidate) == game_state.player and game_state.is_valid_move_down_after_16(
                        Move.play_down(candidate)):
                        candidates.append(candidate)
                candidates = sorted(candidates, key=lambda x: x.row * 9 + x.col)

                white_routes = dict()
                black_candidates = []
                for point, player in game_state.board.get_grid().items():
                    if player == Player.black:
                        black_candidates.append(point)

                for black in black_candidates:
                    coord = black.col + black.row * 9
                    point_s = game_state.king_legal_moves(black)
                    # print(point_s)
                    # 先找哪些限制黑棋的点
                    for score_point in point_s:
                        # 黑棋能跳的点得分高
                        if game_state.board.is_jump(black, score_point):
                            # score_point
                            score_point_coord = score_point.col + score_point.row * 9
                            for white_point in candidates:
                                white_point_coord = white_point.col + white_point.row * 9
                                if (distance(white_point, score_point) == 1 or distance(white_point,
                                                                                        score_point) == sqrt(
                                    2)) and game_state.board.check_move1(white_point,
                                                                         score_point) and game_state.board.check_move1(
                                    white_point, score_point):
                                    white_routes[Move(white_point, score_point)] = 100 + random.random() + 1

                                if (
                                        score_point_coord == 0 or score_point_coord == 8 or score_point_coord == 18 or score_point_coord == 26) and white_point_coord == (
                                        score_point_coord + 18):
                                    white_routes[Move(white_point, score_point)] = 100 + random.random() + 2
                                if (
                                        score_point_coord == 18 or score_point_coord == 26 or score_point_coord == 36 or score_point_coord == 44) and white_point_coord == (
                                        score_point_coord - 18):
                                    white_routes[Move(white_point, score_point)] = 100 + random.random() + 3

                        else:

                            # 中间的棋子
                            # if coord == 19:
                            #     if Point(row=(coord - 1) // 9, col=(coord - 1) % 9) not in game_state.board.get_grid():
                            #         white_routes[
                            #             Point(row=(coord - 1) // 9, col=(coord - 1) % 9),] = 70 + random.random()
                            #     if Point(row=(coord - 9) // 9, col=(coord - 9) % 9) not in game_state.board.get_grid():
                            #         white_routes[
                            #             Point(row=(coord - 9) // 9, col=(coord - 9) % 9)] = 70 + random.random()
                            #     if Point(row=(coord + 9) // 9, col=(coord + 9) % 9) not in game_state.board.get_grid():
                            #         white_routes[
                            #             Point(row=(coord + 9) // 9, col=(coord + 9) % 9)] = 70 + random.random()
                            # if coord == 25:
                            #     if Point(row=(coord + 1) // 9, col=(coord + 1) % 9) not in game_state.board.get_grid():
                            #         white_routes[
                            #             Point(row=(coord + 1) // 9, col=(coord + 1) % 9)] = 70 + random.random()
                            #     if Point(row=(coord - 9) // 9, col=(coord - 9) % 9) not in game_state.board.get_grid():
                            #         white_routes[
                            #             Point(row=(coord - 9) // 9, col=(coord - 9) % 9)] = 70 + random.random()
                            #     if Point(row=(coord + 9) // 9, col=(coord + 9) % 9) not in game_state.board.get_grid():
                            #         white_routes[
                            #             Point(row=(coord + 9) // 9, col=(coord + 9) % 9)] = 70 + random.random()

                            # 黑棋不能跳的点，判断距离 先判断 score_point 的距离
                            coord_ = score_point.col + score_point.row * 9
                            if distance(score_point, black) == 2 or distance(score_point, black) == 2 * sqrt(
                                    2) and not is_border(coord):

                                # 判断

                                if score_point not in game_state.board.get_grid():
                                    # . . X
                                    for white_point in candidates:

                                        if (distance(white_point, score_point) == 1 or distance(white_point,
                                                                                                score_point) == sqrt(
                                            2)) and score_point not in game_state.board.get_grid() and game_state.board.check_move1(
                                            white_point, score_point):
                                            white_routes[Move(white_point, score_point)] = 100 + random.random() + 4
                                elif score_point in game_state.board.get_grid() and game_state.board.get_grid()[
                                    score_point] == Player.white:
                                    # 0 . X
                                    for white_point in candidates:

                                        mid_coord = (coord_ + coord) // 2

                                        if (distance(white_point,
                                                     Point(mid_coord // 9, mid_coord % 9)) == 1 or distance(
                                                white_point, Point(mid_coord // 9, mid_coord % 9)) == sqrt(
                                            2)) and score_point != white_point and Point(mid_coord // 9,
                                                                                         mid_coord % 9) not in game_state.board.get_grid() and game_state.board.check_move1(
                                            white_point, Point(mid_coord // 9,
                                                               mid_coord % 9)):
                                            white_routes[Move(white_point, Point(mid_coord // 9,
                                                                                 mid_coord % 9))] = 100 + random.random() + 5
                            elif is_border(coord) and distance(score_point, black) == 2 and is_border(
                                    score_point):  # 两个都是边界
                                if coord == 18 or coord == 26:
                                    for white in candidates:
                                        if distance(white, score_point) == sqrt(
                                                2) and score_point not in game_state.board.get_grid():
                                            white_routes[Move(white, score_point)] = 100 + random.random() + 6
                                if (coord == 0 or coord == 8) and game_state.board.get_grid()[
                                    coord + 36] == Player.white:
                                    for white in candidates:
                                        if distance(white,
                                                    score_point) == 1 and score_point not in game_state.board.get_grid():
                                            white_routes[Move(white, score_point)] = 100 + random.random() + 7

                                if (coord == 36 or coord == 44) and game_state.board.get_grid()[
                                    coord - 36] == Player.white:
                                    for white in candidates:
                                        if distance(white,
                                                    score_point) == 1 and score_point not in game_state.board.get_grid():
                                            white_routes[Move(white, score_point)] = 100 + random.random() + 8

                            elif distance(score_point, black) == 1 or distance(score_point, black) == sqrt(2):
                                # if coord_<coord:
                                pre_point_coord = ((coord_ * 2) - coord)

                                pre_point = Point(pre_point_coord // 9, pre_point_coord % 9)
                                # print(pre_point_coord)
                                # print(pre_point)
                                # print(game_state.board.get_grid())
                                for white_point in candidates:
                                    if legal_pos(pre_point_coord):
                                        if pre_point in game_state.board.get_grid() and \
                                                game_state.board.get_grid()[pre_point] == Player.white and (
                                                distance(white_point, score_point) == 1 or distance(white_point,
                                                                                                    score_point) == sqrt(
                                            2)) and game_state.board.check_move1(white_point,
                                                                                 score_point) and score_point not in game_state.board.get_grid() and pre_point != white_point:

                                            score_point_c = score_point.col + score_point.row * 9
                                            mid_coord = (score_point_c + coord) // 2
                                            mid_coord_point = Point(mid_coord // 9, mid_coord % 9)
                                            if distance(score_point,
                                                        black) == 2 and mid_coord_point in game_state.board.get_grid() and \
                                                    game_state.board.get_grid()[
                                                        mid_coord_point] == Player.white:  # 还需判断另一个黑棋的距离
                                                white_routes[Move(white_point, score_point)] = 100 + random.random() + 9
                                            else:
                                                white_routes[
                                                    Move(white_point, score_point)] = 100 + random.random() + 10
                                        elif score_point not in game_state.board.get_grid() and game_state.board.check_move1(
                                                white_point, pre_point) and distance(white_point,
                                                                                     pre_point) == 1 or distance(
                                            white_point, pre_point) == sqrt(2):
                                            if pre_point not in game_state.board.get_grid() and game_state.board.check_move1(
                                                    white_point, pre_point):
                                                white_routes[Move(white_point, pre_point)] = 100 + random.random() + 11
                                # elif coord_>coord:
                                #     for white_point in candidates:
                                #         pre_point_coord = ((coord_*2)-coord)
                                #         pre_point = Point(pre_point_coord // 9, pre_point_coord % 9)
                                #         if legal_pos(pre_point_coord) and pre_point in game_state.board.get_grid() and game_state.board.get_grid()[
                                #             pre_point] == Player.white and (
                                #                 distance(white_point, score_point) == 1 or distance(white_point,
                                #                                                                     score_point) == sqrt(
                                #                 2)) and game_state.board.check_move1(white_point,
                                #                                                      score_point) and score_point not in game_state.board.get_grid():
                                #             white_routes[Move(white_point, score_point)] = 100 + random.random()
                                #         elif legal_pos(pre_point_coord) and pre_point not in game_state.board.get_grid() and score_point not in game_state.board.get_grid() and game_state.board.check_move1(white_point, pre_point):
                                #             white_routes[Move(white_point, pre_point)] = 50 + random.random()
                for i in game_state.legal_moves():
                    if i not in white_routes:
                        white_routes[i] = random.random()

                if len(white_routes) != len(game_state.legal_moves()):
                    print(white_routes)
                    print(game_state.legal_moves())
                    print('select move')
                    exit(0)
                # for k, v in white_routes.items():
                #     print(f'({k.__str__()},{v} )')

                white_route = max(white_routes, key=white_routes.get)

                return white_route
        else:
            # 黑棋 国王 检查跳
            black_candidates = []
            for point, player in game_state.board.get_grid().items():
                if player == Player.black:
                    black_candidates.append(point)

            scores_moves = dict()

            black_moves = game_state.legal_moves()

            if len(black_moves) == 0:
                return None
            else:
                for black_move in black_moves:
                    if game_state.board.is_jump(black_move.point, black_move.point_):
                        scores_moves[black_move] = 100
                    else:
                        scores_moves[black_move] = random.random()


                if len(scores_moves) != len(game_state.legal_moves()):
                    print_board(game_state.board)
                    print(scores_moves)
                    print(game_state.legal_moves())
                    exit(0)

                scores_move = max(scores_moves, key=scores_moves.get)

                return scores_move

    
    def select_move(self, game_state):
        if game_state.player == Player.white:
            if game_state.play_out <= 32:
                candidates = []
                for r in range(0, game_state.board.num_rows):
                    for c in range(0, game_state.board.num_cols):
                        candidate = Point(row=r, col=c)
                        # 找到点
                        if game_state.is_None_in_grid(
                                Move.play_down(candidate)):
                            candidates.append(candidate)
                black_candidates = []
                for point, player in game_state.board.get_grid().items():
                    if player == Player.black:
                        black_candidates.append(point)

                scores_moves = dict()
                for black in black_candidates:  # point
                    coord = black.col + black.row * 9
                    point_s = game_state.king_legal_moves(black)
                    # modify 7/17
                    for possible_point in candidates:
                        if (distance(black, possible_point) == 2 or distance(black, possible_point) == 2 * sqrt(
                                2)) and not is_border(possible_point):
                            point_s.append(possible_point)
                    # end add possible distance == 2

                    for score_move in point_s:
                        coord_ = score_move.col + score_move.row * 9

                        # 添加目标点的周围的距离为2*sqrt(2)的位置
                        count = 0
                        for possible_point in candidates:
                            if (distance(score_move, possible_point) == 2 or distance(score_move,
                                                                                      possible_point) == 2 * sqrt(
                                2)) and not is_border(
                                possible_point) and possible_point in game_state.board.get_grid().keys() and \
                                    game_state.board.get_grid()[mid_point] == Player.black:
                                count += 1
                        if count == 2:
                            scores_moves[score_move] = scores_moves[score_move] + 100 + random.random()

                        # end

                        if game_state.board.is_jump(black, score_move):
                            mid_coord = max(coord, coord_) - abs(coord_ - coord) // 2
                            mid_point = Point(row=mid_coord // 9, col=mid_coord % 9)
                            if mid_point in game_state.board.get_grid().keys() and game_state.board.get_grid()[
                                mid_point] == Player.white and game_state.board.check_move1(score_move, mid_point):
                                scores_moves[score_move] = scores_moves[
                                                               score_move] + 150 + random.random() if score_move in scores_moves else 150 + random.random()
                            elif mid_point not in game_state.board.get_grid().keys() and game_state.board.check_move1(
                                    score_move, mid_point):
                                scores_moves[score_move] = scores_moves[
                                                               score_move] + 100 + random.random() if score_move in scores_moves else 100 + random.random()

                        elif (distance(black, score_move) == 2 or distance(black, score_move) == 2 * sqrt(
                                2)) and not is_border(coord):
                            mid_coord = max(coord, coord_) - abs(coord_ - coord) // 2
                            mid_point = Point(row=mid_coord // 9, col=mid_coord % 9)
                            if mid_point not in game_state.board.get_grid().keys() and game_state.board.check_move1(
                                    score_move, mid_point) and game_state.board.check_move1(mid_point, black):
                                scores_moves[score_move] = 100 + random.random()

                        if distance(black, score_move) == sqrt(2) or distance(black, score_move) == 1:
                            if coord_ > coord:
                                max_coord = (max(coord, coord_) + abs(coord - coord_))
                                if legal_pos(max_coord) and Point(row=max_coord // 9,
                                                                  col=max_coord % 9) in game_state.board.get_grid() and \
                                        game_state.board.get_grid()[
                                            Point(row=max_coord // 9,
                                                  col=max_coord % 9)] == Player.white and game_state.board.check_move1(
                                    score_move,
                                    Point(row=max_coord // 9, col=max_coord % 9)) and game_state.board.check_move1(
                                    score_move, black):
                                    scores_moves[score_move] = 100 + random.random()
                                else:
                                    scores_moves[score_move] = -100 + random.random()
                            elif coord_ < coord:
                                min_coord = (min(coord, coord_) - abs(coord - coord_))
                                if legal_pos(min_coord) and Point(row=min_coord // 9,
                                                                  col=min_coord % 9) in game_state.board.get_grid() and \
                                        game_state.board.get_grid()[
                                            Point(row=min_coord // 9,
                                                  col=min_coord % 9)] == Player.white and game_state.board.check_move1(
                                    score_move,
                                    Point(row=min_coord // 9, col=min_coord % 9)) and game_state.board.check_move1(
                                    score_move, black):
                                    scores_moves[score_move] = 100 + random.random()
                                else:
                                    scores_moves[score_move] = -100 + random.random()

                        if (coord == 0 or coord == 8) and coord_ == coord + 36:
                            scores_moves[score_move] = 70 + random.random()
                        if (coord == 44 or coord == 36) and coord_ == coord - 36:
                            scores_moves[score_move] = 70 + random.random()
                        if (coord == 18 or coord == 26) and coord_ == coord - 18:
                            scores_moves[score_move] = 70 + random.random()
                    # 中间的棋子
                    if coord == 19:
                        if Point(row=(coord - 1) // 9, col=(coord - 1) % 9) not in game_state.board.get_grid():
                            scores_moves[Point(row=(coord - 1) // 9, col=(coord - 1) % 9)] = 70 + random.random()
                        if Point(row=(coord - 9) // 9, col=(coord - 9) % 9) not in game_state.board.get_grid():
                            scores_moves[Point(row=(coord - 9) // 9, col=(coord - 9) % 9)] = 70 + random.random()
                        if Point(row=(coord + 9) // 9, col=(coord + 9) % 9) not in game_state.board.get_grid():
                            scores_moves[Point(row=(coord + 9) // 9, col=(coord + 9) % 9)] = 70 + random.random()
                    if coord == 25:
                        if Point(row=(coord + 1) // 9, col=(coord + 1) % 9) not in game_state.board.get_grid():
                            scores_moves[Point(row=(coord + 1) // 9, col=(coord + 1) % 9)] = 70 + random.random()
                        if Point(row=(coord - 9) // 9, col=(coord - 9) % 9) not in game_state.board.get_grid():
                            scores_moves[Point(row=(coord - 9) // 9, col=(coord - 9) % 9)] = 70 + random.random()
                        if Point(row=(coord + 9) // 9, col=(coord + 9) % 9) not in game_state.board.get_grid():
                            scores_moves[Point(row=(coord + 9) // 9, col=(coord + 9) % 9)] = 70 + random.random()

                # print(scores_moves)
                for i in candidates:
                    if i not in scores_moves:
                        scores_moves[i] = random.random()
                # print(scores_moves)

                # 添加(2,2)的分数

                for black in black_candidates:
                    if black.col < 2 and Point(2, 2) in scores_moves and scores_moves[Point(2, 2)] > 1:
                        scores_moves[Point(2, 2)] += 100
                    if black.col > 6 and Point(2, 6) in scores_moves and scores_moves[Point(2, 6)] > 1:
                        scores_moves[Point(2, 6)] += 100

                # end


                if len(scores_moves) != len(game_state.legal_moves()):
                    print_board(game_state.board)
                    print(scores_moves)
                    print(game_state.legal_moves())
                    exit(0)


                scores_move = max(scores_moves, key=scores_moves.get)
                # print(scores_moves[scores_move])



                return Move.play_down(scores_move)
            else:
                candidates = []
                for i in range(0, 45):
                    candidate = Point(row=i // 9, col=i % 9)
                    # 找到点
                    if game_state.board.get(
                            candidate) == game_state.player and game_state.is_valid_move_down_after_16(
                        Move.play_down(candidate)):
                        candidates.append(candidate)
                candidates = sorted(candidates, key=lambda x: x.row * 9 + x.col)

                white_routes = dict()
                black_candidates = []
                for point, player in game_state.board.get_grid().items():
                    if player == Player.black:
                        black_candidates.append(point)

                for black in black_candidates:
                    coord = black.col + black.row * 9
                    point_s = game_state.king_legal_moves(black)
                    # print(point_s)
                    # 先找哪些限制黑棋的点
                    for score_point in point_s:
                        # 黑棋能跳的点得分高
                        if game_state.board.is_jump(black, score_point):
                            # score_point
                            score_point_coord = score_point.col + score_point.row * 9
                            for white_point in candidates:
                                white_point_coord = white_point.col + white_point.row * 9
                                if (distance(white_point, score_point) == 1 or distance(white_point,
                                                                                        score_point) == sqrt(
                                    2)) and game_state.board.check_move1(white_point,
                                                                         score_point) and game_state.board.check_move1(
                                    white_point, score_point):
                                    white_routes[Move(white_point, score_point)] = 100 + random.random() + 1

                                if (
                                        score_point_coord == 0 or score_point_coord == 8 or score_point_coord == 18 or score_point_coord == 26) and white_point_coord == (
                                        score_point_coord + 18):
                                    white_routes[Move(white_point, score_point)] = 100 + random.random() + 2
                                if (
                                        score_point_coord == 18 or score_point_coord == 26 or score_point_coord == 36 or score_point_coord == 44) and white_point_coord == (
                                        score_point_coord - 18):
                                    white_routes[Move(white_point, score_point)] = 100 + random.random() + 3

                        else:

                            # 中间的棋子
                            # if coord == 19:
                            #     if Point(row=(coord - 1) // 9, col=(coord - 1) % 9) not in game_state.board.get_grid():
                            #         white_routes[
                            #             Point(row=(coord - 1) // 9, col=(coord - 1) % 9),] = 70 + random.random()
                            #     if Point(row=(coord - 9) // 9, col=(coord - 9) % 9) not in game_state.board.get_grid():
                            #         white_routes[
                            #             Point(row=(coord - 9) // 9, col=(coord - 9) % 9)] = 70 + random.random()
                            #     if Point(row=(coord + 9) // 9, col=(coord + 9) % 9) not in game_state.board.get_grid():
                            #         white_routes[
                            #             Point(row=(coord + 9) // 9, col=(coord + 9) % 9)] = 70 + random.random()
                            # if coord == 25:
                            #     if Point(row=(coord + 1) // 9, col=(coord + 1) % 9) not in game_state.board.get_grid():
                            #         white_routes[
                            #             Point(row=(coord + 1) // 9, col=(coord + 1) % 9)] = 70 + random.random()
                            #     if Point(row=(coord - 9) // 9, col=(coord - 9) % 9) not in game_state.board.get_grid():
                            #         white_routes[
                            #             Point(row=(coord - 9) // 9, col=(coord - 9) % 9)] = 70 + random.random()
                            #     if Point(row=(coord + 9) // 9, col=(coord + 9) % 9) not in game_state.board.get_grid():
                            #         white_routes[
                            #             Point(row=(coord + 9) // 9, col=(coord + 9) % 9)] = 70 + random.random()

                            # 黑棋不能跳的点，判断距离 先判断 score_point 的距离
                            coord_ = score_point.col + score_point.row * 9
                            if distance(score_point, black) == 2 or distance(score_point, black) == 2 * sqrt(
                                    2) and not is_border(coord):

                                # 判断

                                if score_point not in game_state.board.get_grid():
                                    # . . X
                                    for white_point in candidates:

                                        if (distance(white_point, score_point) == 1 or distance(white_point,
                                                                                                score_point) == sqrt(
                                            2)) and score_point not in game_state.board.get_grid() and game_state.board.check_move1(
                                            white_point, score_point):
                                            white_routes[Move(white_point, score_point)] = 100 + random.random() + 4
                                elif score_point in game_state.board.get_grid() and game_state.board.get_grid()[
                                    score_point] == Player.white:
                                    # 0 . X
                                    for white_point in candidates:

                                        mid_coord = (coord_ + coord) // 2

                                        if (distance(white_point,
                                                     Point(mid_coord // 9, mid_coord % 9)) == 1 or distance(
                                                white_point, Point(mid_coord // 9, mid_coord % 9)) == sqrt(
                                            2)) and score_point != white_point and Point(mid_coord // 9,
                                                                                         mid_coord % 9) not in game_state.board.get_grid() and game_state.board.check_move1(
                                            white_point, Point(mid_coord // 9,
                                                               mid_coord % 9)):
                                            white_routes[Move(white_point, Point(mid_coord // 9,
                                                                                 mid_coord % 9))] = 100 + random.random() + 5
                            elif is_border(coord) and distance(score_point, black) == 2 and is_border(
                                    score_point):  # 两个都是边界
                                if coord == 18 or coord == 26:
                                    for white in candidates:
                                        if distance(white, score_point) == sqrt(
                                                2) and score_point not in game_state.board.get_grid():
                                            white_routes[Move(white, score_point)] = 100 + random.random() + 6
                                if (coord == 0 or coord == 8) and game_state.board.get_grid()[
                                    coord + 36] == Player.white:
                                    for white in candidates:
                                        if distance(white,
                                                    score_point) == 1 and score_point not in game_state.board.get_grid():
                                            white_routes[Move(white, score_point)] = 100 + random.random() + 7

                                if (coord == 36 or coord == 44) and game_state.board.get_grid()[
                                    coord - 36] == Player.white:
                                    for white in candidates:
                                        if distance(white,
                                                    score_point) == 1 and score_point not in game_state.board.get_grid():
                                            white_routes[Move(white, score_point)] = 100 + random.random() + 8

                            elif distance(score_point, black) == 1 or distance(score_point, black) == sqrt(2):
                                # if coord_<coord:
                                pre_point_coord = ((coord_ * 2) - coord)

                                pre_point = Point(pre_point_coord // 9, pre_point_coord % 9)
                                # print(pre_point_coord)
                                # print(pre_point)
                                # print(game_state.board.get_grid())
                                for white_point in candidates:
                                    if legal_pos(pre_point_coord):
                                        if pre_point in game_state.board.get_grid() and \
                                                game_state.board.get_grid()[pre_point] == Player.white and (
                                                distance(white_point, score_point) == 1 or distance(white_point,
                                                                                                    score_point) == sqrt(
                                            2)) and game_state.board.check_move1(white_point,
                                                                                 score_point) and score_point not in game_state.board.get_grid() and pre_point != white_point:

                                            score_point_c = score_point.col + score_point.row * 9
                                            mid_coord = (score_point_c + coord) // 2
                                            mid_coord_point = Point(mid_coord // 9, mid_coord % 9)
                                            if distance(score_point,
                                                        black) == 2 and mid_coord_point in game_state.board.get_grid() and \
                                                    game_state.board.get_grid()[
                                                        mid_coord_point] == Player.white:  # 还需判断另一个黑棋的距离
                                                white_routes[Move(white_point, score_point)] = 100 + random.random() + 9
                                            else:
                                                white_routes[
                                                    Move(white_point, score_point)] = 100 + random.random() + 10
                                        elif score_point not in game_state.board.get_grid() and game_state.board.check_move1(
                                                white_point, pre_point) and distance(white_point,
                                                                                     pre_point) == 1 or distance(
                                            white_point, pre_point) == sqrt(2):
                                            if pre_point not in game_state.board.get_grid() and game_state.board.check_move1(
                                                    white_point, pre_point):
                                                white_routes[Move(white_point, pre_point)] = 100 + random.random() + 11
                                # elif coord_>coord:
                                #     for white_point in candidates:
                                #         pre_point_coord = ((coord_*2)-coord)
                                #         pre_point = Point(pre_point_coord // 9, pre_point_coord % 9)
                                #         if legal_pos(pre_point_coord) and pre_point in game_state.board.get_grid() and game_state.board.get_grid()[
                                #             pre_point] == Player.white and (
                                #                 distance(white_point, score_point) == 1 or distance(white_point,
                                #                                                                     score_point) == sqrt(
                                #                 2)) and game_state.board.check_move1(white_point,
                                #                                                      score_point) and score_point not in game_state.board.get_grid():
                                #             white_routes[Move(white_point, score_point)] = 100 + random.random()
                                #         elif legal_pos(pre_point_coord) and pre_point not in game_state.board.get_grid() and score_point not in game_state.board.get_grid() and game_state.board.check_move1(white_point, pre_point):
                                #             white_routes[Move(white_point, pre_point)] = 50 + random.random()
                for i in game_state.legal_moves():
                    if i not in white_routes:
                        white_routes[i] = random.random()

                if len(white_routes) != len(game_state.legal_moves()):
                    print(white_routes)
                    print(game_state.legal_moves())
                    print('select move')
                    exit(0)
                # for k, v in white_routes.items():
                #     print(f'({k.__str__()},{v} )')

                white_route = max(white_routes, key=white_routes.get)

                return white_route
        else:
            # 黑棋 国王 检查跳
            black_candidates = []
            for point, player in game_state.board.get_grid().items():
                if player == Player.black:
                    black_candidates.append(point)

            scores_moves = dict()

            black_moves = game_state.legal_moves()

            if len(black_moves) == 0:
                return None
            else:
                for black_move in black_moves:
                    if game_state.board.is_jump(black_move.point, black_move.point_):
                        scores_moves[black_move] = 100
                    else:
                        scores_moves[black_move] = random.random()


                if len(scores_moves) != len(game_state.legal_moves()):
                    print_board(game_state.board)
                    print(scores_moves)
                    print(game_state.legal_moves())
                    exit(0)

                scores_move = max(scores_moves, key=scores_moves.get)

                return scores_move

    # def score_moves_old(self, game_state):
    #     if game_state.player == Player.white:
    #         if game_state.play_out <= 32:
    #             candidates = []
    #             for r in range(0, game_state.board.num_rows):
    #                 for c in range(0, game_state.board.num_cols):
    #                     candidate = Point(row=r, col=c)
    #                     # 找到点
    #                     if game_state.is_None_in_grid(
    #                             Move.play_down(candidate)):
    #                         candidates.append(candidate)
    #             black_candidates = []
    #             for point, player in game_state.board.get_grid().items():
    #                 if player == Player.black:
    #                     black_candidates.append(point)
    #
    #             scores_moves = dict()
    #             for black in black_candidates:  # point
    #                 coord = black.col + black.row * 9
    #                 point_s = game_state.king_legal_moves(black)
    #                 for score_move in point_s:
    #                     coord_ = score_move.col + score_move.row * 9
    #                     if game_state.board.is_jump(black, score_move):
    #                         scores_moves[score_move] = 100 + random.random()
    #
    #                     elif (distance(black, score_move) == 2 or distance(black, score_move) == 2 * sqrt(
    #                             2)) and not is_border(coord):
    #                         mid_coord = max(coord, coord_) - abs(coord_ - coord) // 2
    #                         mid_point = Point(row=mid_coord // 9, col=mid_coord % 9)
    #                         if mid_point not in game_state.board.get_grid().keys():
    #                             scores_moves[score_move] = 50 + random.random()
    #                     if distance(black, score_move) == sqrt(2) or distance(black, score_move) == 1:
    #                         if coord_ > coord:
    #                             max_coord = (max(coord, coord_) + abs(coord - coord_))
    #                             if legal_pos(max_coord) and Point(row=max_coord // 9,
    #                                                               col=max_coord % 9) in game_state.board.get_grid() and \
    #                                     game_state.board.get_grid()[
    #                                         Point(row=max_coord // 9, col=max_coord % 9)] == Player.white:
    #                                 scores_moves[score_move] = 50 + random.random()
    #                             # else:
    #                             #     scores_moves[score_move] = -50 + random.random()
    #                         elif coord_ < coord:
    #                             min_coord = (min(coord, coord_) - abs(coord - coord_))
    #                             if legal_pos(min_coord) and Point(row=min_coord // 9,
    #                                                               col=min_coord % 9) in game_state.board.get_grid() and \
    #                                     game_state.board.get_grid()[
    #                                         Point(row=min_coord // 9, col=min_coord % 9)] == Player.white:
    #                                 scores_moves[score_move] = 50 + random.random()
    #                             else:
    #                                 scores_moves[score_move] = -100 + random.random()
    #
    #                     if (coord == 0 or coord == 8) and coord_ == coord + 36:
    #                         scores_moves[score_move] = 100 + random.random()
    #                     if (coord == 44 or coord == 36) and coord_ == coord - 36:
    #                         scores_moves[score_move] = 100 + random.random()
    #                     if (coord == 18 or coord == 26) and coord_ == coord - 18:
    #                         scores_moves[score_move] = 50 + random.random()
    #                 # 中间的棋子
    #                 if coord == 19:
    #                     if Point(row=(coord - 1) // 9, col=(coord - 1) % 9) not in game_state.board.get_grid():
    #                         scores_moves[Point(row=(coord - 1) // 9, col=(coord - 1) % 9)] = 70 + random.random()
    #                     if Point(row=(coord - 9) // 9, col=(coord - 9) % 9) not in game_state.board.get_grid():
    #                         scores_moves[Point(row=(coord - 9) // 9, col=(coord - 9) % 9)] = 70 + random.random()
    #                     if Point(row=(coord + 9) // 9, col=(coord + 9) % 9) not in game_state.board.get_grid():
    #                         scores_moves[Point(row=(coord + 9) // 9, col=(coord + 9) % 9)] = 70 + random.random()
    #                 if coord == 25:
    #                     if Point(row=(coord + 1) // 9, col=(coord + 1) % 9) not in game_state.board.get_grid():
    #                         scores_moves[Point(row=(coord + 1) // 9, col=(coord + 1) % 9)] = 70 + random.random()
    #                     if Point(row=(coord - 9) // 9, col=(coord - 9) % 9) not in game_state.board.get_grid():
    #                         scores_moves[Point(row=(coord - 9) // 9, col=(coord - 9) % 9)] = 70 + random.random()
    #                     if Point(row=(coord + 9) // 9, col=(coord + 9) % 9) not in game_state.board.get_grid():
    #                         scores_moves[Point(row=(coord + 9) // 9, col=(coord + 9) % 9)] = 70 + random.random()
    #             # print(scores_moves)
    #             for i in candidates:
    #                 if i not in scores_moves:
    #                     scores_moves[i] = random.random()
    #             # print(scores_moves)
    #             # scores_move = max(scores_moves, key=scores_moves.get)
    #             scores_moves = {Move.play_down(key): val for key, val in scores_moves.items()}
    #             return scores_moves
    #         else:
    #             candidates = []
    #             for i in range(0, 45):
    #                 candidate = Point(row=i // 9, col=i % 9)
    #                 # 找到点
    #                 if game_state.board.get(
    #                         candidate) == game_state.player and game_state.is_valid_move_down_after_16(
    #                     Move.play_down(candidate)):
    #                     candidates.append(candidate)
    #             candidates = sorted(candidates, key=lambda x: x.row * 9 + x.col)
    #
    #             white_routes = dict()
    #             black_candidates = []
    #             for point, player in game_state.board.get_grid().items():
    #                 if player == Player.black:
    #                     black_candidates.append(point)
    #
    #             for black in black_candidates:
    #                 coord = black.col + black.row * 9
    #                 point_s = game_state.king_legal_moves(black)
    #                 # print(point_s)
    #                 # 先找哪些限制黑棋的点
    #                 for score_point in point_s:
    #                     # 黑棋能跳的点得分高
    #                     if game_state.board.is_jump(black, score_point):
    #                         # score_point
    #                         score_point_coord = score_point.col + score_point.row * 9
    #                         for white_point in candidates:
    #                             white_point_coord = white_point.col + white_point.row * 9
    #                             if distance(white_point, score_point) == 1 or distance(white_point,
    #                                                                                    score_point) == sqrt(2):
    #                                 white_routes[Move(white_point, score_point)] = 100 + random.random()
    #
    #                             if (
    #                                     score_point_coord == 0 or score_point_coord == 8 or score_point_coord == 18 or score_point_coord == 26) and white_point_coord == (
    #                                     score_point_coord + 18):
    #                                 white_routes[Move(white_point, score_point)] = 100 + random.random()
    #                             if (
    #                                     score_point_coord == 18 or score_point_coord == 26 or score_point_coord == 36 or score_point_coord == 44) and white_point_coord == (
    #                                     score_point_coord - 18):
    #                                 white_routes[Move(white_point, score_point)] = 100 + random.random()
    #
    #                     else:
    #
    #                         # 中间的棋子
    #                         # if coord == 19:
    #                         #     if Point(row=(coord - 1) // 9, col=(coord - 1) % 9) not in game_state.board.get_grid():
    #                         #         white_routes[
    #                         #             Point(row=(coord - 1) // 9, col=(coord - 1) % 9),] = 70 + random.random()
    #                         #     if Point(row=(coord - 9) // 9, col=(coord - 9) % 9) not in game_state.board.get_grid():
    #                         #         white_routes[
    #                         #             Point(row=(coord - 9) // 9, col=(coord - 9) % 9)] = 70 + random.random()
    #                         #     if Point(row=(coord + 9) // 9, col=(coord + 9) % 9) not in game_state.board.get_grid():
    #                         #         white_routes[
    #                         #             Point(row=(coord + 9) // 9, col=(coord + 9) % 9)] = 70 + random.random()
    #                         # if coord == 25:
    #                         #     if Point(row=(coord + 1) // 9, col=(coord + 1) % 9) not in game_state.board.get_grid():
    #                         #         white_routes[
    #                         #             Point(row=(coord + 1) // 9, col=(coord + 1) % 9)] = 70 + random.random()
    #                         #     if Point(row=(coord - 9) // 9, col=(coord - 9) % 9) not in game_state.board.get_grid():
    #                         #         white_routes[
    #                         #             Point(row=(coord - 9) // 9, col=(coord - 9) % 9)] = 70 + random.random()
    #                         #     if Point(row=(coord + 9) // 9, col=(coord + 9) % 9) not in game_state.board.get_grid():
    #                         #         white_routes[
    #                         #             Point(row=(coord + 9) // 9, col=(coord + 9) % 9)] = 70 + random.random()
    #
    #                         # 黑棋不能跳的点，判断距离 先判断 score_point 的距离
    #                         coord_ = score_point.col + score_point.row * 9
    #                         if distance(score_point, black) == 2 or distance(score_point, black) == 2 * sqrt(
    #                                 2) and not is_border(coord):
    #
    #                             # 判断
    #
    #                             if score_point not in game_state.board.get_grid():
    #                                 # . . X
    #                                 for white_point in candidates:
    #
    #                                     if distance(white_point, score_point) == 1 or distance(white_point,
    #                                                                                            score_point) == sqrt(
    #                                             2) and score_point not in game_state.board.get_grid():
    #                                         white_routes[Move(white_point, score_point)] = 100 + random.random()
    #                             elif score_point in game_state.board.get_grid() and game_state.board.get_grid()[
    #                                 score_point] == Player.white:
    #                                 # 0 . X
    #                                 for white_point in candidates:
    #
    #                                     mid_coord = (coord_ + coord) // 2
    #
    #                                     if distance(white_point, Point(mid_coord // 9, mid_coord % 9)) == 1 or distance(
    #                                             white_point, Point(mid_coord // 9, mid_coord % 9)) == sqrt(
    #                                             2) and score_point != white_point and Point(mid_coord // 9,
    #                                                                                         mid_coord % 9) not in game_state.board.get_grid():
    #                                         white_routes[Move(white_point, Point(mid_coord // 9,
    #                                                                              mid_coord % 9))] = 100 + random.random()
    #                         elif is_border(coord) and distance(score_point, black) == 2 and is_border(
    #                                 score_point):  # 两个都是边界
    #                             if coord == 18 or coord == 26:
    #                                 for white in candidates:
    #                                     if distance(white, score_point) == sqrt(
    #                                             2) and score_point not in game_state.board.get_grid():
    #                                         white_routes[Move(white, score_point)] = 100 + random.random()
    #                             if (coord == 0 or coord == 8) and game_state.board.get_grid()[
    #                                 coord + 36] == Player.white:
    #                                 for white in candidates:
    #                                     if distance(white,
    #                                                 score_point) == 1 and score_point not in game_state.board.get_grid():
    #                                         white_routes[Move(white, score_point)] = 100 + random.random()
    #
    #                             if (coord == 36 or coord == 44) and game_state.board.get_grid()[
    #                                 coord - 36] == Player.white:
    #                                 for white in candidates:
    #                                     if distance(white,
    #                                                 score_point) == 1 and score_point not in game_state.board.get_grid():
    #                                         white_routes[Move(white, score_point)] = 100 + random.random()
    #
    #                         elif distance(score_point, black) == 1 or distance(score_point, black) == sqrt(2):
    #                             # if coord_<coord:
    #                             pre_point_coord = ((coord_ * 2) - coord)
    #
    #                             pre_point = Point(pre_point_coord // 9, pre_point_coord % 9)
    #                             # print(pre_point_coord)
    #                             # print(pre_point)
    #                             # print(game_state.board.get_grid())
    #                             for white_point in candidates:
    #                                 if legal_pos(pre_point_coord):
    #                                     if pre_point in game_state.board.get_grid() and \
    #                                             game_state.board.get_grid()[pre_point] == Player.white and (
    #                                             distance(white_point, score_point) == 1 or distance(white_point,
    #                                                                                                 score_point) == sqrt(
    #                                             2)) and game_state.board.check_move1(white_point,
    #                                                                                  score_point) and score_point not in game_state.board.get_grid() and pre_point != white_point:
    #
    #                                         score_point_c = score_point.col + score_point.row * 9
    #                                         mid_coord = (score_point_c + coord) // 2
    #                                         mid_coord_point = Point(mid_coord // 9, mid_coord % 9)
    #                                         if distance(score_point, black) == 2 and mid_coord_point in game_state.board.get_grid() and game_state.board.get_grid()[mid_coord_point] == Player.white: # 还需判断另一个黑棋的距离
    #                                             white_routes[Move(white_point, score_point)] = 50 + random.random()
    #                                         else:
    #                                             white_routes[Move(white_point, score_point)] = 100 + random.random()
    #                                     elif score_point not in game_state.board.get_grid() and game_state.board.check_move1(
    #                                             white_point, pre_point) and distance(white_point,
    #                                                                                  pre_point) == 1 or distance(
    #                                             white_point, pre_point) == sqrt(2):
    #                                         if pre_point not in game_state.board.get_grid():
    #                                             white_routes[Move(white_point, pre_point)] = 50 + random.random() + 1
    #                             # elif coord_>coord:
    #                             #     for white_point in candidates:
    #                             #         pre_point_coord = ((coord_*2)-coord)
    #                             #         pre_point = Point(pre_point_coord // 9, pre_point_coord % 9)
    #                             #         if legal_pos(pre_point_coord) and pre_point in game_state.board.get_grid() and game_state.board.get_grid()[
    #                             #             pre_point] == Player.white and (
    #                             #                 distance(white_point, score_point) == 1 or distance(white_point,
    #                             #                                                                     score_point) == sqrt(
    #                             #                 2)) and game_state.board.check_move1(white_point,
    #                             #                                                      score_point) and score_point not in game_state.board.get_grid():
    #                             #             white_routes[Move(white_point, score_point)] = 100 + random.random()
    #                             #         elif legal_pos(pre_point_coord) and pre_point not in game_state.board.get_grid() and score_point not in game_state.board.get_grid() and game_state.board.check_move1(white_point, pre_point):
    #                             #             white_routes[Move(white_point, pre_point)] = 50 + random.random()
    #             for i in game_state.legal_moves():
    #                 if i not in white_routes:
    #                     white_routes[i] = random.random()
    #
    #             # for k, v in white_routes.items():
    #             #     print(f'({k.__str__()},{v} )')
    #
    #             # white_route = max(white_routes, key=white_routes.get)
    #
    #             return white_routes
    #     else:
    #         # 黑棋 国王 检查跳
    #         black_candidates = []
    #         for point, player in game_state.board.get_grid().items():
    #             if player == Player.black:
    #                 black_candidates.append(point)
    #
    #         scores_moves = dict()
    #
    #         black_moves = game_state.legal_moves()
    #
    #         if len(black_moves) == 0:
    #             return None
    #         else:
    #             for black_move in black_moves:
    #                 if game_state.board.is_jump(black_move.point, black_move.point_):
    #                     scores_moves[black_move] = 100
    #                 else:
    #                     scores_moves[black_move] = random.random()
    #                     # pass
    #             # scores_move = max(scores_moves, key=scores_moves.get)
    #
    #             return scores_moves

    def score_moves(self, game_state):
        if game_state.player == Player.white:
            if game_state.play_out <= 32:
                candidates = []
                for r in range(0, game_state.board.num_rows):
                    for c in range(0, game_state.board.num_cols):
                        candidate = Point(row=r, col=c)
                        # 找到点
                        if game_state.is_None_in_grid(
                                Move.play_down(candidate)):
                            candidates.append(candidate)
                black_candidates = []
                for point, player in game_state.board.get_grid().items():
                    if player == Player.black:
                        black_candidates.append(point)

                scores_moves = dict()
                for black in black_candidates:  # point
                    coord = black.col + black.row * 9
                    point_s = game_state.king_legal_moves(black)
                    # modify 7/17
                    for possible_point in candidates:
                        if (distance(black, possible_point) == 2 or distance(black, possible_point) == 2 * sqrt(
                                2)) and not is_border(possible_point):
                            point_s.append(possible_point)
                    # end add possible distance == 2

                    for score_move in point_s:
                        coord_ = score_move.col + score_move.row * 9
                        # 添加目标点的周围的距离为2*sqrt(2)的位置
                        count = 0
                        for possible_point in candidates:
                            if (distance(score_move, possible_point) == 2 or distance(score_move,
                                                                                      possible_point) == 2 * sqrt(
                                2)) and not is_border(
                                possible_point) and possible_point in game_state.board.get_grid().keys() and \
                                    game_state.board.get_grid()[mid_point] == Player.black:
                                count += 1
                        if count == 2:
                            scores_moves[score_move] = scores_moves[score_move] + 100 + random.random()

                        # end
                        if game_state.board.is_jump(black, score_move):
                            mid_coord = max(coord, coord_) - abs(coord_ - coord) // 2
                            mid_point = Point(row=mid_coord // 9, col=mid_coord % 9)
                            if mid_point in game_state.board.get_grid().keys() and game_state.board.get_grid()[
                                mid_point] == Player.white and game_state.board.check_move1(score_move, mid_point):
                                scores_moves[score_move] = scores_moves[
                                                               score_move] + 150 + random.random() if score_move in scores_moves else 150 + random.random()
                            elif mid_point not in game_state.board.get_grid().keys() and game_state.board.check_move1(
                                    score_move, mid_point):
                                scores_moves[score_move] = scores_moves[
                                                               score_move] + 100 + random.random() if score_move in scores_moves else 100 + random.random()



                        elif (distance(black, score_move) == 2 or distance(black, score_move) == 2 * sqrt(
                                2)) and not is_border(coord):
                            mid_coord = max(coord, coord_) - abs(coord_ - coord) // 2
                            mid_point = Point(row=mid_coord // 9, col=mid_coord % 9)
                            if mid_point not in game_state.board.get_grid().keys() and game_state.board.check_move1(
                                    score_move, mid_point) and game_state.board.check_move1(mid_point, black):
                                scores_moves[score_move] = 100 + random.random()

                        if distance(black, score_move) == sqrt(2) or distance(black, score_move) == 1:
                            if coord_ > coord:
                                max_coord = (max(coord, coord_) + abs(coord - coord_))
                                if legal_pos(max_coord) and Point(row=max_coord // 9,
                                                                  col=max_coord % 9) in game_state.board.get_grid() and \
                                        game_state.board.get_grid()[
                                            Point(row=max_coord // 9,
                                                  col=max_coord % 9)] == Player.white and game_state.board.check_move1(
                                    score_move,
                                    Point(row=max_coord // 9, col=max_coord % 9)) and game_state.board.check_move1(
                                    score_move, black):
                                    scores_moves[score_move] = 100 + random.random()
                                else:
                                    scores_moves[score_move] = -100 + random.random()
                            elif coord_ < coord:
                                min_coord = (min(coord, coord_) - abs(coord - coord_))
                                if legal_pos(min_coord) and Point(row=min_coord // 9,
                                                                  col=min_coord % 9) in game_state.board.get_grid() and \
                                        game_state.board.get_grid()[
                                            Point(row=min_coord // 9,
                                                  col=min_coord % 9)] == Player.white and game_state.board.check_move1(
                                    score_move,
                                    Point(row=min_coord // 9, col=min_coord % 9)) and game_state.board.check_move1(
                                    score_move, black):
                                    scores_moves[score_move] = 100 + random.random()
                                else:
                                    scores_moves[score_move] = -100 + random.random()

                        if (coord == 0 or coord == 8) and coord_ == coord + 36:
                            scores_moves[score_move] = 70 + random.random()
                        if (coord == 44 or coord == 36) and coord_ == coord - 36:
                            scores_moves[score_move] = 70 + random.random()
                        if (coord == 18 or coord == 26) and coord_ == coord - 18:
                            scores_moves[score_move] = 70 + random.random()
                    # 中间的棋子
                    if coord == 19:
                        if Point(row=(coord - 1) // 9, col=(coord - 1) % 9) not in game_state.board.get_grid():
                            scores_moves[Point(row=(coord - 1) // 9, col=(coord - 1) % 9)] = 70 + random.random()
                        if Point(row=(coord - 9) // 9, col=(coord - 9) % 9) not in game_state.board.get_grid():
                            scores_moves[Point(row=(coord - 9) // 9, col=(coord - 9) % 9)] = 70 + random.random()
                        if Point(row=(coord + 9) // 9, col=(coord + 9) % 9) not in game_state.board.get_grid():
                            scores_moves[Point(row=(coord + 9) // 9, col=(coord + 9) % 9)] = 70 + random.random()
                    if coord == 25:
                        if Point(row=(coord + 1) // 9, col=(coord + 1) % 9) not in game_state.board.get_grid():
                            scores_moves[Point(row=(coord + 1) // 9, col=(coord + 1) % 9)] = 70 + random.random()
                        if Point(row=(coord - 9) // 9, col=(coord - 9) % 9) not in game_state.board.get_grid():
                            scores_moves[Point(row=(coord - 9) // 9, col=(coord - 9) % 9)] = 70 + random.random()
                        if Point(row=(coord + 9) // 9, col=(coord + 9) % 9) not in game_state.board.get_grid():
                            scores_moves[Point(row=(coord + 9) // 9, col=(coord + 9) % 9)] = 70 + random.random()

                # print(scores_moves)
                for i in candidates:
                    if i not in scores_moves:
                        scores_moves[i] = random.random()
                # print(scores_moves)
                # scores_move = max(scores_moves, key=scores_moves.get)
                # print(scores_moves[scores_move])

                # 添加(2,2)的分数

                for black in black_candidates:
                    if black.col < 2 and Point(2, 2) in scores_moves and scores_moves[Point(2, 2)] > 1:
                        scores_moves[Point(2, 2)] += 100
                    if black.col > 6 and Point(2, 6) in scores_moves and scores_moves[Point(2, 6)] > 1:
                        scores_moves[Point(2, 6)] += 100

                # end

                scores_moves = {Move.play_down(key): val for key, val in scores_moves.items()}

                if len(scores_moves) != len(game_state.legal_moves()):
                    print_board(game_state.board)
                    print(scores_moves)
                    print(game_state.legal_moves())
                    exit(0)

                return scores_moves
            else:
                candidates = []
                for i in range(0, 45):
                    candidate = Point(row=i // 9, col=i % 9)
                    # 找到点
                    if game_state.board.get(
                            candidate) == game_state.player and game_state.is_valid_move_down_after_16(
                        Move.play_down(candidate)):
                        candidates.append(candidate)
                candidates = sorted(candidates, key=lambda x: x.row * 9 + x.col)

                white_routes = dict()
                black_candidates = []
                for point, player in game_state.board.get_grid().items():
                    if player == Player.black:
                        black_candidates.append(point)

                for black in black_candidates:
                    coord = black.col + black.row * 9
                    point_s = game_state.king_legal_moves(black)
                    # print(point_s)
                    # 先找哪些限制黑棋的点
                    for score_point in point_s:
                        # 黑棋能跳的点得分高
                        if game_state.board.is_jump(black, score_point):
                            # score_point
                            score_point_coord = score_point.col + score_point.row * 9
                            for white_point in candidates:
                                white_point_coord = white_point.col + white_point.row * 9
                                if (distance(white_point, score_point) == 1 or distance(white_point,
                                                                                        score_point) == sqrt(
                                    2)) and game_state.board.check_move1(white_point, score_point):
                                    white_routes[Move(white_point, score_point)] = 100 + random.random()

                                if (
                                        score_point_coord == 0 or score_point_coord == 8 or score_point_coord == 18 or score_point_coord == 26) and white_point_coord == (
                                        score_point_coord + 18):
                                    white_routes[Move(white_point, score_point)] = 100 + random.random()
                                if (
                                        score_point_coord == 18 or score_point_coord == 26 or score_point_coord == 36 or score_point_coord == 44) and white_point_coord == (
                                        score_point_coord - 18):
                                    white_routes[Move(white_point, score_point)] = 100 + random.random()

                        else:

                            # 中间的棋子
                            # if coord == 19:
                            #     if Point(row=(coord - 1) // 9, col=(coord - 1) % 9) not in game_state.board.get_grid():
                            #         white_routes[
                            #             Point(row=(coord - 1) // 9, col=(coord - 1) % 9),] = 70 + random.random()
                            #     if Point(row=(coord - 9) // 9, col=(coord - 9) % 9) not in game_state.board.get_grid():
                            #         white_routes[
                            #             Point(row=(coord - 9) // 9, col=(coord - 9) % 9)] = 70 + random.random()
                            #     if Point(row=(coord + 9) // 9, col=(coord + 9) % 9) not in game_state.board.get_grid():
                            #         white_routes[
                            #             Point(row=(coord + 9) // 9, col=(coord + 9) % 9)] = 70 + random.random()
                            # if coord == 25:
                            #     if Point(row=(coord + 1) // 9, col=(coord + 1) % 9) not in game_state.board.get_grid():
                            #         white_routes[
                            #             Point(row=(coord + 1) // 9, col=(coord + 1) % 9)] = 70 + random.random()
                            #     if Point(row=(coord - 9) // 9, col=(coord - 9) % 9) not in game_state.board.get_grid():
                            #         white_routes[
                            #             Point(row=(coord - 9) // 9, col=(coord - 9) % 9)] = 70 + random.random()
                            #     if Point(row=(coord + 9) // 9, col=(coord + 9) % 9) not in game_state.board.get_grid():
                            #         white_routes[
                            #             Point(row=(coord + 9) // 9, col=(coord + 9) % 9)] = 70 + random.random()

                            # 黑棋不能跳的点，判断距离 先判断 score_point 的距离
                            coord_ = score_point.col + score_point.row * 9
                            if distance(score_point, black) == 2 or distance(score_point, black) == 2 * sqrt(
                                    2) and not is_border(coord):

                                # 判断

                                if score_point not in game_state.board.get_grid():
                                    # . . X
                                    for white_point in candidates:

                                        if (distance(white_point, score_point) == 1 or distance(white_point,
                                                                                                score_point) == sqrt(
                                            2)) and score_point not in game_state.board.get_grid() and game_state.board.check_move1(
                                            white_point, score_point):
                                            white_routes[Move(white_point, score_point)] = 100 + random.random()
                                elif score_point in game_state.board.get_grid() and game_state.board.get_grid()[
                                    score_point] == Player.white:
                                    # 0 . X
                                    for white_point in candidates:

                                        mid_coord = (coord_ + coord) // 2

                                        if (distance(white_point,
                                                     Point(mid_coord // 9, mid_coord % 9)) == 1 or distance(
                                                white_point, Point(mid_coord // 9, mid_coord % 9)) == sqrt(
                                            2)) and score_point != white_point and Point(mid_coord // 9,
                                                                                         mid_coord % 9) not in game_state.board.get_grid() and game_state.board.check_move1(
                                            white_point, Point(mid_coord // 9,
                                                               mid_coord % 9)):
                                            white_routes[Move(white_point, Point(mid_coord // 9,
                                                                                 mid_coord % 9))] = 100 + random.random()
                            elif is_border(coord) and distance(score_point, black) == 2 and is_border(
                                    score_point):  # 两个都是边界
                                if coord == 18 or coord == 26:
                                    for white in candidates:
                                        if distance(white, score_point) == sqrt(
                                                2) and score_point not in game_state.board.get_grid():
                                            white_routes[Move(white, score_point)] = 100 + random.random()
                                if (coord == 0 or coord == 8) and game_state.board.get_grid()[
                                    coord + 36] == Player.white:
                                    for white in candidates:
                                        if distance(white,
                                                    score_point) == 1 and score_point not in game_state.board.get_grid():
                                            white_routes[Move(white, score_point)] = 100 + random.random()

                                if (coord == 36 or coord == 44) and game_state.board.get_grid()[
                                    coord - 36] == Player.white:
                                    for white in candidates:
                                        if distance(white,
                                                    score_point) == 1 and score_point not in game_state.board.get_grid():
                                            white_routes[Move(white, score_point)] = 100 + random.random()

                            elif distance(score_point, black) == 1 or distance(score_point, black) == sqrt(2):
                                # if coord_<coord:
                                pre_point_coord = ((coord_ * 2) - coord)

                                pre_point = Point(pre_point_coord // 9, pre_point_coord % 9)
                                # print(pre_point_coord)
                                # print(pre_point)
                                # print(game_state.board.get_grid())
                                for white_point in candidates:
                                    if legal_pos(pre_point_coord):
                                        if pre_point in game_state.board.get_grid() and \
                                                game_state.board.get_grid()[pre_point] == Player.white and (
                                                distance(white_point, score_point) == 1 or distance(white_point,
                                                                                                    score_point) == sqrt(
                                            2)) and game_state.board.check_move1(white_point,
                                                                                 score_point) and score_point not in game_state.board.get_grid() and pre_point != white_point:

                                            score_point_c = score_point.col + score_point.row * 9
                                            mid_coord = (score_point_c + coord) // 2
                                            mid_coord_point = Point(mid_coord // 9, mid_coord % 9)
                                            if distance(score_point,
                                                        black) == 2 and mid_coord_point in game_state.board.get_grid() and \
                                                    game_state.board.get_grid()[
                                                        mid_coord_point] == Player.white:  # 还需判断另一个黑棋的距离
                                                white_routes[Move(white_point, score_point)] = 100 + random.random()
                                            else:
                                                white_routes[Move(white_point, score_point)] = 100 + random.random()
                                        elif score_point not in game_state.board.get_grid() and game_state.board.check_move1(
                                                white_point, pre_point) and distance(white_point,
                                                                                     pre_point) == 1 or distance(
                                            white_point, pre_point) == sqrt(2):
                                            if pre_point not in game_state.board.get_grid() and game_state.board.check_move1(
                                                    white_point, pre_point):
                                                white_routes[Move(white_point, pre_point)] = 100 + random.random()
                                # elif coord_>coord:
                                #     for white_point in candidates:
                                #         pre_point_coord = ((coord_*2)-coord)
                                #         pre_point = Point(pre_point_coord // 9, pre_point_coord % 9)
                                #         if legal_pos(pre_point_coord) and pre_point in game_state.board.get_grid() and game_state.board.get_grid()[
                                #             pre_point] == Player.white and (
                                #                 distance(white_point, score_point) == 1 or distance(white_point,
                                #                                                                     score_point) == sqrt(
                                #                 2)) and game_state.board.check_move1(white_point,
                                #                                                      score_point) and score_point not in game_state.board.get_grid():
                                #             white_routes[Move(white_point, score_point)] = 100 + random.random()
                                #         elif legal_pos(pre_point_coord) and pre_point not in game_state.board.get_grid() and score_point not in game_state.board.get_grid() and game_state.board.check_move1(white_point, pre_point):
                                #             white_routes[Move(white_point, pre_point)] = 50 + random.random()
                for i in game_state.legal_moves():
                    if i not in white_routes:
                        white_routes[i] = random.random()

                if len(white_routes) != len(game_state.legal_moves()):
                    print_board(game_state.board)
                    print(white_routes)
                    print(game_state.legal_moves())
                    exit(0)
                # 增加棋形判断
                #    O  .  X
                #    O  O  O
                #    O  O  O
                # for black in black_candidates:
                #     black_coord = black.col + black.row * 9
                #     if black_coord  == 2:
                #         if coord_point(black_coord + 1) not in game_state.board.get_grid().keys() and is_white_chess(game_state,black_coord+2) and is_white_chess(game_state,black_coord+9) and is_white_chess(game_state,black_coord+10) and is_white_chess(game_state,black_coord+11) and is_white_chess(game_state,black_coord+18) and is_white_chess(game_state,black_coord+19) and is_white_chess(game_state,black_coord+20):
                #             if
                #     if black_coord == 42:
                #     if black_coord == 38:

                # for k, v in white_routes.items():
                #     print(f'({k.__str__()},{v} )')

                # white_route = max(white_routes, key=white_routes.get)

                return white_routes
        else:
            # 黑棋 国王 检查跳
            black_candidates = []
            for point, player in game_state.board.get_grid().items():
                if player == Player.black:
                    black_candidates.append(point)

            scores_moves = dict()

            black_moves = game_state.legal_moves()

            if len(black_moves) == 0:
                # print_board(game_state.board)
                # print('黑棋为空')
                return scores_moves
            else:
                for black_move in black_moves:
                    if game_state.board.is_jump(black_move.point, black_move.point_):
                        scores_moves[black_move] = 100
                    else:
                        scores_moves[black_move] = random.random()

                if len(scores_moves) != len(game_state.legal_moves()):
                    print_board(game_state.board)
                    print(scores_moves)
                    print(game_state.legal_moves())
                    exit(0)

                # scores_move = max(scores_moves, key=scores_moves.get)

                return scores_moves
