import random

# 设置固定的随机种子值
# seed = 1234
# random.seed(seed)
from agent.base import Agent
from fundamental.board import GameState, Move
from fundamental.coordinate import Player, Point
from fundamental.utils import print_board


class Random_agent(Agent):
    def __init__(self):
        super(Random_agent, self).__init__()

    def select_move(self, game_state: GameState):
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
                # random.choice(candidates)
                return Move.play_down(random.choice(candidates))
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
                point = random.choice(candidates)
                # point = candidates[0]
                # print('----------point--------')
                # print(point)
                # # 处理角
                # print('------------point_end-----------')
                points_ = []

                for neighbor in point.border_neighbors()[:point.border_neighbors()[-1]]:
                    if game_state.is_None_in_grid(Move.play_down(neighbor)) and game_state.board.check_move1(point,
                                                                                                             neighbor):
                        points_.append(neighbor)

                # print(points_)
                point_ = random.choice(points_)
                # point_ = points_[0]
                return Move.play_go(point, point_)
        else:
            # 黑棋 国王 检查跳
            black_candidates = []
            for point, player in game_state.board.get_grid().items():
                if player == Player.black:
                    black_candidates.append(point)

            king_index = random.choice([0, 1])
            # black_candidates = sorted(black_candidates, key=lambda x: x.row * 9 + x.col)
            # king_index = 0
            # print_board(game_state.board)
            # print(len(black_candidates))
            # print(black_candidates)
            #
            # print(king_index)

            king = black_candidates[king_index]

            # 需要两个国王都没得地方走了才行

            point_s = game_state.king_legal_moves(king)

            # print(king_index)
            if point_s is None or len(point_s) == 0:
                king_index = 1 if king_index == 0 else 0
                # print(king_index)
                point_s = game_state.king_legal_moves(black_candidates[king_index])
                if point_s is None or len(point_s) == 0:
                    return None
                else:
                    point_ = random.choice(point_s)
                    # point_ = point_s[0]
                    return Move.play_go(black_candidates[king_index], point_)
            else:
                # print(point_s)
                point_ = random.choice(point_s)
                # point_ = point_s[0]
                return Move.play_go(king, point_)

    def get_action(self, game_state: GameState):
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
                # random.choice(candidates)
                return Move.play_down(random.choice(candidates))
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
                point = random.choice(candidates)
                # point = candidates[0]
                # print('----------point--------')
                # print(point)
                # # 处理角
                # print('------------point_end-----------')
                points_ = []

                for neighbor in point.border_neighbors()[:point.border_neighbors()[-1]]:
                    if game_state.is_None_in_grid(Move.play_down(neighbor)) and game_state.board.check_move1(point,
                                                                                                             neighbor):
                        points_.append(neighbor)

                # print(points_)
                point_ = random.choice(points_)
                # point_ = points_[0]
                return Move.play_go(point, point_)
        else:
            # 黑棋 国王 检查跳
            black_candidates = []
            for point, player in game_state.board.get_grid().items():
                if player == Player.black:
                    black_candidates.append(point)

            king_index = random.choice([0, 1])
            # black_candidates = sorted(black_candidates, key=lambda x: x.row * 9 + x.col)
            # king_index = 0
            # print_board(game_state.board)
            # print(len(black_candidates))
            # print(black_candidates)
            #
            # print(king_index)

            king = black_candidates[king_index]

            # 需要两个国王都没得地方走了才行

            point_s = game_state.king_legal_moves(king)

            # print(king_index)
            if point_s is None or len(point_s) == 0:
                king_index = 1 if king_index == 0 else 0
                # print(king_index)
                point_s = game_state.king_legal_moves(black_candidates[king_index])
                if point_s is None or len(point_s) == 0:
                    return None
                else:
                    point_ = random.choice(point_s)
                    # point_ = point_s[0]
                    return Move.play_go(black_candidates[king_index], point_)
            else:
                # print(point_s)
                point_ = random.choice(point_s)
                # point_ = point_s[0]
                return Move.play_go(king, point_)
