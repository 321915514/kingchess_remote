import copy

from agent.base import Agent
from agent.expert_agent import Expert_agent
from fundamental.board import GameState
from fundamental.coordinate import Move


class Alpha_beta(Agent):
    def __init__(self):
        super().__init__()

    def evaluation_state(self, game):
        expert = Expert_agent()
        value = 0
        for move,score in expert.score_moves(game).items():
            if score>10:
                value+=score
        return value



    def minimax(self, game:GameState, alpha, beta, depth, is_max_state):
        game_copy = copy.deepcopy(game)
        if depth == 0 or game_copy.game_over()[0]:
            return self.evaluation_state(game_copy)

        if is_max_state:
            value = -9999
            for move in game_copy.legal_moves():
                game_copy = game_copy.apply_move(move)
                value = max(
                    value,
                    self.minimax(game_copy, alpha, beta, depth - 1, False)
                )
                alpha = max(value, alpha)
                if alpha >= beta:
                    break
            return value
        else:
            value = 9999
            for move in game_copy.legal_moves():
                game_copy = game_copy.apply_move(move)
                value = min(
                    value,
                    self.minimax(game, alpha, beta, depth - 1, True)
                )
                beta = min(value, beta)
                if alpha >= beta:
                    break
            return value






    def select_move(self, game_state):
        game_copy = copy.deepcopy(game_state)
        is_max_state = True if game_copy.player == 1 else False
        best_value = is_max_state and -9999 or 9999

        best_value = Move(-1, -1)

        expert = Expert_agent()
        top_moves = []
        for move,score in expert.score_moves(game_copy).items():
            if score>10:
                top_moves.append(move)


        for move in top_moves:
            game_copy = game_copy.apply_move(move)
            value = self.minimax(game_copy,
                                 -10e5,
                                 10e5,
                                 depth - 1,
                                 not is_max_state)

            if ((is_max_state and value > best_value)
                    or (not is_max_state and value < best_value)):
                best_value = value
                best_move = move
                # print(best_move, best_value)

        if best_move[0] == -1 and best_move[1] == -1:
            return top_moves[0]

        return best_move, best_value