import copy
import math

from agent.mcts import MCTSAgent
from agent.random_agent import Random_agent
from fundamental.board_simple import GameState
from fundamental.coordinate import Player
from fundamental.utils import print_board, print_move, print_move_go
import time


def main():
    row_size = 5
    col_size = 9
    game = GameState.new_game(row_size, col_size)
    bot1 = MCTSAgent(1200, math.sqrt(2))
    bot2 = MCTSAgent(1200, math.sqrt(2))

    # random = Random_agent()

    while True:
        end, winner = game.game_over()
        if end:
            break
        print_board(game.board)
        if game.player == Player.black:

            # print('get black move start')
            # print(game.sum_chess)
            # print(game.left_chess)
            move = bot1.select_move(game)
            # print(game.sum_chess)
            # print(game.left_chess)
            # print('gat black move end')

            # if move is None:
            #     print_board(game.board)
            #     print('dachen win')
            #     break
            print_move_go(game.player, move)
        else:
            # print('get white move start')
            # print(game.sum_chess)
            # print(game.left_chess)

            move = bot2.select_move(game)
            # print(game.sum_chess)
            # print(game.left_chess)
            # print('gat white move end')

            if move.is_down:
                print_move(game.player, move)
            else:
                print_move_go(game.player, move)
        game = game.apply_move(move)
        # if game.eat_chess() >= 11:
        #     print_board(game.board)
        #     print("king win")
        #     break
    print(winner)


if __name__ == '__main__':
    start = time.time()
    # for _ in range(100):
    main()
    end_time = time.time()

    print(end_time - start)
