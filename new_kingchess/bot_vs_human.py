from agent.random_agent import Random_agent
from fundamental.board import GameState, Move
from fundamental.coordinate import Player, Point
from fundamental.utils import print_board, print_move, print_move_go, point_from_coords
import time


def main():
    row_size = 5
    col_size = 9
    game = GameState.new_game(row_size, col_size)
    bot1 = Random_agent()
    # bot2 = Random_agent()

    while True:
        end, winner = game.game_over()
        if end:
            break
        print_board(game.board)
        if game.player == Player.black:
            move = bot1.select_move(game)
            # if move is None:
            #     print('dachen win')
            #     break
            print_move_go(game.player, move)
        else:
            # move = bot2.select_move(game)
            human_move = input('-- ')
            if human_move.strip() == '-1':
                print('game over')
                break
            point = point_from_coords(human_move.strip())
            # print(point)
            if isinstance(point, Point):
                move = Move.play_down(point)
            else:
                move = Move.play_go(point[0], point[1])
        game = game.apply_move(move)
        print("国王吃了{}个兵。".format(game.eat_chess()))
        print('白棋可以落%d个子' % (0 if game.play_out > 32 else 16 - game.play_out // 2))
        # if game.eat_chess() >= 11:
        #     print("king win")
        #     break

    print("国王胜利" if winner == Player.black else "大臣胜利")


if __name__ == '__main__':
    start = time.time()
    # for _ in range(100):
    main()
    end_time = time.time()

    print(end_time - start)
