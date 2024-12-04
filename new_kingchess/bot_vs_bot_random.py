from agent.random_agent import Random_agent
from fundamental.board import GameState, Move
from fundamental.coordinate import Player
from fundamental.utils import print_board, print_move, print_move_go
import time
from net.collect import move2str
def main():
    row_size = 5
    col_size = 9
    game = GameState.new_game(row_size, col_size)
    bot1 = Random_agent()
    bot2 = Random_agent()

    while True:
        print_board(game.board)
        print(game)
        end, winner = game.game_over()
        if end:
            break
        if game.player == Player.black:
            for i in game.legal_moves():
                print(i.__str__())
            move = bot1.select_move(game)
            print_move_go(game.player, move)
            # play_out += 1
        else:
            for i in game.legal_moves():
                print(i.__str__())
            move = bot2.select_move(game)
            if move.is_down:
                print_move(game.player, move)
            else:
                print_move_go(game.player, move)
            # play_out += 1
        game = game.apply_move(move)

        # if game.record:
        #     black_eat_chess += 1
        #     print("国王吃了{}个兵。".format(black_eat_chess))
        #     game.record = False

    print(winner)





def main_simple():

    row_size = 5
    col_size = 9
    game = GameState.new_game(row_size, col_size)
    bot1 = Random_agent()
    bot2 = Random_agent()

    while True:
        # print_board(game.board)

        end, winner = game.game_over()
        if end:
            break
        if game.player == Player.black:
            move = bot1.select_move(game)
            # if move is None:
            #     return Player.white
                # print('dachen win')
                # break
            # print_move_go(game.player, move, game.play_out)
        else:
            move = bot2.select_move(game)
            if move.is_down:
                # print_move(game.player, move, game.play_out)
                pass
            else:
                pass
                # print_move_go(game.player, move,game.play_out)
            # play_out += 1
        game = game.apply_move(move)

        # if game.record:
        #     black_eat_chess += 1
        #     # print("国王吃了{}个兵。".format(black_eat_chess))
        #     game.record = False
        # if game.eat_chess() >= 11:
        #     return Player.black
            # print("king win")

    return winner




if __name__ == '__main__':
    start = time.time()
    # count_white = 0
    main()
    end_time = time.time()

    print(end_time-start)
