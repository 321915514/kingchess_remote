from agent.expert_agent import Expert_agent
from agent.random_agent import Random_agent
from fundamental.board import GameState, Move
from fundamental.coordinate import Player
from fundamental.utils import print_board, print_move, print_move_go
import time
from net.collect import move2str
from agent.alpha_beta import Alpha_beta


def main():
    row_size = 5
    col_size = 9
    game = GameState.new_game(row_size, col_size)
    bot1 = Alpha_beta()
    bot2 = Expert_agent()

    while True:
        print_board(game.board)
        end, winner = game.game_over()
        if end:
            break
        if game.player == Player.black:
            # for i in game.legal_moves():
            #     print(i.__str__())
            move = bot1.select_move(game)
            print_move_go(game.player, move)
            # play_out += 1
        else:
            # for i in game.legal_moves():
            #     print(i.__str__())
            move = bot1.select_move(game)
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
    bot1 = Alpha_beta()
    bot2 = Expert_agent()

    while True:
        end, winner = game.game_over()
        if end:
            break
        if game.player == Player.black:
            move = bot2.select_move(game)
        else:
            move = bot1.select_move(game)
        game = game.apply_move(move)

    return winner

if __name__ == '__main__':
    start = time.time()

    main()
    alpha_black = 0
    alpha_white = 0
    expert_black = 0
    expert_white = 0
    # for i in range(1000):
    #     result = main_simple()
    #     if result == Player.black:
    #         alpha_black += 1
    #     if result == Player.white:
    #         expert_white += 1
    # print(f"alpha black vs expert: alpha black win:{alpha_black},expert white:{expert_white}") #
    #alpha black vs expert: alpha black win:628,expert white:189  1000//

    # for i in range(100):
    #     result = main_simple()
    #     if result == Player.black:
    #         expert_black += 1
    #     if result == Player.white:
    #         alpha_white += 1
    # depth = 4
    print(f"alpha white vs expert black: alpha white win:{alpha_white},expert black:{expert_black}") #
    #alpha white vs expert black: alpha white win:52,expert black:33
    #5406.210613012314
    end_time= time.time()

    print(end_time-start)
