from agent.expert_agent import Expert_agent
from mcts_pure import MCTSPlayer
from fundamental.board import GameState
from fundamental.coordinate import Player
from agent.alpha_beta import Alpha_beta
from fundamental.utils import print_board, print_move, print_move_go
import time


def main():
    row_size = 5
    col_size = 9
    game = GameState.new_game(row_size, col_size)
    bot1 = MCTSPlayer(c_puct=5, n_playout=100)
    # bot2 = MCTSPlayer(c_puct=5, n_playout=100)
    # expert = Expert_agent()
    alpha = Alpha_beta()
    while True:
        end, winner = game.game_over()
        if end:
            break
        # print_board(game.board)
        # game.print_game()
        # print(game.play_out)
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
        else:
            # print('get white move start')
            # print(game.sum_chess)
            # print(game.left_chess)

            move = alpha.select_move(game)
            # print(game.sum_chess)
            # print(game.left_chess)
            # print('gat white move end')

            # if move.is_down:
            #     print_move(game.player, move)
            # else:
            #     print_move_go(game.player, move)

        game = game.apply_move(move)
        # if game.eat_chess() >= 11:
        #     print_board(game.board)
        #     print("king win")
        #     break

    return winner

if __name__ == '__main__':
    start = time.time()
    # mcts black  vs expert
    mcts = 0
    expert = 0
    for _ in range(20):
        if main() == Player.black:
            mcts+=1
        if main() == Player.white:
            expert+=1
    print("{},{}".format(mcts/100,expert/100))
    end_time = time.time()

    print(end_time - start)

    # mcts add expert black 0.56, expert white 0.45 100
    #  # mcts add expert black 0.02, alpha 0.11 50

