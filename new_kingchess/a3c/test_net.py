import time

import torch

# from a3c.supervise_model import Net
# conv
from net.policy_value_net_pytorch import PolicyValueNet, Net
from net.mcts_alphazreo import MCTSPlayer
from net.encoder import encoder_board
# from a3c.discrete_A3C import Net
# from a3c.supervise_model import Net
# from net.policy_value_net_pytorch import PolicyValueNet
from a3c.transfromer import TransformerModel
from a3c.utils import v_wrap
from fundamental.board import GameState
from fundamental.coordinate import Player
from fundamental.utils import print_board, print_move_go
from agent.random_agent import Random_agent
from agent.expert_agent import Expert_agent

N_S = 137  # 目前 153
N_A = 1125
# transformer start
input_size = N_S  # 棋盘状态+剩余棋子位置+黑白棋子数量
d_model = 256
nhead = 16
num_encoder_layers = 6
num_decoder_layers = 6
dim_feedforward = 512
dropout = 0.1


# transformer end


def model_test_black(model_path):
    # net_black = TransformerModel(input_size, d_model, nhead, num_encoder_layers, num_decoder_layers,
    #                              dim_feedforward, dropout)
    # net_white = TransformerModel(input_size, d_model, nhead, num_encoder_layers, num_decoder_layers,
    #                              dim_feedforward, dropout)
    net_black = PolicyValueNet(model_path)

    mcts = MCTSPlayer(net_black.policy_value_fn, c_puct=5, n_playout=10, is_selfplay=False)

    # net_black = Net()

    # random = Random_agent()

    expert = Expert_agent()

    # net_black.load_state_dict(torch.load(model_path))
    # net_white.load_state_dict(torch.load(model_path))

    game = GameState.new_game(5, 9)
    s = game.encoder_board_137()
    # s = encoder_board_(game)
    # start = time.time()
    while True:

        # print_board(game.board)

        end, winner = game.game_over()

        if end:
            break

        if game.player == Player.black:
            move = mcts.select_move(game)

            # print(game.legal_position()[1])
            #
            # print(a)
            #
            # print(game.a_trans_move(a))

            game = game.apply_move(move)

            # game.print_game()
            # s = s_
            #
            # game.decoder_board(s)
            # move = expert.select_move(game)
            # game = game.apply_move(move)


        else:
            # game.decoder_board(s)
            # move = random.select_move(game)
            #
            move = expert.select_move(game)

            game = game.apply_move(move)

            # a = net_white.choose_greedy_action(v_wrap(s[None, :]), game.legal_position()[0])
            #
            # s_, r, done = game.step(a)
            #
            # # game.print_game()
            #
            # s = s_
        # end = time.time()
        #
        # if end - start > 60:
        #     return None

    return winner


def model_test_white(model_path):
    # net_black = TransformerModel(input_size, d_model, nhead, num_encoder_layers, num_decoder_layers,
    #                              dim_feedforward, dropout)
    # net_white = TransformerModel(input_size, d_model, nhead, num_encoder_layers, num_decoder_layers,
    #                              dim_feedforward, dropout)
    net_white = Net(N_S, N_A)
    # net_white = PolicyValueNet(model_path, device='cpu')

    random = Random_agent()

    expert = Expert_agent()

    # net_black.load_state_dict(torch.load(model_path))
    # net_white.load_state_dict(torch.load(model_path))
    game = GameState.new_game(5, 9)
    # s = encoder_board(game)
    # start = time.time()
    while True:

        # print_board(game.board)

        end, winner = game.game_over()

        if end:
            break

        if game.player == Player.black:
            # a = net_black.choose_greedy_action(v_wrap(s[None, :]), game.legal_position()[0])
            #
            # s_, r, done = game.step(a)
            #
            # # game.print_game()
            # s = s_

            # game.decoder_board(s)
            move = expert.select_move(game)
            game = game.apply_move(move)

        else:
            s = game.encoder_board_137()
            # move = random.select_move(game)
            # #
            # # move = expert.select_move(game)
            #
            # game = game.apply_move(move)

            a = net_white.choose_greedy_action(v_wrap(s[None, :]), game.legal_position()[0])

            game = game.apply_move(game.a_trans_move(a))

            # game.print_game()

            # s = s_
        # end = time.time()
        #
        # if end - start > 60:
        #     return None

    return winner


# def model_test(model_path):
#     # net_black = TransformerModel(input_size, d_model, nhead, num_encoder_layers, num_decoder_layers,
#     #                              dim_feedforward, dropout)
#     # net_white = TransformerModel(input_size, d_model, nhead, num_encoder_layers, num_decoder_layers,
#     #                              dim_feedforward, dropout)
#     net_black = Net(N_S, N_A)
#     net_white = Net(N_S, N_A)
#
#     random = Random_agent()
#
#     expert = Expert_agent()
#
#     net_black.load_state_dict(torch.load(model_path))
#     net_white.load_state_dict(torch.load(model_path))
#     game = GameState.new_game(5, 9)
#     s = game.reset()
#     # start = time.time()
#     while True:
#
#         # print_board(game.board)
#
#         end, winner = game.game_over()
#
#         if end:
#             break
#
#         if game.player == Player.black:
#             a = net_black.choose_greedy_action(v_wrap(s[None, :]), game.legal_position()[0])
#
#             s_, r, done = game.step(a)
#
#             # game.print_game()
#             s = s_
#
#             # game.decoder_board(s)
#             # move = expert.select_move(game)
#             # game = game.apply_move(move)
#
#
#         else:
#             # game.decoder_board(s)
#             # move = expert.select_move(game)
#             # #
#             # # move = expert.select_move(game)
#             #
#             # # moves = expert.score_moves(game)
#             # # print(moves)
#             #
#             # game = game.apply_move(move)
#
#             a = net_white.choose_greedy_action(v_wrap(s[None, :]), game.legal_position()[0])
#
#             s_, r, done = game.step(a)
#
#             # game.print_game()
#
#             s = s_
#         # end = time.time()
#
#         # if end - start > 60:
#         #     return None
#
#     return winner


def test_model():
    white = 0
    black = 0
    count_black = 0
    count_white = 0
    for i in range(100):

        result = model_test_black('H:/current_best_black.pth')

        if result == Player.white:
            # white += 1
            count_black += 1
        if result == Player.black:
            black += 1
            count_black += 1
        if result == Player.draw:
            count_black += 1
            continue
        elif result is None:
            count_black += 1
            continue

    # for i in range(100):
    #     result = model_test_white('./alpha_data_model/current.pth')
    #
    #     if result == Player.white:
    #         white += 1
    #         count_white += 1
    #     if result == Player.black:
    #         # black += 1
    #         count_white += 1
    #     if result == Player.draw:
    #         count_white += 1
    #         continue
    #     elif result is None:
    #         count_white += 1
    #         continue
    #
    # print(f'白棋胜:{white / count_white}')
    print(f"黑棋胜：{black / count_black}")


if __name__ == '__main__':
    test_model()
    # model_test('./current.pth')
    # test_model()
    # white = 0
    # black = 0
    # count = 0
    # for i in range(100):
    #
    #     result = model_test('./gnet_7_28_reward_model_1000.pth')
    #
    #     if result == Player.white:
    #         white += 1
    #         count += 1
    #     if result == Player.black:
    #         black += 1
    #         count += 1
    #     if result == Player.draw:
    #         count += 1
    #         continue
    #     elif result is None:
    #         count += 1
    #         continue
    #
    # print(f'白棋胜:{white / count}')
    # print(f"黑棋胜：{black / count}")
    #
    # result = model_test('./gnet_7_21_reward_model_200.pth')
    # print(result)
