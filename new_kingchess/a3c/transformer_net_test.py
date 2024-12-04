from a3c.transfromer import TransformerModel
from a3c.utils import v_wrap
from agent.expert_agent import Expert_agent
from fundamental.board import GameState
from fundamental.coordinate import Player
from agent.alpha_beta import Alpha_beta
from net.encoder import encoder_board


# transformer start
input_size = 153  # 棋盘状态+剩余棋子位置+黑白棋子数量
d_model = 512
nhead = 8
num_encoder_layers = 6
num_decoder_layers = 6
dim_feedforward = 2048
dropout = 0.1
# transformer end


def model_play_black(model_path):
    net_black = TransformerModel(input_size, d_model, nhead, num_encoder_layers, num_decoder_layers,
                                 dim_feedforward, dropout)
    # net_white = TransformerModel(input_size, d_model, nhead, num_encoder_layers, num_decoder_layers,
    #                              dim_feedforward, dropout)
    # net_black = PolicyValueNet(model_path, device='cpu')
    # net_white = Net(N_S, N_A)

    # random = Random_agent()

    expert = Expert_agent()
    alpha = Alpha_beta()

    # net_black.load_state_dict(torch.load(model_path))
    # net_white.load_state_dict(torch.load(model_path))

    game = GameState.new_game(5, 9)
    s = game.encoder_board()
    # s = encoder_board(game)
    # start = time.time()
    while True:

        # print_board(game.board)

        end, winner = game.game_over()

        if end:
            break

        if game.player == Player.black:

            a = net_black.choose_greedy_action(v_wrap(s[None, :]), game.legal_position()[0])


            # print(game.legal_position()[1])
            #
            # print(a)
            #
            # print(game.a_trans_move(a))


            game = game.apply_move(game.a_trans_move(a))

            # game.print_game()
            # s = s_
            #
            # game.decoder_board(s)
            # move = expert.select_move(game)
            # game = game.apply_move(move)


        else:
            game.decoder_board(s)
            # move = random.select_move(game)
            #
            move = alpha.select_move(game)

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

def model_play_white(model_path):
    # net_black = TransformerModel(input_size, d_model, nhead, num_encoder_layers, num_decoder_layers,
    #                              dim_feedforward, dropout)
    net_white = TransformerModel(input_size, d_model, nhead, num_encoder_layers, num_decoder_layers,
                                 dim_feedforward, dropout)
    # net_black = PolicyValueNet(model_path, device='cpu')
    # net_white = Net(N_S, N_A)

    # random = Random_agent()

    expert = Expert_agent()

    # net_black.load_state_dict(torch.load(model_path))
    # net_white.load_state_dict(torch.load(model_path))

    game = GameState.new_game(5, 9)
    s = game.encoder_board()
    # s = encoder_board(game)
    # start = time.time()
    while True:

        # print_board(game.board)

        end, winner = game.game_over()

        if end:
            break

        if game.player == Player.black:
            game.decoder_board(s)

            game = game.apply_move(expert.select_move(game))

        else:
            # game.decoder_board(s)
            # move = random.select_move(game)
            #
            # move = expert.select_move(game)
            #
            # game = game.apply_move(move)

            a = net_white.choose_greedy_action(v_wrap(s[None, :]), game.legal_position()[0])

            s_, r, done = game.step(a)

            # game.print_game()
            s = s_

    return winner







def main():
    # gnet_transformer_8_18.pth play black vs expert white win 1.0 ,play white vs expert black win 0.0
    white = 0
    black = 0
    count_black = 0
    count_white = 0
    for i in range(1000):

        result = model_play_black('./gnet_transformer_8_18.pth')

        if result == Player.white:
            white += 1
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
    #
    # for i in range(100):
    #     result = model_play_white('./gnet_transformer_8_18.pth')
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

    print(f'alpha白棋胜:{white / count_black}')
    print(f"gnet_transformer_8_18黑棋胜：{black / count_black}")
    # alpha白棋胜:0.0
    # gnet_transformer_8_18黑棋胜：1.0

    return white / count_black, black / count_black



if __name__ == '__main__':
    main()