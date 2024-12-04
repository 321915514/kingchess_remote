from a3c.transfromer import TransformerModel
from a3c.utils import v_wrap
from agent.alpha_beta import Alpha_beta
from agent.expert_agent import Expert_agent
from agent.random_agent import Random_agent
from fundamental.board import GameState
from fundamental.coordinate import Player
from fundamental.utils import print_board, print_move

# transformer start
input_size = 153  # 棋盘状态+剩余棋子位置+黑白棋子数量
d_model = 512
nhead = 8
num_encoder_layers = 6
num_decoder_layers = 6
dim_feedforward = 2048
dropout = 0.1
# transformer end


def net_black_vs_expert(model_path):
    net_black = TransformerModel(input_size, d_model, nhead, num_encoder_layers, num_decoder_layers,
                                 dim_feedforward, dropout)
    random = Random_agent()
    expert = Expert_agent()
    alpha = Alpha_beta()

    game = GameState.new_game(5, 9)
    s = game.encoder_board()

    while True:
        # print_board(game.board)
        end, winner = game.game_over()
        if end:
            break
        if game.player == Player.black:
            a = net_black.choose_greedy_action(v_wrap(s[None, :]), game.legal_position()[0])
            # print_move(game.player, game.a_trans_move(a))
            game = game.apply_move(game.a_trans_move(a))
        else:
            # game.decoder_board(s)
            move = expert.select_move(game)
            # print_move(game.player, move)
            game = game.apply_move(move)
    return winner


if __name__ == '__main__':
    net = 0
    other = 0
    for _ in range(100):
        # winner = net_black_vs_alpha('./gnet_transformer_8_18.pth')    # net win: 0.0 # alpha win: 0.88
        winner = net_black_vs_expert('./gnet_transformer_8_18.pth')  # net win: 0.05 expert win: 0.86
        if winner == Player.black:
            net+=1
        if winner == Player.white:
            other+=1

    print('net win:', net/100)
    print('other win:', other/100)
