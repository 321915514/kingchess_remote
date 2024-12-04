import numpy as np
import torch

from fundamental.coordinate import Player
from fundamental.utils import print_board, print_move_go, print_move
from reward_model import RewardModelWithEmbedding, load_data
from fundamental.board import GameState
from agent.random_agent import Random_agent
from agent.expert_agent import Expert_agent

def predict_reward(game,move):

    # mean_reward = 81.63694672371473
    # std_reward = 102.00450246332814 # 53.591406517009425
    embedding_dim = 200  # 嵌入维度
    state_size = 137
    action_size = 1125  # 动作空间大小
    hidden_size = 128
    reward_model = RewardModelWithEmbedding(state_size, action_size, hidden_size, embedding_dim)

    reward_model.load_state_dict(torch.load('./reward_model/reward_model.pth'))
    state = game.encoder_board_137()
    expert = Expert_agent()
    moves = expert.score_moves(game)
    print(move)
    print(moves)
    action = game.move_2_action(move)
    result = reward_model(torch.unsqueeze(torch.tensor(state, dtype=torch.float32),dim=0), torch.unsqueeze(torch.tensor(action, dtype=torch.long),dim=0))
    # 逆标准化
    # original_reward = result * std_reward + mean_reward
    return result



if __name__ == '__main__':

    # data = load_data('./game_data_add_score.pkl')
    #
    # #计算奖励的均值和标准差
    # rewards = [item[2] for item in data]

    random = Random_agent()
    expert = Expert_agent()
    game = GameState.new_game(5,9)

    while True:
        print_board(game.board)
        end, winner = game.game_over()
        if end:
            break
        if game.player == Player.black:
            move = expert.select_move(game)
            print_move_go(game.player, move)
        else:

            move = expert.select_move(game)

            if move.is_down:
                print_move(game.player, move)
            else:
                print_move_go(game.player, move)

        ori, res = predict_reward(game, move)
        print(ori)
        print(res)
        game = game.apply_move(move)

