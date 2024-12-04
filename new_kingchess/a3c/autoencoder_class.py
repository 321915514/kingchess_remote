import os
import pickle

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

from a3c.supervise_model import AutoEncoder
from a3c.test_net import test_model
from fundamental.coordinate import Player

device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")


class GameDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        state, action, score = self.data[idx]
        # 将状态转换为 PyTorch 张量
        state_tensor = torch.from_numpy(state).float()
        # 将动作转换为整数索引（根据具体情况修改）
        actions = np.zeros((1125))
        actions[action] = 1
        action_tensor = torch.tensor(actions, dtype=torch.float32)  # 假设动作是一个整数，表示输出动作的位置
        return state_tensor, action_tensor


def load_data(datapath):
    with open(datapath, 'rb') as f:
        result = pickle.load(f)
        return result





class SimpleNNWithPretrain(nn.Module):
    def __init__(self, autoencoder):
        super(SimpleNNWithPretrain, self).__init__()
        self.encoder = autoencoder.encoder
        self.fc = nn.Linear(16, 1125)

    def forward(self, x):
        with torch.no_grad():
            x = self.encoder(x)
        x = self.fc(x)
        return x

def model_test_white(model_path):
    # net_black = TransformerModel(input_size, d_model, nhead, num_encoder_layers, num_decoder_layers,
    #                              dim_feedforward, dropout)
    # net_white = TransformerModel(input_size, d_model, nhead, num_encoder_layers, num_decoder_layers,
    #                              dim_feedforward, dropout)
    # net_black = Net(N_S, N_A)
    net_white = PolicyValueNet(model_path, device='cpu')

    random = Random_agent()

    expert = Expert_agent()

    # net_black.load_state_dict(torch.load(model_path))
    # net_white.load_state_dict(torch.load(model_path))
    game = GameState.new_game(5, 9)
    s = encoder_board(game)
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
            # game.decoder_board(s)
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



def model_vs_expert():
        white = 0
        black = 0
        count_black = 0
        count_white = 0
        for i in range(100):

            result = model_vs_black('./supervise_model_conv/current.pth')

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

        for i in range(100):
            result = model_vs_white('./supervise_model_conv/current.pth')

            if result == Player.white:
                white += 1
                count_white += 1
            if result == Player.black:
                # black += 1
                count_white += 1
            if result == Player.draw:
                count_white += 1
                continue
            elif result is None:
                count_white += 1
                continue


        return white / count_white, black / count_black





if __name__ == '__main__':

    os.makedirs('autoencoder_class', exist_ok=True)

    data = load_data('./collect_expert_data/game_data_add_score_137.pkl')

    # 创建 Dataset 对象
    dataset = GameDataset(data)
    batch_size = 2048  # 设置批大小
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    autoEncoder = AutoEncoder(137).to(device)

    autoEncoder.load_state_dict(torch.load('./self_train_model/autoencoder.pth'))

    model = SimpleNNWithPretrain(autoEncoder).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    result = {}

    # 训练分类模型
    for epoch in range(1000):
        model.train()
        for X_batch, y_batch in dataloader:
            optimizer.zero_grad()

            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()


        if (epoch+1) % 200 == 0:
            torch.save(model.state_dict(), './supervise_model_137/current_supervise.pth')
            win_white, win_black = model_vs_expert()
            torch.save(model.state_dict(), './supervise_model_137/current_supervise_'+str(epoch+1)+'.pth')
            result[epoch] = (win_white, win_black)
            print(
                f'Epoch [{epoch + 1}/{1000}], Loss: {loss.item():.4f},win_white:{win_white},win_black:{win_black}')



        print(f'Classifier Epoch [{epoch+1}/{500}], Loss: {loss.item():.4f}')

    torch.save(model.state_dict(), './autoencoder_class/autoencoder_class.pth')