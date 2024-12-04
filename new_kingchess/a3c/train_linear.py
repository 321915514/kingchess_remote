import os.path

import numpy as np
from torch import nn, optim

from a3c.discrete_A3C import N_S, N_A
from a3c.supervise_model import Net
from a3c.transfromer import TransformerModel
import pickle

import torch
from torch.utils.data import Dataset, DataLoader
import pickle
from test_net import test_model

device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")


class GameDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        state, action, score = self.data[idx]
        # 将状态转换为 PyTorch 张量
        # state = np.repeat(state[:, np.newaxis], 137, axis=1)
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






if __name__ == '__main__':

    os.makedirs('supervise_model_137', exist_ok=True)

    # transformer start
    input_size = 137  # 棋盘状态+剩余棋子位置+黑白棋子数量
    d_model = 128
    nhead = 4
    num_encoder_layers = 3
    num_decoder_layers = 3
    dim_feedforward = 512
    dropout = 0.1
    # transformer end

    data = load_data('collect_alpha_data/game_data_add_score_137.pkl')

    # 创建 Dataset 对象
    dataset = GameDataset(data)
    batch_size = 32  # 设置批大小
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = Net(137, N_A).to(device)

    # model = TransformerModel(input_size, d_model, nhead, num_encoder_layers, num_decoder_layers,
    #                          dim_feedforward, dropout).to(device)

    model.load_state_dict(torch.load('./supervise_model_137/current_supervise_150.pth'))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.00001)

    result = {}

    # 训练循环
    num_epochs = 300
    for epoch in range(num_epochs):
        for states, actions in dataloader:
            # 将批数据展平成一维向量
            states = states.to(device)

            # 前向传播

            policy = model(states)
            actions = actions.to(device)
            loss = criterion(policy, actions)
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
        if (epoch+1) % 50 == 0:
            # torch.save(model.state_dict(), './supervise_model_137/current_supervise.pth')
            win_white, win_black = test_model()
            torch.save(model.state_dict(), './supervise_model_137/current_supervise_'+str(epoch+1)+'.pth')
            result[epoch] = (win_white, win_black)
            print(
                f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f},win_white:{win_white},win_black:{win_black}')

    torch.save(model.state_dict(), './supervise_model_137/current_supervise_end.pth')
    # {49: (0.0, 0.08), 99: (0.0, 0.07), 149: (0.0, 0.06), 199: (0.0, 0.14), 249: (0.0, 0.08), 299: (0.0, 0.1)}
    print(result)

    print("训练完成")
