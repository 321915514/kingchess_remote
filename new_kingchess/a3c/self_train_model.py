import os
import pickle

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

from a3c.supervise_model import AutoEncoder
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




if __name__ == '__main__':

    os.makedirs('self_train_model', exist_ok=True)

    data = load_data('./collect_expert_data/game_data_add_score_137.pkl')

    # 创建 Dataset 对象
    dataset = GameDataset(data)
    batch_size = 2048  # 设置批大小
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 初始化自监督学习模型、损失函数和优化器
    autoencoder = AutoEncoder(137).to(device)
    autoencoder.load_state_dict(torch.load('./self_train_model/autoencoder.pth'))
    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)

    # 预训练自监督学习模型
    for epoch in range(1000):
        autoencoder.train()
        for X_batch, _ in dataloader:
            optimizer.zero_grad()
            X_batch = X_batch.to(device)
            outputs = autoencoder(X_batch)
            loss = criterion(outputs, X_batch)
            loss.backward()
            optimizer.step()

        print(f'AutoEncoder Epoch [{epoch+1}/{1000}], Loss: {loss.item():.4f}')
    torch.save(autoencoder.state_dict(), './self_train_model/autoencoder.pth')