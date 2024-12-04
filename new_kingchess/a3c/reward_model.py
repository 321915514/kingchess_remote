import os
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# 检查是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_data(datapath):
    with open(datapath, 'rb') as f:
        result = pickle.load(f)
        return result


# 示例数据类
class ExpertDataset(Dataset):
    def __init__(self, data, mean_reward, std_reward):
        self.data = data
        self.mean_reward = mean_reward
        self.std_reward = std_reward

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        state, action, reward = self.data[idx]
        state_tensor = torch.tensor(state, dtype=torch.float32)
        action_tensor = torch.tensor(action, dtype=torch.long)
        reward_tensor = torch.tensor((reward - self.mean_reward) / self.std_reward, dtype=torch.float32)
        reward_tensor = torch.sigmoid(reward_tensor)
        return state_tensor, action_tensor, reward_tensor


class RewardModelWithEmbedding(nn.Module):
    def __init__(self, state_size, action_size, hidden_size, embedding_dim):
        super(RewardModelWithEmbedding, self).__init__()
        self.action_embedding = nn.Embedding(action_size, embedding_dim)
        self.fc1 = nn.Linear(state_size + embedding_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, state, action):
        action_embedded = self.action_embedding(action)
        x = torch.cat((state, action_embedded), dim=1)
        x = torch.relu(self.fc1(x))
        reward = self.fc2(x)
        reward = torch.sigmoid(reward)
        return reward


if __name__ == '__main__':

    os.makedirs('reward_model', exist_ok=True)

    # white_data = load_data('./game_data_add_score_white.pkl')
    data = load_data('./collect_expert_data/game_data_add_score_137.pkl')

    # data = []
    # data.extend(white_data)
    # data.extend(black_data)


    # 计算奖励的均值和标准差
    rewards = [item[2] for item in data]
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)

    # 创建数据集和数据加载器
    dataset = ExpertDataset(data, mean_reward, std_reward)
    dataloader = DataLoader(dataset, batch_size=1024, shuffle=True)



    state_size = len(data[0][0])
    action_size = 1125  # 动作空间大小
    hidden_size = 128
    embedding_dim = 200  # 嵌入维度

    # 初始化奖励模型
    reward_model = RewardModelWithEmbedding(state_size, action_size, hidden_size, embedding_dim).to(device)

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(reward_model.parameters(), lr=0.001)

    # 训练奖励模型
    num_epochs = 1000
    for epoch in range(num_epochs):
        for states, actions, rewards in dataloader:
            states = states.to(device)
            actions = actions.to(device)
            rewards = rewards.to(device).view(-1, 1)

            # 前向传播
            predicted_rewards = reward_model(states, actions)
            loss = criterion(predicted_rewards, rewards)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    print("训练完成")

    # 保存奖励模型
    torch.save(reward_model.state_dict(), './reward_model/reward_model.pth')

    # # 加载奖励模型
    # loaded_reward_model = RewardModel(state_size, action_size, hidden_size).to(device)
    # loaded_reward_model.load_state_dict(torch.load('reward_model.pth'))
