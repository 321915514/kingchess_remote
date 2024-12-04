import torch

import torch.nn as nn
class Net(nn.Module):
    def __init__(self, s_dim=405, a_dim=1125):
        super(Net, self).__init__()
        self.s_dim = s_dim
        self.a_dim = a_dim
        # self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3)
        self.pi1 = nn.Linear(s_dim, 512)
        self.relu = nn.ReLU()
        # self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3)
        self.pi2 = nn.Linear(512, 2048)
        self.dense = nn.Linear(2048, 1125)

        self.v1 = nn.Linear(s_dim, 128)
        self.v2 = nn.Linear(128, 1)
        self.softmax = nn.Softmax(dim=1)
        # set_init([self.pi1, self.mid, self.pi2, self.v1, self.v2])
        self.distribution = torch.distributions.Categorical

    def forward(self, x):
        logits = self.pi1(x)
        logits = self.relu(logits)
        logits = self.pi2(logits)
        logits = self.dense(logits)
        logits = self.softmax(logits)
        v1 = self.v1(x)
        values = torch.tanh(self.v2(v1))
        return logits, values


if __name__ == '__main__':
    net = Net()

    policy, value = net(torch.randn(10, 405))

    print(policy.shape, value.shape)
