import torch
from torch import nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(Net, self).__init__()
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3)
        self.pi1 = nn.Linear(64 * (s_dim - 4), 128)
        self.relu = nn.ReLU()
        self.pi2 = nn.Linear(128, 512)
        self.dense = nn.Linear(512, 1125)
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)  # Flatten
        logits = self.pi1(x)
        logits = self.relu(logits)
        logits = self.pi2(logits)
        logits = self.dense(logits)
        # logits = self.softmax(logits)
        return logits

    def choose_greedy_action(self, s, invalid_actions):
        self.eval()
        logits = self.forward(s)
        logits = F.softmax(logits, dim=-1)
        mask = torch.tensor([invalid_actions])
        masked_prob = logits*mask
        masked_prob[mask == 0] = float('-inf')

        action = torch.argmax(masked_prob, dim=1).numpy()[0]

        return action


class AutoEncoder(nn.Module):
    def __init__(self, input_dim):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded