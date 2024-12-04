import torch
from torch import nn
import torch.nn.functional as F

from fundamental.board import GameState
from fundamental.coordinate import Player

N_S = 137 # 目前 153
N_A = 1125




class TransformerModel(nn.Module):
    def __init__(self, input_size, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout):
        super(TransformerModel, self).__init__()
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout),
            num_encoder_layers
        )
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout),
            num_decoder_layers
        )
        self.fc = nn.Linear(input_size, d_model)
        self.policy = nn.Linear(d_model, N_A)  # 动作空间大小
        self.value = nn.Linear(d_model, 1)
        self.distribution = torch.distributions.Categorical

    def forward(self, src):
        src = self.fc(src)
        memory = self.encoder(src)
        output = self.decoder(memory, src)
        pooled_output = output.mean(dim=1)
        policy = self.policy(pooled_output)
        value = self.value(pooled_output)
        return policy, value

    def choose_action(self, s, invalid_actions, pos_moves, scores, game:GameState):
        self.eval()
        logits, _ = self.forward(s)
        prob = F.softmax(logits, dim=1).data
        mask = torch.tensor([invalid_actions])
        # print(mask)
        masked_prob = prob*mask

        sum_prob = masked_prob.sum(dim=1, keepdim=True)

        # 检查 sum_prob 是否为零，避免除以零
        if torch.all(sum_prob == 0):
            # print("Sum of probabilities is zero, adjusting to avoid division by zero.")
            # 设置一个默认的概率分布，例如均匀分布到合法动作
            masked_prob = mask / mask.sum(dim=1, keepdim=True)
        else:
            # 归一化
            masked_prob = masked_prob / sum_prob

        # print(masked_prob)

        m = self.distribution(masked_prob)


        # if game.player == Player.white:
        #     action = m.sample().numpy()[0]
        #     # while 1:
        #     #     if action not in pos_moves:
        #     #         move = game.a_trans_move(action)
        #     #         if move not in scores or scores[move]<0:
        #     #             action = m.sample().numpy()[0]
        #     #     else:
        #     #         break
        #
        #     while 1:
        #         move = game.a_trans_move(action)
        #
        #         if move in scores and scores[move]>0:
        #             break
        #         else:
        #             action = m.sample().numpy()[0]
        #
        #     return action
        # elif game.player == Player.black:
        #     action = m.sample().numpy()[0]
        #     while 1:
        #         if action not in pos_moves:
        #             action = m.sample().numpy()[0]
        #         else:
        #             break
        #     return action


        m = self.distribution(masked_prob)
        #
        action = m.sample().numpy()[0]
        #
        while action not in pos_moves:
            action = m.sample().numpy()[0]
        return action

    def choose_greedy_action(self, s, invalid_actions):
        self.eval()
        logits, _ = self.forward(s)
        prob = F.softmax(logits, dim=1)
        mask = torch.tensor([invalid_actions])
        # print(mask)
        masked_prob = prob*mask

        # sum_prob = masked_prob.sum(dim=1, keepdim=True)

        # # 检查 sum_prob 是否为零，避免除以零
        # if torch.all(sum_prob == 0):
        #     # print("Sum of probabilities is zero, adjusting to avoid division by zero.")
        #     # 设置一个默认的概率分布，例如均匀分布到合法动作
        #     masked_prob = mask / mask.sum(dim=1, keepdim=True)
        # else:
        #     # 归一化
        #     masked_prob = masked_prob / sum_prob

        masked_prob[mask == 0] = float('-inf')

        # print(masked_prob)

        # masked_prob = masked_prob / sum_prob  # Normalize
        # m = self.distribution(masked_prob)
        action = torch.argmax(masked_prob, dim=1).numpy()[0]
        # action = m.sample().numpy()[0]

        return action


    def loss_func(self, s, a, v_t):
        self.train()
        logits, values = self.forward(s)
        td = v_t - values
        c_loss = td.pow(2)

        probs = F.softmax(logits, dim=1)
        m = self.distribution(probs)
        exp_v = m.log_prob(a) * td.detach().squeeze()
        a_loss = -exp_v
        total_loss = (c_loss + a_loss).mean()
        return total_loss