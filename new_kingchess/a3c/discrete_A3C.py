"""
Reinforcement Learning (A3C) using Pytroch + multiprocessing.
The most simple implementation for continuous action.

View more on my Chinese tutorial page [莫烦Python](https://morvanzhou.github.io/).
"""
import random
import time

import gym
import numpy as np
import torch

from a3c.transfromer import TransformerModel
from fundamental.coordinate import Player

# 设置打印选项以显示所有元素
torch.set_printoptions(threshold=torch.inf)

import torch.nn as nn
from a3c.utils import v_wrap, set_init, record, push_and_pull # v_wrap,push_and_pullset_init
import torch.nn.functional as F
import torch.multiprocessing as mp
from a3c.shared_adam import SharedAdam
from fundamental.board import GameState
from agent.expert_agent import Expert_agent


import os

from fundamental.utils import print_board

os.environ["OMP_NUM_THREADS"] = "1"

UPDATE_GLOBAL_ITER = 5
GAMMA = 0.9
MAX_EP = 10000
# env = gym.make('CartPole-v0')
N_S = 153 # 目前 153
N_A = 1125

# transformer start
input_size = N_S  # 棋盘状态+剩余棋子位置+黑白棋子数量
d_model = 512
nhead = 8
num_encoder_layers = 6
num_decoder_layers = 6
dim_feedforward = 2048
dropout = 0.1
# transformer end

expert = Expert_agent()

class Net_white(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(Net_white, self).__init__()
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.pi1 = nn.Linear(s_dim, 128)
        self.mid = nn.Linear(128, 512)
        self.pi2 = nn.Linear(512, 405)
        self.v1 = nn.Linear(s_dim, 128)
        self.v2 = nn.Linear(128, 1)
        # set_init([self.pi1, self.mid, self.pi2, self.v1, self.v2])
        self.distribution = torch.distributions.Categorical

    def forward(self, x):
        pi1 = torch.tanh(self.pi1(x))
        mid = self.mid(pi1)
        logits = self.pi2(mid)
        v1 = torch.tanh(self.v1(x))
        values = self.v2(v1)
        return logits, values



    # def choose_action(self, s, invalid_actions, pos_moves, scores, game):
    #     self.eval()
    #     logits, _ = self.forward(s)
    #     prob = F.softmax(logits, dim=1).data
    #     mask = torch.tensor([invalid_actions])
    #     # print(mask)
    #     masked_prob = prob*mask
    #
    #     sum_prob = masked_prob.sum(dim=1, keepdim=True)
    #
    #     # 检查 sum_prob 是否为零，避免除以零
    #     if torch.all(sum_prob == 0):
    #         # print("Sum of probabilities is zero, adjusting to avoid division by zero.")
    #         # 设置一个默认的概率分布，例如均匀分布到合法动作
    #         masked_prob = mask / mask.sum(dim=1, keepdim=True)
    #     else:
    #         # 归一化
    #         masked_prob = masked_prob / sum_prob
    #
    #     # print(masked_prob)
    #
    #     # masked_prob = masked_prob / sum_prob  # Normalize
    #
    #     # modify
    #     # m = self.distribution(masked_prob)
    #     # if game.player == Player.white:
    #     #     action = m.sample().numpy()[0]
    #     #     while 1:
    #     #         move = game.a_trans_move(action)
    #     #
    #     #         if move in scores:
    #     #             break
    #     #         else:
    #     #             action = m.sample().numpy()[0]
    #     #
    #     #     return action
    #     # elif game.player == Player.black:
    #     #     action = m.sample().numpy()[0]
    #     #     return action
    #     # modify end
    #     m = self.distribution(masked_prob)
    #     #
    #     action = m.sample().numpy()[0]
    #     #
    #     while action not in pos_moves:
    #         action = m.sample().numpy()[0]
    #     return action

    def choose_greedy_action(self, s, invalid_actions):
        self.eval()
        logits, _ = self.forward(s)
        prob = F.softmax(logits, dim=1)
        mask = torch.tensor([invalid_actions[720:]])
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

        return action+720


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



class Net(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(Net, self).__init__()
        self.s_dim = s_dim
        self.a_dim = a_dim
        # self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3)
        self.pi1 = nn.Linear(s_dim, 128)
        self.relu = nn.ReLU()
        # self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3)
        self.pi2 = nn.Linear(128, 512)
        self.dense = nn.Linear(512, 1125)

        self.v1 = nn.Linear(s_dim, 128)
        self.v2 = nn.Linear(128, 1)
        self.softmax = nn.Softmax(dim=0)
        # set_init([self.pi1, self.mid, self.pi2, self.v1, self.v2])
        self.distribution = torch.distributions.Categorical

    def forward(self, x):
        logits = self.pi1(x)
        logits = self.relu(logits)
        logits = self.pi2(logits)
        logits = self.dense(logits)
        # logits = self.softmax(logits)
        v1 = self.v1(x)
        values = torch.tanh(self.v2(v1))
        return logits, values



    def choose_action(self, s, invalid_actions, pos_moves, scores, game):
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

        # masked_prob = masked_prob / sum_prob  # Normalize

        # modify
        # m = self.distribution(masked_prob)
        # if game.player == Player.white:
        #     action = m.sample().numpy()[0]
        #     while 1:
        #         move = game.a_trans_move(action)
        #
        #         if move in scores:
        #             break
        #         else:
        #             action = m.sample().numpy()[0]
        #
        #     return action
        # elif game.player == Player.black:
        #     action = m.sample().numpy()[0]
        #     return action
        # modify end
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



def normalized_columns_initializer(weights, std=1.0):
    out = torch.randn(weights.size())
    out *= std / torch.sqrt(out.pow(2).sum(1, keepdim=True))
    return out


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)




class Worker(mp.Process):
    def __init__(self, gnet, opt, global_ep, global_ep_r, res_queue, name):
        super(Worker, self).__init__()
        self.name = 'w%02i' % name
        self.g_ep, self.g_ep_r, self.res_queue = global_ep, global_ep_r, res_queue
        self.gnet, self.opt = gnet, opt
        self.lnet = TransformerModel(input_size, d_model, nhead, num_encoder_layers, num_decoder_layers,
                                     dim_feedforward, dropout)
        # self.lnet = Net(N_S, N_A)
        self.env = GameState.new_game(5, 9)

    def run(self):
        total_step = 1
        try:
            while self.g_ep.value < MAX_EP:
                s = self.env.reset()
                buffer_s, buffer_a, buffer_r = [], [], []
                ep_r = 0.
                while True:
                    # if self.name == 'w00':
                    #     self.env.render()
                    pos, pos_moves = self.env.legal_position()
                    scores = expert.score_moves(self.env)
                    # print(pos)
                    if pos is not None:
                        # print_board(self.env.board)
                        # inval = np.all(pos == 0)
                        #
                        # print(inval)

                        a = self.lnet.choose_action(v_wrap(s[None, :]), pos, pos_moves, scores, self.env)
                        s_, r, done = self.env.step(a)
                        # if done:
                        #     print('game over')
                        #     r = -1
                        #     break
                        ep_r += r
                        buffer_a.append(a)
                        buffer_s.append(s)
                        buffer_r.append(r)
                    else:
                        s_, r, done = self.env.step(0)
                        # if done:
                        #     print('game over')
                        #     r = -1
                        #     break

                    if total_step % UPDATE_GLOBAL_ITER == 0 or done:  # update global and assign to local net
                        # sync
                        push_and_pull(self.opt, self.lnet, self.gnet, done, s_, buffer_s, buffer_a, buffer_r, GAMMA)
                        buffer_s, buffer_a, buffer_r = [], [], []

                        if done:  # done and print information
                            # self.env.decoder_board(s_)
                            # print_board(self.env.board)
                            record(self.g_ep, self.g_ep_r, ep_r, self.res_queue, self.name)
                            break
                    s = s_
                    total_step += 1
        except KeyboardInterrupt:
            # now = time.time()
            # torch.save(gnet.state_dict(), f'./gnet_linear_{str(now)}.pth')

            print('\n\rquit')
            self.res_queue.put(None)


if __name__ == "__main__":

    gnet = TransformerModel(input_size, d_model, nhead, num_encoder_layers, num_decoder_layers,
                                     dim_feedforward, dropout)

    # gnet = Net(N_S, N_A)        # global network

    weights_init(gnet)

    # gnet.load_state_dict(torch.load('./gnet_7_17.pth'))

    gnet.share_memory()         # share the global parameters in multiprocessing
    opt = SharedAdam(gnet.parameters(), lr=1e-3, betas=(0.92, 0.999))      # global optimizer
    global_ep, global_ep_r, res_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()

    # parallel training
    workers = [Worker(gnet, opt, global_ep, global_ep_r, res_queue, i) for i in range(4)] # mp.cpu_count()
    [w.start() for w in workers]
    res = []                    # record episode reward to plot
    while True:
        r = res_queue.get()
        if r is not None:
            with open('r.log', 'a') as f:
                f.write(str(r))
            res.append(r)
        else:
            break
    [w.join() for w in workers]

    torch.save(gnet.state_dict(), './gnet_transformer_9_8.pth')

    import matplotlib.pyplot as plt
    plt.plot(res)
    plt.ylabel('Moving average ep reward')
    plt.xlabel('Step')
    plt.show()
    pass
