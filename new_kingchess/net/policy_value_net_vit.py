# -*- coding: utf-8 -*-
"""
An implementation of the policyValueNet in PyTorch
Tested in PyTorch 0.2.0 and 0.3.0

@author: Junxiao Song
"""
import os
import random

import os
import shutil
import sys
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)



import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.autograd import Variable
import numpy as np
import torchvision.models as models
from torch.nn import init
from torch.nn.utils.rnn import pad_sequence
from net.config import CONFIG_LDCONV
from net.encoder import encoder_board
from fundamental.board import GameState

# 设置打印选项以显示所有元素
torch.set_printoptions(threshold=torch.inf)
from vit_l import ViT

def set_learning_rate(optimizer, lr):
    """Sets the learning rate to the given value"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# class Net(nn.Module):
#     """policy-value network module"""
#
#     def __init__(self):
#         super(Net, self).__init__()
#
#         # common layers
#         self.conv1 = nn.Conv2d(5, 32, kernel_size=1) # ks=3, padding=1
#         # self.conv2 = nn.Conv2d(32, 64, kernel_size=1)
#         # self.conv3 = nn.Conv2d(64, 128, kernel_size=1)
#         # action policy layers
#         self.act_conv1 = nn.Conv2d(32, 16, kernel_size=1)
#         self.act_fc1 = nn.Linear(16 * 5 * 9, 1125)
#         # state value layers
#         self.val_conv1 = nn.Conv2d(32, 2, kernel_size=1)
#         self.val_fc1 = nn.Linear(2 * 5 * 9, 64)
#         self.val_fc2 = nn.Linear(64, 1)
#
#     def forward(self, state_input):
#         # common layers
#         x = F.relu(self.conv1(state_input))
#         # x = F.relu(self.conv2(x))
#         # x = F.relu(self.conv3(x))
#         # action policy layers
#         x_act = F.relu(self.act_conv1(x))
#         x_act = x_act.view(-1,  16*5*9)
#         x_act = F.softmax(self.act_fc1(x_act), dim=1)
#         # state value layers
#         x_val = F.relu(self.val_conv1(x))
#         x_val = x_val.view(-1, 2*5*9)
#         x_val = F.relu(self.val_fc1(x_val))
#         x_val = torch.tanh(self.val_fc2(x_val))
#         return x_act, x_val

# 搭建残差块
class ResBlock(nn.Module):

    def __init__(self, num_filters=256):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=num_filters, out_channels=num_filters, kernel_size=(3, 3), stride=(1, 1),
                               padding=1, bias=False)
        self.conv1_bn = nn.BatchNorm2d(num_filters, )
        self.conv1_act = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=num_filters, out_channels=num_filters, kernel_size=(3, 3), stride=(1, 1),
                               padding=1, bias=False)
        self.conv2_bn = nn.BatchNorm2d(num_filters, )
        self.conv2_act = nn.ReLU()

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv1_bn(y)
        y = self.conv1_act(y)
        y = self.conv2(y)
        y = self.conv2_bn(y)
        y = x + y
        return self.conv2_act(y)



class WhiteFeatureExtractor(nn.Module):
    def __init__(self):
        super(WhiteFeatureExtractor, self).__init__()
        self.white_conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.white_bn1 = nn.BatchNorm2d(32)
        self.white_conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.white_bn2 = nn.BatchNorm2d(64)

    def forward(self, x):
        # 假设输入 x 的一个通道对应白棋
        white_channel = x[:, 1:2, :, :]
        x = self.white_conv1(white_channel)
        x = self.white_bn1(x)
        x = nn.ReLU()(x)
        x = self.white_conv2(x)
        x = self.white_bn2(x)
        x = nn.ReLU()(x)
        return x


class AttentionModule(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(AttentionModule, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)

    def forward(self, x):
        x, _ = self.attention(x, x, x)
        return x





# 搭建骨干网络，输入：N, 5, 5, 9 --> N, C, H, W
class Net(nn.Module):

    def __init__(self, num_channels=128, num_res_blocks=5):
        super().__init__()
        # 全局特征
        # self.global_conv = nn.Conv2D(in_channels=9, out_channels=512, kernel_size=(10, 9))
        # self.global_bn = nn.BatchNorm2D(512)
        # 初始化特征
        self.conv_block = nn.Conv2d(in_channels=32, out_channels=num_channels, kernel_size=(3, 3), stride=(1, 1),
                                    padding=1)
        self.conv_block_bn = nn.BatchNorm2d(128)
        self.conv_block_act = nn.ReLU()
        # 残差块抽取特征
        self.res_blocks = nn.ModuleList([ResBlock(num_filters=num_channels) for _ in range(num_res_blocks)])
        # 策略头
        self.policy_conv = nn.Conv2d(in_channels=num_channels, out_channels=16, kernel_size=(1, 1), stride=(1, 1),
                                     bias=False)
        self.policy_bn = nn.BatchNorm2d(16)
        self.policy_act = nn.ReLU()

        self.policy_fc = nn.Linear(16 * 5 * 9, 1125)
        # 价值头
        self.value_conv = nn.Conv2d(in_channels=num_channels, out_channels=8, kernel_size=(1, 1), stride=(1, 1),
                                    bias=False)
        self.value_bn = nn.BatchNorm2d(8)
        self.value_act1 = nn.ReLU()
        self.value_fc1 = nn.Linear(8 * 5 * 9, 128)
        # self.value_act2 = nn.ReLU()
        self.value_fc2 = nn.Linear(128, 1)

    # 定义前向传播
    def forward(self, x):
        # 公共头
        x = self.conv_block(x)
        x = self.conv_block_bn(x)
        x = self.conv_block_act(x)
        for layer in self.res_blocks:
            x = layer(x)
        # 策略头
        policy = self.policy_conv(x)
        policy = self.policy_bn(policy)
        policy = self.policy_act(policy)
        policy = torch.reshape(policy, [-1, 16 * 5 * 9])
        policy = self.policy_fc(policy)
        # policy = F.softmax(policy, dim=1)
        # 价值头
        value = self.value_conv(x)
        value = self.value_bn(value)
        value = self.value_act1(value)
        value = torch.reshape(value, [-1, 8 * 5 * 9])
        value = self.value_fc1(value)
        value = self.value_act1(value)
        value = self.value_fc2(value)
        value = torch.tanh(value)

        return policy, value


# # transformer start
# input_size = 153  # 棋盘状态+剩余棋子位置+黑白棋子数量
# d_model = 512
# nhead = 8
# num_encoder_layers = 6
# num_decoder_layers = 6
# dim_feedforward = 2048
# dropout = 0.1
# # transformer end


# class Net(nn.Module):
#     def __init__(self, input_size=243, d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048, dropout=0.01):
#         super(Net, self).__init__()
#         self.encoder = nn.TransformerEncoder(
#             nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout),
#             num_encoder_layers
#         )
#         self.decoder = nn.TransformerDecoder(
#             nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout),
#             num_decoder_layers
#         )
#         self.fc = nn.Linear(input_size, d_model)
#         self.policy = nn.Linear(d_model, 1125)  # 动作空间大小
#         self.value = nn.Linear(d_model, 1)
#         # self.distribution = torch.distributions.Categorical
#
#     def forward(self, src):
#         src = self.fc(src)
#         memory = self.encoder(src)
#         output = self.decoder(memory, src)
#         policy = F.softmax(self.policy(output))
#         value = torch.tanh(self.value(output))
#         return policy, value



# class Bottle2neck(nn.Module):
#     expansion = 1
#
#     def __init__(self, inplanes, planes, stride=1, downsample=None, baseWidth=26, scale=4, stype='normal'):
#         super(Bottle2neck, self).__init__()
#
#         width = int(math.floor(planes * (baseWidth / 64.0)))
#         self.conv1 = nn.Conv2d(inplanes, width * scale, kernel_size=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(width * scale)
#
#         if scale == 1:
#             self.nums = 1
#         else:
#             self.nums = scale - 1
#         if stype == 'stage':
#             self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)
#         convs = []
#         bns = []
#         for i in range(self.nums):
#             convs.append(nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, bias=False))
#             bns.append(nn.BatchNorm2d(width))
#         self.convs = nn.ModuleList(convs)
#         self.bns = nn.ModuleList(bns)
#
#         self.conv3 = nn.Conv2d(width * scale, planes * self.expansion, kernel_size=1, bias=False)
#         self.bn3 = nn.BatchNorm2d(planes * self.expansion)
#
#         self.relu = nn.ReLU(inplace=True)
#         self.downsample = downsample
#         self.stype = stype
#         self.scale = scale
#         self.width = width
#
#     def forward(self, x):
#         residual = x
#
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#
#         spx = torch.split(out, self.width, 1)
#         for i in range(self.nums):
#             if i == 0 or self.stype == 'stage':
#                 sp = spx[i]
#             else:
#                 sp = sp + spx[i]
#             sp = self.convs[i](sp)
#             sp = self.relu(self.bns[i](sp))
#             if i == 0:
#                 out = sp
#             else:
#                 out = torch.cat((out, sp), 1)
#         if self.scale!= 1 and self.stype == 'normal':
#             out = torch.cat((out, spx[self.nums]), 1)
#         elif self.scale!= 1 and self.stype == 'stage':
#             out = torch.cat((out, self.pool(spx[self.nums])), 1)
#
#         out = self.conv3(out)
#         out = self.bn3(out)
#         if self.downsample is not None:
#             residual = self.downsample(x)
#         # else:
#         #     # If no downsample, adjust residual if needed
#         #     if out.shape != residual.shape:
#         #         device = x.device
#         #         conv_layer = nn.Conv2d(residual.shape[1], out.shape[1], kernel_size=1, stride=1, padding=0,
#         #                                bias=False).to(device)
#         #         # 修改这里，根据输入张量的形状调整卷积层的输入通道数
#         #         conv_layer.weight = nn.Parameter(
#         #             conv_layer.weight.repeat(1, out.shape[1] // conv_layer.weight.shape[0], 1, 1))
#         #         residual = conv_layer(residual)
#         out += residual
#         out = self.relu(out)
#         return out

# class Net(nn.Module):
#     def __init__(self, num_channels=128, num_res_blocks=5):
#         super().__init__()
#         # 白棋特征提取器
#         self.white_feature_extractor = WhiteFeatureExtractor()
#         # 注意力机制模块
#         self.attention_module = AttentionModule(embed_dim=45, num_heads=3)
#         # 初始化特征
#         self.conv_block = nn.Conv2d(in_channels=21, out_channels=num_channels, kernel_size=(3, 3), stride=(1, 1),
#                                     padding=1)
#         self.conv_block_bn = nn.BatchNorm2d(128)
#         self.conv_block_act = nn.ReLU()
#         # 残差块抽取特征
#         self.res_blocks = nn.ModuleList([ResBlock(num_filters=num_channels) for _ in range(num_res_blocks)])
#         # 残差块抽取特征改为 Res2Net
#         # self.res_blocks = nn.ModuleList(
#         #     [Bottle2neck(inplanes=num_channels, planes=num_channels) for _ in range(num_res_blocks)])
#         # 策略头
#         self.policy_conv = nn.Conv2d(in_channels=num_channels, out_channels=16, kernel_size=(1, 1), stride=(1, 1),
#                                      bias=False)
#         self.policy_bn = nn.BatchNorm2d(16)
#         self.policy_act = nn.ReLU()
#         self.policy_fc = nn.Linear(16 * 5 * 9 + 64 * 5 * 9, 1125)  # 调整输入大小以融合白棋特征
#         # 价值头
#         self.value_conv = nn.Conv2d(in_channels=num_channels, out_channels=8, kernel_size=(1, 1), stride=(1, 1),
#                                     bias=False)
#         self.value_bn = nn.BatchNorm2d(8)
#         self.value_act1 = nn.ReLU()
#         self.value_fc1 = nn.Linear(8 * 5 * 9 + 64 * 5 * 9, 128)  # 调整输入大小以融合白棋特征
#         self.value_fc2 = nn.Linear(128, 1)

#     def forward(self, x):
#         # 提取白棋特征
#         white_features = self.white_feature_extractor(x)
#         # 应用注意力机制
#         white_features = self.attention_module(white_features.view(white_features.size(0), white_features.size(1), -1))
#         white_features = white_features.view(white_features.size(0), -1)
#         # 公共头
#         x = self.conv_block(x)
#         x = self.conv_block_bn(x)
#         x = self.conv_block_act(x)
#         for layer in self.res_blocks:
#             x = layer(x)
#         # 策略头
#         policy = self.policy_conv(x)
#         policy = self.policy_bn(policy)
#         policy = self.policy_act(policy)
#         policy = torch.reshape(policy, [-1, 16 * 5 * 9])
#         # 融合白棋特征
#         policy = torch.cat((policy, white_features), dim=1)
#         policy = self.policy_fc(policy)
#         #policy = F.softmax(policy, dim=1)
#         # 价值头
#         value = self.value_conv(x)
#         value = self.value_bn(value)
#         value = self.value_act1(value)
#         value = torch.reshape(value, [-1, 8 * 5 * 9])
#         # 融合白棋特征
#         value = torch.cat((value, white_features), dim=1)
#         value = self.value_fc1(value)
#         value = self.value_act1(value)
#         value = self.value_fc2(value)
#         value = torch.tanh(value)

#         return policy, value

# class Net_5(nn.Module):
#     def __init__(self, num_channels=128, num_res_blocks=5):
#         super().__init__()
#         # 白棋特征提取器
#         self.white_feature_extractor = WhiteFeatureExtractor()
#         # 注意力机制模块
#         self.attention_module = AttentionModule(embed_dim=45, num_heads=3)
#         # 初始化特征
#         self.conv_block = nn.Conv2d(in_channels=5, out_channels=num_channels, kernel_size=(3, 3), stride=(1, 1),
#                                     padding=1)
#         self.conv_block_bn = nn.BatchNorm2d(128)
#         self.conv_block_act = nn.ReLU()
#         # 残差块抽取特征
#         self.res_blocks = nn.ModuleList([ResBlock(num_filters=num_channels) for _ in range(num_res_blocks)])
#         # 残差块抽取特征改为 Res2Net
#         # self.res_blocks = nn.ModuleList(
#         #     [Bottle2neck(inplanes=num_channels, planes=num_channels) for _ in range(num_res_blocks)])
#         # 策略头
#         self.policy_conv = nn.Conv2d(in_channels=num_channels, out_channels=16, kernel_size=(1, 1), stride=(1, 1),
#                                      bias=False)
#         self.policy_bn = nn.BatchNorm2d(16)
#         self.policy_act = nn.ReLU()
#         self.policy_fc = nn.Linear(16 * 5 * 9 + 64 * 5 * 9, 1125)  # 调整输入大小以融合白棋特征
#         # 价值头
#         self.value_conv = nn.Conv2d(in_channels=num_channels, out_channels=8, kernel_size=(1, 1), stride=(1, 1),
#                                     bias=False)
#         self.value_bn = nn.BatchNorm2d(8)
#         self.value_act1 = nn.ReLU()
#         self.value_fc1 = nn.Linear(8 * 5 * 9 + 64 * 5 * 9, 128)  # 调整输入大小以融合白棋特征
#         self.value_fc2 = nn.Linear(128, 1)

#     def forward(self, x):
#         # 提取白棋特征
#         white_features = self.white_feature_extractor(x)
#         # 应用注意力机制
#         white_features = self.attention_module(white_features.view(white_features.size(0), white_features.size(1), -1))
#         white_features = white_features.view(white_features.size(0), -1)
#         # 公共头
#         x = self.conv_block(x)
#         x = self.conv_block_bn(x)
#         x = self.conv_block_act(x)
#         for layer in self.res_blocks:
#             x = layer(x)
#         # 策略头
#         policy = self.policy_conv(x)
#         policy = self.policy_bn(policy)
#         policy = self.policy_act(policy)
#         policy = torch.reshape(policy, [-1, 16 * 5 * 9])
#         # 融合白棋特征
#         policy = torch.cat((policy, white_features), dim=1)
#         policy = self.policy_fc(policy)
#         #policy = F.softmax(policy, dim=1)
#         # 价值头
#         value = self.value_conv(x)
#         value = self.value_bn(value)
#         value = self.value_act1(value)
#         value = torch.reshape(value, [-1, 8 * 5 * 9])
#         # 融合白棋特征
#         value = torch.cat((value, white_features), dim=1)
#         value = self.value_fc1(value)
#         value = self.value_act1(value)
#         value = self.value_fc2(value)
#         value = torch.tanh(value)

#         return policy, value















class PolicyValueNet():
    """policy-value network """

    def __init__(self, model_file=None, use_gpu=True, device='cuda:7'):
        self.device = device
        # self.use_gpu = use_gpu
        # self.board_width = board_width
        # self.board_height = board_height
        self.l2_const = 2e-3  # coef of l2 penalty
        # the policy value net module
        # if self.use_gpu:
        self.policy_value_net = ViT().to(self.device)
        # else:
        #     self.policy_value_net = Net()
        self.optimizer = optim.AdamW(self.policy_value_net.parameters(), lr=1e-6, betas=(0.9, 0.999), eps=1e-8,
                                    weight_decay=self.l2_const)

        if model_file:
            self.policy_value_net.load_state_dict(torch.load(model_file, map_location=self.device))

    def policy_value(self, state_batch):
        """
        input: a batch of states
        output: a batch of action probabilities and state values
        """
        self.policy_value_net.eval()

        state_batch = torch.tensor(state_batch).to(self.device)
        log_act_probs, value = self.policy_value_net(state_batch)
        log_act_probs, value = log_act_probs.cpu(), value.cpu()
        act_probs = log_act_probs.detach().numpy()
        return act_probs, value.detach().numpy()
        # else:
        #     state_batch = torch.tensor(state_batch)
        #     log_act_probs, value = self.policy_value_net(state_batch)
        #     act_probs = np.exp(log_act_probs.data.numpy())
        #     return act_probs, value.data.numpy()

    # def choose_greedy_action(self, s, invalid_actions):
    #     self.policy_value_net.eval()
    #     logits , _ = self.policy_value_net(s)
    #
    #     mask = torch.tensor([invalid_actions])
    #
    #     masked_logits = logits.masked_fill(mask == 0, float('-inf'))
    #
    #     probs = F.softmax(masked_logits, dim=-1)
    #
    #
    #     # masked_prob[mask == 0] = float('-inf')
    #
    #     action = torch.argmax(probs, dim=1).numpy()[0]
    #
    #     return action

    def accuracy(self, y_true, y_pred):
        true_argmax = torch.argmax(y_true, dim=1)
        pre_argmax = torch.argmax(y_pred, dim=1)
        return torch.mean((true_argmax == pre_argmax).float())

    def policy_value_fn(self, game: GameState):
        """
        input: board
        output: a list of (action, probability) tuples for each available
        action and the score of the board state
        """
        self.policy_value_net.eval()
        legal_positions = [game.move_2_action(move) for move in game.legal_moves()]
        # current_state = np.ascontiguousarray(encoder_board(game).reshape(-1, 5, 9, 9)).astype(
        #     'float16')  # .reshape(-1, 3)

        current_state = np.ascontiguousarray(encoder_board(game).reshape(-1, 32, 5, 9)).astype('float32')  # .reshape(-1, 3)

        current_state = torch.as_tensor(current_state).to(self.device)

        # log_act_probs, value = self.policy_value_net(current_state)

        # with autocast():  # 半精度fp16
        log_act_probs, value = self.policy_value_net(current_state)

        log_act_probs, value = log_act_probs.cpu(), value.cpu()

        act_probs = log_act_probs.detach().numpy().astype('float32').flatten()  #

        act_probs = zip(legal_positions, act_probs[legal_positions])

        # pos, pos_moves = game.legal_position()
        #
        # if pos is not None:
        # #     # print(mask)
        # #     masked_prob = act_probs * pos
        # #
        # #     for i in pos_moves:
        # #         if masked_prob[i] != 0:
        # #             continue
        # #         else:
        # #             masked_prob[i] = random.random()
        #
        #     # act_probs = zip(legal_positions, act_probs[:len(legal_positions)])
        #
        #     # print(act_probs)
        #     # print(value)
        #
        #     mask = torch.tensor(pos).to(self.device)
        #
        #     log_act_probs = log_act_probs.flatten()
        #
        #     masked_logits = log_act_probs.masked_fill(mask == 0, float('-inf'))
        #
        #     probs = F.softmax(masked_logits, dim=-1)
        #
        #     return probs.cpu().detach().numpy(), value.cpu().detach().numpy()
        # else:
        #     return np.zeros([1125], dtype=int), 0

        return act_probs, value.cpu().detach().numpy()


    def center_area(self, state): # 512, 5, 5, 9
        center_area = [(1, 3), (1, 4), (1, 5), (2, 3),(2,4),(2,5),(3,3),(3,4),(3,5)]
        white_chess_in_center = 0
        total_white_chess = 0
        for i in range(5):
            for j in range(9):
                if state[1, i, j] == 1:  # 白棋层判断
                    total_white_chess += 1
                    if (i, j) in center_area:
                        white_chess_in_center += 1
        return white_chess_in_center / total_white_chess

    def state_batch_center_loss(self, state_batch):
        batch = state_batch.shape[0]
        batch_loss = 0
        for i in range(batch):
            state = state_batch[i]
            one_center_loss = self.center_area(state)
            batch_loss += one_center_loss

        return batch_loss/batch


    def train_step(self, state_batch, mcts_probs, winner_batch, lr=1e-6):
        """perform a training step"""
        # wrap in Variable
        self.policy_value_net.train()
        # 将内部列表转换为torch.Tensor
        # mcts_probs_tensors = [[lst[idx] if idx < len(lst) else 0 for idx in range(81)] for lst in mcts_probs]

        # # 将补齐后的列表转换为PyTorch张量
        # tensor = torch.tensor(padded_lists, dtype=torch.float32)
        # state_batch = np.array(state_batch)

        state_batch = torch.tensor(state_batch, dtype=torch.float32).to(self.device)  # 确保 state_batch 是 numpy 数组
        #print(state_batch[:,2,:,:].shape)
        #color = state_batch[:,2,0,0]
        # mcts_probs = Variable(torch.FloatTensor(mcts_probs).cuda())
        # 使用pad_sequence补齐张量到最长序列的长度
        # 将补齐后的列表转换为PyTorch张量
        mcts_probs = torch.tensor(mcts_probs, dtype=torch.float32).to(self.device)
        # mcts_probs = pad_sequence(mcts_probs, batch_first=True, padding_value=0).cuda()
        winner_batch = torch.tensor(winner_batch, dtype=torch.float32).to(self.device)  # 确保 state_batch 是 numpy 数组
        # else:
        #     state_batch = torch.tensor(state_batch, dtype=torch.float)
        #     # mcts_probs = Variable(torch.FloatTensor(mcts_probs))
        #     # 将补齐后的列表转换为PyTorch张量
        #     mcts_probs = torch.tensor(mcts_probs_tensors, dtype=torch.float32)
        #     winner_batch = torch.tensor(winner_batch, dtype=torch.float)

        # zero the parameter gradients
        self.optimizer.zero_grad()
        # set learning rate
        # for params in self.optimizer.param_groups:
        #     params['lr'] = lr

        # forward
        log_act_probs, value = self.policy_value_net(state_batch)  # (512,5,9,9)
        # define the loss = (z - v)^2 - pi^T * log(p) + c||theta||^2
        # Note: the L2 penalty is incorporated in optimizer
        value = torch.reshape(value, shape=[-1])
        criterion = nn.CrossEntropyLoss()

        #print(winner_batch)


        #winners = winner_batch
        
        
        #print((torch.mean(winners*color)+1)/2)

        value_loss = F.mse_loss(value, winner_batch)
        # log_act_probs = F.softmax(log_act_probs, dim=1)





        accuracy = self.accuracy(mcts_probs, torch.softmax(log_act_probs,dim=-1))

        #policy_loss = -torch.mean(torch.sum(mcts_probs * torch.log_softmax(log_act_probs, dim=-1), 1), dim=-1)

#        print(log_act_probs)

        #print(mcts_probs)

        #return

        policy_loss = criterion(log_act_probs, mcts_probs)

 #       print(policy_loss)

        #center_loss = self.state_batch_center_loss(state_batch)

        loss = value_loss + policy_loss
        # backward and optimize
        loss.backward()
        self.optimizer.step()
        # calc policy entropy, for monitoring only
        # 策略的熵
        
        with torch.no_grad():
            softmax_act_probs = torch.softmax(log_act_probs,dim=-1)
            entropy = -torch.mean(
                torch.sum(torch.log(softmax_act_probs) * softmax_act_probs, 1)
            )

        # return loss.data[0], entropy.data[0]
        return accuracy.item(), loss.item(), policy_loss.item(), value_loss.item(), entropy.item(), 0
        # for pytorch version >= 0.5 please use the following line instead.
        # return loss.item(), entropy.item()

    def save_model(self, model_file):
        """ save model params to file """
        # get model params
        torch.save(self.policy_value_net.state_dict(), model_file)


def initialize_weights(module):
    if isinstance(module, nn.Conv2d):
        init.kaiming_normal_(module.weight, nonlinearity='relu')
    elif isinstance(module, nn.Linear):
        init.xavier_uniform_(module.weight)
    elif isinstance(module, nn.BatchNorm2d):
        init.constant_(module.weight, 1)
        init.constant_(module.bias, 0)


if __name__ == '__main__':
    # from net.config import CONFIG, CONFIG_TRAIN, CONFIG_TRANSFORMER
    # from fvcore.nn import FlopCountAnalysis, parameter_count_table
    #
    net = ViT().to('cuda:0')
    #
    # board = (torch.randn(1, 5, 9, 9),)
    #
    # flog = FlopCountAnalysis(net, board)
    #
    # print(flog.total())
    #
    # print(parameter_count_table(net))
    #
    # # os.makedirs('model_torch', exist_ok=True)
    #
    # net.apply(initialize_weights)
    #
    torch.save(net.state_dict(), CONFIG_LDCONV['pytorch_model_path'])

    # net.load_state_dict(torch.load('E:/pytorch.pt'))
    # net.eval()
    # net.cuda()
    #
    # # print(net)
    board = torch.zeros((10, 32, 5, 9))
    # game = GameState.new_game(5, 9)
    #
    # current_state = np.ascontiguousarray(encoder_board(game).reshape(-1, 5, 9, 9)).astype('float32')  # .reshape(-1, 3)
    #
    current_state = torch.as_tensor(board).to('cuda:0')
    #
    p, v = net(current_state)
    #
    print(p.shape)
    print(v.shape)
