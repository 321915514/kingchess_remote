import pickle

import torch
import torch.nn as nn

from a3c.transfromer import TransformerModel

# # 定义嵌入层
# action_size = 5  # 动作数量
# embedding_dim = 3  # 每个动作的嵌入向量的维度
# embedding = nn.Embedding(action_size, embedding_dim)
#
# # 打印嵌入矩阵（随机初始化的）
# print("Embedding Matrix (Randomly Initialized):")
# print(embedding.weight)
#
# # 示例输入（动作索引）
# action_indices = torch.tensor([0, 2, 4])
#
# # 获取对应的嵌入向量
# action_embeddings = embedding(action_indices)
#
# print("\nAction Embeddings:")
# print(action_embeddings)

# def load_data(datapath):
#     with open(datapath, 'rb') as f:
#         result = pickle.load(f)
#         return result
#
# if __name__ == '__main__':
#     select_data = []
#     data = load_data('./game_data_add_score_white.pkl')
#     for state, move, score in data:
#         if score > 10:
#             select_data.append((state, move, score))
#
#     with open('game_data_add_score_white.pkl', 'wb') as f:
#         pickle.dump(select_data, f)



if __name__ == '__main__':
    # transformer start
    input_size = 137  # 棋盘状态+剩余棋子位置+黑白棋子数量
    d_model = 512
    nhead = 8
    num_encoder_layers = 6
    num_decoder_layers = 6
    dim_feedforward = 2048
    dropout = 0.1
    # transformer end

    model = TransformerModel(input_size, d_model, nhead, num_encoder_layers, num_decoder_layers,
                             dim_feedforward, dropout).to('cuda:7')


    src = torch.randn(64, 137, 137).to('cuda:7')

    src = src.permute(0, 2, 1)

    # 前向传播
    policy_logits, value = model(src)

