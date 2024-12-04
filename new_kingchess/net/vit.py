import torch
import torch.nn as nn
#from torchsummary import summary
from einops import rearrange


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


# 定义多头注意力机制模块
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        # x 的形状为 (batch_size, num_patches + 1, total_embed_dim)
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# 定义 MLP 模块
class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# 定义残差连接模块
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


# 定义 Transformer Encoder 模块
class TransformerEncoderBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, drop=0., attn_drop=0., drop_path=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Residual(Attention(dim, num_heads, attn_drop=attn_drop, proj_drop=drop))
        self.drop_path = nn.Identity() if drop_path == 0. else DropPath(drop_path)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Residual(MLP(dim, mlp_hidden_dim, drop=drop))

    def forward(self, x):
        x = self.norm1(x)
        x = self.attn(x)
        x = self.drop_path(x)
        x = self.norm2(x)
        x = self.mlp(x)
        return x


# 定义 Patch Embedding 模块
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=32, patch_size=2, embed_dim=128):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        if len(x.shape) == 5:
            # 如果输入是五维张量，假设最后一维是无用维度，将其移除
            x = x.squeeze(-1)
        batch_size, channels, height, width = x.shape
        x = self.proj(x)
        num_patches = (height // self.proj.kernel_size[0]) * (width // self.proj.kernel_size[1])
        x = rearrange(x, 'b e h w -> b (h w) e')
        x = self.norm(x)
        return x


# 定义完整的 ViT 模型
class ViT(nn.Module):
    def __init__(self):
        super().__init__()
        # 假设这里的输入通道数为 32
        in_channels = 32
        # 假设 patch 大小为 2（可以根据实际情况调整）
        patch_size = 4
        embed_dim = 128
        num_heads = 4
        num_encoders = 4
        mlp_ratio = 4.0
        drop_rate = 0.1
        drop_path_rate = 0.1
        self.patch_embedding = PatchEmbedding(in_channels, patch_size, embed_dim)
        num_patches = ((5 // patch_size) * (9 // patch_size))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))
        self.dropout = nn.Dropout(drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_encoders)]
        self.encoder_blocks = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads, mlp_ratio, drop_rate, drop_rate, dpr[i])
            for i in range(num_encoders)
        ])
        # 添加新的输出层用于 policy 和 value
        self.policy_fc = nn.Linear(embed_dim, 1125)
        self.value_fc = nn.Linear(embed_dim, 1)

    def forward(self, x):
#        print("Input shape:", x.shape)
        x = self.patch_embedding(x)
        batch_size = x.shape[0]

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embedding
        x = self.dropout(x)

        for encoder_block in self.encoder_blocks:
            x = encoder_block(x)

        # 得到 policy 和 value
        policy = self.policy_fc(x[:, 0])
        value = self.value_fc(x[:, 0])
        return policy, value


if __name__ == '__main__':
    # 实例化模型
    model = ViT()
    # 查看模型结构

    policy, value = model(torch.randn(1, 32, 5, 9))

    print(policy.shape)
    print(value.shape)
