import torch
import torch.nn.functional as F
from torch import layer_norm, nn
import numpy as np
import math
from utils import *
from models.ST_GCN2 import GCN_main

def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


class TemporalSELayer(nn.Module):
    def __init__(self,
                 c=21,  # 时间帧数量（T_seq），对应要学习的权重数量
                 r=4,
                 use_max_pooling=False):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1) if not use_max_pooling else nn.AdaptiveMaxPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(c+1, c // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c // r, c+1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        bs, t, d = x.shape
        # x = x.permute(0, 2, 1)
        y = self.squeeze(x)
        y = y.view(bs, t)
        y = self.excitation(y)
        y = y.view(bs, t, 1)
        return x * y.expand_as(x)


class SpatialSELayer(nn.Module):
    def __init__(self,
                 c=17,  # 关节数（N），对应要学习的权重数量
                 r=4,
                 use_max_pooling=False):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1) if not use_max_pooling else nn.AdaptiveMaxPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(c, c // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c // r, c, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        bs, t, s = x.shape
        x = x.permute(0, 2, 1)
        y = self.squeeze(x).view(bs, s)
        y = self.excitation(y)
        y = y.view(bs, s, 1).permute(0, 2, 1)
        x = x.permute(0, 2, 1)
        return x * y.expand_as(x)


class FFN(nn.Module):

    def __init__(self, latent_dim, ffn_dim, dropout):
        super().__init__()
        # self.linear1 = nn.Linear(latent_dim, ffn_dim)
        # self.linear2 = zero_module(nn.Linear(ffn_dim, latent_dim))
        self.linear1 = nn.Linear(latent_dim, 256)
        self.linear2 = zero_module(nn.Linear(256, latent_dim))
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, x):
        y = self.linear2(self.dropout(self.activation(self.linear1(self.norm(x)))))
        y = self.dropout(y)
        y = x + y
        return y



class TemporalSelfAttention(nn.Module):

    def __init__(self, latent_dim, num_head, dropout):
        super().__init__()
        self.num_head = num_head
        self.norm = nn.LayerNorm(latent_dim)
        self.query = nn.Linear(latent_dim, latent_dim, bias=False)
        self.key = nn.Linear(latent_dim, latent_dim, bias=False)
        self.value = nn.Linear(latent_dim, latent_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: B, T, D
        """
        B, T, D = x.shape
        H = self.num_head

        # B, T, 1, D
        query = self.query(self.norm(x)).unsqueeze(2)
        # B, 1, T, D
        key = self.key(self.norm(x)).unsqueeze(1)
        query = query.view(B, T, H, -1)
        key = key.view(B, T, H, -1)
        # B, T, T, H
        attention = torch.einsum('bnhd,bmhd->bnmh', query, key) / math.sqrt(D // H)
        weight = self.dropout(F.softmax(attention, dim=2))
        value = self.value(self.norm(x)).view(B, T, H, -1)
        y = torch.einsum('bnmh,bmhd->bnhd', weight, value).reshape(B, T, D)

        y = x + y
        return y
    
class TemporalDiffusionTransformerDecoderLayer(nn.Module):

    def __init__(self,
                 cfg,
                 latent_dim=32,
                 time_embed_dim=128,
                 ffn_dim=256,
                 num_head=4,
                 dropout=0.2,
                 skip=False,
                 se_dim_t=20,
                 se_dim_s=512,
                 se_r=4
                 ):
        super().__init__()
        if skip:
            self.skip_linear = nn.Linear(2 * latent_dim, latent_dim)

        self.sa_block = FuzzyGuidedTemporalAttention(
            latent_dim, num_head, dropout)
        # self.sa_block = TemporalSelfAttention(
        #     latent_dim, num_head, dropout)

        self.ffn = FFN(latent_dim, ffn_dim, dropout)

        self.se_block_s = SpatialSELayer(
            c=se_dim_s,  # 压缩维度（对应关节数N）
            r=se_r,  # 缩减率
            use_max_pooling=False
        )
        self.se_block_t = TemporalSELayer(
            c=se_dim_t,  # 压缩维度（对应num_frames+1）
            r=se_r,  # 缩减率
            use_max_pooling=False,  # 用平均池化（更稳定，也可改max）
        )

    def forward(self, x, skip=None):
        # if skip is not None:
        #     x = self.skip_linear(torch.cat([x, skip], dim=-1))
        # x = x + self.se_block_t(x)  # 实现了残差连接,模型在训练过程中更容易地学习到特征的增量信息，避免梯度消失问题，同时保留了原始特征图的信息。
        # x = x + self.se_block_s(x)
        x = self.sa_block(x)
        x = self.ffn(x)
        return x


class MotionTransformerV(nn.Module):
    def __init__(self,
                 cfg,
                 input_feats,
                 num_frames=240,
                 latent_dim=512,
                 ff_size=1024,
                 num_layers=8,
                 num_heads=8,
                 dropout=0.2,
                 activation="gelu",
                 **kargs):
        super().__init__()
        self.cfg = cfg
        self.num_frames = num_frames
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.ff_size = ff_size
        self.dropout = dropout
        self.activation = activation
        self.input_feats = input_feats
        self.time_embed_dim = latent_dim
        self.sequence_embedding = nn.Parameter(torch.randn(num_frames+ 2, latent_dim))

        # Input Embedding
        self.joint_embed = nn.Linear(self.input_feats, self.latent_dim)
        
        self.cond_embed = nn.Linear(self.input_feats * self.num_frames, self.time_embed_dim)

        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, self.time_embed_dim),
            nn.SiLU(),
            nn.Linear(self.time_embed_dim, self.time_embed_dim),
        )

        self.in_blocks = nn.ModuleList([
            TemporalDiffusionTransformerDecoderLayer(
                    self.cfg,
                    latent_dim=latent_dim,
                    time_embed_dim=self.time_embed_dim,
                    ffn_dim=ff_size,
                    num_head=num_heads,
                    dropout=dropout,)
            for i in range(self.num_layers // 2)])

        self.mid_block = TemporalDiffusionTransformerDecoderLayer(
                                self.cfg,
                                latent_dim=latent_dim,
                                time_embed_dim=self.time_embed_dim,
                                ffn_dim=ff_size,
                                num_head=num_heads,
                                dropout=dropout,)

        self.out_blocks = nn.ModuleList([
            TemporalDiffusionTransformerDecoderLayer(
                    self.cfg,
                    latent_dim=latent_dim,
                    time_embed_dim=self.time_embed_dim,
                    ffn_dim=ff_size,
                    num_head=num_heads,
                    dropout=dropout,
                    skip=True)
            for i in range(self.num_layers // 2)])

        self.out = zero_module(nn.Linear(self.latent_dim, self.input_feats))

        #GCN
        self.gcn_model = GCN_main(self.cfg).to(device=self.cfg.device)
        #

    def forward(self, x, timesteps, mod=None):
        """
        x: B, T, D
        """
        B, T = x.shape[0], x.shape[1]
        # x =self.gcn_model(x)
        emb = self.time_embed(timestep_embedding(timesteps, self.latent_dim)).unsqueeze(1)

        if mod is not None:
            mod_proj = self.cond_embed(mod.reshape(B, -1)).unsqueeze(1)
            emb = emb + mod_proj

        h = self.joint_embed(x)
        h = torch.cat([emb, h], dim=1)
        h = h + self.sequence_embedding.unsqueeze(0)[:, :T+1, :]

        skips = []

        for blk in self.in_blocks:
            h = blk(h)
            skips.append(h)

        h = self.mid_block(h)

        for blk in self.out_blocks:
            h = blk(h, skips.pop())

        output = self.out(h[:, 1:, :]).view(B, T, -1).contiguous()
        return output

class FuzzyGuidedTemporalAttention(nn.Module):
    def __init__(self, in_channels=512, num_heads=8, dropout=0.1, num_rules=16):
        super().__init__()
        assert in_channels % num_heads == 0, f"in_channels ({in_channels}) must be divisible by num_heads ({num_heads})"

        self.in_channels = in_channels
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads
        self.num_rules = num_rules

        # === Fuzzy motion saliency module (operates on frame-level motion magnitude) ===
        self.u_center = nn.Parameter(torch.randn(num_rules))
        self.u_sigma = nn.Parameter(torch.ones(num_rules))

        # === Temporal self-attention (keeps channel dimension unchanged) ===
        self.norm = nn.LayerNorm(in_channels)
        self.q_proj = nn.Linear(in_channels, in_channels, bias=False)
        self.k_proj = nn.Linear(in_channels, in_channels, bias=False)
        self.v_proj = nn.Linear(in_channels, in_channels, bias=False)
        self.out_proj = nn.Linear(in_channels, in_channels)
        self.dropout = nn.Dropout(dropout)

    def compute_motion_saliency(self, x):
        """
        x: (B, T, C)
        Returns: (B, T) — motion magnitude per frame
        """
        motion = torch.zeros_like(x)
        motion[:, 1:] = x[:, 1:] - x[:, :-1]
        return torch.norm(motion, dim=-1)  # L2 norm over C

    def generate_fuzzy_bias(self, saliency):
        """
        saliency: (B, T)
        Returns: (B, T, T) attention bias
        """
        B, T = saliency.shape
        s = saliency.unsqueeze(1)  # (B, 1, T)

        # Compute membership for each fuzzy rule
        diff = s - self.u_center.view(1, -1, 1)  # (B, R, T)
        sigma = self.u_sigma.view(1, -1, 1)  # (B, R, 1)
        membership = torch.exp(-0.5 * (diff / (sigma + 1e-6)) ** 2)  # (B, R, T)

        # Defuzzify to enhanced saliency
        weighted = (membership * s).sum(dim=1)  # (B, T)
        weights_sum = membership.sum(dim=1) + 1e-6
        enhanced = weighted / weights_sum  # (B, T)

        # Symmetric attention bias: high-saliency frames attend to each other
        bias = enhanced.unsqueeze(1) + enhanced.unsqueeze(2)  # (B, T, T)
        return bias

    def forward(self, x, return_attn=False):
        """
        x: (B, T, C) with C = in_channels (e.g., 512)
        Returns: (B, T, C)
        """
        B, T, C = x.shape

        # Step 1: Compute motion-based attention bias
        saliency = self.compute_motion_saliency(x)  # (B, T)
        fuzzy_bias = self.generate_fuzzy_bias(saliency)  # (B, T, T)

        # Step 2: Pre-norm
        residual = x
        x_norm = self.norm(x)

        # Step 3: Multi-head projections
        q = self.q_proj(x_norm).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, T, Dh)
        k = self.k_proj(x_norm).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x_norm).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Step 4: Scaled dot-product + fuzzy bias
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, H, T, T)
        attn_scores = attn_scores + fuzzy_bias.unsqueeze(1)  # (B, H, T, T)

        # Step 5: Softmax + dropout
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Step 6: Weighted sum + output projection
        attn_out = torch.matmul(attn_weights, v)  # (B, H, T, Dh)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, C)
        attn_out = self.out_proj(attn_out)

        # Step 7: Residual connection
        output = residual + attn_out  # (B, T, C)

        if return_attn:
            # Return average attention over heads: (B, T, T)
            return output, attn_weights.mean(dim=1)
        else:
            return output