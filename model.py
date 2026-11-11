"""从零实现的小型 Transformer 编码器，用来判断邮件是不是广告。"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizer import Tokenizer


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)  # 一次算出 Q、K、V
        self.out = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=-1)
        # 拆成多个头：(B, T, C) -> (B, 头数, T, d_k)
        q = q.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        # 缩放点积注意力：softmax(QK^T / sqrt(d_k)) V
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_k)
        scores = scores.masked_fill(mask[:, None, None, :] == 0, float("-inf"))  # 不看填充位
        attn = self.drop(F.softmax(scores, dim=-1))

        y = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)  # 合并各个头
        return self.out(y)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(),
            nn.Linear(4 * d_model, d_model), nn.Dropout(dropout),
        )

    def forward(self, x, mask):
        x = x + self.attn(self.ln1(x), mask)  # 残差连接 + 前置 LayerNorm
        x = x + self.ffn(self.ln2(x))
        return x


class AdClassifier(nn.Module):
    def __init__(self, vocab_size, max_len, d_model=128, n_heads=4, n_layers=2, dropout=0.1):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, d_model)  # 可学习的位置编码
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(TransformerBlock(d_model, n_heads, dropout) for _ in range(n_layers))
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 2)  # 两类：0=正常，1=广告

    def forward(self, ids, mask):
        pos = torch.arange(ids.size(1), device=ids.device)
        x = self.drop(self.tok_emb(ids) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x, mask)
        return self.head(self.ln(x[:, 0]))  # 取 <cls> 位置的向量做分类


def load_classifier(path, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    model = AdClassifier(**ckpt["model_cfg"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, Tokenizer(ckpt["vocab"])


@torch.no_grad()
def ad_probability(model, tok, text, max_len):
    ids, mask = tok.encode(text, max_len)
    device = next(model.parameters()).device
    logits = model(torch.tensor([ids], device=device), torch.tensor([mask], device=device))
    return torch.softmax(logits, dim=-1)[0, 1].item()
