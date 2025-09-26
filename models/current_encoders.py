import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================================================
# 公用模块
# =========================================================

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(4, channels // reduction)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: [B,C,L]
        w = self.pool(x).squeeze(-1)
        w = self.fc(w).unsqueeze(-1)
        return x * w


class ResidualDilatedBlock(nn.Module):
    def __init__(self, channels, dilation, kernel_size=7, dropout=0.1, use_se=True):
        super().__init__()
        padding = (kernel_size - 1) // 2 * dilation
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.se = SEBlock(channels) if use_se else nn.Identity()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        identity = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.drop(out)
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out = self.act(out + identity)
        return out


# =========================================================
# 电流编码器 1：ResDilaSE
# =========================================================

class ResDilaSECurrentEncoder(nn.Module):
    def __init__(self, feature_dim=256, base_channels=64,
                 dilations=(1, 2, 4, 8, 16, 32), dropout=0.1):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, base_channels, 9, padding=4),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(),
            nn.Conv1d(base_channels, base_channels, 9, padding=4),
            nn.BatchNorm1d(base_channels),
            nn.ReLU()
        )
        blocks = [
            ResidualDilatedBlock(base_channels, d, kernel_size=7, dropout=dropout, use_se=True)
            for d in dilations
        ]
        self.blocks = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(base_channels, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

    def forward(self, x, return_feature=False):
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [B,1,L]
        x = self.stem(x)
        x = self.blocks(x)
        pooled = self.pool(x).squeeze(-1)   # [B,C]
        feat = self.proj(pooled)            # [B,feature_dim]
        if return_feature:
            return feat, pooled
        return feat


# =========================================================
# 电流编码器 2：ConvFormer (轻量版)
# =========================================================

class PatchEmbedding1D(nn.Module):
    def __init__(self, in_ch=1, embed_dim=128, patch_size=32, stride=32):
        super().__init__()
        self.proj = nn.Conv1d(
            in_ch, embed_dim, kernel_size=patch_size, stride=stride, padding=patch_size // 4
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)       # [B,E,Lp]
        x = x.transpose(1, 2)  # [B,Lp,E]
        x = self.norm(x)
        return x


class SimpleSpectralTokens(nn.Module):
    def __init__(self, num_scales=3, top_k=64, embed_dim=128):
        super().__init__()
        self.num_scales = num_scales
        self.top_k = top_k
        self.proj = nn.Linear(num_scales * top_k, embed_dim)

    def forward(self, x):
        # x: [B,1,L]
        feats = []
        for s in range(self.num_scales):
            pool = 2 ** s
            if pool > 1:
                p = F.avg_pool1d(x, pool, pool)
            else:
                p = x
            fft = torch.fft.rfft(p, dim=-1)
            mag = torch.abs(fft)[:, :, :self.top_k]
            feats.append(mag)
        feat = torch.cat(feats, dim=-1).squeeze(1)  # [B, num_scales*top_k]
        tok = self.proj(feat).unsqueeze(1)          # [B,1,E]
        return tok


class ConvFormerCurrentEncoder(nn.Module):
    def __init__(self, feature_dim=256, embed_dim=128, depth=4,
                 num_heads=8, mlp_ratio=2.0, patch_size=32, stride=32,
                 spectral_tokens=True, spectral_top_k=64, dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding1D(1, embed_dim, patch_size, stride)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, 512, embed_dim) * 0.02)
        self.use_spectral = spectral_tokens
        if spectral_tokens:
            self.spectral = SimpleSpectralTokens(3, spectral_top_k, embed_dim)

        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=int(embed_dim * mlp_ratio),
                dropout=dropout,
                activation='gelu',
                batch_first=True,
                norm_first=True
            ) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        nn.init.normal_(self.cls_token, std=0.02)

    def forward(self, x, return_feature=False):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        tok = self.patch_embed(x)
        if self.use_spectral:
            tok = torch.cat([self.spectral(x), tok], dim=1)
        B, Lp, E = tok.shape
        cls = self.cls_token.expand(B, -1, -1)
        tok = torch.cat([cls, tok], dim=1)
        if tok.size(1) > self.pos_embed.size(1):
            pos = F.interpolate(
                self.pos_embed.transpose(1, 2),
                size=tok.size(1),
                mode='linear',
                align_corners=False
            ).transpose(1, 2)
        else:
            pos = self.pos_embed[:, :tok.size(1)]
        x = tok + pos
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        cls_feat = x[:, 0]
        feat = self.head(cls_feat)
        if return_feature:
            return feat, cls_feat
        return feat


# =========================================================
# 工厂函数
# =========================================================

def build_current_encoder(kind: str,
                          feature_dim: int,
                          **kwargs):
    k = kind.lower()
    if k in ["res", "resdila", "resdilase"]:
        return ResDilaSECurrentEncoder(
            feature_dim=feature_dim,
            base_channels=kwargs.get("base_channels", 64),
            dilations=kwargs.get("dilations", (1, 2, 4, 8, 16, 32)),
            dropout=kwargs.get("dropout", 0.1)
        )
    elif k in ["convformer", "ct", "convtransformer"]:
        return ConvFormerCurrentEncoder(
            feature_dim=feature_dim,
            embed_dim=kwargs.get("embed_dim", 128),
            depth=kwargs.get("depth", 4),
            num_heads=kwargs.get("num_heads", 8),
            mlp_ratio=kwargs.get("mlp_ratio", 2.0),
            patch_size=kwargs.get("patch_size", 32),
            stride=kwargs.get("stride", 32),
            spectral_tokens=kwargs.get("spectral_tokens", True),
            spectral_top_k=kwargs.get("spectral_top_k", 64),
            dropout=kwargs.get("dropout", 0.1)
        )
    else:
        raise ValueError(f"未知电流编码器类型: {kind}")