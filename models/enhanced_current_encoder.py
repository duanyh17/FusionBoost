"""
Enhanced Current Encoder for FusionBoost
Direct replacement for the current encoder in bottleneck_fusion.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EnhancedCurrentEncoder(nn.Module):
    """
    增强版电流信号编码器 - 专门处理焊接电流的复杂时序特征

    主要特点:
    1. 多尺度时序卷积 - 捕获不同时间窗口的电流模式
    2. 通道注意力机制 - 自动关注重要特征通道
    3. 时序注意力机制 - 关注关键时间点
    4. 频域增强 - 结合频域特征分析
    5. 残差连接 - 保持梯度流动
    """

    def __init__(self, feature_dim=256):
        super().__init__()
        self.feature_dim = feature_dim

        # 1. 多尺度卷积分支 - 并行处理不同时间尺度
        self.multi_scale_convs = nn.ModuleList([
            # 短期特征 (kernel_size=3,5)
            self._create_conv_branch(1, 32, kernel_size=3),
            self._create_conv_branch(1, 32, kernel_size=5),
            # 中期特征 (kernel_size=7,11)
            self._create_conv_branch(1, 24, kernel_size=7),
            self._create_conv_branch(1, 24, kernel_size=11),
            # 长期特征 (kernel_size=15,21)
            self._create_conv_branch(1, 16, kernel_size=15),
            self._create_conv_branch(1, 16, kernel_size=21)
        ])

        # 计算多尺度特征的总维度
        total_channels = 32 + 32 + 24 + 24 + 16 + 16  # 144

        # 2. 特征融合层
        self.feature_fusion = nn.Sequential(
            nn.Conv1d(total_channels, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True)
        )

        # 3. 通道注意力模块 - 关注重要特征通道
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(64, 16, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(16, 64, 1),
            nn.Sigmoid()
        )

        # 4. 时序注意力模块 - 关注重要时间点
        self.temporal_attention = nn.Sequential(
            nn.Conv1d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # 5. 频域分析分支
        self.freq_analyzer = FrequencyAnalyzer()

        # 6. 最终特征投影
        self.final_projection = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64 + 32, 256),  # 64(时域) + 32(频域)
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, feature_dim),
            nn.LayerNorm(feature_dim)
        )

        # 7. 可学习的电流增强因子
        self.current_boost_factor = nn.Parameter(torch.tensor(1.5))

    def _create_conv_branch(self, in_channels, out_channels, kernel_size):
        """创建单个卷积分支"""
        padding = kernel_size // 2
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2)
        )

    def forward(self, x):
        """
        前向传播
        Args:
            x: [batch_size, seq_len] 或 [batch_size, 1, seq_len]
        Returns:
            enhanced_features: [batch_size, feature_dim]
        """
        # 确保输入维度正确
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # [B, 1, seq_len]

        # 1. 多尺度特征提取
        multi_scale_features = []
        for conv_branch in self.multi_scale_convs:
            branch_features = conv_branch(x)  # [B, channels, reduced_seq_len]
            multi_scale_features.append(branch_features)

        # 对齐序列长度 (选择最小长度)
        min_length = min([feat.size(-1) for feat in multi_scale_features])
        aligned_features = []
        for feat in multi_scale_features:
            if feat.size(-1) > min_length:
                # 中心裁切
                start_idx = (feat.size(-1) - min_length) // 2
                feat = feat[:, :, start_idx:start_idx + min_length]
            aligned_features.append(feat)

        # 拼接多尺度特征
        combined_features = torch.cat(aligned_features, dim=1)  # [B, total_channels, seq_len]

        # 2. 特征融合
        fused_features = self.feature_fusion(combined_features)  # [B, 64, seq_len]

        # 3. 通道注意力
        channel_weights = self.channel_attention(fused_features)  # [B, 64, 1]
        channel_attended = fused_features * channel_weights  # [B, 64, seq_len]

        # 4. 时序注意力
        temporal_weights = self.temporal_attention(channel_attended)  # [B, 1, seq_len]
        temporal_attended = channel_attended * temporal_weights  # [B, 64, seq_len]

        # 5. 频域特征提取
        freq_features = self.freq_analyzer(x.squeeze(1))  # [B, 32]

        # 6. 时域特征聚合
        temporal_features = F.adaptive_avg_pool1d(temporal_attended, 1).squeeze(-1)  # [B, 64]

        # 7. 时频域特征融合
        combined_temporal_freq = torch.cat([temporal_features, freq_features], dim=1)  # [B, 96]

        # 8. 最终特征投影
        final_features = self.final_projection(
            combined_temporal_freq.unsqueeze(-1)
        ).squeeze(-1)  # [B, feature_dim]

        # 9. 应用电流增强因子
        enhanced_features = final_features * self.current_boost_factor

        return enhanced_features


class FrequencyAnalyzer(nn.Module):
    """频域分析模块"""

    def __init__(self):
        super().__init__()

        self.freq_encoder = nn.Sequential(
            nn.Linear(2048, 128),  # 假设输入长度为4096，FFT后为2048
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32)
        )

    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len] 时域信号
        Returns:
            freq_features: [batch_size, 32] 频域特征
        """
        # FFT变换
        fft_result = torch.fft.rfft(x, dim=-1)

        # 计算幅度谱
        magnitude_spectrum = torch.abs(fft_result)

        # 对数变换和归一化
        log_magnitude = torch.log(magnitude_spectrum + 1e-8)
        normalized_spectrum = F.normalize(log_magnitude, p=2, dim=-1)

        # 调整维度以匹配全连接层
        if normalized_spectrum.size(-1) > 2048:
            # 截断到2048
            normalized_spectrum = normalized_spectrum[:, :2048]
        elif normalized_spectrum.size(-1) < 2048:
            # 零填充到2048
            pad_size = 2048 - normalized_spectrum.size(-1)
            normalized_spectrum = F.pad(normalized_spectrum, (0, pad_size))

        # 频域特征提取
        freq_features = self.freq_encoder(normalized_spectrum)

        return freq_features