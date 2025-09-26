"""
Enhanced three-stage fusion model with proper interface
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Dict, List, Optional, Tuple
import logging

class ThreeStageWeldingFusion(nn.Module):
    """Complete three-stage fusion model with proper interface"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.current_stage = 1

        # Build encoders
        self._build_encoders()

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(config.feature_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_dim, config.num_classes)
        )

        # Dynamic fusion components
        self.fusion_weights_net = nn.Sequential(
            nn.Linear(config.feature_dim * config.num_modalities, 128),
            nn.ReLU(),
            nn.Linear(128, config.num_modalities),
            nn.Softmax(dim=-1)
        )

        # Beta parameter for dynamic weighting
        self.register_buffer('beta', torch.tensor(config.beta_init))

        # TLA tracking
        self.register_buffer('combination_performance', torch.zeros(2 ** config.num_modalities))
        self.register_buffer('combination_counts', torch.zeros(2 ** config.num_modalities))

        logging.info(f"Model initialized with {sum(p.numel() for p in self.parameters())} parameters")

    def _build_encoders(self):
        """Build modality-specific encoders"""

        # Image encoder (MobileNetV3-Small) - fix deprecation warning
        mobilenet = models.mobilenet_v3_small(weights='IMAGENET1K_V1')
        self.image_encoder = nn.Sequential(*list(mobilenet.children())[:-1])

        self.image_projection = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(576, self.config.feature_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # Sound encoder (1D CNN)
        self.sound_encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),

            nn.Flatten(),
            nn.Linear(128, self.config.feature_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # Current encoder (1D CNN)
        self.current_encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),

            nn.Flatten(),
            nn.Linear(128, self.config.feature_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

    # Replace the set_stage method in your model file:

    def set_stage(self, stage: int):
        """Set current training stage - Windows compatible"""
        self.current_stage = stage
        logging.info(f"[MODEL] Model set to Stage {stage}")  # Removed emoji

    def encode_modalities(self, batch_data: Dict) -> Dict[str, torch.Tensor]:
        """Encode available modalities"""

        encoded_features = {}

        # Process each modality if available
        for modality in self.config.modalities:
            if modality in batch_data['modalities']:
                data = batch_data['modalities'][modality]

                # Check if modality data is meaningful (not all zeros)
                if torch.sum(torch.abs(data)) > 1e-6:
                    if modality == 'image':
                        features = self.image_encoder(data)
                        features = self.image_projection(features)
                    elif modality == 'sound':
                        if len(data.shape) == 2:
                            data = data.unsqueeze(1)
                        features = self.sound_encoder(data)
                    elif modality == 'current':
                        if len(data.shape) == 2:
                            data = data.unsqueeze(1)
                        features = self.current_encoder(data)

                    encoded_features[modality] = features

        return encoded_features

    def generate_bottleneck_tokens(self, available_features: Dict[str, torch.Tensor],
                                  missing_modalities: List[str]) -> Dict[str, torch.Tensor]:
        """Generate bottleneck tokens for missing modalities"""

        if not missing_modalities or not available_features:
            return available_features

        batch_size = list(available_features.values())[0].size(0)
        device = list(available_features.values())[0].device

        # Simple bottleneck token: mean of available features
        if available_features:
            available_stack = torch.stack(list(available_features.values()), dim=1)
            bottleneck_token = torch.mean(available_stack, dim=1)
        else:
            bottleneck_token = torch.zeros(batch_size, self.config.feature_dim, device=device)

        # Add bottleneck tokens for missing modalities
        complete_features = available_features.copy()
        for modality in missing_modalities:
            complete_features[modality] = bottleneck_token

        return complete_features

    def dynamic_weighted_fusion(self, features: Dict[str, torch.Tensor],
                               available_modalities: List[List[str]],
                               beta: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """Dynamic weighted fusion using your formula"""

        batch_size = list(features.values())[0].size(0)
        device = list(features.values())[0].device

        # Stack all features
        feature_stack = torch.stack([features[mod] for mod in self.config.modalities], dim=1)

        # Compute confidence scores (simple version)
        confidences = torch.sigmoid(self.fusion_weights_net(feature_stack.flatten(1, 2)))

        # Apply your dynamic weighting formula: ω_i = 1 / (1 + exp(β * Σ(1 - p_n)))
        fusion_weights = torch.zeros_like(confidences)

        for i in range(self.config.num_modalities):
            # Sum of (1 - confidence) for other modalities
            other_indices = [j for j in range(self.config.num_modalities) if j != i]
            sum_other_unreliable = torch.sum(1.0 - confidences[:, other_indices], dim=1)

            # Apply your formula
            fusion_weights[:, i] = 1.0 / (1.0 + torch.exp(beta * sum_other_unreliable))

        # Apply availability mask
        availability_mask = torch.zeros_like(fusion_weights)
        for batch_idx, available_mods in enumerate(available_modalities):
            for mod_idx, modality in enumerate(self.config.modalities):
                if modality in available_mods:
                    availability_mask[batch_idx, mod_idx] = 1.0

        fusion_weights = fusion_weights * availability_mask
        fusion_weights = fusion_weights / (fusion_weights.sum(dim=1, keepdim=True) + 1e-8)

        # Weighted fusion
        fused_features = torch.sum(fusion_weights.unsqueeze(-1) * feature_stack, dim=1)

        return fused_features, fusion_weights

    def get_combination_index(self, available_modalities: List[str]) -> int:
        """Convert available modalities to combination index"""
        index = 0
        for i, modality in enumerate(self.config.modalities):
            if modality in available_modalities:
                index += 2 ** i
        return index

    def update_tla_tracking(self, available_modalities_batch: List[List[str]],
                           predictions: torch.Tensor, labels: torch.Tensor):
        """Update TLA performance tracking"""

        if not self.training:
            return

        with torch.no_grad():
            correct = (predictions.argmax(dim=1) == labels).float()

            for available_mods, is_correct in zip(available_modalities_batch, correct):
                combo_idx = self.get_combination_index(available_mods)

                if combo_idx < len(self.combination_performance):
                    # Exponential moving average
                    alpha = 0.1
                    if self.combination_counts[combo_idx] > 0:
                        self.combination_performance[combo_idx] = (
                            (1 - alpha) * self.combination_performance[combo_idx] +
                            alpha * is_correct
                        )
                    else:
                        self.combination_performance[combo_idx] = is_correct

                    self.combination_counts[combo_idx] += 1

    def forward(self, batch_data: Dict, labels: Optional[torch.Tensor] = None) -> Dict:
        """
        Forward pass through three-stage model
        CRITICAL FIX: Always return all required keys
        """

        # Encode available modalities
        available_features = self.encode_modalities(batch_data)

        # Determine missing modalities
        all_missing = set()
        available_modalities_batch = batch_data.get('available_modalities', [])

        for available_mods in available_modalities_batch:
            missing_mods = [mod for mod in self.config.modalities if mod not in available_mods]
            all_missing.update(missing_mods)

        # Stage 1: Generate bottleneck tokens for missing modalities
        complete_features = self.generate_bottleneck_tokens(available_features, list(all_missing))

        # Stage 2+: Dynamic weighted fusion
        if self.current_stage >= 2:
            fused_features, fusion_weights = self.dynamic_weighted_fusion(
                complete_features,
                available_modalities_batch,
                beta=float(self.beta)
            )
        else:
            # Simple average fusion for Stage 1
            if complete_features:
                feature_stack = torch.stack(list(complete_features.values()), dim=1)
                fused_features = torch.mean(feature_stack, dim=1)
                fusion_weights = torch.ones(fused_features.size(0), self.config.num_modalities,
                                          device=fused_features.device) / self.config.num_modalities
            else:
                batch_size = batch_data['labels'].size(0)
                device = batch_data['labels'].device
                fused_features = torch.zeros(batch_size, self.config.feature_dim, device=device)
                fusion_weights = torch.zeros(batch_size, self.config.num_modalities, device=device)

        # Classification
        logits = self.classifier(fused_features)

        # Base output dictionary - ALWAYS include all keys
        output = {
            'logits': logits,
            'fused_features': fused_features,
            'fusion_weights': fusion_weights,
            'confidences': fusion_weights,  # Use fusion weights as confidence proxy
            'beta': float(self.beta),
            'available_features': available_features,
            'complete_features': complete_features,
            'lazy_indicators': torch.zeros(fused_features.size(0), device=fused_features.device),  # CRITICAL FIX
            'tla_loss': torch.tensor(0.0, device=fused_features.device),  # CRITICAL FIX
            'kd_loss': torch.tensor(0.0, device=fused_features.device)  # CRITICAL FIX
        }

        # Stage 3: TLA mechanism
        if self.current_stage >= 3:
            # Generate lazy indicators based on combination performance
            lazy_indicators = torch.zeros(fused_features.size(0), device=fused_features.device)

            for batch_idx, available_mods in enumerate(available_modalities_batch):
                combo_idx = self.get_combination_index(available_mods)
                if combo_idx < len(self.combination_performance) and self.combination_counts[combo_idx] > 0:
                    performance = self.combination_performance[combo_idx]
                    lazy_indicators[batch_idx] = float(performance < 0.6)  # Lazy threshold

            output['lazy_indicators'] = lazy_indicators

            # Simple TLA loss
            tla_loss = torch.mean(lazy_indicators * torch.sigmoid(fusion_weights.sum(dim=1)))
            output['tla_loss'] = tla_loss

            # Update tracking if labels available
            if labels is not None and self.training:
                self.update_tla_tracking(available_modalities_batch, logits, labels)

        # Stage 2+: Knowledge distillation loss (simplified)
        if self.current_stage >= 2:
            # Simple KD loss based on confidence
            kd_loss = torch.mean(torch.sum(fusion_weights * torch.log(fusion_weights + 1e-8), dim=1))
            output['kd_loss'] = kd_loss

        return output

    def update_beta(self, decay_factor: float = None):
        """Update beta parameter"""
        if decay_factor is None:
            decay_factor = getattr(self.config, 'beta_decay', 0.95)

        new_beta = max(self.beta * decay_factor, getattr(self.config, 'beta_min', 0.1))
        self.beta.fill_(new_beta)