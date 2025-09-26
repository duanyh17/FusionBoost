"""
Enhanced trainer with current modality emphasis
"""

import torch
import torch.nn as nn
import torch.optim as optim
from training.three_stage_trainer import ThreeStageTrainer
import logging


class BalancedThreeStageTrainer(ThreeStageTrainer):
    """Enhanced trainer with modality balancing"""

    def __init__(self, model, config, device):
        super().__init__(model, config, device)

        # Current modality emphasis parameters
        self.current_emphasis_factor = getattr(config, 'current_emphasis_factor', 2.0)
        self.balanced_missing_prob = getattr(config, 'balanced_missing_prob', 0.15)

    def compute_enhanced_loss(self, model_output: dict, labels: torch.Tensor,
                              stage: int, available_modalities: list) -> tuple:
        """Enhanced loss computation with current emphasis"""

        # Base classification loss
        cls_loss = self.criterion(model_output['logits'], labels)
        total_loss = cls_loss
        loss_components = {'cls_loss': cls_loss.item()}

        # Modality balancing loss
        if 'fusion_weights' in model_output:
            fusion_weights = model_output['fusion_weights']

            # Encourage current utilization
            current_idx = self.config.modalities.index('current')
            current_weights = fusion_weights[:, current_idx]

            # Current emphasis loss (encourage higher current weights)
            current_emphasis_loss = -torch.mean(torch.log(current_weights + 1e-8))
            total_loss += 0.1 * current_emphasis_loss * self.current_emphasis_factor
            loss_components['current_emphasis'] = current_emphasis_loss.item()

            # Prevent complete dominance (entropy regularization)
            entropy = -torch.sum(fusion_weights * torch.log(fusion_weights + 1e-8), dim=1)
            entropy_reg = -torch.mean(entropy) * 0.1
            total_loss += entropy_reg
            loss_components['entropy_reg'] = entropy_reg.item()

        # Stage-specific enhancements
        if stage >= 2:
            # Current reconstruction loss
            if 'current' in model_output.get('available_features', {}):
                current_features = model_output['available_features']['current']
                if 'complete_features' in model_output and 'current' in model_output['complete_features']:
                    reconstructed_current = model_output['complete_features']['current']
                    current_recon_loss = F.mse_loss(reconstructed_current, current_features)
                    total_loss += 0.05 * current_recon_loss
                    loss_components['current_recon'] = current_recon_loss.item()

        # Update performance tracking
        if hasattr(self.model, 'adaptive_fusion') and hasattr(self.model.adaptive_fusion, 'importance_balancer'):
            self.model.adaptive_fusion.importance_balancer.update_performance_tracking(
                fusion_weights, model_output['logits'], labels
            )

        loss_components['total_loss'] = total_loss.item()
        return total_loss, loss_components

    def apply_balanced_missing_modalities(self, batch_data: dict) -> dict:
        """Apply balanced missing modality simulation during training"""

        if self.model.training and hasattr(self.config, 'balanced_missing_prob'):
            import random

            # Each sample has a chance of missing modalities
            batch_size = batch_data['labels'].size(0)

            for sample_idx in range(batch_size):
                if random.random() < self.balanced_missing_prob:
                    # Randomly choose modalities to remove (higher chance for non-current)
                    modalities_to_remove = []

                    # Current has lower probability of being removed
                    if random.random() < 0.1:  # 10% chance to remove current
                        modalities_to_remove.append('current')

                    # Other modalities have higher probability
                    if random.random() < 0.3:  # 30% chance to remove image
                        modalities_to_remove.append('image')

                    if random.random() < 0.3:  # 30% chance to remove sound
                        modalities_to_remove.append('sound')

                    # Apply removal
                    for modality in modalities_to_remove:
                        if modality in batch_data['modalities'] and batch_data['modalities'][modality] is not None:
                            batch_data['modalities'][modality][sample_idx] = torch.zeros_like(
                                batch_data['modalities'][modality][sample_idx]
                            )

                    # Update available modalities
                    if 'available_modalities' in batch_data:
                        original_available = batch_data['available_modalities'][sample_idx]
                        batch_data['available_modalities'][sample_idx] = [
                            mod for mod in original_available if mod not in modalities_to_remove
                        ]

        return batch_data

    def train_epoch(self, train_loader, stage: int):
        """Enhanced training epoch with balanced modality emphasis"""

        logging.info(f"[BALANCED_TRAIN] Starting enhanced training for stage {stage}")

        self.model.train()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        loss_components_sum = {}

        for batch_idx, batch_data in enumerate(train_loader):
            try:
                # Apply balanced missing modality simulation
                batch_data = self.apply_balanced_missing_modalities(batch_data)

                # Move data to device
                for modality in batch_data['modalities']:
                    if batch_data['modalities'][modality] is not None:
                        batch_data['modalities'][modality] = batch_data['modalities'][modality].to(self.device)

                labels = batch_data['labels'].to(self.device)

                # Zero gradients
                self.optimizer.zero_grad()

                # Forward pass
                with torch.cuda.amp.autocast(enabled=self.config.mixed_precision):
                    model_output = self.model(batch_data, labels)

                    # Enhanced loss computation
                    loss, loss_components = self.compute_enhanced_loss(
                        model_output, labels, stage, batch_data.get('available_modalities', [])
                    )

                # Backward pass
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()

                # Track metrics
                total_loss += loss.item()
                predictions = torch.argmax(model_output['logits'], dim=1)
                correct_predictions += (predictions == labels).sum().item()
                total_samples += labels.size(0)

                # Accumulate loss components
                for key, value in loss_components.items():
                    if key not in loss_components_sum:
                        loss_components_sum[key] = 0.0
                    loss_components_sum[key] += value

                # Log current fusion weights periodically
                if batch_idx % 100 == 0 and 'fusion_weights' in model_output:
                    fusion_weights = model_output['fusion_weights'].mean(0)
                    current_idx = self.config.modalities.index('current')
                    current_weight = fusion_weights[current_idx].item()
                    logging.info(f"[BALANCED_TRAIN] Batch {batch_idx} - Current fusion weight: {current_weight:.3f}")

            except Exception as e:
                logging.error(f"[BALANCED_TRAIN] Error in batch {batch_idx}: {e}")
                continue

        avg_loss = total_loss / len(train_loader)
        accuracy = correct_predictions / total_samples
        avg_loss_components = {key: value / len(train_loader) for key, value in loss_components_sum.items()}

        return avg_loss, accuracy, avg_loss_components