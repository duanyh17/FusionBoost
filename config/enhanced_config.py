"""
Enhanced configuration with current modality emphasis
"""

from config.config import Config


class EnhancedConfig(Config):
    def __init__(self):
        super().__init__()

        # Current modality enhancement parameters
        self.current_emphasis_factor = 2.5  # Boost current importance
        self.balanced_missing_prob = 0.15  # Probability of missing modalities during training

        # Enhanced current encoder parameters
        self.current_multi_scale = True
        self.current_domain_features = True
        self.current_attention = True

        # Balanced fusion parameters
        self.modality_importance_balancing = True
        self.cross_modality_attention = True
        self.current_boosting_factor = 1.5

        # Training enhancements
        self.current_reconstruction_loss_weight = 0.05
        self.entropy_regularization_weight = 0.1
        self.current_emphasis_loss_weight = 0.1

        # Performance tracking
        self.track_modality_performance = True
        self.adaptive_importance_balancing = True

        # Stage-specific current emphasis
        self.stage_current_emphasis = {
            1: 1.0,  # Base level
            2: 1.5,  # Increased in knowledge distillation
            3: 2.0  # Maximum in TLA stage
        }