"""
Updated configuration with Windows compatibility and better early stopping
"""

import os
import platform

class Config:
    def __init__(self):
        # System detection
        self.is_windows = platform.system() == 'Windows'

        # Dataset configuration
        self.data_root = r"E:\11_weld_data\dataset"

        # Class information
        self.class_names = [
            'burn_through',        # 4304 samples
            'lack_of_penetration', # 6983 samples
            'misalignment',        # 3654 samples
            'normal',              # 6376 samples
            'over_penetration',    # 6950 samples
            'stomata'              # 4706 samples
        ]
        self.num_classes = len(self.class_names)

        # Modality configuration
        self.modalities = ['image', 'sound', 'current']
        self.num_modalities = len(self.modalities)

        # Data preprocessing
        self.image_size = (224, 224)
        self.audio_length = 8000
        self.audio_freq = 16000   # Reduced for stability
        self.current_length = 4000

        # Data splits
        self.train_ratio = 0.7
        self.val_ratio = 0.15
        self.test_ratio = 0.15

        # Model architecture
        self.feature_dim = 256
        self.bottleneck_dim = 128
        self.hidden_dim = 512
        self.dropout_rate = 0.3
        self.num_bottleneck_tokens = 8

        # Three-stage training configuration
        self.stage1_epochs = 40
        self.stage2_epochs = 50
        self.stage3_epochs = 30

        # Training parameters
        self.batch_size = 32
        self.learning_rate = 1e-3
        self.weight_decay = 1e-4

        # CRITICAL FIX: Improved early stopping
        self.min_delta = 1e-4
        self.monitor_metric = 'val_acc'

        # CRITICAL FIX: Windows compatibility
        if self.is_windows:
            print("🪟 Windows detected - Optimizing for single-threaded data loading")

        # Dynamic fusion parameters
        self.beta_init = 1.0
        self.beta_decay = 0.95
        self.beta_min = 0.1
        self.temperature = 4.0

        # TLA configuration
        self.tla_lazy_threshold = 0.6

        # Missing data simulation
        self.train_missing_prob = 0.1

        # Output directories
        self.output_dir = "outputs"
        self.model_save_dir = os.path.join(self.output_dir, "models")
        self.log_dir = os.path.join(self.output_dir, "logs")
        self.results_dir = os.path.join(self.output_dir, "results")
        self.visualization_dir = os.path.join(self.output_dir, "visualizations")

        # Create directories
        for dir_path in [self.output_dir, self.model_save_dir, self.log_dir,
                        self.results_dir, self.visualization_dir]:
            os.makedirs(dir_path, exist_ok=True)

        # Training control
        self.do_train = True
        self.do_eval = True
        self.save_model = True

        # System settings
        self.seed = 42
        self.mixed_precision = True

        # Logging
        self.log_level = "INFO"