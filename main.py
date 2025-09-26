"""
Main training script for three-stage multi-modal welding quality detection
Fixed for Windows compatibility and proper error handling
"""

import torch
import torch.backends.cudnn as cudnn
import numpy as np
import random
import logging
import os
import platform
from pathlib import Path

# Import your modules
from config.config import Config
from data.dataset import MultiModalWeldingDataset
from models.bottleneck_fusion import ThreeStageWeldingFusion
from training.three_stage_trainer import ThreeStageTrainer

def setup_logging(config):
    """Setup logging configuration"""

    os.makedirs(config.log_dir, exist_ok=True)

    logging.basicConfig(
        level=getattr(logging, config.log_level),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(config.log_dir, 'training.log')),
            logging.StreamHandler()
        ]
    )

def set_seed(seed):
    """Set random seeds for reproducibility"""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

def get_device():
    """Get available device with proper CUDA setup"""

    if torch.cuda.is_available():
        device = torch.device('cuda')
        logging.info(f"Using device: {device}")
        logging.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logging.info(f"CUDA Version: {torch.version.cuda}")
    else:
        device = torch.device('cpu')
        logging.info("CUDA not available, using CPU")

    return device

def count_parameters(model):
    """Count model parameters"""

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return total_params, trainable_params

def main():
    """Main training function with comprehensive error handling"""

    try:
        # System information
        print(f"System: {platform.system()} {platform.release()}")
        print(f"Python: {platform.python_version()}")
        print(f"PyTorch: {torch.__version__}")

        # Load configuration
        config = Config()

        # Setup logging
        setup_logging(config)
        logging.info("Starting Multi-Modal Welding Quality Detection Training")

        # Set random seed
        set_seed(config.seed)

        # Get device
        device = get_device()

        # Initialize dataset
        logging.info("Initializing dataset...")
        try:
            dataset = MultiModalWeldingDataset(config.data_root, config, split='train')
            train_loader, val_loader, test_loader = dataset.get_data_loaders()

            logging.info(f"Dataset loaded: {len(dataset.samples)} total samples")
            logging.info(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")

        except Exception as e:
            logging.error(f"Error loading dataset: {e}")
            raise e

        # Initialize model
        logging.info("Initializing three-stage fusion model...")
        try:
            model = ThreeStageWeldingFusion(config).to(device)

            # Count parameters
            total_params, trainable_params = count_parameters(model)
            logging.info(f"Model initialized:")
            logging.info(f"  Total parameters: {total_params:,}")
            logging.info(f"  Trainable parameters: {trainable_params:,}")

        except Exception as e:
            logging.error(f"Error initializing model: {e}")
            raise e

        # Initialize trainer
        try:
            trainer = ThreeStageTrainer(model, config, device)
        except Exception as e:
            logging.error(f"Error initializing trainer: {e}")
            raise e

        # Start training
        if config.do_train:
            logging.info("Starting three-stage training...")
            try:
                best_model_path = trainer.train_three_stages(train_loader, val_loader)
                logging.info(f"Training completed! Best model saved at: {best_model_path}")

            except Exception as e:
                logging.error(f"Error during training: {e}")
                raise e

        # Evaluation
        if config.do_eval:
            logging.info("Starting evaluation...")
            # Add evaluation code here if needed

        logging.info("All processes completed successfully!")  # Removed emoji

    except KeyboardInterrupt:
        logging.info("Training interrupted by user")
        return

    except Exception as e:
        logging.error(f"Error in main execution: {e}")
        raise e

if __name__ == "__main__":
    main()