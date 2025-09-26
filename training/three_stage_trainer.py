"""
Enhanced three-stage trainer with Windows console compatibility
Removed emoji characters and added comprehensive debug information
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
import numpy as np
import logging
import os
import time
from tqdm import tqdm
from typing import Dict, Tuple, Optional
import sys

class ImprovedEarlyStopping:
    """Enhanced early stopping with better patience"""

    def __init__(self, patience: int = 15, min_delta: float = 1e-4, monitor: str = 'val_acc'):
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor

        self.counter = 0
        self.best_score = None
        self.should_stop = False

        # Determine comparison operation
        if monitor == 'val_acc':
            self.monitor_op = np.greater
            self.min_delta *= 1
        else:
            self.monitor_op = np.less
            self.min_delta *= -1

    def __call__(self, current_score: float) -> bool:
        """Check if training should stop"""

        if self.best_score is None:
            self.best_score = current_score
        elif self.monitor_op(current_score, self.best_score + self.min_delta):
            self.best_score = current_score
            self.counter = 0
            logging.info(f"[BEST] New best {self.monitor}: {current_score:.4f}")
        else:
            self.counter += 1
            logging.info(f"[EARLY_STOP] Counter: {self.counter}/{self.patience}")

            if self.counter >= self.patience:
                self.should_stop = True
                logging.info(f"[EARLY_STOP] Training stopped after {self.counter} epochs without improvement")

        return self.should_stop

class ThreeStageTrainer:
    """Enhanced trainer with Windows compatibility and debug information"""

    def __init__(self, model, config, device):
        self.model = model
        self.config = config
        self.device = device

        # Fix GradScaler deprecation warning
        self.scaler = GradScaler('cuda', enabled=config.mixed_precision)

        # Setup logging with ASCII-only format
        self.setup_windows_compatible_logging()

        # Debug counters
        self.total_batches_processed = 0
        self.total_samples_processed = 0
        self.stage_start_time = None

        logging.info("[INIT] ThreeStageTrainer initialized successfully")

    def setup_windows_compatible_logging(self):
        """Setup logging that works with Windows console encoding"""

        # Configure logging with ASCII-only format
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # Update existing handlers to use ASCII-safe format
        for handler in logging.getLogger().handlers:
            handler.setFormatter(formatter)

    def setup_stage(self, stage: int):
        """Setup training for specific stage with detailed logging"""

        logging.info(f"[STAGE_{stage}] Setting up stage {stage} training")

        # Set model stage
        try:
            self.model.set_stage(stage)
            logging.info(f"[STAGE_{stage}] Model stage set successfully")
        except Exception as e:
            logging.error(f"[STAGE_{stage}] Error setting model stage: {e}")
            raise e

        # Setup optimizer
        try:
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
            logging.info(f"[STAGE_{stage}] Optimizer configured - LR: {self.config.learning_rate}")
        except Exception as e:
            logging.error(f"[STAGE_{stage}] Error setting up optimizer: {e}")
            raise e

        # Stage-specific early stopping patience
        patience_map = {1: 15, 2: 20, 3: 12}
        patience = patience_map.get(stage, 15)

        try:
            self.early_stopping = ImprovedEarlyStopping(
                patience=patience,
                min_delta=self.config.min_delta,
                monitor='val_acc'
            )
            logging.info(f"[STAGE_{stage}] Early stopping configured - Patience: {patience}")
        except Exception as e:
            logging.error(f"[STAGE_{stage}] Error setting up early stopping: {e}")
            raise e

        # Setup loss function
        try:
            self.criterion = nn.CrossEntropyLoss()
            logging.info(f"[STAGE_{stage}] Loss function configured")
        except Exception as e:
            logging.error(f"[STAGE_{stage}] Error setting up loss function: {e}")
            raise e

        # Reset debug counters
        self.total_batches_processed = 0
        self.total_samples_processed = 0
        self.stage_start_time = time.time()

        logging.info(f"[STAGE_{stage}] Setup completed successfully")

    def compute_stage_specific_loss(self, model_output: Dict, labels: torch.Tensor, stage: int) -> Tuple[torch.Tensor, Dict]:
        """Compute loss with proper error handling and debug info"""

        # Classification loss (always present)
        try:
            cls_loss = self.criterion(model_output['logits'], labels)
            total_loss = cls_loss
            loss_components = {'cls_loss': cls_loss.item()}

            logging.debug(f"[LOSS] Stage {stage} - Classification loss: {cls_loss.item():.4f}")

        except Exception as e:
            logging.error(f"[LOSS] Error computing classification loss: {e}")
            raise e

        try:
            if stage == 1:
                # Stage 1: Bottleneck regularization
                fusion_weights = model_output.get('fusion_weights', None)
                if fusion_weights is not None:
                    bottleneck_reg = torch.mean(torch.sum(fusion_weights ** 2, dim=1))
                    reg_loss = 0.001 * bottleneck_reg
                    total_loss += reg_loss
                    loss_components['bottleneck_reg'] = bottleneck_reg.item()
                    logging.debug(f"[LOSS] Stage 1 - Bottleneck reg: {bottleneck_reg.item():.6f}")

            elif stage == 2:
                # Stage 2: Knowledge distillation + fusion regularization
                kd_loss = model_output.get('kd_loss', torch.tensor(0.0, device=self.device))
                kd_weighted = 0.4 * kd_loss
                total_loss += kd_weighted
                loss_components['kd_loss'] = kd_loss.item()

                # Fusion regularization
                fusion_weights = model_output.get('fusion_weights', None)
                if fusion_weights is not None:
                    eps = 1e-8
                    entropy = -torch.sum(fusion_weights * torch.log(fusion_weights + eps), dim=1)
                    fusion_reg = -torch.mean(entropy)
                    fusion_weighted = 0.1 * fusion_reg
                    total_loss += fusion_weighted
                    loss_components['fusion_reg'] = fusion_reg.item()
                    logging.debug(f"[LOSS] Stage 2 - KD: {kd_loss.item():.4f}, Fusion: {fusion_reg.item():.4f}")

            elif stage == 3:
                # Stage 3: TLA + knowledge distillation
                kd_loss = model_output.get('kd_loss', torch.tensor(0.0, device=self.device))
                tla_loss = model_output.get('tla_loss', torch.tensor(0.0, device=self.device))

                kd_weighted = 0.3 * kd_loss
                tla_weighted = 0.2 * tla_loss
                total_loss += kd_weighted + tla_weighted

                loss_components['kd_loss'] = kd_loss.item()
                loss_components['tla_loss'] = tla_loss.item()
                logging.debug(f"[LOSS] Stage 3 - KD: {kd_loss.item():.4f}, TLA: {tla_loss.item():.4f}")

        except Exception as e:
            logging.warning(f"[LOSS] Warning in stage-specific loss computation: {e}")

        loss_components['total_loss'] = total_loss.item()
        return total_loss, loss_components

    def train_epoch(self, train_loader, stage: int) -> Tuple[float, float, Dict]:
        """Train for one epoch with comprehensive debug information"""

        logging.info(f"[TRAIN] Starting training epoch for stage {stage}")
        epoch_start_time = time.time()

        self.model.train()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        loss_components_sum = {}

        # Debug: Check data loader
        logging.info(f"[TRAIN] Total batches in loader: {len(train_loader)}")

        # Create progress bar with detailed info
        progress_bar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Stage {stage} Training",
            leave=False,
            ascii=True,  # Use ASCII characters for Windows compatibility
            file=sys.stdout
        )

        batch_times = []

        for batch_idx, batch_data in progress_bar:
            batch_start_time = time.time()

            try:
                # Debug: Log batch processing start
                if batch_idx == 0:
                    logging.info(f"[TRAIN] Processing first batch - Size: {len(batch_data['labels'])}")
                    logging.info(f"[TRAIN] Available modalities sample: {batch_data['available_modalities'][0] if batch_data['available_modalities'] else 'None'}")

                # Move data to device with error checking
                try:
                    for modality in batch_data['modalities']:
                        if batch_data['modalities'][modality] is not None:
                            batch_data['modalities'][modality] = batch_data['modalities'][modality].to(self.device, non_blocking=True)

                    labels = batch_data['labels'].to(self.device, non_blocking=True)

                except Exception as e:
                    logging.error(f"[TRAIN] Error moving batch {batch_idx} to device: {e}")
                    continue

                # Zero gradients
                self.optimizer.zero_grad()

                # Forward pass with error checking
                try:
                    with autocast('cuda', enabled=self.config.mixed_precision):
                        model_output = self.model(batch_data, labels)

                        # Debug: Check model output keys
                        if batch_idx == 0:
                            logging.info(f"[TRAIN] Model output keys: {list(model_output.keys())}")

                        loss, loss_components = self.compute_stage_specific_loss(model_output, labels, stage)

                except Exception as e:
                    logging.error(f"[TRAIN] Error in forward pass for batch {batch_idx}: {e}")
                    continue

                # Backward pass with error checking
                try:
                    self.scaler.scale(loss).backward()

                    # Gradient clipping
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                except Exception as e:
                    logging.error(f"[TRAIN] Error in backward pass for batch {batch_idx}: {e}")
                    continue

                # Update beta for dynamic fusion
                if stage >= 2:
                    try:
                        self.model.update_beta()
                    except Exception as e:
                        logging.warning(f"[TRAIN] Warning updating beta: {e}")

                # Track metrics
                total_loss += loss.item()
                predictions = torch.argmax(model_output['logits'], dim=1)
                batch_correct = (predictions == labels).sum().item()
                correct_predictions += batch_correct
                total_samples += labels.size(0)

                # Update debug counters
                self.total_batches_processed += 1
                self.total_samples_processed += labels.size(0)

                # Accumulate loss components
                for key, value in loss_components.items():
                    if key not in loss_components_sum:
                        loss_components_sum[key] = 0.0
                    loss_components_sum[key] += value

                # Calculate metrics
                current_acc = correct_predictions / total_samples
                current_loss = total_loss / (batch_idx + 1)
                batch_acc = batch_correct / labels.size(0)

                # Track batch processing time
                batch_time = time.time() - batch_start_time
                batch_times.append(batch_time)

                # Update progress bar with safe characters
                beta_value = self.model.beta.item() if hasattr(self.model, 'beta') else 0.0
                progress_bar.set_postfix({
                    'Loss': f'{current_loss:.4f}',
                    'Acc': f'{current_acc:.3f}',
                    'BatchAcc': f'{batch_acc:.3f}',
                    'Beta': f'{beta_value:.3f}',
                    'Time': f'{batch_time:.2f}s'
                })

                # Periodic detailed logging
                if (batch_idx + 1) % 50 == 0:
                    avg_batch_time = np.mean(batch_times[-50:])
                    logging.info(f"[TRAIN] Batch {batch_idx + 1}/{len(train_loader)} - "
                               f"Loss: {current_loss:.4f}, Acc: {current_acc:.4f}, "
                               f"BatchTime: {avg_batch_time:.2f}s")

                # Check if first batch completed successfully
                if batch_idx == 0:
                    logging.info(f"[TRAIN] First batch completed successfully in {batch_time:.2f}s")

            except Exception as e:
                logging.error(f"[TRAIN] Unexpected error in batch {batch_idx}: {e}")
                continue

        # Calculate final metrics
        avg_loss = total_loss / len(train_loader) if len(train_loader) > 0 else 0.0
        accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0
        avg_loss_components = {key: value / len(train_loader) for key, value in loss_components_sum.items()}

        # Log epoch summary
        epoch_time = time.time() - epoch_start_time
        avg_batch_time = np.mean(batch_times) if batch_times else 0.0

        logging.info(f"[TRAIN] Epoch completed - Time: {epoch_time:.2f}s, "
                    f"AvgBatchTime: {avg_batch_time:.2f}s, "
                    f"TotalSamples: {total_samples}")

        return avg_loss, accuracy, avg_loss_components

    def validate_epoch(self, val_loader, stage: int) -> Tuple[float, float]:
        """Validate for one epoch with debug information"""

        logging.info(f"[VALID] Starting validation for stage {stage}")
        validation_start_time = time.time()

        self.model.eval()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        progress_bar = tqdm(
            val_loader,
            desc=f"Stage {stage} Validation",
            leave=False,
            ascii=True,
            file=sys.stdout
        )

        with torch.no_grad():
            for batch_idx, batch_data in enumerate(progress_bar):
                try:
                    # Move data to device
                    for modality in batch_data['modalities']:
                        if batch_data['modalities'][modality] is not None:
                            batch_data['modalities'][modality] = batch_data['modalities'][modality].to(self.device, non_blocking=True)

                    labels = batch_data['labels'].to(self.device, non_blocking=True)

                    # Forward pass
                    with autocast('cuda', enabled=self.config.mixed_precision):
                        model_output = self.model(batch_data, labels)
                        loss, _ = self.compute_stage_specific_loss(model_output, labels, stage)

                    # Track metrics
                    total_loss += loss.item()
                    predictions = torch.argmax(model_output['logits'], dim=1)
                    correct_predictions += (predictions == labels).sum().item()
                    total_samples += labels.size(0)

                    # Update progress bar
                    current_acc = correct_predictions / total_samples
                    current_loss = total_loss / (batch_idx + 1)

                    progress_bar.set_postfix({
                        'Loss': f'{current_loss:.4f}',
                        'Acc': f'{current_acc:.3f}'
                    })

                except Exception as e:
                    logging.error(f"[VALID] Error in validation batch {batch_idx}: {e}")
                    continue

        avg_loss = total_loss / len(val_loader) if len(val_loader) > 0 else 0.0
        accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0

        validation_time = time.time() - validation_start_time
        logging.info(f"[VALID] Validation completed - Time: {validation_time:.2f}s, "
                    f"Samples: {total_samples}")

        return avg_loss, accuracy

    def train_stage(self, stage: int, train_loader, val_loader, epochs: int) -> Dict:
        """Train a specific stage with comprehensive logging"""

        logging.info(f"[STAGE_{stage}] Starting stage {stage} training")
        logging.info(f"[STAGE_{stage}] Target epochs: {epochs}")
        logging.info(f"[STAGE_{stage}] Train batches: {len(train_loader)}")
        logging.info(f"[STAGE_{stage}] Validation batches: {len(val_loader)}")

        # Setup stage
        try:
            self.setup_stage(stage)
        except Exception as e:
            logging.error(f"[STAGE_{stage}] Error in stage setup: {e}")
            raise e

        logging.info(f"[STAGE_{stage}] ====== Stage {stage} Training Started ======")

        best_val_acc = 0.0
        best_model_state = None
        stage_results = {
            'train_losses': [],
            'val_losses': [],
            'train_accs': [],
            'val_accs': []
        }

        for epoch in range(epochs):
            epoch_overall_start = time.time()

            try:
                logging.info(f"[STAGE_{stage}] ---- Epoch {epoch + 1}/{epochs} ----")

                # Training phase
                logging.info(f"[STAGE_{stage}] Starting training phase")
                train_loss, train_acc, train_loss_components = self.train_epoch(train_loader, stage)

                # Validation phase
                logging.info(f"[STAGE_{stage}] Starting validation phase")
                val_loss, val_acc = self.validate_epoch(val_loader, stage)

                # Track metrics
                stage_results['train_losses'].append(train_loss)
                stage_results['val_losses'].append(val_loss)
                stage_results['train_accs'].append(train_acc)
                stage_results['val_accs'].append(val_acc)

                # Calculate epoch time
                epoch_time = time.time() - epoch_overall_start

                # Log comprehensive results
                logging.info(f"[STAGE_{stage}] Epoch {epoch + 1} Results:")
                logging.info(f"[STAGE_{stage}]   Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
                logging.info(f"[STAGE_{stage}]   Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
                logging.info(f"[STAGE_{stage}]   Epoch Time: {epoch_time:.2f}s")
                logging.info(f"[STAGE_{stage}]   Loss Components: {train_loss_components}")

                # Save best model
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_model_state = self.model.state_dict().copy()

                    # Save checkpoint
                    try:
                        os.makedirs(self.config.model_save_dir, exist_ok=True)
                        model_path = os.path.join(
                            self.config.model_save_dir,
                            f"best_stage_{stage}_epoch_{epoch + 1}_acc_{val_acc:.4f}.pth"
                        )
                        torch.save({
                            'model_state_dict': best_model_state,
                            'epoch': epoch + 1,
                            'val_acc': val_acc,
                            'stage': stage,
                            'train_loss': train_loss,
                            'val_loss': val_loss
                        }, model_path)
                        logging.info(f"[STAGE_{stage}] New best model saved: {model_path}")
                    except Exception as e:
                        logging.error(f"[STAGE_{stage}] Error saving model: {e}")

                # Early stopping check
                if self.early_stopping(val_acc):
                    logging.info(f"[STAGE_{stage}] Early stopping triggered at epoch {epoch + 1}")
                    break

            except Exception as e:
                logging.error(f"[STAGE_{stage}] Error in epoch {epoch + 1}: {e}")
                continue

        # Load best model
        if best_model_state is not None:
            try:
                self.model.load_state_dict(best_model_state)
                logging.info(f"[STAGE_{stage}] Best model loaded successfully")
            except Exception as e:
                logging.error(f"[STAGE_{stage}] Error loading best model: {e}")

        # Calculate stage summary
        stage_time = time.time() - self.stage_start_time
        epochs_completed = len(stage_results['train_losses'])

        logging.info(f"[STAGE_{stage}] ====== Stage {stage} Completed ======")
        logging.info(f"[STAGE_{stage}] Best validation accuracy: {best_val_acc:.4f}")
        logging.info(f"[STAGE_{stage}] Epochs completed: {epochs_completed}/{epochs}")
        logging.info(f"[STAGE_{stage}] Total stage time: {stage_time:.2f}s")
        logging.info(f"[STAGE_{stage}] Total batches processed: {self.total_batches_processed}")
        logging.info(f"[STAGE_{stage}] Total samples processed: {self.total_samples_processed}")

        # Update results dictionary
        stage_results.update({
            'best_val_acc': best_val_acc,
            'epochs_trained': epochs_completed,
            'stage_time': stage_time,
            'total_batches_processed': self.total_batches_processed,
            'total_samples_processed': self.total_samples_processed
        })

        return stage_results

    def train_three_stages(self, train_loader, val_loader) -> str:
        """Execute complete three-stage training pipeline with comprehensive logging"""

        pipeline_start_time = time.time()

        logging.info("=" * 60)
        logging.info("[PIPELINE] Starting Three-Stage Training Pipeline")
        logging.info("=" * 60)
        logging.info(f"[PIPELINE] Stage 1: {self.config.stage1_epochs} epochs (Pre-training)")
        logging.info(f"[PIPELINE] Stage 2: {self.config.stage2_epochs} epochs (Knowledge Distillation)")
        logging.info(f"[PIPELINE] Stage 3: {self.config.stage3_epochs} epochs (TLA Activation)")
        logging.info(f"[PIPELINE] Total planned epochs: {self.config.stage1_epochs + self.config.stage2_epochs + self.config.stage3_epochs}")

        stage_results = {}

        try:
            # Stage 1: Pre-training with bottleneck tokens
            logging.info("[PIPELINE] Starting Stage 1: Pre-training")
            stage_results[1] = self.train_stage(1, train_loader, val_loader, self.config.stage1_epochs)

            # Stage 2: Knowledge distillation + dynamic fusion
            logging.info("[PIPELINE] Starting Stage 2: Knowledge Distillation")
            stage_results[2] = self.train_stage(2, train_loader, val_loader, self.config.stage2_epochs)

            # Stage 3: TLA activation
            logging.info("[PIPELINE] Starting Stage 3: TLA Activation")
            stage_results[3] = self.train_stage(3, train_loader, val_loader, self.config.stage3_epochs)

        except Exception as e:
            logging.error(f"[PIPELINE] Critical error in three-stage training: {e}")
            raise e

        # Save final model
        try:
            os.makedirs(self.config.model_save_dir, exist_ok=True)
            final_model_path = os.path.join(self.config.model_save_dir, "final_three_stage_model.pth")
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'stage_results': stage_results,
                'config': self.config,
                'pipeline_time': time.time() - pipeline_start_time
            }, final_model_path)
            logging.info(f"[PIPELINE] Final model saved: {final_model_path}")
        except Exception as e:
            logging.error(f"[PIPELINE] Error saving final model: {e}")
            final_model_path = "ERROR_SAVING_MODEL"

        # Print comprehensive summary
        pipeline_time = time.time() - pipeline_start_time
        total_epochs_trained = sum(results['epochs_trained'] for results in stage_results.values())

        logging.info("=" * 60)
        logging.info("[PIPELINE] THREE-STAGE TRAINING COMPLETED")
        logging.info("=" * 60)
        logging.info(f"[PIPELINE] Total pipeline time: {pipeline_time:.2f}s ({pipeline_time/60:.1f} minutes)")
        logging.info(f"[PIPELINE] Total epochs trained: {total_epochs_trained}")

        for stage, results in stage_results.items():
            logging.info(f"[PIPELINE] Stage {stage}: Best Acc = {results['best_val_acc']:.4f} "
                        f"({results['epochs_trained']} epochs, {results['stage_time']:.1f}s)")

        logging.info(f"[PIPELINE] Final model path: {final_model_path}")
        logging.info("[PIPELINE] Training pipeline completed successfully!")

        return final_model_path