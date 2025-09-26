"""
Comprehensive missing modality testing framework for three-stage fusion model
Fixed for PyTorch 2.6 compatibility and Windows systems
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import logging
import os
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import time

# Import your modules
from config.config import Config
from data.dataset import MultiModalWeldingDataset
from models.bottleneck_fusion import ThreeStageWeldingFusion

class MissingModalityTester:
    """Comprehensive tester for missing modality scenarios"""

    def __init__(self, config: Config, model_path: str, device: torch.device):
        self.config = config
        self.model_path = model_path
        self.device = device

        # Setup logging
        self.setup_logging()

        # Load trained model (FIXED for PyTorch 2.6)
        self.model = self._load_trained_model()

        # Define test scenarios
        self.test_scenarios = {
            'complete': [],
            'missing_image': ['image'],
            'missing_sound': ['sound'],
            'missing_current': ['current'],
            'missing_image_sound': ['image', 'sound'],
            'missing_image_current': ['image', 'current'],
            'missing_sound_current': ['sound', 'current']
        }

        # Results storage
        self.results = {}

        logging.info("[TESTER] Missing modality tester initialized successfully")

    def setup_logging(self):
        """Setup logging for testing"""

        os.makedirs(self.config.results_dir, exist_ok=True)

        # Create test-specific log file
        log_path = os.path.join(self.config.results_dir, 'missing_modality_test.log')

        # Setup file handler
        file_handler = logging.FileHandler(log_path)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)

        # Add to logger
        logger = logging.getLogger()
        logger.addHandler(file_handler)

        logging.info(f"[TESTER] Test logging setup - Log file: {log_path}")

    def _load_trained_model(self) -> nn.Module:
        """Load the trained three-stage model with PyTorch 2.6 fix"""

        logging.info(f"[TESTER] Loading trained model from: {self.model_path}")

        try:
            # Check if model file exists
            if not Path(self.model_path).exists():
                raise FileNotFoundError(f"Model file not found: {self.model_path}")

            # CRITICAL FIX: Load with weights_only=False for PyTorch 2.6
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)

            # Initialize model
            model = ThreeStageWeldingFusion(self.config).to(self.device)

            # Load state dict
            model.load_state_dict(checkpoint['model_state_dict'])

            # Set to evaluation mode and final stage
            model.eval()
            model.set_stage(3)  # Use the final stage for testing

            # Log model info
            total_params = sum(p.numel() for p in model.parameters())
            logging.info(f"[TESTER] Model loaded successfully - Parameters: {total_params:,}")

            # Log training results if available
            if 'stage_results' in checkpoint:
                stage_results = checkpoint['stage_results']
                logging.info("[TESTER] Training results:")
                for stage, results in stage_results.items():
                    logging.info(f"[TESTER]   Stage {stage}: {results['best_val_acc']:.4f} "
                                f"({results['epochs_trained']} epochs)")

            return model

        except Exception as e:
            logging.error(f"[TESTER] Error loading model: {e}")
            raise e

    def _create_test_dataset(self, missing_scenario: List[str]):
        """Create test dataset for specific missing scenario"""

        logging.info(f"[TESTER] Creating test dataset for missing: {missing_scenario}")

        try:
            # Create full dataset first
            full_dataset = MultiModalWeldingDataset(
                data_root=self.config.data_root,
                config=self.config,
                split='train'  # Use full dataset to get all splits
            )

            # Get data loaders
            train_loader, val_loader, test_loader = full_dataset.get_data_loaders()

            # Return test loader for evaluation
            return test_loader

        except Exception as e:
            logging.error(f"[TESTER] Error creating test dataset: {e}")
            raise e

    def _simulate_missing_modalities(self, batch_data: Dict, missing_modalities: List[str]) -> Dict:
        """Simulate missing modalities by zeroing out data"""

        modified_batch = batch_data.copy()

        for modality in missing_modalities:
            if modality in modified_batch['modalities']:
                # Zero out the modality data
                if modified_batch['modalities'][modality] is not None:
                    modified_batch['modalities'][modality] = torch.zeros_like(
                        modified_batch['modalities'][modality]
                    )

        # Update available modalities list
        if 'available_modalities' in modified_batch:
            new_available_modalities = []
            for sample_available in modified_batch['available_modalities']:
                new_sample_available = [m for m in sample_available if m not in missing_modalities]
                new_available_modalities.append(new_sample_available)
            modified_batch['available_modalities'] = new_available_modalities

        return modified_batch

    def _evaluate_scenario(self, scenario_name: str, missing_modalities: List[str]) -> Dict:
        """Evaluate model on specific missing modality scenario"""

        logging.info(f"[TESTER] ===== Testing Scenario: {scenario_name} =====")
        logging.info(f"[TESTER] Missing modalities: {missing_modalities}")

        scenario_start_time = time.time()

        # Create test dataloader (using complete data, we'll simulate missing modalities)
        test_loader = self._create_test_dataset(missing_modalities)

        # Prediction storage
        all_predictions = []
        all_labels = []
        all_confidences = []
        all_fusion_weights = []

        # Batch processing
        total_samples = 0
        correct_predictions = 0

        with torch.no_grad():
            progress_bar = tqdm(
                test_loader,
                desc=f"Testing {scenario_name}",
                leave=False,
                ascii=True
            )

            for batch_idx, batch_data in enumerate(progress_bar):
                try:
                    # Simulate missing modalities
                    if missing_modalities:
                        batch_data = self._simulate_missing_modalities(batch_data, missing_modalities)

                    # Move data to device
                    for modality in batch_data['modalities']:
                        if batch_data['modalities'][modality] is not None:
                            batch_data['modalities'][modality] = batch_data['modalities'][modality].to(
                                self.device, non_blocking=True
                            )

                    labels = batch_data['labels'].to(self.device, non_blocking=True)

                    # Forward pass
                    with torch.cuda.amp.autocast(enabled=self.config.mixed_precision):
                        model_output = self.model(batch_data, labels)

                    # Extract predictions and metrics
                    logits = model_output['logits']
                    predictions = torch.argmax(logits, dim=1)
                    confidences = torch.softmax(logits, dim=1)
                    fusion_weights = model_output.get('fusion_weights', None)

                    # Move to CPU for storage
                    batch_predictions = predictions.cpu().numpy()
                    batch_labels = labels.cpu().numpy()
                    batch_confidences = confidences.cpu().numpy()

                    if fusion_weights is not None:
                        batch_fusion_weights = fusion_weights.cpu().numpy()
                        all_fusion_weights.append(batch_fusion_weights)

                    # Store results
                    all_predictions.append(batch_predictions)
                    all_labels.append(batch_labels)
                    all_confidences.append(batch_confidences)

                    # Update counters
                    batch_correct = (predictions == labels).sum().item()
                    correct_predictions += batch_correct
                    total_samples += labels.size(0)

                    # Update progress bar
                    current_acc = correct_predictions / total_samples
                    progress_bar.set_postfix({
                        'Acc': f'{current_acc:.4f}',
                        'Batch': f'{batch_idx + 1}/{len(test_loader)}'
                    })

                    # Limit to reasonable number of batches for testing
                    if batch_idx >= 50:  # Test on ~1600 samples (50 * 32)
                        break

                except Exception as e:
                    logging.error(f"[TESTER] Error in batch {batch_idx} for scenario {scenario_name}: {e}")
                    continue

        # Concatenate all results
        all_predictions = np.concatenate(all_predictions) if all_predictions else np.array([])
        all_labels = np.concatenate(all_labels) if all_labels else np.array([])
        all_confidences = np.concatenate(all_confidences, axis=0) if all_confidences else np.array([])

        if all_fusion_weights:
            all_fusion_weights = np.concatenate(all_fusion_weights, axis=0)
        else:
            all_fusion_weights = np.array([])

        # Calculate metrics
        scenario_time = time.time() - scenario_start_time

        if len(all_predictions) > 0 and len(all_labels) > 0:
            # Overall accuracy
            accuracy = accuracy_score(all_labels, all_predictions)

            # Per-class metrics
            precision, recall, f1, support = precision_recall_fscore_support(
                all_labels, all_predictions, average=None, labels=range(self.config.num_classes), zero_division=0
            )

            # Macro averages
            macro_precision = np.mean(precision)
            macro_recall = np.mean(recall)
            macro_f1 = np.mean(f1)

            # Confusion matrix
            conf_matrix = confusion_matrix(all_labels, all_predictions, labels=range(self.config.num_classes))

            # Classification report
            class_report = classification_report(
                all_labels, all_predictions,
                target_names=self.config.class_names,
                output_dict=True,
                zero_division=0
            )

        else:
            logging.error(f"[TESTER] No valid predictions for scenario {scenario_name}")
            accuracy = 0.0
            precision = recall = f1 = np.zeros(self.config.num_classes)
            macro_precision = macro_recall = macro_f1 = 0.0
            conf_matrix = np.zeros((self.config.num_classes, self.config.num_classes))
            class_report = {}

        # Compile results
        scenario_results = {
            'scenario_name': scenario_name,
            'missing_modalities': missing_modalities,
            'accuracy': accuracy,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'macro_f1': macro_f1,
            'per_class_precision': precision.tolist(),
            'per_class_recall': recall.tolist(),
            'per_class_f1': f1.tolist(),
            'per_class_support': support.tolist() if 'support' in locals() else [0] * self.config.num_classes,
            'confusion_matrix': conf_matrix.tolist(),
            'classification_report': class_report,
            'total_samples': total_samples,
            'correct_predictions': correct_predictions,
            'test_time': scenario_time,
            'fusion_weights': all_fusion_weights.tolist() if all_fusion_weights.size > 0 else []
        }

        # Log results
        logging.info(f"[TESTER] Scenario {scenario_name} completed:")
        logging.info(f"[TESTER]   Accuracy: {accuracy:.4f}")
        logging.info(f"[TESTER]   Macro Precision: {macro_precision:.4f}")
        logging.info(f"[TESTER]   Macro Recall: {macro_recall:.4f}")
        logging.info(f"[TESTER]   Macro F1: {macro_f1:.4f}")
        logging.info(f"[TESTER]   Total Samples: {total_samples}")
        logging.info(f"[TESTER]   Test Time: {scenario_time:.2f}s")

        return scenario_results

    def run_all_tests(self) -> Dict:
        """Run tests for all missing modality scenarios"""

        logging.info("[TESTER] ========================================")
        logging.info("[TESTER] Starting Missing Modality Test Suite")
        logging.info("[TESTER] ========================================")

        total_start_time = time.time()

        # Test each scenario
        for scenario_name, missing_modalities in self.test_scenarios.items():
            try:
                scenario_results = self._evaluate_scenario(scenario_name, missing_modalities)
                self.results[scenario_name] = scenario_results

            except Exception as e:
                logging.error(f"[TESTER] Error in scenario {scenario_name}: {e}")
                # Add empty results to maintain structure
                self.results[scenario_name] = {
                    'scenario_name': scenario_name,
                    'missing_modalities': missing_modalities,
                    'accuracy': 0.0,
                    'error': str(e)
                }

        # Calculate total test time
        total_test_time = time.time() - total_start_time

        logging.info("[TESTER] ========================================")
        logging.info("[TESTER] All Tests Completed")
        logging.info(f"[TESTER] Total Test Time: {total_test_time:.2f}s")
        logging.info("[TESTER] ========================================")

        return self.results

    def generate_report(self) -> str:
        """Generate comprehensive performance report"""

        logging.info("[TESTER] Generating performance report")

        # Create report
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("MISSING MODALITY PERFORMANCE REPORT")
        report_lines.append("Three-Stage Multi-Modal Welding Quality Detection")
        report_lines.append("=" * 80)
        report_lines.append(f"Model: {self.model_path}")
        report_lines.append(f"Test Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")

        # Summary table
        report_lines.append("ACCURACY SUMMARY:")
        report_lines.append("-" * 60)
        report_lines.append(f"{'Scenario':<20} {'Accuracy':<10} {'Missing Modalities'}")
        report_lines.append("-" * 60)

        # Sort scenarios by accuracy for better presentation
        sorted_scenarios = sorted(
            self.results.items(),
            key=lambda x: x[1].get('accuracy', 0),
            reverse=True
        )

        for scenario_name, results in sorted_scenarios:
            if 'accuracy' in results:
                accuracy = results['accuracy']
                missing = results['missing_modalities']
                missing_str = ', '.join(missing) if missing else 'None'
                report_lines.append(f"{scenario_name:<20} {accuracy:>8.2%} {missing_str}")

        report_lines.append("")

        # Detailed analysis for missing image (key scenario)
        if 'missing_image' in self.results and 'complete' in self.results:
            complete_acc = self.results['complete']['accuracy']
            missing_image_acc = self.results['missing_image']['accuracy']
            image_impact = (complete_acc - missing_image_acc) * 100

            report_lines.append("KEY FINDING - MISSING IMAGE IMPACT:")
            report_lines.append("-" * 40)
            report_lines.append(f"Complete Data Accuracy:      {complete_acc:8.2%}")
            report_lines.append(f"Missing Image Accuracy:      {missing_image_acc:8.2%}")
            report_lines.append(f"Performance Drop:            {image_impact:8.2f} percentage points")

            if image_impact < 5:
                report_lines.append("Assessment: EXCELLENT robustness to missing image data")
            elif image_impact < 10:
                report_lines.append("Assessment: GOOD robustness to missing image data")
            elif image_impact < 20:
                report_lines.append("Assessment: MODERATE robustness to missing image data")
            else:
                report_lines.append("Assessment: LIMITED robustness to missing image data")

            report_lines.append("")

        # Class-wise performance for missing image
        if 'missing_image' in self.results and 'classification_report' in self.results['missing_image']:
            report_lines.append("MISSING IMAGE - CLASS-WISE PERFORMANCE:")
            report_lines.append("-" * 50)
            report_lines.append(f"{'Class':<20} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
            report_lines.append("-" * 50)

            class_report = self.results['missing_image']['classification_report']
            for class_name in self.config.class_names:
                if class_name in class_report:
                    metrics = class_report[class_name]
                    report_lines.append(
                        f"{class_name[:19]:<20} "
                        f"{metrics['precision']:>8.3f}  "
                        f"{metrics['recall']:>8.3f}  "
                        f"{metrics['f1-score']:>8.3f}"
                    )

            report_lines.append("")

        # Overall robustness analysis
        missing_scenarios = {k: v for k, v in self.results.items() if k != 'complete' and 'accuracy' in v}
        if missing_scenarios and 'complete' in self.results:
            complete_acc = self.results['complete']['accuracy']

            report_lines.append("OVERALL ROBUSTNESS ANALYSIS:")
            report_lines.append("-" * 35)

            for scenario_name, results in missing_scenarios.items():
                accuracy_drop = (complete_acc - results['accuracy']) * 100
                report_lines.append(f"{scenario_name:<25}: {accuracy_drop:>6.2f}pp drop")

            # Average robustness
            avg_drop = np.mean([(complete_acc - results['accuracy']) * 100
                               for results in missing_scenarios.values()])
            report_lines.append("-" * 35)
            report_lines.append(f"{'Average Performance Drop':<25}: {avg_drop:>6.2f}pp")

            if avg_drop < 5:
                robustness_level = "EXCELLENT"
            elif avg_drop < 10:
                robustness_level = "GOOD"
            elif avg_drop < 20:
                robustness_level = "MODERATE"
            else:
                robustness_level = "LIMITED"

            report_lines.append(f"Overall Robustness Level: {robustness_level}")

        report_lines.append("")
        report_lines.append("=" * 80)

        report_content = '\n'.join(report_lines)

        # Save report to file
        report_path = os.path.join(self.config.results_dir, 'missing_modality_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)

        logging.info(f"[TESTER] Performance report saved: {report_path}")

        return report_content

    def create_visualizations(self):
        """Create performance visualizations"""

        logging.info("[TESTER] Creating visualizations")

        # Create visualization directory
        viz_dir = os.path.join(self.config.results_dir, 'visualizations')
        os.makedirs(viz_dir, exist_ok=True)

        try:
            # 1. Accuracy comparison bar chart
            scenarios = []
            accuracies = []
            colors = []

            # Define colors for different scenarios
            color_map = {
                'complete': '#2E8B57',  # Sea Green
                'missing_image': '#DC143C',  # Crimson - KEY SCENARIO
                'missing_sound': '#4169E1',  # Royal Blue
                'missing_current': '#FF8C00',  # Dark Orange
                'missing_image_sound': '#8B0000',  # Dark Red
                'missing_image_current': '#B22222',  # Fire Brick
                'missing_sound_current': '#191970'  # Midnight Blue
            }

            for scenario_name, results in self.results.items():
                if 'accuracy' in results:
                    display_name = scenario_name.replace('_', '\n')
                    if scenario_name == 'missing_image':
                        display_name += '\n⭐ KEY TEST'

                    scenarios.append(display_name)
                    accuracies.append(results['accuracy'] * 100)
                    colors.append(color_map.get(scenario_name, '#696969'))

            # Create plot
            plt.figure(figsize=(14, 8))
            bars = plt.bar(scenarios, accuracies, color=colors, alpha=0.8,
                          edgecolor='black', linewidth=1.5)

            # Add value labels on bars
            for bar, acc in zip(bars, accuracies):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                        f'{acc:.1f}%', ha='center', va='bottom',
                        fontsize=11, fontweight='bold')

            plt.title('Three-Stage Fusion Model: Missing Modality Performance\n'
                     'Welding Quality Detection Robustness Test',
                     fontsize=16, fontweight='bold', pad=20)
            plt.xlabel('Missing Modality Scenario', fontsize=12)
            plt.ylabel('Accuracy (%)', fontsize=12)
            plt.ylim(0, max(accuracies) + 10)
            plt.grid(axis='y', alpha=0.3)
            plt.xticks(rotation=0, ha='center')

            # Add baseline line (complete data performance)
            if 'complete' in self.results:
                baseline = self.results['complete']['accuracy'] * 100
                plt.axhline(y=baseline, color='green', linestyle='--', alpha=0.7,
                           label=f'Baseline (Complete): {baseline:.1f}%')
                plt.legend()

            plt.tight_layout()

            # Save plot
            plot_path = os.path.join(viz_dir, 'missing_modality_performance.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()

            logging.info(f"[TESTER] Performance chart saved: {plot_path}")

            # 2. Missing Image Confusion Matrix (Key Analysis)
            if 'missing_image' in self.results and 'confusion_matrix' in self.results['missing_image']:
                plt.figure(figsize=(10, 8))

                conf_matrix = np.array(self.results['missing_image']['confusion_matrix'])

                sns.heatmap(
                    conf_matrix,
                    annot=True,
                    fmt='d',
                    cmap='Blues',
                    xticklabels=self.config.class_names,
                    yticklabels=self.config.class_names,
                    cbar_kws={'label': 'Number of Samples'}
                )

                accuracy = self.results['missing_image']['accuracy']
                plt.title(f'Missing Image Scenario - Confusion Matrix\n'
                         f'Accuracy: {accuracy:.2%}',
                         fontsize=14, fontweight='bold')
                plt.xlabel('Predicted Class', fontsize=12)
                plt.ylabel('True Class', fontsize=12)
                plt.xticks(rotation=45, ha='right')
                plt.yticks(rotation=0)
                plt.tight_layout()

                # Save plot
                confusion_path = os.path.join(viz_dir, 'missing_image_confusion_matrix.png')
                plt.savefig(confusion_path, dpi=300, bbox_inches='tight')
                plt.close()

                logging.info(f"[TESTER] Confusion matrix saved: {confusion_path}")

        except Exception as e:
            logging.error(f"[TESTER] Error creating visualizations: {e}")

    def export_results(self):
        """Export results to files"""

        logging.info("[TESTER] Exporting results")

        try:
            # Export complete results to JSON
            json_path = os.path.join(self.config.results_dir, 'missing_modality_results.json')
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(self.results, f, indent=2, ensure_ascii=False)

            logging.info(f"[TESTER] Complete results exported: {json_path}")

            # Export summary to CSV
            summary_data = []
            for scenario_name, results in self.results.items():
                if 'accuracy' in results:
                    row = {
                        'Scenario': scenario_name,
                        'Missing_Modalities': ', '.join(results['missing_modalities']) if results['missing_modalities'] else 'None',
                        'Accuracy': results['accuracy'],
                        'Macro_Precision': results.get('macro_precision', 0),
                        'Macro_Recall': results.get('macro_recall', 0),
                        'Macro_F1': results.get('macro_f1', 0),
                        'Total_Samples': results.get('total_samples', 0),
                        'Test_Time_Seconds': results.get('test_time', 0)
                    }
                    summary_data.append(row)

            # Create summary DataFrame
            summary_df = pd.DataFrame(summary_data)
            csv_path = os.path.join(self.config.results_dir, 'missing_modality_summary.csv')
            summary_df.to_csv(csv_path, index=False)

            logging.info(f"[TESTER] Summary CSV saved: {csv_path}")

        except Exception as e:
            logging.error(f"[TESTER] Error exporting results: {e}")

    def run_complete_test_suite(self) -> str:
        """Run complete missing modality test suite"""

        logging.info("[TESTER] Starting complete missing modality test suite")

        try:
            # Run all tests
            self.run_all_tests()

            # Generate performance report
            report = self.generate_report()

            # Create visualizations
            self.create_visualizations()

            # Export results
            self.export_results()

            # Print summary to console
            print("\n" + "=" * 80)
            print("🔬 MISSING MODALITY TEST RESULTS SUMMARY")
            print("=" * 80)
            print("Three-Stage Multi-Modal Welding Quality Detection")
            print("")

            # Highlight key findings
            if 'complete' in self.results and 'missing_image' in self.results:
                complete_acc = self.results['complete']['accuracy']
                missing_image_acc = self.results['missing_image']['accuracy']

                print(f"🔍 KEY FINDING:")
                print(f"   Complete Data:    {complete_acc:8.2%}")
                print(f"   Missing Image:    {missing_image_acc:8.2%}")
                print(f"   Performance Drop: {(complete_acc - missing_image_acc)*100:8.2f} percentage points")
                print("")

            print("📊 ALL SCENARIOS:")
            for scenario_name, results in sorted(self.results.items(),
                                                key=lambda x: x[1].get('accuracy', 0), reverse=True):
                if 'accuracy' in results:
                    accuracy = results['accuracy']
                    missing = results['missing_modalities']
                    missing_str = ', '.join(missing) if missing else 'All modalities'

                    marker = "⭐" if scenario_name == 'missing_image' else "📋"
                    print(f"{marker} {scenario_name:20}: {accuracy:7.2%} ({missing_str})")

            print("\n" + "=" * 80)
            print(f"📁 Detailed results saved to: {self.config.results_dir}")
            print("=" * 80)

            logging.info("[TESTER] Complete test suite finished successfully")

            return report

        except Exception as e:
            logging.error(f"[TESTER] Error in complete test suite: {e}")
            raise e

def main():
    """Main function to run missing modality tests"""

    print("🔬 Missing Modality Testing Framework")
    print("Three-Stage Multi-Modal Welding Quality Detection")
    print("=" * 50)

    # Load configuration
    config = Config()

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Model path
    model_path = os.path.join(config.model_save_dir, 'final_three_stage_model.pth')

    if not Path(model_path).exists():
        print(f"❌ Error: Trained model not found at {model_path}")
        print("Please ensure the model has been trained and saved.")
        return

    print(f"✅ Found trained model: {model_path}")
    print(f"🖥️  Using device: {device}")
    print("")

    # Initialize tester
    try:
        tester = MissingModalityTester(config, model_path, device)
        print("✅ Tester initialized successfully")
    except Exception as e:
        print(f"❌ Error initializing tester: {e}")
        return

    # Run complete test suite
    try:
        print("🚀 Starting missing modality evaluation...")
        print("This will test performance with:")
        print("   - Complete data (baseline)")
        print("   - Missing image only ⭐ KEY TEST")
        print("   - Missing sound only")
        print("   - Missing current only")
        print("   - Various combinations")
        print("")

        report = tester.run_complete_test_suite()
        print("\n✅ Testing completed successfully!")
        print("📊 Check the results directory for detailed analysis and visualizations.")

    except Exception as e:
        print(f"❌ Error during testing: {e}")
        logging.error(f"Testing failed: {e}")

if __name__ == "__main__":
    main()