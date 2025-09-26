"""
Comprehensive evaluation metrics for multi-modal welding quality detection
"""

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from tqdm import tqdm
import logging
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import pandas as pd


class ModelEvaluator:
    """Comprehensive model evaluation with missing modality analysis"""

    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Create results directory
        os.makedirs(config.results_dir, exist_ok=True)

    def evaluate_model(self, model, test_loader, device: torch.device) -> Dict:
        """Comprehensive model evaluation with missing modality scenarios"""

        logging.info("Starting comprehensive model evaluation...")

        # Standard evaluation (complete data)
        logging.info("Evaluating with complete data...")
        complete_results = self._evaluate_complete_data(model, test_loader, device)

        # Missing modality evaluation
        logging.info("Evaluating missing modality scenarios...")
        missing_results = self._evaluate_missing_modalities(model, test_loader, device)

        # Confidence analysis
        logging.info("Analyzing prediction confidence...")
        confidence_analysis = self._analyze_prediction_confidence(model, test_loader, device)

        # Fusion weight analysis
        logging.info("Analyzing fusion weights...")
        fusion_analysis = self._analyze_fusion_weights(model, test_loader, device)

        # Combine all results
        comprehensive_results = {
            **complete_results,
            'missing_modality_results': missing_results,
            'confidence_analysis': confidence_analysis,
            'fusion_analysis': fusion_analysis,
            'evaluation_summary': self._create_evaluation_summary(
                complete_results, missing_results, confidence_analysis
            )
        }

        logging.info("Model evaluation completed!")
        return comprehensive_results

    def _evaluate_complete_data(self, model, test_loader, device: torch.device) -> Dict:
        """Evaluate model performance with complete data"""

        model.eval()
        all_predictions = []
        all_labels = []
        all_probabilities = []
        all_sample_ids = []
        all_class_names = []

        total_loss = 0.0
        criterion = torch.nn.CrossEntropyLoss()

        with torch.no_grad():
            for batch_idx, batch_data in enumerate(tqdm(test_loader, desc="Complete data evaluation")):
                # Move data to device
                for modality in batch_data['modalities']:
                    if batch_data['modalities'][modality] is not None:
                        batch_data['modalities'][modality] = batch_data['modalities'][modality].to(device)

                labels = batch_data['labels'].to(device)

                # Forward pass
                model_output = model(batch_data, labels)
                logits = model_output['logits']

                # Calculate loss
                loss = criterion(logits, labels)
                total_loss += loss.item()

                # Get predictions and probabilities
                probabilities = F.softmax(logits, dim=1)
                predictions = torch.argmax(logits, dim=1)

                # Store results
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                all_sample_ids.extend(batch_data['sample_ids'])
                all_class_names.extend(batch_data['class_names'])

        # Calculate comprehensive metrics
        metrics = self._calculate_comprehensive_metrics(
            all_labels, all_predictions, all_probabilities
        )

        # Add additional information
        metrics.update({
            'avg_loss': total_loss / len(test_loader),
            'predictions': {
                'sample_ids': all_sample_ids,
                'class_names': all_class_names,
                'true_labels': all_labels,
                'predicted_labels': all_predictions,
                'probabilities': all_probabilities
            }
        })

        return metrics

    def _evaluate_missing_modalities(self, model, test_loader, device: torch.device) -> Dict:
        """Evaluate model performance under different missing modality scenarios"""

        missing_results = {}

        for scenario_idx, missing_modalities in enumerate(self.config.missing_scenarios):
            scenario_name = f"scenario_{scenario_idx}"

            if missing_modalities:
                logging.info(f"Evaluating scenario {scenario_idx}: Missing {missing_modalities}")
            else:
                logging.info(f"Evaluating scenario {scenario_idx}: Complete data")

            all_predictions = []
            all_labels = []
            all_probabilities = []
            fusion_weights_list = []
            confidence_scores_list = []

            model.eval()
            with torch.no_grad():
                for batch_data in tqdm(test_loader, desc=f"Missing scenario {scenario_idx}"):
                    # Simulate missing modalities
                    modified_batch = self._simulate_missing_modalities(batch_data, missing_modalities)

                    # Move data to device
                    for modality in modified_batch['modalities']:
                        if modified_batch['modalities'][modality] is not None:
                            modified_batch['modalities'][modality] = modified_batch['modalities'][modality].to(device)

                    labels = modified_batch['labels'].to(device)

                    # Forward pass
                    model_output = model(modified_batch, labels)
                    logits = model_output['logits']

                    # Get predictions and probabilities
                    probabilities = F.softmax(logits, dim=1)
                    predictions = torch.argmax(logits, dim=1)

                    # Store results
                    all_predictions.extend(predictions.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    all_probabilities.extend(probabilities.cpu().numpy())

                    # Store fusion analysis data
                    if 'fusion_weights' in model_output:
                        fusion_weights_list.append(model_output['fusion_weights'].cpu().numpy())
                    if 'confidences' in model_output:
                        confidence_scores_list.append(model_output['confidences'].cpu().numpy())

            # Calculate metrics for this scenario
            scenario_metrics = self._calculate_comprehensive_metrics(
                all_labels, all_predictions, all_probabilities
            )

            # Add scenario-specific information
            scenario_metrics.update({
                'missing_modalities': missing_modalities,
                'scenario_index': scenario_idx,
                'fusion_weights': np.concatenate(fusion_weights_list, axis=0) if fusion_weights_list else None,
                'confidence_scores': np.concatenate(confidence_scores_list, axis=0) if confidence_scores_list else None
            })

            missing_results[scenario_name] = scenario_metrics

        return missing_results

    def _simulate_missing_modalities(self, batch_data: Dict, missing_modalities: List[str]) -> Dict:
        """Simulate missing modalities by modifying batch data"""

        modified_batch = {
            'modalities': {},
            'labels': batch_data['labels'],
            'sample_ids': batch_data.get('sample_ids', []),
            'class_names': batch_data.get('class_names', []),
            'available_modalities': [],
            'existing_modalities': batch_data.get('existing_modalities', [])
        }

        # Update available modalities list
        for available_mods in batch_data.get('available_modalities', []):
            new_available = [m for m in available_mods if m not in missing_modalities]
            # Ensure at least one modality is available
            if not new_available and available_mods:
                new_available = [available_mods[0]]  # Keep first available modality
            modified_batch['available_modalities'].append(new_available)

        # Modify modality data
        for modality in self.config.modalities:
            if modality in missing_modalities:
                # Zero out missing modality
                if modality == 'image':
                    modified_batch['modalities'][modality] = torch.zeros_like(batch_data['modalities'][modality])
                elif modality == 'sound':
                    modified_batch['modalities'][modality] = torch.zeros_like(batch_data['modalities'][modality])
                elif modality == 'current':
                    modified_batch['modalities'][modality] = torch.zeros_like(batch_data['modalities'][modality])
            else:
                # Keep original modality
                modified_batch['modalities'][modality] = batch_data['modalities'][modality]

        return modified_batch

    def _calculate_comprehensive_metrics(self, true_labels: List, predicted_labels: List,
                                         probabilities: List) -> Dict:
        """Calculate comprehensive evaluation metrics"""

        # Convert to numpy arrays
        y_true = np.array(true_labels)
        y_pred = np.array(predicted_labels)
        y_prob = np.array(probabilities)

        # Basic classification metrics
        accuracy = accuracy_score(y_true, y_pred)

        # Macro-averaged metrics
        precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)

        # Weighted metrics (account for class imbalance)
        precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)

        # Per-class metrics
        precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        # AUROC calculation
        try:
            if self.config.num_classes == 2:
                # Binary classification
                auroc = roc_auc_score(y_true, y_prob[:, 1])
                auprc = average_precision_score(y_true, y_prob[:, 1])
            else:
                # Multi-class classification
                y_true_binarized = label_binarize(y_true, classes=list(range(self.config.num_classes)))
                auroc = roc_auc_score(y_true_binarized, y_prob, multi_class='ovr', average='macro')
                auprc = average_precision_score(y_true_binarized, y_prob, average='macro')
        except Exception as e:
            logging.warning(f"Could not calculate AUROC/AUPRC: {e}")
            auroc = 0.0
            auprc = 0.0

        # Top-k accuracy (for multi-class)
        top2_accuracy = 0.0
        top3_accuracy = 0.0
        if self.config.num_classes > 2:
            top2_predictions = np.argsort(y_prob, axis=1)[:, -2:]
            top3_predictions = np.argsort(y_prob, axis=1)[:, -3:]

            top2_accuracy = np.mean([label in pred for label, pred in zip(y_true, top2_predictions)])
            top3_accuracy = np.mean([label in pred for label, pred in zip(y_true, top3_predictions)])

        # Classification report
        report = classification_report(
            y_true, y_pred,
            target_names=self.config.class_names,
            output_dict=True,
            zero_division=0
        )

        # Per-class confusion matrix statistics
        class_stats = {}
        for i, class_name in enumerate(self.config.class_names):
            if i < len(cm):
                tp = cm[i, i]
                fp = cm[:, i].sum() - tp
                fn = cm[i, :].sum() - tp
                tn = cm.sum() - tp - fp - fn

                class_stats[class_name] = {
                    'true_positives': int(tp),
                    'false_positives': int(fp),
                    'false_negatives': int(fn),
                    'true_negatives': int(tn),
                    'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0.0,
                    'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0.0,
                    'support': int(cm[i, :].sum())
                }

        return {
            'accuracy': accuracy,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_score': f1_macro,
            'precision_weighted': precision_weighted,
            'recall_weighted': recall_weighted,
            'f1_weighted': f1_weighted,
            'precision_per_class': precision_per_class.tolist(),
            'recall_per_class': recall_per_class.tolist(),
            'f1_per_class': f1_per_class.tolist(),
            'auroc': auroc,
            'auprc': auprc,
            'top2_accuracy': top2_accuracy,
            'top3_accuracy': top3_accuracy,
            'confusion_matrix': cm.tolist(),
            'classification_report': report,
            'class_statistics': class_stats
        }

    def _analyze_prediction_confidence(self, model, test_loader, device: torch.device) -> Dict:
        """Analyze prediction confidence distribution"""

        model.eval()
        confidence_data = {
            'all_confidences': [],
            'correct_confidences': [],
            'incorrect_confidences': [],
            'class_confidences': {class_name: [] for class_name in self.config.class_names},
            'confidence_accuracy_bins': {}
        }

        with torch.no_grad():
            for batch_data in tqdm(test_loader, desc="Confidence analysis"):
                # Move data to device
                for modality in batch_data['modalities']:
                    if batch_data['modalities'][modality] is not None:
                        batch_data['modalities'][modality] = batch_data['modalities'][modality].to(device)

                labels = batch_data['labels'].to(device)

                # Forward pass
                model_output = model(batch_data, labels)
                logits = model_output['logits']

                # Get predictions and confidence
                probabilities = F.softmax(logits, dim=1)
                predictions = torch.argmax(logits, dim=1)
                confidences = torch.max(probabilities, dim=1)[0]

                # Convert to numpy
                labels_np = labels.cpu().numpy()
                predictions_np = predictions.cpu().numpy()
                confidences_np = confidences.cpu().numpy()

                # Analyze confidence
                for i in range(len(labels_np)):
                    confidence = confidences_np[i]
                    is_correct = labels_np[i] == predictions_np[i]
                    class_name = self.config.class_names[labels_np[i]]

                    confidence_data['all_confidences'].append(confidence)
                    confidence_data['class_confidences'][class_name].append(confidence)

                    if is_correct:
                        confidence_data['correct_confidences'].append(confidence)
                    else:
                        confidence_data['incorrect_confidences'].append(confidence)

        # Calculate confidence statistics
        confidence_stats = {
            'mean_confidence': np.mean(confidence_data['all_confidences']),
            'std_confidence': np.std(confidence_data['all_confidences']),
            'mean_correct_confidence': np.mean(confidence_data['correct_confidences']) if confidence_data[
                'correct_confidences'] else 0,
            'mean_incorrect_confidence': np.mean(confidence_data['incorrect_confidences']) if confidence_data[
                'incorrect_confidences'] else 0,
            'confidence_accuracy_correlation': self._calculate_confidence_accuracy_correlation(
                confidence_data['all_confidences'],
                [c in confidence_data['correct_confidences'] for c in confidence_data['all_confidences']]
            )
        }

        # Confidence binning analysis
        bins = np.linspace(0, 1, 11)  # 10 bins
        for i in range(len(bins) - 1):
            bin_start, bin_end = bins[i], bins[i + 1]
            bin_confidences = [c for c in confidence_data['all_confidences'] if bin_start <= c < bin_end]
            bin_correct = [c for c in confidence_data['correct_confidences'] if bin_start <= c < bin_end]

            bin_accuracy = len(bin_correct) / len(bin_confidences) if bin_confidences else 0
            confidence_data['confidence_accuracy_bins'][f'{bin_start:.1f}-{bin_end:.1f}'] = {
                'count': len(bin_confidences),
                'accuracy': bin_accuracy
            }

        return {
            'confidence_data': confidence_data,
            'confidence_statistics': confidence_stats
        }

    def _analyze_fusion_weights(self, model, test_loader, device: torch.device) -> Dict:
        """Analyze fusion weight patterns"""

        model.eval()
        fusion_data = {
            'all_weights': [],
            'weights_by_scenario': {},
            'weights_by_class': {class_name: [] for class_name in self.config.class_names}
        }

        with torch.no_grad():
            for batch_data in tqdm(test_loader, desc="Fusion weight analysis"):
                # Move data to device
                for modality in batch_data['modalities']:
                    if batch_data['modalities'][modality] is not None:
                        batch_data['modalities'][modality] = batch_data['modalities'][modality].to(device)

                labels = batch_data['labels'].to(device)

                # Forward pass
                model_output = model(batch_data, labels)

                if 'fusion_weights' in model_output:
                    weights = model_output['fusion_weights'].cpu().numpy()
                    labels_np = labels.cpu().numpy()

                    fusion_data['all_weights'].append(weights)

                    # Analyze by class
                    for i, label in enumerate(labels_np):
                        class_name = self.config.class_names[label]
                        fusion_data['weights_by_class'][class_name].append(weights[i])

        # Calculate fusion statistics
        if fusion_data['all_weights']:
            all_weights = np.concatenate(fusion_data['all_weights'], axis=0)

            fusion_stats = {
                'mean_weights': np.mean(all_weights, axis=0).tolist(),
                'std_weights': np.std(all_weights, axis=0).tolist(),
                'weight_entropy': [-np.sum(w * np.log(w + 1e-8)) for w in np.mean(all_weights, axis=0)],
                'modality_dominance': {
                    modality: float(np.mean(all_weights[:, i]))
                    for i, modality in enumerate(self.config.modalities)
                }
            }

            # Per-class fusion statistics
            class_fusion_stats = {}
            for class_name, class_weights in fusion_data['weights_by_class'].items():
                if class_weights:
                    class_weights_array = np.array(class_weights)
                    class_fusion_stats[class_name] = {
                        'mean_weights': np.mean(class_weights_array, axis=0).tolist(),
                        'dominant_modality': self.config.modalities[np.argmax(np.mean(class_weights_array, axis=0))]
                    }

            return {
                'fusion_statistics': fusion_stats,
                'class_fusion_statistics': class_fusion_stats,
                'raw_data': fusion_data
            }
        else:
            return {'fusion_statistics': None, 'class_fusion_statistics': None, 'raw_data': fusion_data}

    def _calculate_confidence_accuracy_correlation(self, confidences: List[float],
                                                   is_correct: List[bool]) -> float:
        """Calculate correlation between confidence and accuracy"""

        try:
            from scipy.stats import pearsonr
            correlation, p_value = pearsonr(confidences, is_correct)
            return correlation
        except ImportError:
            # Fallback calculation without scipy
            conf_array = np.array(confidences)
            correct_array = np.array(is_correct, dtype=float)

            conf_mean = np.mean(conf_array)
            correct_mean = np.mean(correct_array)

            numerator = np.sum((conf_array - conf_mean) * (correct_array - correct_mean))
            denominator = np.sqrt(np.sum((conf_array - conf_mean) ** 2) * np.sum((correct_array - correct_mean) ** 2))

            return numerator / denominator if denominator != 0 else 0.0

    def _create_evaluation_summary(self, complete_results: Dict, missing_results: Dict,
                                   confidence_analysis: Dict) -> Dict:
        """Create a comprehensive evaluation summary"""

        summary = {
            'overall_performance': {
                'accuracy': complete_results['accuracy'],
                'f1_macro': complete_results['f1_score'],
                'f1_weighted': complete_results['f1_weighted'],
                'auroc': complete_results['auroc']
            },
            'robustness_analysis': {},
            'confidence_analysis': {
                'mean_confidence': confidence_analysis['confidence_statistics']['mean_confidence'],
                'confidence_accuracy_gap': (
                        confidence_analysis['confidence_statistics']['mean_correct_confidence'] -
                        confidence_analysis['confidence_statistics']['mean_incorrect_confidence']
                )
            },
            'class_performance_ranking': []
        }

        # Robustness analysis
        complete_accuracy = complete_results['accuracy']
        max_drop = 0.0
        avg_drop = 0.0
        robust_scenarios = 0

        for scenario_name, scenario_results in missing_results.items():
            missing_mods = scenario_results['missing_modalities']
            if missing_mods:  # Skip complete data scenario
                accuracy_drop = (complete_accuracy - scenario_results['accuracy']) / complete_accuracy * 100
                max_drop = max(max_drop, accuracy_drop)
                avg_drop += accuracy_drop

                if accuracy_drop < 10.0:  # Less than 10% drop is considered robust
                    robust_scenarios += 1

                summary['robustness_analysis'][f"missing_{'+'.join(missing_mods)}"] = {
                    'accuracy': scenario_results['accuracy'],
                    'f1_score': scenario_results['f1_score'],
                    'performance_drop_percent': accuracy_drop
                }

        num_missing_scenarios = len([s for s in missing_results.values() if s['missing_modalities']])
        summary['robustness_analysis']['summary'] = {
            'max_performance_drop_percent': max_drop,
            'avg_performance_drop_percent': avg_drop / num_missing_scenarios if num_missing_scenarios > 0 else 0,
            'robust_scenarios_count': robust_scenarios,
            'robustness_score': robust_scenarios / num_missing_scenarios if num_missing_scenarios > 0 else 1.0
        }

        # Class performance ranking
        for i, class_name in enumerate(self.config.class_names):
            if i < len(complete_results['f1_per_class']):
                summary['class_performance_ranking'].append({
                    'class_name': class_name,
                    'f1_score': complete_results['f1_per_class'][i],
                    'precision': complete_results['precision_per_class'][i],
                    'recall': complete_results['recall_per_class'][i]
                })

        # Sort by F1 score
        summary['class_performance_ranking'].sort(key=lambda x: x['f1_score'], reverse=True)

        return summary

    def save_results(self, results: Dict, filename: str = 'evaluation_results.json'):
        """Save comprehensive evaluation results"""

        # Prepare results for JSON serialization
        json_results = {
            'evaluation_timestamp': pd.Timestamp.now().isoformat(),
            'dataset_info': {
                'num_classes': self.config.num_classes,
                'class_names': self.config.class_names,
                'modalities': self.config.modalities
            },
            'overall_metrics': {
                'accuracy': results['accuracy'],
                'precision_macro': results['precision_macro'],
                'recall_macro': results['recall_macro'],
                'f1_score': results['f1_score'],
                'precision_weighted': results['precision_weighted'],
                'recall_weighted': results['recall_weighted'],
                'f1_weighted': results['f1_weighted'],
                'auroc': results['auroc'],
                'auprc': results['auprc'],
                'top2_accuracy': results.get('top2_accuracy', 0.0),
                'top3_accuracy': results.get('top3_accuracy', 0.0)
            },
            'per_class_metrics': {
                'precision': results['precision_per_class'],
                'recall': results['recall_per_class'],
                'f1_score': results['f1_per_class']
            },
            'confusion_matrix': results['confusion_matrix'],
            'classification_report': results['classification_report'],
            'class_statistics': results['class_statistics'],
            'missing_modality_results': {},
            'evaluation_summary': results['evaluation_summary']
        }

        # Add missing modality results
        for scenario_name, scenario_results in results['missing_modality_results'].items():
            json_results['missing_modality_results'][scenario_name] = {
                'missing_modalities': scenario_results['missing_modalities'],
                'accuracy': scenario_results['accuracy'],
                'f1_score': scenario_results['f1_score'],
                'auroc': scenario_results['auroc'],
                'precision_macro': scenario_results['precision_macro'],
                'recall_macro': scenario_results['recall_macro']
            }

        # Add confidence analysis (simplified)
        if 'confidence_analysis' in results:
            json_results['confidence_analysis'] = results['confidence_analysis']['confidence_statistics']

        # Add fusion analysis (simplified)
        if 'fusion_analysis' in results and results['fusion_analysis']['fusion_statistics']:
            json_results['fusion_analysis'] = results['fusion_analysis']['fusion_statistics']

        # Save main results
        results_file = os.path.join(self.config.results_dir, filename)
        with open(results_file, 'w') as f:
            json.dump(json_results, f, indent=2)

        logging.info(f"Evaluation results saved to: {results_file}")

        # Save detailed predictions if available
        if 'predictions' in results:
            predictions_file = os.path.join(self.config.results_dir, 'detailed_predictions.json')
            predictions_data = {
                'sample_ids': results['predictions']['sample_ids'],
                'class_names': results['predictions']['class_names'],
                'true_labels': [int(x) for x in results['predictions']['true_labels']],
                'predicted_labels': [int(x) for x in results['predictions']['predicted_labels']],
                'probabilities': [[float(y) for y in x] for x in results['predictions']['probabilities']]
            }

            with open(predictions_file, 'w') as f:
                json.dump(predictions_data, f, indent=2)

            logging.info(f"Detailed predictions saved to: {predictions_file}")

        # Create and save performance summary table
        self._create_performance_summary_table(results)

        return results_file

    def _create_performance_summary_table(self, results: Dict):
        """Create a summary table of key performance metrics"""

        # Create performance summary DataFrame
        summary_data = []

        # Overall performance
        summary_data.append({
            'Metric': 'Overall Accuracy',
            'Value': f"{results['accuracy']:.4f}",
            'Description': 'Overall classification accuracy'
        })

        summary_data.append({
            'Metric': 'Macro F1-Score',
            'Value': f"{results['f1_score']:.4f}",
            'Description': 'Macro-averaged F1 score'
        })

        summary_data.append({
            'Metric': 'Weighted F1-Score',
            'Value': f"{results['f1_weighted']:.4f}",
            'Description': 'Weighted F1 score (accounts for class imbalance)'
        })

        summary_data.append({
            'Metric': 'AUROC',
            'Value': f"{results['auroc']:.4f}",
            'Description': 'Area Under ROC Curve'
        })

        # Robustness metrics
        if 'evaluation_summary' in results:
            robustness = results['evaluation_summary']['robustness_analysis']['summary']
            summary_data.append({
                'Metric': 'Max Performance Drop',
                'Value': f"{robustness['max_performance_drop_percent']:.2f}%",
                'Description': 'Maximum performance drop with missing modalities'
            })

            summary_data.append({
                'Metric': 'Robustness Score',
                'Value': f"{robustness['robustness_score']:.4f}",
                'Description': 'Fraction of scenarios with <10% performance drop'
            })

        # Create DataFrame and save
        df = pd.DataFrame(summary_data)
        summary_file = os.path.join(self.config.results_dir, 'performance_summary.csv')
        df.to_csv(summary_file, index=False)

        logging.info(f"Performance summary table saved to: {summary_file}")

    def generate_evaluation_report(self, results: Dict) -> str:
        """Generate a comprehensive text evaluation report"""

        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("MULTI-MODAL WELDING QUALITY DETECTION - EVALUATION REPORT")
        report_lines.append("=" * 80)
        report_lines.append("")

        # Overall Performance
        report_lines.append("OVERALL PERFORMANCE:")
        report_lines.append("-" * 40)
        report_lines.append(f"Accuracy: {results['accuracy']:.4f}")
        report_lines.append(f"Precision (Macro): {results['precision_macro']:.4f}")
        report_lines.append(f"Recall (Macro): {results['recall_macro']:.4f}")
        report_lines.append(f"F1-Score (Macro): {results['f1_score']:.4f}")
        report_lines.append(f"F1-Score (Weighted): {results['f1_weighted']:.4f}")
        report_lines.append(f"AUROC: {results['auroc']:.4f}")
        report_lines.append("")

        # Per-Class Performance
        report_lines.append("PER-CLASS PERFORMANCE:")
        report_lines.append("-" * 40)
        for i, class_name in enumerate(self.config.class_names):
            if i < len(results['f1_per_class']):
                report_lines.append(
                    f"{class_name:20} | F1: {results['f1_per_class'][i]:.4f} | "
                    f"Prec: {results['precision_per_class'][i]:.4f} | "
                    f"Rec: {results['recall_per_class'][i]:.4f}"
                )
        report_lines.append("")

        # Missing Modality Robustness
        report_lines.append("MISSING MODALITY ROBUSTNESS:")
        report_lines.append("-" * 40)
        for scenario_name, scenario_results in results['missing_modality_results'].items():
            missing_mods = scenario_results['missing_modalities']
            if missing_mods:
                missing_str = ', '.join(missing_mods)
                report_lines.append(
                    f"Missing [{missing_str:15}] | Acc: {scenario_results['accuracy']:.4f} | "
                    f"F1: {scenario_results['f1_score']:.4f}"
                )
            else:
                report_lines.append(
                    f"Complete Data        | Acc: {scenario_results['accuracy']:.4f} | "
                    f"F1: {scenario_results['f1_score']:.4f}"
                )
        report_lines.append("")

        # Evaluation Summary
        if 'evaluation_summary' in results:
            summary = results['evaluation_summary']
            report_lines.append("EVALUATION SUMMARY:")
            report_lines.append("-" * 40)

            robustness = summary['robustness_analysis']['summary']
            report_lines.append(f"Maximum Performance Drop: {robustness['max_performance_drop_percent']:.2f}%")
            report_lines.append(f"Average Performance Drop: {robustness['avg_performance_drop_percent']:.2f}%")
            report_lines.append(
                f"Robust Scenarios: {robustness['robust_scenarios_count']}/{len([s for s in results['missing_modality_results'].values() if s['missing_modalities']])}")
            report_lines.append(f"Robustness Score: {robustness['robustness_score']:.4f}")
            report_lines.append("")

            # Best and worst performing classes
            class_ranking = summary['class_performance_ranking']
            report_lines.append("CLASS PERFORMANCE RANKING:")
            report_lines.append(f"Best:  {class_ranking[0]['class_name']} (F1: {class_ranking[0]['f1_score']:.4f})")
            report_lines.append(f"Worst: {class_ranking[-1]['class_name']} (F1: {class_ranking[-1]['f1_score']:.4f})")

        report_lines.append("")
        report_lines.append("=" * 80)

        # Save report
        report_text = "\n".join(report_lines)
        report_file = os.path.join(self.config.results_dir, 'evaluation_report.txt')
        with open(report_file, 'w') as f:
            f.write(report_text)

        logging.info(f"Evaluation report saved to: {report_file}")
        return report_text