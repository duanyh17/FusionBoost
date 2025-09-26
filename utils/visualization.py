"""
Enhanced visualization utilities for the three-stage fusion system
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from sklearn.metrics import confusion_matrix
import pandas as pd
import logging
from typing import Dict, List, Optional
import torch


class ResultVisualizer:
    def __init__(self, config):
        self.config = config
        plt.style.use('default')
        sns.set_palette("husl")

        # Create visualization directory
        os.makedirs(config.visualization_dir, exist_ok=True)

    def plot_training_curves(self, train_losses: List[float], val_losses: List[float],
                             train_accuracies: List[float] = None, val_accuracies: List[float] = None,
                             stage_transitions: List[int] = None):
        """Plot training and validation curves with stage transitions"""

        fig_height = 10 if train_accuracies is not None else 5
        fig, axes = plt.subplots(2 if train_accuracies is not None else 1, 1, figsize=(12, fig_height))

        if train_accuracies is None:
            axes = [axes]

        epochs = range(len(train_losses))

        # Loss curves
        axes[0].plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
        axes[0].plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
        axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Mark stage transitions
        if stage_transitions:
            stage_names = ['Stage 1\n(Pre-training)', 'Stage 2\n(Knowledge Distillation)', 'Stage 3\n(TLA Activation)']
            colors = ['green', 'orange', 'purple']

            for i, transition in enumerate(stage_transitions):
                if transition < len(epochs):
                    axes[0].axvline(x=transition, color=colors[i % len(colors)],
                                    linestyle='--', alpha=0.7, linewidth=2)
                    if i < len(stage_names):
                        axes[0].text(transition + 1, max(train_losses) * 0.9, stage_names[i],
                                     rotation=90, verticalalignment='top', fontsize=10)

        # Accuracy curves
        if train_accuracies is not None:
            axes[1].plot(epochs, train_accuracies, 'b-', label='Training Accuracy', linewidth=2)
            axes[1].plot(epochs, val_accuracies, 'r-', label='Validation Accuracy', linewidth=2)
            axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Accuracy')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

            # Mark stage transitions
            if stage_transitions:
                for i, transition in enumerate(stage_transitions):
                    if transition < len(epochs):
                        axes[1].axvline(x=transition, color=colors[i % len(colors)],
                                        linestyle='--', alpha=0.7, linewidth=2)

        plt.tight_layout()

        # Save plot
        plot_path = os.path.join(self.config.visualization_dir, 'training_curves.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        logging.info(f"Training curves saved to: {plot_path}")

    def plot_confusion_matrix(self, cm: np.ndarray, class_names: List[str], normalize: bool = True):
        """Plot confusion matrix with enhanced styling"""

        fig, axes = plt.subplots(1, 2, figsize=(20, 8))

        # Raw confusion matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names,
                    ax=axes[0], cbar_kws={'label': 'Count'})
        axes[0].set_title('Confusion Matrix (Raw Counts)', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Predicted Class')
        axes[0].set_ylabel('True Class')

        # Normalized confusion matrix
        if normalize:
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            sns.heatmap(cm_normalized, annot=True, fmt='.3f', cmap='Blues',
                        xticklabels=class_names, yticklabels=class_names,
                        ax=axes[1], cbar_kws={'label': 'Proportion'})
            axes[1].set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
            axes[1].set_xlabel('Predicted Class')
            axes[1].set_ylabel('True Class')

        plt.tight_layout()

        # Save plot
        plot_path = os.path.join(self.config.visualization_dir, 'confusion_matrix.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        logging.info(f"Confusion matrix saved to: {plot_path}")

    def plot_missing_modality_performance(self, missing_results: Dict):
        """Plot performance under different missing modality scenarios"""

        scenarios = []
        accuracies = []
        f1_scores = []
        aurocs = []

        for scenario_name, results in missing_results.items():
            missing_mods = results.get('missing_modalities', [])
            if missing_mods:
                scenario_label = f"Missing: {', '.join(missing_mods)}"
            else:
                scenario_label = "Complete"

            scenarios.append(scenario_label)
            accuracies.append(results['accuracy'])
            f1_scores.append(results['f1_score'])
            aurocs.append(results['auroc'])

        # Create subplot
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        x_pos = np.arange(len(scenarios))

        # Accuracy plot
        bars1 = axes[0].bar(x_pos, accuracies, color='skyblue', alpha=0.7)
        axes[0].set_title('Accuracy by Missing Modality Scenario', fontweight='bold')
        axes[0].set_ylabel('Accuracy')
        axes[0].set_xticks(x_pos)
        axes[0].set_xticklabels(scenarios, rotation=45, ha='right')
        axes[0].grid(True, alpha=0.3)

        # Add value labels on bars
        for i, bar in enumerate(bars1):
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                         f'{height:.3f}', ha='center', va='bottom')

        # F1 Score plot
        bars2 = axes[1].bar(x_pos, f1_scores, color='lightcoral', alpha=0.7)
        axes[1].set_title('F1-Score by Missing Modality Scenario', fontweight='bold')
        axes[1].set_ylabel('F1-Score')
        axes[1].set_xticks(x_pos)
        axes[1].set_xticklabels(scenarios, rotation=45, ha='right')
        axes[1].grid(True, alpha=0.3)

        for i, bar in enumerate(bars2):
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                         f'{height:.3f}', ha='center', va='bottom')

        # AUROC plot
        bars3 = axes[2].bar(x_pos, aurocs, color='lightgreen', alpha=0.7)
        axes[2].set_title('AUROC by Missing Modality Scenario', fontweight='bold')
        axes[2].set_ylabel('AUROC')
        axes[2].set_xticks(x_pos)
        axes[2].set_xticklabels(scenarios, rotation=45, ha='right')
        axes[2].grid(True, alpha=0.3)

        for i, bar in enumerate(bars3):
            height = bar.get_height()
            axes[2].text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                         f'{height:.3f}', ha='center', va='bottom')

        plt.tight_layout()

        # Save plot
        plot_path = os.path.join(self.config.visualization_dir, 'missing_modality_performance.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        logging.info(f"Missing modality performance plot saved to: {plot_path}")

    def plot_class_performance(self, results: Dict):
        """Plot per-class performance metrics"""

        class_names = self.config.class_names
        precision_scores = results['precision_per_class']
        recall_scores = results['recall_per_class']
        f1_scores = results['f1_per_class']

        x = np.arange(len(class_names))
        width = 0.25

        fig, ax = plt.subplots(figsize=(14, 8))

        bars1 = ax.bar(x - width, precision_scores, width, label='Precision', alpha=0.8)
        bars2 = ax.bar(x, recall_scores, width, label='Recall', alpha=0.8)
        bars3 = ax.bar(x + width, f1_scores, width, label='F1-Score', alpha=0.8)

        ax.set_xlabel('Welding Defect Classes')
        ax.set_ylabel('Score')
        ax.set_title('Per-Class Performance Metrics', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add value labels on bars
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=8)

        plt.tight_layout()

        # Save plot
        plot_path = os.path.join(self.config.visualization_dir, 'class_performance.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        logging.info(f"Class performance plot saved to: {plot_path}")

    def plot_fusion_weight_evolution(self, fusion_weights_history: List[np.ndarray]):
        """Plot evolution of fusion weights during training"""

        if not fusion_weights_history:
            return

        # Sample weights at regular intervals
        num_samples = min(len(fusion_weights_history), 20)
        sample_indices = np.linspace(0, len(fusion_weights_history) - 1, num_samples).astype(int)

        sampled_weights = [fusion_weights_history[i] for i in sample_indices]
        sampled_epochs = sample_indices

        # Calculate mean weights for each epoch
        mean_weights_per_epoch = []
        for weights_batch in sampled_weights:
            mean_weights = np.mean(weights_batch, axis=0)
            mean_weights_per_epoch.append(mean_weights)

        mean_weights_per_epoch = np.array(mean_weights_per_epoch)

        # Plot weight evolution
        fig, ax = plt.subplots(figsize=(12, 8))

        for i, modality in enumerate(self.config.modalities):
            ax.plot(sampled_epochs, mean_weights_per_epoch[:, i],
                    marker='o', linewidth=2, label=f'{modality.capitalize()}')

        ax.set_title('Evolution of Fusion Weights During Training', fontsize=14, fontweight='bold')
        ax.set_xlabel('Training Epoch')
        ax.set_ylabel('Average Fusion Weight')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save plot
        plot_path = os.path.join(self.config.visualization_dir, 'fusion_weight_evolution.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        logging.info(f"Fusion weight evolution plot saved to: {plot_path}")

    def plot_tla_performance_tracking(self, tla_performance_history: List[np.ndarray]):
        """Plot TLA combination performance tracking"""

        if not tla_performance_history:
            return

        # Get final performance
        final_performance = tla_performance_history[-1]

        # Create combination labels
        combination_labels = []
        for i in range(len(final_performance)):
            # Convert index to binary representation for modalities
            binary_repr = format(i, f'0{len(self.config.modalities)}b')
            available_mods = []
            for j, bit in enumerate(binary_repr):
                if bit == '1':
                    available_mods.append(self.config.modalities[j][0].upper())  # First letter

            if available_mods:
                combination_labels.append('+'.join(available_mods))
            else:
                combination_labels.append('None')

        # Filter out combinations with zero performance (not used)
        valid_combinations = final_performance > 0
        valid_performance = final_performance[valid_combinations]
        valid_labels = [combination_labels[i] for i in range(len(combination_labels)) if valid_combinations[i]]

        if len(valid_performance) == 0:
            return

        # Plot performance
        fig, ax = plt.subplots(figsize=(12, 6))

        colors = ['red' if perf < 0.6 else 'orange' if perf < 0.75 else 'green'
                  for perf in valid_performance]

        bars = ax.bar(range(len(valid_performance)), valid_performance, color=colors, alpha=0.7)

        ax.set_title('TLA: Modality Combination Performance Tracking', fontsize=14, fontweight='bold')
        ax.set_xlabel('Modality Combinations')
        ax.set_ylabel('Performance Score')
        ax.set_xticks(range(len(valid_labels)))
        ax.set_xticklabels(valid_labels, rotation=45, ha='right')

        # Add horizontal line for lazy threshold
        ax.axhline(y=0.6, color='red', linestyle='--', alpha=0.8, label='Lazy Threshold')
        ax.axhline(y=0.75, color='orange', linestyle='--', alpha=0.8, label='Medium Threshold')

        # Add value labels on bars
        for bar, perf in zip(bars, valid_performance):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)

        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save plot
        plot_path = os.path.join(self.config.visualization_dir, 'tla_performance_tracking.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        logging.info(f"TLA performance tracking plot saved to: {plot_path}")

    def create_summary_report(self, results: Dict, model_info: Dict = None):
        """Create a comprehensive summary report"""

        report_path = os.path.join(self.config.visualization_dir, 'summary_report.html')

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Welding Quality Detection - Model Performance Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #2E86AB; color: white; padding: 20px; text-align: center; }}
                .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                .metric {{ display: inline-block; margin: 10px; padding: 10px; background-color: #f9f9f9; border-radius: 5px; }}
                .good {{ color: #27ae60; }}
                .warning {{ color: #f39c12; }}
                .poor {{ color: #e74c3c; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Multi-Modal Welding Quality Detection</h1>
                <h2>Three-Stage Fusion Architecture Performance Report</h2>
            </div>

            <div class="section">
                <h3>Overall Performance</h3>
                <div class="metric">
                    <strong>Accuracy:</strong> 
                    <span class="{'good' if results['accuracy'] > 0.85 else 'warning' if results['accuracy'] > 0.7 else 'poor'}">
                        {results['accuracy']:.4f}
                    </span>
                </div>
                <div class="metric">
                    <strong>F1-Score (Macro):</strong> 
                    <span class="{'good' if results['f1_score'] > 0.80 else 'warning' if results['f1_score'] > 0.65 else 'poor'}">
                        {results['f1_score']:.4f}
                    </span>
                </div>
                <div class="metric">
                    <strong>AUROC:</strong> 
                    <span class="{'good' if results['auroc'] > 0.85 else 'warning' if results['auroc'] > 0.7 else 'poor'}">
                        {results['auroc']:.4f}
                    </span>
                </div>
            </div>

            <div class="section">
                <h3>Per-Class Performance</h3>
                <table>
                    <tr>
                        <th>Class</th>
                        <th>Precision</th>
                        <th>Recall</th>
                        <th>F1-Score</th>
                    </tr>
        """

        for i, class_name in enumerate(self.config.class_names):
            if i < len(results['precision_per_class']):
                precision = results['precision_per_class'][i]
                recall = results['recall_per_class'][i]
                f1 = results['f1_per_class'][i]

                html_content += f"""
                    <tr>
                        <td>{class_name.replace('_', ' ').title()}</td>
                        <td>{precision:.4f}</td>
                        <td>{recall:.4f}</td>
                        <td>{f1:.4f}</td>
                    </tr>
                """

        html_content += """
                </table>
            </div>

            <div class="section">
                <h3>Missing Modality Robustness</h3>
                <table>
                    <tr>
                        <th>Scenario</th>
                        <th>Accuracy</th>
                        <th>F1-Score</th>
                        <th>Performance Drop</th>
                    </tr>
        """

        complete_accuracy = None
        for scenario_name, scenario_results in results['missing_modality_results'].items():
            missing_mods = scenario_results['missing_modalities']
            accuracy = scenario_results['accuracy']
            f1_score = scenario_results['f1_score']

            if not missing_mods:  # Complete data scenario
                complete_accuracy = accuracy
                scenario_desc = "Complete Data"
                performance_drop = "N/A"
            else:
                scenario_desc = f"Missing: {', '.join(missing_mods)}"
                if complete_accuracy is not None:
                    drop = (complete_accuracy - accuracy) / complete_accuracy * 100
                    performance_drop = f"{drop:.1f}%"
                else:
                    performance_drop = "N/A"

            html_content += f"""
                <tr>
                    <td>{scenario_desc}</td>
                    <td>{accuracy:.4f}</td>
                    <td>{f1_score:.4f}</td>
                    <td>{performance_drop}</td>
                </tr>
            """

        html_content += """
                </table>
            </div>

            <div class="section">
                <h3>Model Architecture Highlights</h3>
                <ul>
                    <li><strong>Three-Stage Training:</strong> Pre-training → Knowledge Distillation → TLA Activation</li>
                    <li><strong>Adaptive Bottleneck Tokens:</strong> Learnable representations for missing modalities</li>
                    <li><strong>Dynamic Weight Fusion:</strong> Confidence-based modality weighting</li>
                    <li><strong>Traceable Laziness Activation:</strong> Performance tracking and lazy combination enhancement</li>
                    <li><strong>Industrial Ready:</strong> Real-time inference with sensor failure robustness</li>
                </ul>
            </div>
        </body>
        </html>
        """

        with open(report_path, 'w') as f:
            f.write(html_content)

        logging.info(f"Summary report saved to: {report_path}")