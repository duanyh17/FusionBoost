"""
增强版电流编码器完整训练对比脚本 - 修复版本
修复了torch.uniform错误、数据形状不一致、多模态数据增强等问题
"""

import os
import json
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)


def safe_json_convert(obj):
    """安全的JSON类型转换"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy().tolist()
    elif isinstance(obj, dict):
        return {key: safe_json_convert(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [safe_json_convert(item) for item in obj]
    else:
        return obj


class CurrentDataset(Dataset):
    """电流数据集类 - 修复版本，解决数据形状不一致问题"""

    def __init__(self, data_list, labels, target_length=None, scaler=None, is_training=False, enable_augment=False):
        """
        Args:
            data_list: 原始数据列表，每个元素可能长度不同
            labels: 标签列表
            target_length: 目标序列长度，如果为None则使用最大长度
            scaler: 标准化器
            is_training: 是否为训练模式
            enable_augment: 是否启用数据增强（对多模态数据谨慎使用）
        """
        self.labels = np.array(labels, dtype=np.int64)
        self.is_training = is_training
        self.enable_augment = enable_augment

        # 1. 确定目标长度
        if target_length is None:
            self.target_length = max(len(seq) for seq in data_list)
            logger.info(f"自动确定目标序列长度: {self.target_length}")
        else:
            self.target_length = target_length

        # 2. 统一序列长度
        logger.info("正在统一序列长度...")
        processed_data = []
        for i, seq in enumerate(data_list):
            if len(seq.shape) > 1:
                seq = seq.flatten()

            # 长度调整
            if len(seq) > self.target_length:
                # 截取中间部分
                start_idx = (len(seq) - self.target_length) // 2
                seq = seq[start_idx:start_idx + self.target_length]
            elif len(seq) < self.target_length:
                # 使用边缘值填充
                pad_width = self.target_length - len(seq)
                seq = np.pad(seq, (0, pad_width), mode='edge')

            processed_data.append(seq.astype(np.float32))

        # 3. 转换为numpy数组
        self.data = np.array(processed_data, dtype=np.float32)
        logger.info(f"数据形状: {self.data.shape}")

        # 4. 数据标准化
        if scaler is None:
            self.scaler = StandardScaler()
            # 重塑数据进行标准化
            reshaped_data = self.data.reshape(-1, self.data.shape[-1])
            self.scaler.fit(reshaped_data)
            normalized_data = self.scaler.transform(reshaped_data)
            self.data = normalized_data.reshape(self.data.shape)
            logger.info("已创建新的标准化器")
        else:
            self.scaler = scaler
            reshaped_data = self.data.reshape(-1, self.data.shape[-1])
            normalized_data = self.scaler.transform(reshaped_data)
            self.data = normalized_data.reshape(self.data.shape)
            logger.info("使用现有的标准化器")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx].copy()
        label = self.labels[idx]

        # 数据增强（仅在训练时且启用时使用）
        if self.is_training and self.enable_augment:
            data = self._apply_augmentation(data)

        return torch.FloatTensor(data), torch.LongTensor([label]).squeeze()

    def _apply_augmentation(self, data):
        """
        谨慎的数据增强策略，适合多模态对应的数据
        只使用不会破坏时序结构的轻微变换
        """
        # 1. 轻微高斯噪声（标准差很小）
        if np.random.random() < 0.3:
            noise_std = 0.01 * np.std(data)
            noise = np.random.normal(0, noise_std, data.shape)
            data = data + noise

        # 2. 非常轻微的幅度缩放（避免破坏模态对应关系）
        if np.random.random() < 0.2:
            # 修复：使用正确的随机数生成方法
            scale = 0.98 + 0.04 * np.random.random()  # 0.98 到 1.02 之间
            data = data * scale

        # 3. 轻微的DC偏移
        if np.random.random() < 0.2:
            offset = 0.01 * np.std(data) * (np.random.random() - 0.5)
            data = data + offset

        return data


class OriginalCurrentEncoder(nn.Module):
    """原版电流编码器"""

    def __init__(self, input_dim, hidden_dim=128, num_classes=6):
        super().__init__()

        # 1D卷积特征提取
        self.conv_layers = nn.Sequential(
            # 第一层卷积
            nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            # 第二层卷积
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            # 第三层卷积
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        # 分类器
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        # 添加通道维度
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # [B, 1, L]

        # 特征提取
        features = self.conv_layers(x)

        # 分类
        output = self.classifier(features)

        return output


class EnhancedCurrentEncoder(nn.Module):
    """增强版电流编码器 - 改进架构"""

    def __init__(self, input_dim, hidden_dim=256, num_classes=6):
        super().__init__()

        # 多尺度卷积分支
        self.short_conv = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        self.medium_conv = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        self.long_conv = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=15, padding=7),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=15, padding=7),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        # 特征融合
        self.fusion = nn.Sequential(
            nn.Conv1d(192, 256, kernel_size=3, padding=1),  # 64*3 = 192
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(16)  # 固定输出长度
        )

        # 自注意力机制
        self.self_attention = nn.MultiheadAttention(
            embed_dim=128, num_heads=8, batch_first=True
        )

        # 位置编码
        self.pos_encoding = nn.Parameter(torch.randn(1, 16, 128))

        # 分类器
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x):
        # 添加通道维度
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # [B, 1, L]

        batch_size = x.size(0)

        # 多尺度特征提取
        short_feat = self.short_conv(x)
        medium_feat = self.medium_conv(x)
        long_feat = self.long_conv(x)

        # 确保特征长度一致
        min_len = min(short_feat.size(-1), medium_feat.size(-1), long_feat.size(-1))
        short_feat = short_feat[:, :, :min_len]
        medium_feat = medium_feat[:, :, :min_len]
        long_feat = long_feat[:, :, :min_len]

        # 特征融合
        combined = torch.cat([short_feat, medium_feat, long_feat], dim=1)
        fused = self.fusion(combined)  # [B, 128, 16]

        # 自注意力
        fused_transposed = fused.transpose(1, 2)  # [B, 16, 128]
        fused_transposed = fused_transposed + self.pos_encoding[:, :fused_transposed.size(1), :]

        attended, _ = self.self_attention(
            fused_transposed, fused_transposed, fused_transposed
        )

        # 分类
        output = self.classifier(attended)

        return output


def load_current_data(data_dir, max_samples_per_class=None):
    """
    加载电流数据 - 改进版本
    """
    logger.info(f"正在从 {data_dir} 加载电流数据...")

    if not os.path.exists(data_dir):
        # 尝试一些可能的路径
        possible_paths = [
            r"E:\11_weld_data\dataset\current",
            "current_data",
            "../current_data",
            "./current_data"
        ]

        for path in possible_paths:
            if os.path.exists(path):
                data_dir = path
                logger.info(f"找到数据目录: {data_dir}")
                break
        else:
            raise FileNotFoundError(f"找不到数据目录。尝试过的路径: {possible_paths}")

    current_data = []
    labels = []
    class_names = []

    # 获取所有类别目录
    class_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    class_dirs.sort()

    for class_idx, class_name in enumerate(class_dirs):
        class_path = os.path.join(data_dir, class_name)
        class_names.append(class_name)

        # 获取所有.npy文件
        npy_files = [f for f in os.listdir(class_path) if f.endswith('.npy')]

        # 限制样本数量（如果指定）
        if max_samples_per_class is not None:
            npy_files = npy_files[:max_samples_per_class]

        class_count = 0
        for npy_file in npy_files:
            try:
                file_path = os.path.join(class_path, npy_file)
                data = np.load(file_path)

                # 确保数据是有效的
                if data.size == 0:
                    continue

                current_data.append(data)  # 保持原始形状，在Dataset中处理
                labels.append(class_idx)
                class_count += 1

            except Exception as e:
                logger.warning(f"加载文件失败 {file_path}: {e}")
                continue

        logger.info(f"类别 {class_name}: 加载了 {class_count} 个样本")

    logger.info(f"总计加载 {len(current_data)} 个电流样本")
    return current_data, labels, class_names


class CurrentEncoderTrainer:
    """电流编码器训练器"""

    def __init__(self, device='cuda', output_dir='current_encoder_results_fixed'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        logger.info(f"使用设备: {self.device}")
        logger.info(f"输出目录: {output_dir}")

    def count_parameters(self, model):
        """计算模型参数量"""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def prepare_data(self, data_dir, max_samples_per_class=None, enable_augment=False):
        """准备训练数据"""
        # 加载原始数据
        current_data, labels, class_names = load_current_data(data_dir, max_samples_per_class)

        # 数据划分
        X_train_raw, X_temp, y_train, y_temp = train_test_split(
            current_data, labels, test_size=0.4, random_state=42, stratify=labels
        )

        X_val_raw, X_test_raw, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
        )

        logger.info(f"数据划分: 训练集 {len(X_train_raw)}, 验证集 {len(X_val_raw)}, 测试集 {len(X_test_raw)}")

        # 确定统一的序列长度
        all_lengths = [len(seq.flatten()) for seq in current_data]
        target_length = max(all_lengths)
        logger.info(f"统一序列长度: {target_length}")

        # 创建数据集
        train_dataset = CurrentDataset(
            X_train_raw, y_train,
            target_length=target_length,
            is_training=True,
            enable_augment=enable_augment
        )

        val_dataset = CurrentDataset(
            X_val_raw, y_val,
            target_length=target_length,
            scaler=train_dataset.scaler,
            is_training=False
        )

        test_dataset = CurrentDataset(
            X_test_raw, y_test,
            target_length=target_length,
            scaler=train_dataset.scaler,
            is_training=False
        )

        return train_dataset, val_dataset, test_dataset, class_names

    def train_epoch(self, model, train_loader, criterion, optimizer, epoch):
        """训练一个epoch"""
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}')

        for batch_idx, (data, target) in enumerate(progress_bar):
            data, target = data.to(self.device), target.to(self.device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

            # 更新进度条
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100. * correct / total:.2f}%'
            })

        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total

        return avg_loss, accuracy

    def validate(self, model, val_loader, criterion):
        """验证模型"""
        model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                loss = criterion(output, target)

                total_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)

                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())

        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total
        f1_macro = f1_score(all_targets, all_preds, average='macro')

        return avg_loss, accuracy, f1_macro

    def train_model(self, model, train_dataset, val_dataset, model_name, epochs=100):
        """训练模型"""
        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)

        # 优化器和损失函数
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

        # 训练历史
        history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [], 'val_f1': []
        }

        best_val_acc = 0
        best_model_state = None
        patience_counter = 0
        max_patience = 15

        logger.info(f"\n开始训练 {model_name}")
        logger.info("=" * 60)

        for epoch in range(1, epochs + 1):
            # 训练
            train_loss, train_acc = self.train_epoch(
                model, train_loader, criterion, optimizer, epoch
            )

            # 验证
            val_loss, val_acc, val_f1 = self.validate(
                model, val_loader, criterion
            )

            # 学习率调度
            scheduler.step()

            # 记录历史
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['val_f1'].append(val_f1)

            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = model.state_dict().copy()
                patience_counter = 0

                # 保存检查点
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': best_model_state,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_f1': val_f1
                }, os.path.join(self.output_dir, f'{model_name.replace(" ", "_")}_best.pth'))

            else:
                patience_counter += 1

            # 早停
            if patience_counter >= max_patience:
                logger.info(f"早停触发 (patience={max_patience})")
                break

            # 每5个epoch打印进度
            if epoch % 5 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                logger.info(
                    f'Epoch {epoch}: Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}, Val F1={val_f1:.4f}, LR={current_lr:.6f}')

        # 加载最佳模型
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        logger.info(f"{model_name} 训练完成！最佳验证准确率: {best_val_acc:.4f}")

        return history, best_val_acc

    def evaluate_model(self, model, test_dataset, class_names):
        """评估模型"""
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)

        model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                pred = output.argmax(dim=1)

                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())

        # 计算指标
        test_acc = accuracy_score(all_targets, all_preds)
        test_f1 = f1_score(all_targets, all_preds, average='macro')

        # 详细报告
        report = classification_report(
            all_targets, all_preds,
            target_names=class_names,
            output_dict=True
        )

        # 混淆矩阵
        cm = confusion_matrix(all_targets, all_preds)

        return {
            'test_acc': test_acc,
            'test_f1': test_f1,
            'report': report,
            'confusion_matrix': cm,
            'predictions': all_preds,
            'targets': all_targets
        }

    def plot_training_curves(self, histories, model_names, save_path):
        """绘制训练曲线"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('电流编码器训练对比', fontsize=16, fontweight='bold')

        colors = ['blue', 'red']

        for i, (history, name, color) in enumerate(zip(histories, model_names, colors)):
            # 损失曲线
            axes[0, 0].plot(history['train_loss'], color=color, label=f'{name} 训练', alpha=0.7)
            axes[0, 0].plot(history['val_loss'], color=color, linestyle='--', label=f'{name} 验证')

            # 准确率曲线
            axes[0, 1].plot(history['train_acc'], color=color, label=f'{name} 训练', alpha=0.7)
            axes[0, 1].plot(history['val_acc'], color=color, linestyle='--', label=f'{name} 验证')

            # F1分数
            axes[1, 0].plot(history['val_f1'], color=color, label=f'{name}')

        # 设置图表
        axes[0, 0].set_title('损失曲线')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].set_title('准确率曲线')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        axes[1, 0].set_title('F1分数曲线')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 最终性能对比
        final_accs = [max(history['val_acc']) for history in histories]
        bars = axes[1, 1].bar(model_names, final_accs, color=colors, alpha=0.7)
        axes[1, 1].set_title('最佳验证准确率对比')
        axes[1, 1].set_ylabel('Accuracy')
        axes[1, 1].set_ylim(0, max(final_accs) * 1.1)

        # 添加数值标签
        for bar, acc in zip(bars, final_accs):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                            f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')

        axes[1, 1].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"训练曲线已保存: {save_path}")

    def plot_confusion_matrices(self, results, class_names, model_names, save_path):
        """绘制混淆矩阵对比"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('混淆矩阵对比', fontsize=16, fontweight='bold')

        for i, (result, name) in enumerate(zip(results, model_names)):
            cm = result['confusion_matrix']

            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=class_names, yticklabels=class_names,
                        ax=axes[i])
            axes[i].set_title(f'{name}\n准确率: {result["test_acc"]:.3f}')
            axes[i].set_xlabel('预测类别')
            axes[i].set_ylabel('真实类别')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"混淆矩阵已保存: {save_path}")

    def run_complete_comparison(self, data_dir, max_samples_per_class=None, enable_augment=False, epochs=100):
        """运行完整的对比实验"""
        logger.info("🚀 开始完整的电流编码器训练对比实验")
        logger.info("=" * 80)

        # 准备数据
        train_dataset, val_dataset, test_dataset, class_names = self.prepare_data(
            data_dir, max_samples_per_class, enable_augment
        )

        # 获取输入维度
        sample_data, _ = train_dataset[0]
        input_dim = sample_data.shape[0]
        num_classes = len(class_names)

        logger.info(f"输入维度: {input_dim}")
        logger.info(f"类别数量: {num_classes}")
        logger.info(f"类别名称: {class_names}")

        # 创建模型
        models = {
            '原版编码器': OriginalCurrentEncoder(input_dim=input_dim, num_classes=num_classes).to(self.device),
            '增强版编码器': EnhancedCurrentEncoder(input_dim=input_dim, num_classes=num_classes).to(self.device)
        }

        # 显示模型参数量
        for name, model in models.items():
            param_count = self.count_parameters(model)
            logger.info(f"{name} 参数量: {param_count:,}")

        # 训练和评估
        histories = []
        results = []
        model_names = list(models.keys())

        for model_name, model in models.items():
            # 训练模型
            history, best_val_acc = self.train_model(
                model, train_dataset, val_dataset, model_name, epochs
            )
            histories.append(history)

            # 评估模型
            result = self.evaluate_model(model, test_dataset, class_names)
            results.append(result)

            logger.info(f"\n{model_name} 最终结果:")
            logger.info(f"  测试准确率: {result['test_acc']:.4f}")
            logger.info(f"  测试F1分数: {result['test_f1']:.4f}")
            logger.info(f"  最佳验证准确率: {best_val_acc:.4f}")

        # 生成可视化
        self.plot_training_curves(histories, model_names, os.path.join(self.output_dir, 'training_curves.png'))
        self.plot_confusion_matrices(results, class_names, model_names,
                                     os.path.join(self.output_dir, 'confusion_matrices.png'))

        # 生成报告
        self.generate_summary_report(histories, results, model_names, class_names)

        return histories, results

    def generate_summary_report(self, histories, results, model_names, class_names):
        """生成总结报告"""
        logger.info("\n" + "=" * 80)
        logger.info("🎯 完整训练对比总结")
        logger.info("=" * 80)

        # 性能对比
        orig_result, enh_result = results[0], results[1]
        orig_best_val = max(histories[0]['val_acc'])
        enh_best_val = max(histories[1]['val_acc'])

        # 计算改进百分比
        test_acc_imp = (enh_result['test_acc'] - orig_result['test_acc']) / orig_result['test_acc'] * 100
        f1_imp = (enh_result['test_f1'] - orig_result['test_f1']) / orig_result['test_f1'] * 100
        val_acc_imp = (enh_best_val - orig_best_val) / orig_best_val * 100

        logger.info("📊 性能指标对比:")
        logger.info(
            f"  测试准确率:     原版 {orig_result['test_acc']:.4f} → 增强版 {enh_result['test_acc']:.4f} ({test_acc_imp:+.1f}%)")
        logger.info(
            f"  Macro F1-Score: 原版 {orig_result['test_f1']:.4f} → 增强版 {enh_result['test_f1']:.4f} ({f1_imp:+.1f}%)")
        logger.info(f"  验证准确率:     原版 {orig_best_val:.4f} → 增强版 {enh_best_val:.4f} ({val_acc_imp:+.1f}%)")

        # 模型复杂度
        orig_params = self.count_parameters(OriginalCurrentEncoder(input_dim=1000))
        enh_params = self.count_parameters(EnhancedCurrentEncoder(input_dim=1000))
        param_inc = (enh_params / orig_params - 1) * 100

        logger.info(f"\n🔧 模型复杂度:")
        logger.info(f"  参数量:         原版 {orig_params:,} → 增强版 {enh_params:,} ({param_inc:+.1f}%)")

        # 总体评估
        logger.info(f"\n🎯 总体评估:")
        if test_acc_imp > 5 and f1_imp > 5:
            logger.info("✅ 增强版编码器显著提升了性能，强烈推荐使用！")
        elif test_acc_imp > 2 or f1_imp > 2:
            logger.info("✅ 增强版编码器有明显改进，建议使用。")
        elif test_acc_imp > 0 or f1_imp > 0:
            logger.info("⚡ 增强版编码器有轻微改进，可以考虑使用。")
        else:
            logger.info("❌ 增强版编码器改进有限，建议进一步优化架构。")

        # 保存完整结果
        complete_results = {
            'experiment_info': {
                'timestamp': datetime.now().isoformat(),
                'device': str(self.device),
                'class_names': class_names,
                'num_classes': len(class_names),
                'data_augmentation_enabled': False  # 标明未使用数据增强
            },
            'performance_comparison': {
                'original': {
                    'test_accuracy': orig_result['test_acc'],
                    'test_f1': orig_result['test_f1'],
                    'best_val_acc': orig_best_val
                },
                'enhanced': {
                    'test_accuracy': enh_result['test_acc'],
                    'test_f1': enh_result['test_f1'],
                    'best_val_acc': enh_best_val
                },
                'improvements': {
                    'test_acc_improvement_pct': test_acc_imp,
                    'f1_improvement_pct': f1_imp,
                    'val_acc_improvement_pct': val_acc_imp,
                    'parameter_increase_pct': param_inc
                }
            },
            'detailed_results': safe_json_convert({
                model_name: {
                    'training_history': history,
                    'test_evaluation': {k: v for k, v in result.items()
                                        if k not in ['predictions', 'targets']}
                }
                for model_name, history, result in zip(model_names, histories, results)
            })
        }

        # 保存结果
        results_file = os.path.join(self.output_dir, 'complete_training_results.json')
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(complete_results, f, indent=2, ensure_ascii=False)

        logger.info(f"✅ 完整实验结果已保存到: {results_file}")
        logger.info(f"📁 所有文件保存在目录: {self.output_dir}")


def main():
    """主函数"""
    try:
        # 设置参数
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        data_dir = r"E:\11_weld_data\dataset\current"  # 使用您的正确路径
        output_dir = "current_encoder_results_fixed"
        max_samples_per_class = None  # 使用所有样本
        enable_augment = False  # 禁用数据增强（适合多模态数据）
        epochs = 100

        logger.info(f"使用设备: {device}")
        logger.info(f"数据目录: {data_dir}")
        logger.info(f"输出目录: {output_dir}")
        logger.info(f"数据增强: {'启用' if enable_augment else '禁用'}")

        # 创建训练器
        trainer = CurrentEncoderTrainer(device=device, output_dir=output_dir)

        # 运行完整对比实验
        histories, results = trainer.run_complete_comparison(
            data_dir=data_dir,
            max_samples_per_class=max_samples_per_class,
            enable_augment=enable_augment,
            epochs=epochs
        )

        logger.info("🎉 实验完成！所有结果已保存。")

    except Exception as e:
        logger.error(f"实验过程中出错: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()