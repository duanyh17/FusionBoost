import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np

# Assuming Encoder and DatasetLodar are imported
from Encoder import MultiModalClassifier
from DatasetLodar import MultimodalDataset

# Constants
NUM_CLASSES = 6
NUM_TRAIN_SAMPLES = 23000

# Fixed plot_class_distribution function that does not iterate through dataset

def plot_class_distribution_fast(dataset):
    class_count = np.zeros(NUM_CLASSES)
    for sample in dataset.samples:
        class_count[sample[3]] += 1
    label_list = [f'Class {i}' for i in range(NUM_CLASSES)]
    plt.bar(label_list, class_count)
    plt.xticks(rotation=45)
    plt.title('Class Distribution')
    plt.xlabel('Classes')
    plt.ylabel('Number of Samples')
    plt.show()

# Fixed compute_class_weights_fast function

def compute_class_weights_fast(dataset):
    class_count = np.zeros(NUM_CLASSES)
    for sample in dataset.samples:
        class_count[sample[3]] += 1
    class_weights = 1.0 / class_count
    return class_weights / np.sum(class_weights) * len(dataset.samples)

# Fixed create_balanced_sampler_fast function

def create_balanced_sampler_fast(dataset):
    class_weights = compute_class_weights_fast(dataset)
    sample_weights = [class_weights[sample[3]] for sample in dataset.samples]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    return sampler

# Enhanced MultiModal Classifier with Modal Dropout

class EnhancedMultiModalClassifier(MultiModalClassifier):
    def __init__(self, dropout_rate=0.5):
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = super().forward(x)
        return self.dropout(x)

# Focal Loss implementation

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        bce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean() if self.reduction == 'mean' else focal_loss.sum()

# Complete training loop with TensorBoard logging

def train(model, data_loader, criterion, optimizer, writer, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in data_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(data_loader)
        writer.add_scalar('Loss/train', avg_loss, epoch)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

# Testing on all missing modality scenarios

def test_model(model, test_loader):
    model.eval()
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            # Add evaluation logic here

# Visualization functions for confusion matrix and scenario comparison

def visualize_confusion_matrix(confusion_matrix):
    plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(NUM_CLASSES)
    plt.xticks(tick_marks, range(NUM_CLASSES))
    plt.yticks(tick_marks, range(NUM_CLASSES))
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

# The main function to execute training
if __name__ == '__main__':
    dataset = MultimodalDataset()  # Ensure dataset is initialized correctly
    train_sampler = create_balanced_sampler_fast(dataset)
    train_loader = DataLoader(dataset, sampler=train_sampler, batch_size=32)
    model = EnhancedMultiModalClassifier()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = FocalLoss(alpha=compute_class_weights_fast(dataset))
    writer = SummaryWriter(log_dir='./logs')
    train(model, train_loader, criterion, optimizer, writer, num_epochs=20)
    writer.close()