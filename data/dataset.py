"""
Multi-modal dataset loader for welding quality detection
Fixed for Windows multiprocessing compatibility
"""

import os
import torch
import torchaudio
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import random
from sklearn.model_selection import train_test_split
import logging
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import platform

class WindowsCompatibleTransforms:
    """Transforms that can be pickled for Windows multiprocessing"""

    @staticmethod
    def resize_and_crop(size=(224, 224)):
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(size),
        ])

    @staticmethod
    def validation_resize(size=(224, 224)):
        return transforms.Resize(size)

    @staticmethod
    def normalize():
        return transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

class AudioAugmentation:
    """Audio augmentation functions that work with Windows multiprocessing"""

    def __init__(self, noise_prob=0.3, shift_prob=0.3):
        self.noise_prob = noise_prob
        self.shift_prob = shift_prob

    def add_noise(self, waveform, noise_factor=0.01):
        if random.random() < self.noise_prob:
            return waveform + torch.randn_like(waveform) * noise_factor
        return waveform

    def time_shift(self, waveform, max_shift=100):
        if random.random() < self.shift_prob:
            shift = random.randint(-max_shift, max_shift)
            return torch.roll(waveform, shifts=shift)
        return waveform

    def __call__(self, waveform):
        waveform = self.add_noise(waveform)
        waveform = self.time_shift(waveform)
        return waveform

class MultiModalWeldingDataset(Dataset):
    """Multi-modal dataset with Windows compatibility"""

    def __init__(self, data_root: str, config, split: str = 'train', missing_scenario: List[str] = None):
        self.data_root = Path(data_root)
        self.config = config
        self.split = split
        self.missing_scenario = missing_scenario or []
        self.is_windows = platform.system() == 'Windows'

        if not self.data_root.exists():
            raise FileNotFoundError(f"Data root not found: {data_root}")

        self._setup_transforms()
        self.samples = self._load_samples()

        if not self.samples:
            raise ValueError(f"No samples found in {data_root}")

        logging.info(f"Loaded {len(self.samples)} samples for {split} split")
        self._print_class_distribution()

    def _setup_transforms(self):
        """Setup transforms without lambda functions"""

        if self.split == 'train':
            self.image_transform = transforms.Compose([
                WindowsCompatibleTransforms.resize_and_crop(self.config.image_size),
                transforms.RandomHorizontalFlip(p=0.3),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
                transforms.ToTensor(),
                WindowsCompatibleTransforms.normalize()
            ])
            self.audio_augmentation = AudioAugmentation()
        else:
            self.image_transform = transforms.Compose([
                WindowsCompatibleTransforms.validation_resize(self.config.image_size),
                transforms.ToTensor(),
                WindowsCompatibleTransforms.normalize()
            ])
            self.audio_augmentation = None

    def _load_samples(self) -> List[Dict]:
        """Load all sample paths"""
        samples = []

        for class_idx, class_name in enumerate(self.config.class_names):
            image_dir = self.data_root / 'image' / class_name

            if not image_dir.exists():
                logging.warning(f"Image directory not found: {image_dir}")
                continue

            image_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))

            for image_file in image_files:
                sample_id = image_file.stem

                sample_paths = {
                    'image': str(image_file),
                    'sound': str(self.data_root / 'sound' / class_name / f"{sample_id}.wav"),
                    'current': str(self.data_root / 'current' / class_name / f"{sample_id}.npy")
                }

                existing_modalities = []
                for modality, path in sample_paths.items():
                    if Path(path).exists():
                        existing_modalities.append(modality)

                if existing_modalities:
                    samples.append({
                        'paths': sample_paths,
                        'existing_modalities': existing_modalities,
                        'label': class_idx,
                        'class_name': class_name,
                        'sample_id': sample_id
                    })

        return samples

    def _print_class_distribution(self):
        """Print class distribution"""
        class_counts = [0] * len(self.config.class_names)
        for sample in self.samples:
            class_counts[sample['label']] += 1

        logging.info(f"\nClass distribution for {self.split} split:")
        for class_name, count in zip(self.config.class_names, class_counts):
            percentage = (count / len(self.samples)) * 100 if self.samples else 0
            logging.info(f"  {class_name}: {count} samples ({percentage:.1f}%)")

    def _load_image(self, path: str) -> torch.Tensor:
        """Load and preprocess image"""
        try:
            if not Path(path).exists():
                return torch.zeros((3, *self.config.image_size))

            image = Image.open(path).convert('RGB')
            return self.image_transform(image)

        except Exception as e:
            logging.error(f"Error loading image {path}: {e}")
            return torch.zeros((3, *self.config.image_size))

    def _load_audio(self, path: str) -> torch.Tensor:
        """Load and preprocess audio"""
        try:
            if not Path(path).exists():
                return torch.zeros(self.config.audio_length)

            waveform, sample_rate = torchaudio.load(path)

            if sample_rate != self.config.audio_freq:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sample_rate,
                    new_freq=self.config.audio_freq
                )
                waveform = resampler(waveform)

            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            waveform = waveform.squeeze(0)

            if len(waveform) > self.config.audio_length:
                if self.split == 'train':
                    start_idx = random.randint(0, len(waveform) - self.config.audio_length)
                else:
                    start_idx = (len(waveform) - self.config.audio_length) // 2
                waveform = waveform[start_idx:start_idx + self.config.audio_length]
            else:
                pad_length = self.config.audio_length - len(waveform)
                waveform = torch.nn.functional.pad(waveform, (0, pad_length))

            if self.audio_augmentation:
                waveform = self.audio_augmentation(waveform)

            if waveform.std() > 1e-8:
                waveform = (waveform - waveform.mean()) / waveform.std()

            return waveform

        except Exception as e:
            logging.error(f"Error loading audio {path}: {e}")
            return torch.zeros(self.config.audio_length)

    def _load_current(self, path: str) -> torch.Tensor:
        """Load and preprocess current signal"""
        try:
            if not Path(path).exists():
                return torch.zeros(self.config.current_length)

            current_data = np.load(path)
            current_tensor = torch.FloatTensor(current_data)

            if len(current_tensor.shape) > 1:
                current_tensor = current_tensor.flatten()

            if len(current_tensor) > self.config.current_length:
                if self.split == 'train':
                    start_idx = random.randint(0, len(current_tensor) - self.config.current_length)
                else:
                    start_idx = (len(current_tensor) - self.config.current_length) // 2
                current_tensor = current_tensor[start_idx:start_idx + self.config.current_length]
            else:
                pad_length = self.config.current_length - len(current_tensor)
                current_tensor = torch.nn.functional.pad(current_tensor, (0, pad_length))

            if self.split == 'train' and random.random() < 0.2:
                noise = torch.randn_like(current_tensor) * 0.01
                current_tensor = current_tensor + noise

            if current_tensor.std() > 1e-8:
                current_tensor = (current_tensor - current_tensor.mean()) / current_tensor.std()

            return current_tensor

        except Exception as e:
            logging.error(f"Error loading current {path}: {e}")
            return torch.zeros(self.config.current_length)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        """Get a single sample"""
        sample = self.samples[idx]
        available_modalities = sample['existing_modalities'].copy()

        if self.missing_scenario:
            available_modalities = [m for m in available_modalities if m not in self.missing_scenario]
        elif self.split == 'train':
            missing_prob = getattr(self.config, 'train_missing_prob', 0.1)
            if random.random() < missing_prob and len(available_modalities) > 1:
                num_to_remove = random.randint(1, min(2, len(available_modalities) - 1))
                modalities_to_remove = random.sample(available_modalities, num_to_remove)
                available_modalities = [m for m in available_modalities if m not in modalities_to_remove]

        modalities = {}
        for modality in self.config.modalities:
            if modality in available_modalities and modality in sample['existing_modalities']:
                if modality == 'image':
                    modalities[modality] = self._load_image(sample['paths'][modality])
                elif modality == 'sound':
                    modalities[modality] = self._load_audio(sample['paths'][modality])
                elif modality == 'current':
                    modalities[modality] = self._load_current(sample['paths'][modality])
            else:
                modalities[modality] = None

        return {
            'modalities': modalities,
            'label': torch.tensor(sample['label'], dtype=torch.long),
            'sample_id': sample['sample_id'],
            'class_name': sample['class_name'],
            'available_modalities': available_modalities,
            'existing_modalities': sample['existing_modalities']
        }

    def get_data_loaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Create data loaders with Windows compatibility"""

        train_samples, temp_samples = train_test_split(
            self.samples,
            test_size=(1 - self.config.train_ratio),
            stratify=[s['label'] for s in self.samples],
            random_state=self.config.seed
        )

        val_samples, test_samples = train_test_split(
            temp_samples,
            test_size=(self.config.test_ratio / (self.config.val_ratio + self.config.test_ratio)),
            stratify=[s['label'] for s in temp_samples],
            random_state=self.config.seed
        )

        datasets = {}
        for split_name, split_samples in [('train', train_samples), ('val', val_samples), ('test', test_samples)]:
            split_dataset = MultiModalWeldingDataset.__new__(MultiModalWeldingDataset)
            split_dataset.__dict__.update(self.__dict__)
            split_dataset.samples = split_samples
            split_dataset.split = split_name
            split_dataset._setup_transforms()
            datasets[split_name] = split_dataset

        # Windows-compatible loader settings
        loader_kwargs = {
            'batch_size': self.config.batch_size,
            'collate_fn': self._collate_fn,
            'num_workers': 0 if self.is_windows else 4,  # Force single-thread on Windows
            'pin_memory': not self.is_windows,  # Disable on Windows
        }

        train_loader = DataLoader(datasets['train'], shuffle=True, drop_last=True, **loader_kwargs)
        val_loader = DataLoader(datasets['val'], shuffle=False, **loader_kwargs)
        test_loader = DataLoader(datasets['test'], shuffle=False, **loader_kwargs)

        return train_loader, val_loader, test_loader

    def _collate_fn(self, batch: List[Dict]) -> Dict:
        """Custom collate function"""
        batch_data = {
            'modalities': {modality: [] for modality in self.config.modalities},
            'labels': [],
            'sample_ids': [],
            'class_names': [],
            'available_modalities': [],
            'existing_modalities': []
        }

        for sample in batch:
            batch_data['labels'].append(sample['label'])
            batch_data['sample_ids'].append(sample['sample_id'])
            batch_data['class_names'].append(sample['class_name'])
            batch_data['available_modalities'].append(sample['available_modalities'])
            batch_data['existing_modalities'].append(sample['existing_modalities'])

            for modality in self.config.modalities:
                if sample['modalities'][modality] is not None:
                    batch_data['modalities'][modality].append(sample['modalities'][modality])
                else:
                    if modality == 'image':
                        placeholder = torch.zeros((3, *self.config.image_size))
                    elif modality == 'sound':
                        placeholder = torch.zeros(self.config.audio_length)
                    elif modality == 'current':
                        placeholder = torch.zeros(self.config.current_length)
                    else:
                        placeholder = torch.zeros(self.config.feature_dim)

                    batch_data['modalities'][modality].append(placeholder)

        for modality in self.config.modalities:
            if batch_data['modalities'][modality]:
                batch_data['modalities'][modality] = torch.stack(batch_data['modalities'][modality])

        batch_data['labels'] = torch.stack(batch_data['labels'])
        return batch_data