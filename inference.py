"""
Real-time inference system for welding quality detection
"""

import os
import torch
import numpy as np
import argparse
from PIL import Image
import torchaudio
import torchvision.transforms as transforms
import json
import logging
from datetime import datetime
from pathlib import Path

from config.config import Config
from models.bottleneck_fusion import ThreeStageWeldingFusion


class WeldingQualityInference:
    """Real-time inference system for welding quality detection"""

    def __init__(self, model_path: str, config_path: str = None):
        """
        Initialize inference system

        Args:
            model_path: Path to trained model
            config_path: Path to config file (optional)
        """
        # Load configuration
        if config_path and os.path.exists(config_path):
            # Load custom config if provided
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            self.config = Config()
            # Update config with loaded values
            for key, value in config_dict.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
        else:
            self.config = Config()

        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize model
        self.model = ThreeStageWeldingFusion(self.config).to(self.device)

        # Load trained weights
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            logging.info(f"Model loaded from: {model_path}")
        else:
            raise FileNotFoundError(f"Model not found: {model_path}")

        # Setup preprocessing transforms
        self._setup_transforms()

        # Confidence thresholds for quality assessment
        self.confidence_thresholds = {
            'high': 0.8,
            'medium': 0.6,
            'low': 0.4
        }

    def _setup_transforms(self):
        """Setup preprocessing transforms"""

        self.image_transform = transforms.Compose([
            transforms.Resize(self.config.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """Preprocess image data"""

        try:
            if isinstance(image_path, str):
                image = Image.open(image_path).convert('RGB')
            else:
                image = image_path  # Assume it's already a PIL image

            return self.image_transform(image).unsqueeze(0)  # Add batch dimension
        except Exception as e:
            logging.error(f"Error preprocessing image: {e}")
            return torch.zeros((1, 3, *self.config.image_size))

    def preprocess_audio(self, audio_path: str) -> torch.Tensor:
        """Preprocess audio data"""

        try:
            if isinstance(audio_path, str):
                waveform, sr = torchaudio.load(audio_path)
            else:
                waveform, sr = audio_path, self.config.audio_freq

            # Resample if necessary
            if sr != self.config.audio_freq:
                resampler = torchaudio.transforms.Resample(sr, self.config.audio_freq)
                waveform = resampler(waveform)

            # Take first channel if stereo
            if waveform.shape[0] > 1:
                waveform = waveform[0:1]

            # Trim or pad to desired length
            if waveform.shape[1] > self.config.audio_length:
                # Center crop
                start_idx = (waveform.shape[1] - self.config.audio_length) // 2
                waveform = waveform[:, start_idx:start_idx + self.config.audio_length]
            else:
                # Pad with zeros
                pad_length = self.config.audio_length - waveform.shape[1]
                waveform = torch.nn.functional.pad(waveform, (0, pad_length))

            return waveform.squeeze(0).unsqueeze(0)  # Remove channel dim, add batch dim

        except Exception as e:
            logging.error(f"Error preprocessing audio: {e}")
            return torch.zeros((1, self.config.audio_length))

    def preprocess_current(self, current_path: str) -> torch.Tensor:
        """Preprocess current signal data"""

        try:
            if isinstance(current_path, str):
                current_data = np.load(current_path)
            else:
                current_data = current_path

            # Convert to tensor
            current_tensor = torch.FloatTensor(current_data)

            # Handle different input shapes
            if len(current_tensor.shape) > 1:
                current_tensor = current_tensor.flatten()

            # Trim or pad to desired length
            if len(current_tensor) > self.config.current_length:
                # Center crop
                start_idx = (len(current_tensor) - self.config.current_length) // 2
                current_tensor = current_tensor[start_idx:start_idx + self.config.current_length]
            else:
                # Pad with zeros
                pad_length = self.config.current_length - len(current_tensor)
                current_tensor = torch.nn.functional.pad(current_tensor, (0, pad_length))

            # Normalize
            current_tensor = (current_tensor - current_tensor.mean()) / (current_tensor.std() + 1e-8)

            return current_tensor.unsqueeze(0)  # Add batch dimension

        except Exception as e:
            logging.error(f"Error preprocessing current: {e}")
            return torch.zeros((1, self.config.current_length))

    def predict_single(self, image_path=None, sound_path=None, current_path=None):
        """
        Predict welding quality for a single sample

        Args:
            image_path: Path to image file or PIL Image
            sound_path: Path to sound file or waveform data
            current_path: Path to current file or numpy array

        Returns:
            Dictionary with prediction results
        """

        # Prepare batch data
        batch_data = {
            'modalities': {},
            'available_modalities': [[]]
        }

        available_modalities = []

        # Load and preprocess each modality
        if image_path is not None:
            batch_data['modalities']['image'] = self.preprocess_image(image_path).to(self.device)
            available_modalities.append('image')
        else:
            batch_data['modalities']['image'] = torch.zeros((1, 3, *self.config.image_size)).to(self.device)

        if sound_path is not None:
            batch_data['modalities']['sound'] = self.preprocess_audio(sound_path).to(self.device)
            available_modalities.append('sound')
        else:
            batch_data['modalities']['sound'] = torch.zeros((1, self.config.audio_length)).to(self.device)

        if current_path is not None:
            batch_data['modalities']['current'] = self.preprocess_current(current_path).to(self.device)
            available_modalities.append('current')
        else:
            batch_data['modalities']['current'] = torch.zeros((1, self.config.current_length)).to(self.device)

        batch_data['available_modalities'] = [available_modalities]

        # Ensure at least one modality is available
        if not available_modalities:
            raise ValueError("At least one modality must be provided")

        # Perform inference
        with torch.no_grad():
            model_output = self.model(batch_data)

            # Get predictions and confidence
            logits = model_output['logits']
            probabilities = torch.softmax(logits, dim=1)
            predicted_class_idx = torch.argmax(logits, dim=1).item()
            confidence = probabilities[0, predicted_class_idx].item()

            # Get all class probabilities
            class_probabilities = {
                class_name: prob.item()
                for class_name, prob in zip(self.config.class_names, probabilities[0])
            }

            # Determine confidence level
            if confidence >= self.confidence_thresholds['high']:
                confidence_level = 'High'
            elif confidence >= self.confidence_thresholds['medium']:
                confidence_level = 'Medium'
            else:
                confidence_level = 'Low'

            # Get fusion weights and other intermediate results
            fusion_weights = model_output['fusion_weights'][0].cpu().numpy()
            modality_confidences = model_output['confidences'][0].cpu().numpy()

        # Prepare results
        results = {
            'predicted_class': self.config.class_names[predicted_class_idx],
            'predicted_class_idx': predicted_class_idx,
            'confidence': confidence,
            'confidence_level': confidence_level,
            'class_probabilities': class_probabilities,
            'available_modalities': available_modalities,
            'fusion_weights': {
                modality: weight for modality, weight in zip(self.config.modalities, fusion_weights)
            },
            'modality_confidences': {
                modality: conf for modality, conf in zip(self.config.modalities, modality_confidences)
            },
            'prediction_timestamp': datetime.now().isoformat(),
            'beta_parameter': model_output.get('beta', 'N/A')
        }

        return results

    def predict_batch(self, input_dir: str, output_file: str = None):
        """
        Predict welding quality for a batch of samples

        Args:
            input_dir: Directory containing input files
            output_file: Path to save batch results (optional)
        """

        input_path = Path(input_dir)
        if not input_path.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")

        # Find all sample files
        image_files = list(input_path.glob("*.jpg")) + list(input_path.glob("*.png"))
        sound_files = list(input_path.glob("*.wav"))
        current_files = list(input_path.glob("*.npy"))

        # Group files by sample ID (assuming naming convention)
        samples = {}

        for img_file in image_files:
            sample_id = img_file.stem
            if sample_id not in samples:
                samples[sample_id] = {}
            samples[sample_id]['image'] = str(img_file)

        for sound_file in sound_files:
            sample_id = sound_file.stem
            if sample_id not in samples:
                samples[sample_id] = {}
            samples[sample_id]['sound'] = str(sound_file)

        for current_file in current_files:
            sample_id = current_file.stem
            if sample_id not in samples:
                samples[sample_id] = {}
            samples[sample_id]['current'] = str(current_file)

        # Process each sample
        batch_results = []

        for sample_id, file_paths in samples.items():
            logging.info(f"Processing sample: {sample_id}")

            try:
                result = self.predict_single(
                    image_path=file_paths.get('image'),
                    sound_path=file_paths.get('sound'),
                    current_path=file_paths.get('current')
                )
                result['sample_id'] = sample_id
                batch_results.append(result)

            except Exception as e:
                logging.error(f"Error processing sample {sample_id}: {e}")
                batch_results.append({
                    'sample_id': sample_id,
                    'error': str(e),
                    'prediction_timestamp': datetime.now().isoformat()
                })

        # Save results if output file is specified
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(batch_results, f, indent=2)
            logging.info(f"Batch results saved to: {output_file}")

        return batch_results


def main():
    """Command line interface for inference"""

    parser = argparse.ArgumentParser(description="Welding Quality Detection Inference")
    parser.add_argument('--model_path', required=True, help='Path to trained model')
    parser.add_argument('--input_dir', help='Input directory for batch processing')
    parser.add_argument('--image_path', help='Path to single image file')
    parser.add_argument('--sound_path', help='Path to single sound file')
    parser.add_argument('--current_path', help='Path to single current file')
    parser.add_argument('--output_file', help='Output file for results')
    parser.add_argument('--config_path', help='Path to config file')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')

    args = parser.parse_args()

    # Setup logging
    log_level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Initialize inference system
    inference_system = WeldingQualityInference(args.model_path, args.config_path)

    if args.input_dir:
        # Batch processing
        logging.info(f"Starting batch processing: {args.input_dir}")
        results = inference_system.predict_batch(args.input_dir, args.output_file)

        # Print summary
        print(f"\nProcessed {len(results)} samples")
        if results:
            predictions = {}
            for result in results:
                if 'predicted_class' in result:
                    pred_class = result['predicted_class']
                    predictions[pred_class] = predictions.get(pred_class, 0) + 1

            print("Prediction summary:")
            for class_name, count in predictions.items():
                print(f"  {class_name}: {count}")

    else:
        # Single sample processing
        if not any([args.image_path, args.sound_path, args.current_path]):
            print(
                "Error: Must provide either --input_dir or at least one of --image_path, --sound_path, --current_path")
            return

        logging.info("Processing single sample...")
        result = inference_system.predict_single(
            image_path=args.image_path,
            sound_path=args.sound_path,
            current_path=args.current_path
        )

        # Print results
        print(f"\nWelding Quality Detection Results:")
        print(f"{'=' * 50}")
        print(f"Predicted Class: {result['predicted_class']}")
        print(f"Confidence: {result['confidence']:.4f} ({result['confidence_level']})")
        print(f"Available Modalities: {', '.join(result['available_modalities'])}")
        print(f"\nClass Probabilities:")
        for class_name, prob in result['class_probabilities'].items():
            print(f"  {class_name}: {prob:.4f}")
        print(f"\nFusion Weights:")
        for modality, weight in result['fusion_weights'].items():
            print(f"  {modality}: {weight:.4f}")

        # Save single result if output file specified
        if args.output_file:
            with open(args.output_file, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"\nResults saved to: {args.output_file}")


if __name__ == "__main__":
    main()