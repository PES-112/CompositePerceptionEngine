"""
threat_prioritizer_finetuner.py — Fine-tune threat prioritizer on SANPO data.

This module:
1. Loads training metrics accumulated from SANPO preprocessing
2. Fine-tunes threat_prioritizer model state_dict
3. Uses Ground-truth kinetic scores as supervision signal
4. Saves trained model weights

Ground-truth objective:
    Input: (distance_m, velocity_mps, ttc_s, class_name)
    Output: kinetic_score ∈ [0, 10]
    Loss: MSE between predicted_score and ground_truth_score

The threat_prioritizer is a small MLP that learns to weight different
threat components based on SANPO real-world distributions.
"""

import json
import logging
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)


class ThreatDataset(Dataset):
    """PyTorch dataset for threat scoring training."""
    
    def __init__(self, training_cache_path: str):
        """
        Args:
            training_cache_path: Path to training_cache.pkl from preprocessing
        """
        with open(training_cache_path, 'rb') as f:
            self.training_cache = pickle.load(f)
        
        # Extract all valid detection samples
        self.samples = []
        for frame in self.training_cache:
            for det in frame["detections"]:
                self.samples.append({
                    "distance_m": det["distance_m"],
                    "velocity_mps": det["velocity_mps"],
                    "ttc_s": det["ttc_s"] or 0.0,
                    "class_name": det["class_name"],
                    "kinetic_score": det["kinetic_score"]
                })
        
        logger.info(f"ThreatDataset: {len(self.samples)} samples")
        
        # Class encoding for one-hot
        self.classes = list(set(s["class_name"] for s in self.samples))
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Build feature vector: [distance, velocity, ttc, one_hot_class]
        class_idx = self.class_to_idx[sample["class_name"]]
        class_one_hot = np.zeros(len(self.classes), dtype=np.float32)
        class_one_hot[class_idx] = 1.0
        
        # Normalize inputs (important for neural nets)
        distance_norm = min(1.0, sample["distance_m"] / 100.0)
        velocity_norm = np.tanh(sample["velocity_mps"] / 10.0)  # Tanh for unbounded velocity
        ttc_norm = min(1.0, sample["ttc_s"] / 10.0) if sample["ttc_s"] > 0 else 0.0
        
        features = np.concatenate([
            [distance_norm],
            [velocity_norm],
            [ttc_norm],
            class_one_hot
        ]).astype(np.float32)
        
        target = np.array([sample["kinetic_score"] / 10.0], dtype=np.float32)  # Normalize to [0, 1]
        
        return torch.from_numpy(features), torch.from_numpy(target)


class ThreatPrioritizerMLP(nn.Module):
    """
    Small MLP for threat scoring.
    
    Architecture:
    - Input: [distance, velocity, ttc, one_hot_class] (~15-20 dims)
    - Hidden: 64 → 32 → 16 neurons (ReLU)
    - Output: 1 (threat score)
    """
    
    def __init__(self, num_classes: int = 5):
        """
        Args:
            num_classes: Number of object classes (vehicle, person, etc.)
        """
        super().__init__()
        input_dim = 3 + num_classes  # distance, velocity, ttc + one_hot
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.2),
            
            nn.Linear(32, 16),
            nn.ReLU(),
            
            nn.Linear(16, 1),
            nn.Sigmoid()  # Output [0, 1]
        )
    
    def forward(self, x):
        """
        Args:
            x: [batch_size, feature_dim]
        
        Returns:
            [batch_size, 1] threat scores ∈ [0, 1]
        """
        return self.net(x)


class ThreatPrioritizerFinetuner:
    """Fine-tune threat prioritizer model on SANPO training data."""
    
    def __init__(self,
                 training_cache_path: str,
                 model_output_path: str = "./threat_prioritizer_finetuned.pt",
                 device: str = "cpu"):
        """
        Args:
            training_cache_path: Path to training_cache.pkl from preprocessing
            model_output_path: Where to save fine-tuned model
            device: "cpu", "cuda", or "mps"
        """
        self.training_cache_path = training_cache_path
        self.model_output_path = Path(model_output_path)
        self.device = torch.device(device)
        
        # Load dataset
        self.dataset = ThreatDataset(training_cache_path)
        self.num_classes = len(self.dataset.classes)
        
        logger.info(f"Classes: {self.dataset.classes}")
        logger.info(f"Device: {self.device}")
    
    def finetune(self,
                 epochs: int = 10,
                 batch_size: int = 32,
                 learning_rate: float = 1e-3,
                 validation_split: float = 0.2):
        """
        Fine-tune the threat prioritizer model.
        
        Args:
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Adam learning rate
            validation_split: Fraction of data for validation
        """
        logger.info(f"\n{'='*60}")
        logger.info("THREAT PRIORITIZER FINE-TUNING")
        logger.info(f"{'='*60}")
        logger.info(f"Dataset size: {len(self.dataset)}")
        logger.info(f"Batch size: {batch_size}")
        logger.info(f"Epochs: {epochs}")
        
        # Split dataset
        n_train = int(len(self.dataset) * (1 - validation_split))
        n_val = len(self.dataset) - n_train
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            self.dataset, [n_train, n_val]
        )
        
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False
        )
        
        # Initialize model
        model = ThreatPrioritizerMLP(num_classes=self.num_classes)
        model = model.to(self.device)
        
        # Loss and optimizer
        loss_fn = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=2, verbose=True
        )
        
        # Training loop
        best_val_loss = float('inf')
        history = {
            "train_loss": [],
            "val_loss": []
        }
        
        for epoch in range(epochs):
            # Training
            train_loss = self._train_epoch(model, train_loader, loss_fn, optimizer)
            
            # Validation
            val_loss = self._validate_epoch(model, val_loader, loss_fn)
            
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            
            logger.info(f"Epoch {epoch+1}/{epochs}: "
                       f"Train loss={train_loss:.4f}, Val loss={val_loss:.4f}")
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self._save_model(model)
                logger.info(f"  ✓ Saved best model (loss={val_loss:.4f})")
        
        logger.info(f"{'='*60}")
        logger.info(f"Fine-tuning complete!")
        logger.info(f"Best validation loss: {best_val_loss:.4f}")
        logger.info(f"Model saved to: {self.model_output_path}")
        
        return history
    
    def _train_epoch(self, model, train_loader, loss_fn, optimizer):
        """Train for one epoch."""
        model.train()
        total_loss = 0.0
        
        for features, targets in train_loader:
            features = features.to(self.device)
            targets = targets.to(self.device)
            
            # Forward
            outputs = model(features)
            loss = loss_fn(outputs, targets)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * len(features)
        
        return total_loss / len(train_loader.dataset)
    
    def _validate_epoch(self, model, val_loader, loss_fn):
        """Validate for one epoch."""
        model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for features, targets in val_loader:
                features = features.to(self.device)
                targets = targets.to(self.device)
                
                outputs = model(features)
                loss = loss_fn(outputs, targets)
                
                total_loss += loss.item() * len(features)
        
        return total_loss / len(val_loader.dataset)
    
    def _save_model(self, model):
        """Save model state_dict to file."""
        state_dict = {
            "model_state": model.state_dict(),
            "num_classes": self.num_classes,
            "classes": self.dataset.classes,
            "architecture": "ThreatPrioritizerMLP"
        }
        torch.save(state_dict, self.model_output_path)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python threat_prioritizer_finetuner.py <training_cache.pkl> "
              "[model_output.pt] [device]")
        print("Example: python threat_prioritizer_finetuner.py training_cache.pkl "
              "threat_prioritizer_finetuned.pt cpu")
        sys.exit(1)
    
    training_cache = sys.argv[1]
    model_output = sys.argv[2] if len(sys.argv) > 2 else "threat_prioritizer_finetuned.pt"
    device = sys.argv[3] if len(sys.argv) > 3 else "cpu"
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    finetuner = ThreatPrioritizerFinetuner(
        training_cache_path=training_cache,
        model_output_path=model_output,
        device=device
    )
    
    finetuner.finetune(epochs=10, batch_size=32, learning_rate=1e-3)
