import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from pathlib import Path
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from tqdm import tqdm
import json
from eeg_classifier import EEGClassifier

class EEGDataset(Dataset):
    """Dataset class for EEG data"""
    def __init__(self, data_files: Dict[str, List[str]], transform=None):
        """
        Args:
            data_files: Dictionary mapping labels to lists of file paths
            transform: Optional transform to be applied on a sample
        """
        self.samples = []
        self.transform = transform
        self.label_to_idx = {}
        self.idx_to_label = {}
        
        # Create label mappings
        for idx, label in enumerate(sorted(data_files.keys())):
            self.label_to_idx[label] = idx
            self.idx_to_label[idx] = label
        
        # Load all data
        classifier = EEGClassifier()  # Temporary instance for data loading
        for label, files in data_files.items():
            for file_path in files:
                if Path(file_path).exists():
                    data = classifier.load_eeg_data(file_path)
                    preprocessed_data = classifier.preprocess_data(data)
                    self.samples.append((preprocessed_data, self.label_to_idx[label]))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        data, label = self.samples[idx]
        if self.transform:
            data = self.transform(data)
        return data, label

class EEGClassifierTrainer:
    def __init__(self, 
                 model_path: Optional[str] = None,
                 num_classes: int = 2,
                 learning_rate: float = 1e-4,
                 batch_size: int = 32,
                 num_epochs: int = 10,
                 device: Optional[str] = None):
        """
        Initialize the trainer
        
        Args:
            model_path: Path to pretrained model or None
            num_classes: Number of classes to classify
            learning_rate: Learning rate for optimization
            batch_size: Batch size for training
            num_epochs: Number of training epochs
            device: Device to use for training (cuda/cpu)
        """
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize the base classifier
        self.classifier = EEGClassifier(model_path)
        
        # Add classification head
        self.classification_head = nn.Sequential(
            nn.Linear(768, 384),  # 768 is Data2Vec's hidden size
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(384, num_classes)
        ).to(self.device)
        
        # Initialize optimizer with different learning rates
        self.optimizer = optim.AdamW([
            {'params': self.classifier.model.parameters(), 'lr': learning_rate * 0.1},
            {'params': self.classification_head.parameters(), 'lr': learning_rate}
        ])
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def train(self, 
              train_data: Dict[str, List[str]],
              val_data: Optional[Dict[str, List[str]]] = None,
              save_dir: str = 'models'):
        """
        Train the classifier
        
        Args:
            train_data: Dictionary mapping labels to lists of training file paths
            val_data: Optional dictionary mapping labels to lists of validation file paths
            save_dir: Directory to save model checkpoints
        """
        # Create datasets and dataloaders
        train_dataset = EEGDataset(train_data)
        train_loader = DataLoader(train_dataset, 
                                batch_size=self.batch_size,
                                shuffle=True,
                                num_workers=4)
        
        if val_data:
            val_dataset = EEGDataset(val_data)
            val_loader = DataLoader(val_dataset,
                                  batch_size=self.batch_size,
                                  shuffle=False,
                                  num_workers=4)
        
        # Create save directory
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save label mappings
        with open(save_dir / 'label_mapping.json', 'w') as f:
            json.dump(train_dataset.label_to_idx, f)
        
        best_val_acc = 0.0
        for epoch in range(self.num_epochs):
            self.logger.info(f"Epoch {epoch+1}/{self.num_epochs}")
            
            # Training phase
            self.classifier.model.train()
            self.classification_head.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_data, batch_labels in tqdm(train_loader, desc="Training"):
                # Prepare EEG signals for the model
                batch_signals = torch.stack([
                    self.classifier._prepare_signal_for_model(data)[0]
                    for data in batch_data
                ]).to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.classifier.model(batch_signals)
                features = outputs.last_hidden_state.mean(dim=1)  # Average over sequence length
                logits = self.classification_head(features)
                
                # Calculate loss and update weights
                loss = self.criterion(logits, batch_labels)
                loss.backward()
                self.optimizer.step()
                
                # Calculate accuracy
                _, predicted = torch.max(logits.data, 1)
                train_total += batch_labels.size(0)
                train_correct += (predicted == batch_labels).sum().item()
                train_loss += loss.item()
            
            train_acc = 100 * train_correct / train_total
            train_loss = train_loss / len(train_loader)
            self.logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            
            # Validation phase
            if val_data:
                val_acc = self.evaluate(val_loader)
                self.logger.info(f"Validation Acc: {val_acc:.2f}%")
                
                # Save best model
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    self.save_model(save_dir / 'best_model.pth')
            
            # Save checkpoint
            self.save_model(save_dir / f'checkpoint_epoch_{epoch+1}.pth')
    
    def evaluate(self, val_loader: DataLoader) -> float:
        """Evaluate the model on validation data"""
        self.classifier.model.eval()
        self.classification_head.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_data, batch_labels in tqdm(val_loader, desc="Validation"):
                # Prepare EEG signals for the model
                batch_signals = torch.stack([
                    self.classifier._prepare_signal_for_model(data)[0]
                    for data in batch_data
                ]).to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                # Forward pass
                outputs = self.classifier.model(batch_signals)
                features = outputs.last_hidden_state.mean(dim=1)
                logits = self.classification_head(features)
                
                # Calculate accuracy
                _, predicted = torch.max(logits.data, 1)
                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()
        
        return 100 * correct / total
    
    def save_model(self, path: str):
        """Save model checkpoint"""
        torch.save({
            'model_state_dict': self.classifier.model.state_dict(),
            'classification_head_state_dict': self.classification_head.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
    
    def load_model(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path)
        self.classifier.model.load_state_dict(checkpoint['model_state_dict'])
        self.classification_head.load_state_dict(checkpoint['classification_head_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

def main():
    """Example usage of the trainer"""
    # Example data organization
    train_data = {
        'class1': ['data/train/class1_1.csv', 'data/train/class1_2.csv'],
        'class2': ['data/train/class2_1.csv', 'data/train/class2_2.csv']
    }
    
    val_data = {
        'class1': ['data/val/class1_1.csv', 'data/val/class1_2.csv'],
        'class2': ['data/val/class2_1.csv', 'data/val/class2_2.csv']
    }
    
    # Initialize trainer with Data2Vec Audio model
    trainer = EEGClassifierTrainer(
        model_path="facebook/data2vec-audio-base-960h",
        num_classes=len(train_data),
        learning_rate=1e-4,
        batch_size=32,
        num_epochs=10
    )
    
    # Train the model
    trainer.train(train_data, val_data, save_dir='models')

if __name__ == "__main__":
    main() 