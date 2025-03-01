import pandas as pd
import numpy as np
import torch
from pathlib import Path
from typing import List, Tuple, Optional
from sklearn.preprocessing import StandardScaler
import mne
from transformers import Data2VecAudioModel, Data2VecAudioConfig, Data2VecAudioFeatureExtractor
from torch import nn

class EEGClassifier:
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the EEG classifier with Data2Vec Audio model
        
        Args:
            model_path: Path to a pretrained Data2Vec Audio model (optional)
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize Data2Vec Audio model
        if model_path:
            self.model = Data2VecAudioModel.from_pretrained(model_path)
        else:
            # Initialize with custom configuration for EEG data
            config = Data2VecAudioConfig(
                hidden_size=768,
                num_hidden_layers=12,
                num_attention_heads=12,
                intermediate_size=3072,
                vocab_size=32,  # Not used for continuous input
                conv_dim=(512, 512, 512),  # Adjust based on EEG characteristics
                conv_stride=(5, 2, 2),
                conv_kernel=(10, 3, 3),
                conv_bias=True,
            )
            self.model = Data2VecAudioModel(config)
            
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize feature extractor for Data2Vec Audio
        self.feature_extractor = Data2VecAudioFeatureExtractor()
        self.scaler = StandardScaler()
        
    def load_eeg_data(self, csv_path: str, channel_names: Optional[List[str]] = None) -> np.ndarray:
        """
        Load EEG data from a CSV file
        
        Args:
            csv_path: Path to the CSV file containing EEG data
            channel_names: List of channel names (optional)
            
        Returns:
            numpy array of shape (n_samples, n_channels)
        """
        df = pd.read_csv(csv_path)
        
        if channel_names is None:
            # Assume all columns except any obvious non-EEG columns are channels
            non_eeg_cols = ['time', 'timestamp', 'label', 'class', 'event']
            channel_names = [col for col in df.columns if col.lower() not in non_eeg_cols]
        
        eeg_data = df[channel_names].values
        return eeg_data
    
    def preprocess_data(self, eeg_data: np.ndarray, sfreq: float = 250.0) -> np.ndarray:
        """
        Preprocess EEG data
        
        Args:
            eeg_data: Raw EEG data of shape (n_samples, n_channels)
            sfreq: Sampling frequency in Hz
            
        Returns:
            Preprocessed EEG data
        """
        # Create MNE raw object for filtering
        ch_names = [f'CH{i+1}' for i in range(eeg_data.shape[1])]
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
        raw = mne.io.RawArray(eeg_data.T, info)
        
        # Apply bandpass filter (1-45 Hz)
        raw.filter(l_freq=1, h_freq=45)
        
        # Get filtered data back
        filtered_data = raw.get_data().T
        
        # Standardize the data
        preprocessed_data = self.scaler.fit_transform(filtered_data)
        return preprocessed_data
    
    def _prepare_signal_for_model(self, eeg_data: np.ndarray) -> torch.Tensor:
        """
        Prepare EEG signal for Data2Vec Audio model
        
        Args:
            eeg_data: EEG data of shape (n_samples, n_channels)
            
        Returns:
            Tensor of shape (1, n_channels, n_samples)
        """
        # Ensure minimum sequence length (adjust if needed)
        min_length = 512
        if eeg_data.shape[0] < min_length:
            pad_size = min_length - eeg_data.shape[0]
            eeg_data = np.pad(eeg_data, ((0, pad_size), (0, 0)), mode='constant')
        
        # Convert to format expected by Data2Vec Audio
        # Shape: (batch_size, n_channels, sequence_length)
        signal = torch.FloatTensor(eeg_data.T).unsqueeze(0)
        
        # Process using Data2Vec Audio feature extractor
        inputs = self.feature_extractor(
            signal,
            sampling_rate=250,  # Default EEG sampling rate
            return_tensors="pt"
        )
        
        return inputs.input_values
    
    def extract_features(self, eeg_data: np.ndarray) -> torch.Tensor:
        """
        Extract features using Data2Vec Audio model
        
        Args:
            eeg_data: Preprocessed EEG data of shape (n_samples, n_channels)
            
        Returns:
            Feature embeddings
        """
        with torch.no_grad():
            # Prepare signal for model
            signal = self._prepare_signal_for_model(eeg_data).to(self.device)
            
            # Get embeddings from Data2Vec Audio
            outputs = self.model(signal)
            # Use the mean of the last hidden state as the feature vector
            embeddings = outputs.last_hidden_state.mean(dim=1)
            
        return embeddings.cpu()
    
    def few_shot_classification(self, 
                              support_data: List[Tuple[str, np.ndarray]], 
                              query_data: np.ndarray,
                              k: int = 5) -> Tuple[str, float]:
        """
        Perform few-shot classification using cosine similarity
        
        Args:
            support_data: List of (label, data) tuples for support set
            query_data: Query EEG data to classify
            k: Number of nearest neighbors to consider
            
        Returns:
            Predicted label and confidence score
        """
        # Preprocess and extract features for support set
        support_features = []
        support_labels = []
        
        for label, data in support_data:
            preprocessed_data = self.preprocess_data(data)
            features = self.extract_features(preprocessed_data)
            support_features.append(features)
            support_labels.append(label)
            
        support_features = torch.cat(support_features, dim=0)
        
        # Preprocess and extract features for query
        preprocessed_query = self.preprocess_data(query_data)
        query_features = self.extract_features(preprocessed_query)
        
        # Calculate cosine similarities
        similarities = torch.nn.functional.cosine_similarity(
            query_features.unsqueeze(1),
            support_features.unsqueeze(0),
            dim=2
        )
        
        # Get top-k nearest neighbors
        _, indices = similarities.topk(k, dim=1)
        nearest_labels = [support_labels[idx] for idx in indices[0]]
        
        # Get most common label and its frequency as confidence
        from collections import Counter
        label_counts = Counter(nearest_labels)
        predicted_label, count = label_counts.most_common(1)[0]
        confidence = count / k
        
        return predicted_label, confidence

def main():
    """
    Example usage of the EEGClassifier
    """
    # Initialize classifier with pretrained Data2Vec Audio model
    classifier = EEGClassifier("facebook/data2vec-audio-base-960h")
    
    # Example: Load support set data
    support_data = []
    support_csv_files = {
        'class1': ['path/to/class1_example1.csv', 'path/to/class1_example2.csv'],
        'class2': ['path/to/class2_example1.csv', 'path/to/class2_example2.csv']
    }
    
    for label, file_paths in support_csv_files.items():
        for file_path in file_paths:
            if Path(file_path).exists():
                data = classifier.load_eeg_data(file_path)
                support_data.append((label, data))
    
    # Example: Load and classify query data
    query_file = 'path/to/query.csv'
    if Path(query_file).exists():
        query_data = classifier.load_eeg_data(query_file)
        predicted_label, confidence = classifier.few_shot_classification(
            support_data, query_data
        )
        print(f"Predicted label: {predicted_label}")
        print(f"Confidence: {confidence:.2f}")

if __name__ == "__main__":
    main() 