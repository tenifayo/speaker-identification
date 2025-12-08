"""Speaker embedding extraction using SpeechBrain ECAPA-TDNN (Emphasized Channel Attention, Propagation and Aggregation in TDNN)."""

import numpy as np
import torch
from typing import Union, List
from pathlib import Path

from .config import config
from .preprocessing import preprocess_audio


class EmbeddingExtractor:
    """Extract speaker embeddings using pretrained ECAPA-TDNN model."""
    
    _instance = None
    
    def __new__(cls):
        """Singleton pattern to avoid loading model multiple times."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        from speechbrain.inference import EncoderClassifier
        
        self.device = config.embedding.device
        self.model = EncoderClassifier.from_hparams(
            source=config.embedding.model_source,
            run_opts={"device": self.device}
        )
        self._initialized = True
    
    def extract(self, audio: np.ndarray, sr: int = None) -> np.ndarray:
        """
        Extract speaker embedding from audio signal.
        
        Args:
            audio: Preprocessed audio signal
            sr: Sample rate (default from config)
        
        Returns:
            Speaker embedding as numpy array (192-dim for ECAPA-TDNN)
        """
        sr = sr or config.audio.sample_rate
        
        # Convert to torch tensor
        audio_tensor = torch.tensor(audio).unsqueeze(0).float()
        
        # Extract embedding
        with torch.no_grad():
            embedding = self.model.encode_batch(audio_tensor)
        
        return embedding.squeeze().cpu().numpy()
    
    def extract_from_file(
        self,
        audio_path: Union[str, Path],
        preprocess: bool = True
    ) -> np.ndarray:
        """
        Extract speaker embedding from audio file.
        
        Args:
            audio_path: Path to audio file
            preprocess: Whether to apply preprocessing pipeline
        
        Returns:
            Speaker embedding as numpy array
        """
        if preprocess:
            audio = preprocess_audio(audio_path)
        else:
            import librosa
            audio, _ = librosa.load(audio_path, sr=config.audio.sample_rate)
        
        return self.extract(audio)
    
    def extract_batch(
        self,
        audio_paths: List[Union[str, Path]],
        preprocess: bool = True
    ) -> List[np.ndarray]:
        """
        Extract embeddings from multiple audio files.
        
        Args:
            audio_paths: List of audio file paths
            preprocess: Whether to apply preprocessing pipeline
        
        Returns:
            List of speaker embeddings
        """
        return [self.extract_from_file(p, preprocess) for p in audio_paths]


def get_extractor() -> EmbeddingExtractor:
    """Get singleton embedding extractor instance."""
    return EmbeddingExtractor()


def compute_average_embedding(embeddings: List[np.ndarray]) -> np.ndarray:
    """
    Compute average embedding from multiple samples.
    
    Args:
        embeddings: List of speaker embeddings
    
    Returns:
        Averaged and normalized embedding
    """
    avg_embedding = np.mean(embeddings, axis=0)
    # L2 normalize
    return avg_embedding / np.linalg.norm(avg_embedding)
