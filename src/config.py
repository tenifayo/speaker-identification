"""Speaker Identification System - Configuration Module."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal
import os


@dataclass
class AudioConfig:
    """Audio preprocessing configuration."""
    sample_rate: int = 16000  #16kHz
    mono: bool = True
    target_lufs: float = -23.0
    vad_aggressiveness: int = 2  # 0-3, higher = more aggressive


@dataclass
class EmbeddingConfig:
    """Speaker embedding extraction configuration."""
    model_source: str = "speechbrain/spkrec-ecapa-voxceleb"
    embedding_dim: int = 192
    device: Literal["cpu", "cuda"] = "cpu"


@dataclass
class VerificationConfig:
    """Speaker verification configuration."""
    similarity_threshold: float = 0.7
    min_enrollment_samples: int = 3


@dataclass
class DatabaseConfig:
    """Database configuration."""
    db_path: Path = field(default_factory=lambda: Path("data/users.db"))


@dataclass
class Config:
    """Main application configuration."""
    audio: AudioConfig = field(default_factory=AudioConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    verification: VerificationConfig = field(default_factory=VerificationConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)

    def __post_init__(self):
        # Auto-detect CUDA availability
        try:
            import torch
            if torch.cuda.is_available():
                self.embedding.device = "cuda"
        except ImportError:
            pass


# Global configuration instance
config = Config()
