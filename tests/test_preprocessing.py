"""Tests for preprocessing module."""

import pytest
import numpy as np
import os
import tempfile
import soundfile as sf

# Skip if dependencies not installed
pytest.importorskip("librosa")
pytest.importorskip("noisereduce")


class TestPreprocessing:
    """Test preprocessing functions."""
    
    @pytest.fixture
    def sample_audio(self):
        """Create a sample audio file for testing."""
        sr = 16000
        duration = 2.0
        # Generate a simple sine wave
        t = np.linspace(0, duration, int(sr * duration))
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz tone
        
        # Save to temp file
        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        sf.write(path, audio, sr)
        
        yield path, audio, sr
        
        # Cleanup
        if os.path.exists(path):
            os.remove(path)
    
    def test_load_and_resample(self, sample_audio):
        """Test audio loading and resampling."""
        from src.preprocessing import load_and_resample
        
        path, _, _ = sample_audio
        audio = load_and_resample(path, sr=16000)
        
        assert isinstance(audio, np.ndarray)
        assert audio.ndim == 1  # Mono
        assert len(audio) > 0
    
    def test_reduce_noise(self, sample_audio):
        """Test noise reduction."""
        from src.preprocessing import load_and_resample, reduce_noise
        
        path, _, _ = sample_audio
        audio = load_and_resample(path)
        denoised = reduce_noise(audio)
        
        assert isinstance(denoised, np.ndarray)
        assert len(denoised) == len(audio)
    
    def test_normalize_loudness(self, sample_audio):
        """Test loudness normalization."""
        from src.preprocessing import load_and_resample, normalize_loudness
        
        path, _, _ = sample_audio
        audio = load_and_resample(path)
        normalized = normalize_loudness(audio, target_lufs=-23)
        
        assert isinstance(normalized, np.ndarray)
        assert len(normalized) == len(audio)
    
    def test_preprocess_pipeline(self, sample_audio):
        """Test full preprocessing pipeline."""
        from src.preprocessing import preprocess_audio
        
        path, _, _ = sample_audio
        processed = preprocess_audio(
            path,
            apply_noise_reduction=True,
            apply_vad_filter=False,  # Skip VAD for synthetic audio
            apply_normalization=True
        )
        
        assert isinstance(processed, np.ndarray)
        assert len(processed) > 0
