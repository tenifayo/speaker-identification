"""Audio preprocessing pipeline for speaker identification."""

import numpy as np
import librosa
import noisereduce as nr
import webrtcvad
import pyloudnorm as pyln
import soundfile as sf
from pathlib import Path
from typing import Union

from .config import config


def load_and_resample(
    audio_path: Union[str, Path],
    sr: int = None
) -> np.ndarray:
    """
    Load audio file and resample to target sample rate.
    
    Args:
        audio_path: Path to audio file
        sr: Target sample rate (default from config)
    
    Returns:
        Mono audio signal as numpy array
    """
    sr = sr or config.audio.sample_rate
    audio, _ = librosa.load(audio_path, sr=sr, mono=config.audio.mono)
    return audio


def reduce_noise(
    audio: np.ndarray,
    sr: int = None,
    stationary: bool = True
) -> np.ndarray:
    """
    Apply spectral gating noise reduction.
    
    Args:
        audio: Input audio signal
        sr: Sample rate (default from config)
        stationary: Use stationary noise reduction
    
    Returns:
        Denoised audio signal
    """
    sr = sr or config.audio.sample_rate
    return nr.reduce_noise(
        y=audio,
        sr=sr,
        stationary=stationary,
        prop_decrease=0.8
    )


def apply_vad(
    audio: np.ndarray,
    sr: int = None,
    aggressiveness: int = None,
    frame_duration_ms: int = 30
) -> np.ndarray:
    """
    Apply Voice Activity Detection to remove silence.
    
    Args:
        audio: Input audio signal (must be 16-bit PCM compatible)
        sr: Sample rate (must be 8000, 16000, 32000, or 48000)
        aggressiveness: VAD aggressiveness level (0-3)
        frame_duration_ms: Frame duration (10, 20, or 30 ms)
    
    Returns:
        Audio with silence removed
    """
    sr = sr or config.audio.sample_rate
    aggressiveness = aggressiveness or config.audio.vad_aggressiveness
    
    # WebRTC VAD requires specific sample rates
    if sr not in [8000, 16000, 32000, 48000]:
        raise ValueError(f"VAD requires sample rate in [8000, 16000, 32000, 48000], got {sr}")
    
    vad = webrtcvad.Vad(aggressiveness)
    
    # Convert to 16-bit PCM (Pulse Code Modulation)
    audio_int16 = (audio * 32767).astype(np.int16)
    
    # Calculate frame size
    frame_size = int(sr * frame_duration_ms / 1000)
    
    # Process frames
    voiced_frames = []
    for i in range(0, len(audio_int16) - frame_size, frame_size):
        frame = audio_int16[i:i + frame_size]
        if vad.is_speech(frame.tobytes(), sr):
            voiced_frames.append(frame)
    
    if not voiced_frames:
        # Return original if no speech detected
        return audio
    
    # Concatenate voiced frames and convert back to float
    voiced_audio = np.concatenate(voiced_frames)
    return voiced_audio.astype(np.float32) / 32767


def normalize_loudness(
    audio: np.ndarray,
    sr: int = None,
    target_lufs: float = None
) -> np.ndarray:
    """
    Normalize audio to target loudness level.
    
    Args:
        audio: Input audio signal
        sr: Sample rate (default from config)
        target_lufs: Target loudness in LUFS (default from config)
    
    Returns:
        Loudness-normalized audio
    """
    sr = sr or config.audio.sample_rate
    target_lufs = target_lufs or config.audio.target_lufs
    
    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(audio)
    
    # Handle silence or very quiet audio
    if np.isinf(loudness) or loudness < -70:
        return audio
    
    return pyln.normalize.loudness(audio, loudness, target_lufs)


def preprocess_audio(
    audio_path: Union[str, Path],
    apply_noise_reduction: bool = True,
    apply_vad_filter: bool = True,
    apply_normalization: bool = True
) -> np.ndarray:
    """
    Full preprocessing pipeline for speaker identification.
    
    Args:
        audio_path: Path to input audio file
        apply_noise_reduction: Whether to apply noise reduction
        apply_vad_filter: Whether to apply VAD filtering
        apply_normalization: Whether to apply loudness normalization
    
    Returns:
        Preprocessed audio signal ready for embedding extraction
    """
    # Load and resample
    audio = load_and_resample(audio_path)
    
    # Noise reduction
    if apply_noise_reduction:
        audio = reduce_noise(audio)
    
    # Voice activity detection
    if apply_vad_filter:
        audio = apply_vad(audio)
    
    # Loudness normalization
    if apply_normalization:
        audio = normalize_loudness(audio)
    
    return audio


def save_audio(
    audio: np.ndarray,
    output_path: Union[str, Path],
    sr: int = None
) -> None:
    """
    Save audio to file.
    
    Args:
        audio: Audio signal to save
        output_path: Output file path
        sr: Sample rate (default from config)
    """
    sr = sr or config.audio.sample_rate
    sf.write(output_path, audio, sr)
