"""Speaker verification module."""

import numpy as np
from typing import Union, Tuple, Optional
from pathlib import Path
from dataclasses import dataclass

from .config import config
from .embeddings import get_extractor
from .database import get_database


@dataclass
class VerificationResult:
    """Result of speaker verification."""
    user_id: str
    is_verified: bool
    score: float
    threshold: float
    decision: str  # 'granted' | 'denied'


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.
    
    Args:
        a: First embedding vector
        b: Second embedding vector
    
    Returns:
        Similarity score in range [-1, 1]
    """
    a_norm = a / np.linalg.norm(a)
    b_norm = b / np.linalg.norm(b)
    return float(np.dot(a_norm, b_norm))


def verify_speaker(
    audio_path: Union[str, Path],
    claimed_user_id: str,
    threshold: float = None,
    log_access: bool = True
) -> VerificationResult:
    """
    Verify if audio matches claimed speaker identity.
    
    Args:
        audio_path: Path to test audio file
        claimed_user_id: ID of claimed speaker
        threshold: Similarity threshold (default from config)
        log_access: Whether to log the access attempt
    
    Returns:
        VerificationResult with decision and score
    
    Raises:
        ValueError: If claimed user not found
    """
    threshold = threshold or config.verification.similarity_threshold
    
    db = get_database()
    stored_embedding = db.get_user_embedding(claimed_user_id)
    
    if stored_embedding is None:
        raise ValueError(f"User '{claimed_user_id}' not found")
    
    # Extract embedding from test audio
    extractor = get_extractor()
    test_embedding = extractor.extract_from_file(audio_path, preprocess=True)
    
    # Compute similarity
    score = cosine_similarity(test_embedding, stored_embedding)
    
    # Make decision
    is_verified = score >= threshold
    decision = "granted" if is_verified else "denied"
    
    # Log access attempt
    if log_access:
        db.log_access(claimed_user_id, decision, score, threshold)
    
    return VerificationResult(
        user_id=claimed_user_id,
        is_verified=is_verified,
        score=score,
        threshold=threshold,
        decision=decision
    )


def verify_speaker_from_embedding(
    test_embedding: np.ndarray,
    claimed_user_id: str,
    threshold: float = None,
    log_access: bool = True
) -> VerificationResult:
    """
    Verify using pre-extracted embedding.
    
    Args:
        test_embedding: Pre-extracted test embedding
        claimed_user_id: ID of claimed speaker
        threshold: Similarity threshold
        log_access: Whether to log the access attempt
    
    Returns:
        VerificationResult with decision and score
    """
    threshold = threshold or config.verification.similarity_threshold
    
    db = get_database()
    stored_embedding = db.get_user_embedding(claimed_user_id)
    
    if stored_embedding is None:
        raise ValueError(f"User '{claimed_user_id}' not found")
    
    score = cosine_similarity(test_embedding, stored_embedding)
    is_verified = score >= threshold
    decision = "granted" if is_verified else "denied"
    
    if log_access:
        db.log_access(claimed_user_id, decision, score, threshold)
    
    return VerificationResult(
        user_id=claimed_user_id,
        is_verified=is_verified,
        score=score,
        threshold=threshold,
        decision=decision
    )


def identify_speaker(
    audio_path: Union[str, Path],
    threshold: float = None,
    top_n: int = 1
) -> list:
    """
    Identify speaker from database (1:N matching).
    
    Args:
        audio_path: Path to test audio
        threshold: Minimum similarity threshold
        top_n: Number of top matches to return
    
    Returns:
        List of (user_id, name, score) tuples, sorted by score descending
    """
    threshold = threshold or config.verification.similarity_threshold
    
    db = get_database()
    extractor = get_extractor()
    
    # Get test embedding
    test_embedding = extractor.extract_from_file(audio_path, preprocess=True)
    
    # Compare against all enrolled users
    users = db.list_users()
    scores = []
    
    for user_info in users:
        user = db.get_user(user_info['id'])
        score = cosine_similarity(test_embedding, user.embedding)
        
        if score >= threshold:
            scores.append({
                "user_id": user.id,
                "name": user.name,
                "score": score
            })
    
    # Sort by score descending and return top N
    scores.sort(key=lambda x: x['score'], reverse=True)
    return scores[:top_n]
