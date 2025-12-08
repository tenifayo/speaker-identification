"""Speaker enrollment module."""

import numpy as np
from typing import List, Union
from pathlib import Path

from .config import config
from .embeddings import get_extractor, compute_average_embedding
from .database import get_database


class EnrollmentError(Exception):
    """Enrollment-related errors."""
    pass


def enroll_user(
    user_id: str,
    name: str,
    audio_paths: List[Union[str, Path]],
    min_samples: int = None
) -> dict:
    """
    Enroll a new user with multiple audio samples.
    
    Args:
        user_id: Unique identifier for the user
        name: Display name for the user
        audio_paths: List of paths to audio samples
        min_samples: Minimum required samples (default from config)
    
    Returns:
        Dict with enrollment result details
    
    Raises:
        EnrollmentError: If enrollment fails
    """
    min_samples = min_samples or config.verification.min_enrollment_samples
    
    # Validate sample count
    if len(audio_paths) < min_samples:
        raise EnrollmentError(
            f"Insufficient samples: got {len(audio_paths)}, need at least {min_samples}"
        )
    
    db = get_database()
    
    # Check if user already exists
    if db.user_exists(user_id):
        raise EnrollmentError(f"User '{user_id}' already exists")
    
    # Extract embeddings from all samples
    extractor = get_extractor()
    embeddings = extractor.extract_batch(audio_paths, preprocess=True)
    
    # Compute averaged embedding
    avg_embedding = compute_average_embedding(embeddings)
    
    # Store in database
    success = db.add_user(
        user_id=user_id,
        name=name,
        embedding=avg_embedding,
        num_samples=len(audio_paths)
    )
    
    if not success:
        raise EnrollmentError(f"Failed to add user '{user_id}' to database")
    
    return {
        "user_id": user_id,
        "name": name,
        "num_samples": len(audio_paths),
        "embedding_dim": len(avg_embedding),
        "status": "enrolled"
    }


def update_enrollment(
    user_id: str,
    audio_paths: List[Union[str, Path]],
    replace: bool = False
) -> dict:
    """
    Update existing user's enrollment with new samples.
    
    Args:
        user_id: User to update
        audio_paths: New audio samples
        replace: If True, replace embedding; if False, combine with existing
    
    Returns:
        Dict with update result details
    """
    db = get_database()
    user = db.get_user(user_id)
    
    if user is None:
        raise EnrollmentError(f"User '{user_id}' not found")
    
    extractor = get_extractor()
    new_embeddings = extractor.extract_batch(audio_paths, preprocess=True)
    
    if replace:
        # Replace with new embedding
        new_avg = compute_average_embedding(new_embeddings)
        new_count = len(audio_paths)
    else:
        # Weighted combination of old and new
        old_weight = user.num_samples
        new_weight = len(audio_paths)
        total_weight = old_weight + new_weight
        
        new_avg = (
            (user.embedding * old_weight) + 
            (compute_average_embedding(new_embeddings) * new_weight)
        ) / total_weight
        # Re-normalize
        new_avg = new_avg / np.linalg.norm(new_avg)
        new_count = total_weight
    
    db.update_user_embedding(user_id, new_avg, new_count)
    
    return {
        "user_id": user_id,
        "num_samples": new_count,
        "status": "updated"
    }


def delete_enrollment(user_id: str) -> bool:
    """
    Remove user enrollment.
    
    Args:
        user_id: User to delete
    
    Returns:
        True if deleted, False if user not found
    """
    db = get_database()
    return db.delete_user(user_id)


def list_enrolled_users() -> list:
    """List all enrolled users."""
    db = get_database()
    return db.list_users()
