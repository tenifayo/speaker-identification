"""Liveness detection module for speaker verification."""

from datetime import datetime, timedelta
from typing import Optional, Tuple
from fuzzywuzzy import fuzz
from pathlib import Path

from .config import config
from .sentence_generator import SentenceGenerator
from .transcription import get_transcriber, TranscriptionError
from .database import get_database


class LivenessError(Exception):
    """Raised when liveness detection fails."""
    pass


class LivenessResult:
    """Result of liveness detection."""
    
    def __init__(
        self,
        passed: bool,
        challenge_id: str,
        expected_sentence: str,
        transcribed_text: str,
        similarity_score: float,
        threshold: float,
        reason: Optional[str] = None
    ):
        self.passed = passed
        self.challenge_id = challenge_id
        self.expected_sentence = expected_sentence
        self.transcribed_text = transcribed_text
        self.similarity_score = similarity_score
        self.threshold = threshold
        self.reason = reason or ("Passed" if passed else "Failed")
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "passed": self.passed,
            "challenge_id": self.challenge_id,
            "expected_sentence": self.expected_sentence,
            "transcribed_text": self.transcribed_text,
            "similarity_score": self.similarity_score,
            "threshold": self.threshold,
            "reason": self.reason
        }


def calculate_sentence_similarity(expected: str, actual: str) -> float:
    """
    Calculate similarity between expected and actual sentences.
    
    Uses fuzzy string matching to handle minor transcription errors.
    
    Args:
        expected: Expected sentence
        actual: Actual transcribed sentence
        
    Returns:
        Similarity score from 0.0 to 1.0
    """
    # Normalize both strings
    expected_norm = expected.lower().strip()
    actual_norm = actual.lower().strip()
    
    # Use token sort ratio for better handling of word order variations
    similarity = fuzz.token_sort_ratio(expected_norm, actual_norm)
    
    # Convert from 0-100 to 0.0-1.0
    return similarity / 100.0


def generate_challenge(
    user_id: Optional[str] = None,
    complexity: str = None
) -> Tuple[str, str, datetime]:
    """
    Generate a new liveness challenge.
    
    Args:
        user_id: Optional user ID for verification challenges
        complexity: Sentence complexity level
        
    Returns:
        Tuple of (challenge_id, sentence, expires_at)
    """
    complexity = complexity or config.liveness.sentence_complexity
    
    # Generate sentence
    generator = SentenceGenerator(complexity)
    challenge_id, sentence = generator.generate()
    
    # Calculate expiration
    expires_at = datetime.now() + timedelta(
        seconds=config.liveness.challenge_expiry_seconds
    )
    
    # Store in database
    db = get_database()
    db.create_challenge(
        challenge_id=challenge_id,
        user_id=user_id,
        sentence=sentence,
        expires_at=expires_at
    )
    
    return challenge_id, sentence, expires_at


def validate_challenge(
    challenge_id: str,
    audio_path: Path,
    language: str = "en"
) -> LivenessResult:
    """
    Validate a liveness challenge by transcribing audio and comparing.
    
    Args:
        challenge_id: Challenge ID to validate
        audio_path: Path to audio file with user's response
        language: Language code for transcription
        
    Returns:
        LivenessResult with validation details
        
    Raises:
        LivenessError: If challenge is invalid or expired
    """
    db = get_database()
    
    # Get challenge from database
    challenge = db.get_challenge(challenge_id)
    if challenge is None:
        raise LivenessError(f"Challenge '{challenge_id}' not found")
    
    # Check if already used
    if challenge.used:
        raise LivenessError("Challenge has already been used")
    
    # Check expiration
    if datetime.now() > challenge.expires_at:
        raise LivenessError("Challenge has expired")
    
    # Transcribe audio
    try:
        transcriber = get_transcriber()
        transcribed_text = transcriber.transcribe(audio_path, language=language)
    except TranscriptionError as e:
        raise LivenessError(f"Transcription failed: {str(e)}")
    
    # Calculate similarity
    similarity = calculate_sentence_similarity(challenge.sentence, transcribed_text)
    threshold = config.liveness.match_threshold
    
    # Determine if passed
    passed = similarity >= threshold
    reason = None
    
    if not passed:
        if similarity < 0.3:
            reason = "Transcribed text does not match expected sentence"
        else:
            reason = f"Sentence similarity ({similarity:.2%}) below threshold ({threshold:.2%})"
    
    # Mark challenge as used
    db.mark_challenge_used(challenge_id)
    
    return LivenessResult(
        passed=passed,
        challenge_id=challenge_id,
        expected_sentence=challenge.sentence,
        transcribed_text=transcribed_text,
        similarity_score=similarity,
        threshold=threshold,
        reason=reason
    )


def verify_with_liveness(
    challenge_id: str,
    audio_path: Path,
    language: str = "en"
) -> LivenessResult:
    """
    Convenience function for liveness verification.
    
    Args:
        challenge_id: Challenge ID
        audio_path: Path to audio file
        language: Language code
        
    Returns:
        LivenessResult
    """
    return validate_challenge(challenge_id, audio_path, language)


if __name__ == "__main__":
    # Demo usage
    print("Generating challenge...")
    challenge_id, sentence, expires_at = generate_challenge()
    print(f"Challenge ID: {challenge_id}")
    print(f"Sentence: {sentence}")
    print(f"Expires at: {expires_at}")
    
    print("\nTesting sentence similarity:")
    test_cases = [
        ("The blue car is outside", "The blue car is outside", "Exact match"),
        ("The blue car is outside", "the blue car is outside", "Case difference"),
        ("The blue car is outside", "The blue car is parked outside", "Minor difference"),
        ("The blue car is outside", "Blue car outside", "Missing words"),
        ("The blue car is outside", "The red car is inside", "Different content"),
    ]
    
    for expected, actual, description in test_cases:
        similarity = calculate_sentence_similarity(expected, actual)
        print(f"{description}: {similarity:.2%}")
