"""Speech-to-text transcription using Whisper via Groq API."""

import os
from pathlib import Path
from typing import Union, Optional
from groq import Groq

from .config import config


class TranscriptionError(Exception):
    """Raised when transcription fails."""
    pass


class Transcriber:
    """Transcribe audio to text using Whisper via Groq API."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "whisper-large-v3"):
        """
        Initialize transcriber.
        
        Args:
            api_key: Groq API key (defaults to GROQ_API_KEY env var)
            model: Whisper model to use
        """
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Groq API key not provided. Set GROQ_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        self.model = model
        self.client = Groq(api_key=self.api_key)
    
    def transcribe(
        self,
        audio_path: Union[str, Path],
        language: str = "en",
        prompt: Optional[str] = None
    ) -> str:
        """
        Transcribe audio file to text.
        
        Args:
            audio_path: Path to audio file
            language: Language code (e.g., 'en' for English)
            prompt: Optional prompt to guide transcription
            
        Returns:
            Transcribed text
            
        Raises:
            TranscriptionError: If transcription fails
        """
        audio_path = Path(audio_path)
        
        if not audio_path.exists():
            raise TranscriptionError(f"Audio file not found: {audio_path}")
        
        try:
            with open(audio_path, "rb") as audio_file:
                transcription = self.client.audio.transcriptions.create(
                    file=(audio_path.name, audio_file.read()),
                    model=self.model,
                    language=language,
                    prompt=prompt,
                    response_format="text"
                )
            
            # Groq returns the text directly when response_format="text"
            return transcription.strip()
            
        except Exception as e:
            raise TranscriptionError(f"Transcription failed: {str(e)}")
    
    def transcribe_with_confidence(
        self,
        audio_path: Union[str, Path],
        language: str = "en"
    ) -> dict:
        """
        Transcribe audio and get detailed response with timestamps.
        
        Args:
            audio_path: Path to audio file
            language: Language code
            
        Returns:
            Dictionary with transcription details
        """
        audio_path = Path(audio_path)
        
        if not audio_path.exists():
            raise TranscriptionError(f"Audio file not found: {audio_path}")
        
        try:
            with open(audio_path, "rb") as audio_file:
                transcription = self.client.audio.transcriptions.create(
                    file=(audio_path.name, audio_file.read()),
                    model=self.model,
                    language=language,
                    response_format="verbose_json"
                )
            
            return {
                "text": transcription.text.strip(),
                "language": transcription.language,
                "duration": transcription.duration,
                "segments": transcription.segments if hasattr(transcription, 'segments') else []
            }
            
        except Exception as e:
            raise TranscriptionError(f"Transcription failed: {str(e)}")


# Global transcriber instance
_transcriber_instance: Optional[Transcriber] = None


def get_transcriber() -> Transcriber:
    """Get singleton transcriber instance."""
    global _transcriber_instance
    if _transcriber_instance is None:
        _transcriber_instance = Transcriber(
            model=config.transcription.model
        )
    return _transcriber_instance


def transcribe_audio(
    audio_path: Union[str, Path],
    language: str = "en"
) -> str:
    """
    Convenience function to transcribe audio.
    
    Args:
        audio_path: Path to audio file
        language: Language code
        
    Returns:
        Transcribed text
    """
    transcriber = get_transcriber()
    return transcriber.transcribe(audio_path, language=language)


if __name__ == "__main__":
    # Demo usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python transcription.py <audio_file>")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    
    try:
        transcriber = Transcriber()
        result = transcriber.transcribe_with_confidence(audio_file)
        print(f"Transcription: {result['text']}")
        print(f"Language: {result['language']}")
        print(f"Duration: {result['duration']:.2f}s")
    except TranscriptionError as e:
        print(f"Error: {e}")
        sys.exit(1)
