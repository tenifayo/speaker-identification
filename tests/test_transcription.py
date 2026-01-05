"""Tests for transcription module."""

import pytest
from unittest.mock import MagicMock, patch, mock_open
from src.transcription import Transcriber, TranscriptionError

class TestTranscriber:
    """Test Transcriber class."""
    
    @pytest.fixture
    def mock_groq(self):
        with patch('src.transcription.Groq') as mock:
            yield mock
            
    @pytest.fixture
    def transcriber(self, mock_groq):
        return Transcriber(api_key="test_key")
    
    def test_init_no_key(self):
        """Test initialization without API key."""
        with patch.dict('os.environ', clear=True):
            with pytest.raises(ValueError, match="Groq API key not provided"):
                Transcriber(api_key=None)
                
    def test_init_with_env_key(self):
        """Test initialization with environment variable."""
        with patch.dict('os.environ', {'GROQ_API_KEY': 'env_key'}):
            with patch('src.transcription.Groq') as mock_groq:
                t = Transcriber()
                mock_groq.assert_called_with(api_key='env_key')

    def test_transcribe_success(self, transcriber, tmp_path):
        """Test successful transcription."""
        # Setup mock response
        mock_client = transcriber.client
        mock_client.audio.transcriptions.create.return_value = "Hello world"
        
        # Create dummy file
        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"dummy audio")
        
        # Call method
        result = transcriber.transcribe(audio_file)
        
        assert result == "Hello world"
        mock_client.audio.transcriptions.create.assert_called_once()
        
    def test_transcribe_file_not_found(self, transcriber):
        """Test transcription with missing file."""
        with pytest.raises(TranscriptionError, match="Audio file not found"):
            transcriber.transcribe("nonexistent.wav")
            
    def test_transcribe_api_error(self, transcriber, tmp_path):
        """Test transcription with API error."""
        # Setup mock to raise exception
        mock_client = transcriber.client
        mock_client.audio.transcriptions.create.side_effect = Exception("API Error")
        
        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"dummy audio")
        
        with pytest.raises(TranscriptionError, match="Transcription failed"):
            transcriber.transcribe(audio_file)

    def test_transcribe_with_confidence(self, transcriber, tmp_path):
        """Test verbose transcription."""
        # Mock response object
        mock_response = MagicMock()
        mock_response.text = "Hello world"
        mock_response.language = "en"
        mock_response.duration = 2.5
        mock_response.segments = []
        
        mock_client = transcriber.client
        mock_client.audio.transcriptions.create.return_value = mock_response
        
        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"dummy audio")
        
        result = transcriber.transcribe_with_confidence(audio_file)
        
        assert result['text'] == "Hello world"
        assert result['language'] == "en"
        assert result['duration'] == 2.5
        mock_client.audio.transcriptions.create.assert_called_once()
