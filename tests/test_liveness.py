"""Tests for liveness detection module."""

import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta
from src.liveness import (
    generate_challenge, 
    validate_challenge, 
    calculate_sentence_similarity,
    LivenessError
)

class TestLiveness:
    """Test liveness detection logic."""
    
    def test_calculate_similarity(self):
        """Test fuzzy matching similarity calculation."""
        # Exact match
        assert calculate_sentence_similarity("Hello world", "Hello world") == 1.0
        
        # Case insensitive
        assert calculate_sentence_similarity("Hello world", "hello world") == 1.0
        
        # Minor difference
        assert calculate_sentence_similarity("The blue car", "The blue cat") > 0.8
        
        # Completely different
        assert calculate_sentence_similarity("Hello world", "Goodbye moon") < 0.5
        
    @patch('src.liveness.get_database')
    @patch('src.liveness.SentenceGenerator')
    def test_generate_challenge(self, MockGenerator, mock_get_db):
        """Test challenge generation."""
        # Setup mocks
        mock_db = MagicMock()
        mock_get_db.return_value = mock_db
        
        mock_gen_instance = MockGenerator.return_value
        mock_gen_instance.generate.return_value = ("test_id", "Test sentence")
        
        # Call function
        cid, sentence, expires = generate_challenge(user_id="user1")
        
        assert cid == "test_id"
        assert sentence == "Test sentence"
        assert expires > datetime.now()
        
        # Verify DB call
        mock_db.create_challenge.assert_called_once()
        call_args = mock_db.create_challenge.call_args[1]
        assert call_args['user_id'] == "user1"
        assert call_args['challenge_id'] == "test_id"

    @patch('src.liveness.get_database')
    @patch('src.liveness.get_transcriber')
    def test_validate_challenge_success(self, mock_get_transcriber, mock_get_db, tmp_path):
        """Test successful challenge validation."""
        # Mock DB
        mock_db = MagicMock()
        mock_get_db.return_value = mock_db
        
        mock_challenge = MagicMock()
        mock_challenge.sentence = "The blue car"
        mock_challenge.used = False
        mock_challenge.expires_at = datetime.now() + timedelta(minutes=5)
        mock_db.get_challenge.return_value = mock_challenge
        
        # Mock Transcription
        mock_transcriber = MagicMock()
        mock_get_transcriber.return_value = mock_transcriber
        mock_transcriber.transcribe.return_value = "The blue car"
        
        # Test
        audio_path = tmp_path / "test.wav"
        result = validate_challenge("test_id", audio_path)
        
        assert result.passed
        assert result.similarity_score == 1.0
        mock_db.mark_challenge_used.assert_called_with("test_id")

    @patch('src.liveness.get_database')
    @patch('src.liveness.get_transcriber')
    def test_validate_challenge_fail_mismatch(self, mock_get_transcriber, mock_get_db, tmp_path):
        """Test validation failure due to mismatch."""
        # Mock DB
        mock_db = MagicMock()
        mock_challenge = MagicMock()
        mock_challenge.sentence = "The blue car"
        mock_challenge.used = False
        mock_challenge.expires_at = datetime.now() + timedelta(minutes=5)
        mock_db.get_challenge.return_value = mock_challenge
        mock_get_db.return_value = mock_db
        
        # Mock Transcription (different text)
        mock_transcriber = MagicMock()
        mock_transcriber.transcribe.return_value = "The red truck"
        mock_get_transcriber.return_value = mock_transcriber
        
        # Test
        result = validate_challenge("test_id", tmp_path / "test.wav")
        
        assert not result.passed
        assert result.similarity_score < 0.8
        
    @patch('src.liveness.get_database')
    def test_validate_challenge_expired(self, mock_get_db, tmp_path):
        """Test validation with expired challenge."""
        mock_db = MagicMock()
        mock_challenge = MagicMock()
        mock_challenge.used = False
        mock_challenge.expires_at = datetime.now() - timedelta(minutes=1)
        mock_db.get_challenge.return_value = mock_challenge
        mock_get_db.return_value = mock_db
        
        with pytest.raises(LivenessError, match="expired"):
            validate_challenge("test_id", tmp_path / "test.wav")

    @patch('src.liveness.get_database')
    def test_validate_challenge_used(self, mock_get_db, tmp_path):
        """Test validation with used challenge."""
        mock_db = MagicMock()
        mock_challenge = MagicMock()
        mock_challenge.used = True
        mock_db.get_challenge.return_value = mock_challenge
        mock_get_db.return_value = mock_db
        
        with pytest.raises(LivenessError, match="already been used"):
            validate_challenge("test_id", tmp_path / "test.wav")
