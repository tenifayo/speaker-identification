"""Tests for API endpoints."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
from src.api import app

client = TestClient(app)

class TestAPI:
    """Test API endpoints."""
    
    @patch('src.api.generate_challenge')
    def test_create_challenge(self, mock_generate):
        """Test challenge generation endpoint."""
        from datetime import datetime
        mock_generate.return_value = ("test_id", "Test sentence", datetime.now())
        
        response = client.post("/challenge/generate")
        assert response.status_code == 200
        data = response.json()
        assert data['challenge_id'] == "test_id"
        assert data['sentence'] == "Test sentence"
        
    @patch('src.api.verify_speaker')
    @patch('src.api.save_upload_files')
    @patch('src.api.cleanup_temp_files')
    def test_verify_with_challenge(self, mock_cleanup, mock_save, mock_verify):
        """Test verification with challenge ID."""
        # Setup mocks
        mock_save.return_value = ["dummy/path.wav"]
        
        mock_result = MagicMock()
        mock_result.user_id = "user1"
        mock_result.is_verified = True
        mock_result.score = 0.9
        mock_result.threshold = 0.7
        mock_result.decision = "granted"
        mock_result.liveness_result = {"passed": True}
        mock_verify.return_value = mock_result
        
        # Send request
        with patch('builtins.open', mock_open_file()):
            response = client.post(
                "/verify",
                data={"user_id": "user1", "challenge_id": "cid"},
                files={"audio_file": ("test.wav", b"dummy", "audio/wav")}
            )
            
        assert response.status_code == 200
        data = response.json()
        assert data['is_verified']
        assert data['liveness_result']['passed']
        
        # Verify call args
        mock_verify.assert_called_once()
        print(mock_verify.call_args)
        assert mock_verify.call_args[1]['challenge_id'] == 'cid'

def mock_open_file():
    """Helper to mock file opening in save_upload_files."""
    from unittest.mock import mock_open
    return mock_open()
