"""Tests for verification module."""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch

class TestVerification:
    """Test verification functions."""
    
    def test_cosine_similarity_identical(self):
        """Test cosine similarity with identical vectors."""
        from src.verification import cosine_similarity
        
        a = np.array([1.0, 2.0, 3.0])
        score = cosine_similarity(a, a)
        
        assert score == pytest.approx(1.0, abs=1e-6)
    
    def test_cosine_similarity_orthogonal(self):
        """Test cosine similarity with orthogonal vectors."""
        from src.verification import cosine_similarity
        
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        score = cosine_similarity(a, b)
        
        assert score == pytest.approx(0.0, abs=1e-6)
    
    def test_cosine_similarity_opposite(self):
        """Test cosine similarity with opposite vectors."""
        from src.verification import cosine_similarity
        
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([-1.0, -2.0, -3.0])
        score = cosine_similarity(a, b)
        
        assert score == pytest.approx(-1.0, abs=1e-6)
    
    def test_cosine_similarity_range(self):
        """Test that cosine similarity is in valid range."""
        from src.verification import cosine_similarity
        
        np.random.seed(42)
        for _ in range(100):
            a = np.random.randn(192)
            b = np.random.randn(192)
            score = cosine_similarity(a, b)
            
            assert -1.0 <= score <= 1.0
            
    @patch('src.verification.get_database')
    @patch('src.verification.get_extractor')
    @patch('src.liveness.validate_challenge')
    def test_verify_speaker_with_liveness(self, mock_validate, mock_get_extractor, mock_get_db, tmp_path):
        """Test verification with liveness check."""
        from src.verification import verify_speaker
        
        # Mock DB
        mock_db = MagicMock()
        mock_db.get_user_embedding.return_value = np.ones(192)
        mock_get_db.return_value = mock_db
        
        # Mock Extractor
        mock_extractor = MagicMock()
        mock_extractor.extract_from_file.return_value = np.ones(192) # Perfect match
        mock_get_extractor.return_value = mock_extractor
        
        # Mock Liveness (Passed)
        mock_liveness_result = MagicMock()
        mock_liveness_result.passed = True
        mock_liveness_result.to_dict.return_value = {"passed": True}
        mock_validate.return_value = mock_liveness_result
        
        # Test passed
        result = verify_speaker(
            audio_path=tmp_path / "test.wav",
            claimed_user_id="user1",
            challenge_id="valid_challenge"
        )
        assert result.is_verified
        assert result.liveness_result["passed"]
        
        # Mock Liveness (Failed)
        mock_liveness_result.passed = False
        mock_liveness_result.to_dict.return_value = {"passed": False}
        
        result = verify_speaker(
            audio_path=tmp_path / "test.wav",
            claimed_user_id="user1",
            challenge_id="valid_challenge"
        )
        assert not result.is_verified
        assert not result.liveness_result["passed"]


class TestDatabase:
    """Test database operations."""
    
    @pytest.fixture
    def temp_db(self, tmp_path):
        """Create a temporary database."""
        from src.database import Database
        db_path = tmp_path / "test.db"
        return Database(db_path)
    
    def test_add_and_get_user(self, temp_db):
        """Test adding and retrieving a user."""
        embedding = np.random.randn(192).astype(np.float32)
        
        success = temp_db.add_user("test1", "Test User", embedding, 3)
        assert success
        
        user = temp_db.get_user("test1")
        assert user is not None
        assert user.id == "test1"
        assert user.name == "Test User"
        assert user.num_samples == 3
        # Use almost equal for float comparison
        np.testing.assert_array_almost_equal(user.embedding, embedding)
    
    def test_duplicate_user(self, temp_db):
        """Test that duplicate users are rejected."""
        embedding = np.random.randn(192).astype(np.float32)
        
        temp_db.add_user("test1", "Test User", embedding, 1)
        success = temp_db.add_user("test1", "Another User", embedding, 1)
        
        assert not success
    
    def test_delete_user(self, temp_db):
        """Test deleting a user."""
        embedding = np.random.randn(192).astype(np.float32)
        temp_db.add_user("test1", "Test User", embedding, 1)
        
        deleted = temp_db.delete_user("test1")
        assert deleted
        
        user = temp_db.get_user("test1")
        assert user is None
    
    def test_list_users(self, temp_db):
        """Test listing users."""
        embedding = np.random.randn(192).astype(np.float32)
        temp_db.add_user("user1", "User One", embedding, 1)
        temp_db.add_user("user2", "User Two", embedding, 2)
        
        users = temp_db.list_users()
        assert len(users) == 2
    
    def test_access_logging(self, temp_db):
        """Test access logging."""
        embedding = np.random.randn(192).astype(np.float32)
        temp_db.add_user("test1", "Test User", embedding, 1)
        
        temp_db.log_access("test1", "granted", 0.85, 0.5)
        temp_db.log_access("test1", "denied", 0.35, 0.5)
        
        logs = temp_db.get_access_logs("test1")
        assert len(logs) == 2
    
    def test_challenge_operations(self, temp_db):
        """Test challenge table operations."""
        from datetime import datetime, timedelta
        
        expires = datetime.now() + timedelta(minutes=5)
        success = temp_db.create_challenge("c1", "test sentence", expires, "user1")
        assert success
        
        challenge = temp_db.get_challenge("c1")
        assert challenge is not None
        assert challenge.sentence == "test sentence"
        assert not challenge.used
        
        # Mark used
        temp_db.mark_challenge_used("c1")
        challenge = temp_db.get_challenge("c1")
        assert challenge.used
