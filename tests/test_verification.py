"""Tests for verification module."""

import pytest
import numpy as np


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
        assert np.allclose(user.embedding, embedding)
    
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
