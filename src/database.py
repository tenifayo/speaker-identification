"""SQLite database for user storage and access logging."""

import sqlite3
import json
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

from .config import config


@dataclass
class User:
    """User data model."""
    id: str
    name: str
    embedding: np.ndarray
    num_samples: int
    created_at: datetime
    updated_at: datetime


@dataclass
class AccessLog:
    """Access log entry."""
    id: int
    user_id: str
    timestamp: datetime
    decision: str  # 'granted' | 'denied'
    score: float
    threshold: float
    challenge_id: Optional[str] = None
    transcription: Optional[str] = None
    sentence_match: Optional[bool] = None
    liveness_passed: Optional[bool] = None


@dataclass
class Challenge:
    """Liveness challenge entry."""
    id: str
    user_id: Optional[str]
    sentence: str
    created_at: datetime
    expires_at: datetime
    used: bool


class Database:
    """SQLite database manager for speaker identification system."""
    
    def __init__(self, db_path: Path = None):
        self.db_path = db_path or config.database.db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection with row factory."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def _init_db(self):
        """Initialize database schema."""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    num_samples INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS access_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    decision TEXT NOT NULL,
                    score REAL NOT NULL,
                    threshold REAL NOT NULL,
                    challenge_id TEXT,
                    transcription TEXT,
                    sentence_match BOOLEAN,
                    liveness_passed BOOLEAN,
                    FOREIGN KEY (user_id) REFERENCES users(id),
                    FOREIGN KEY (challenge_id) REFERENCES challenges(id)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS challenges (
                    id TEXT PRIMARY KEY,
                    user_id TEXT,
                    sentence TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP NOT NULL,
                    used BOOLEAN DEFAULT 0,
                    FOREIGN KEY (user_id) REFERENCES users(id)
                )
            """)
            conn.commit()
    
    @staticmethod
    def _serialize_embedding(embedding: np.ndarray) -> bytes:
        """Serialize numpy array to bytes."""
        return embedding.astype(np.float32).tobytes()
    
    @staticmethod
    def _deserialize_embedding(data: bytes) -> np.ndarray:
        """Deserialize bytes to numpy array."""
        return np.frombuffer(data, dtype=np.float32)
    
    # ==================== User Operations ====================
    
    def add_user(
        self,
        user_id: str,
        name: str,
        embedding: np.ndarray,
        num_samples: int = 1
    ) -> bool:
        """
        Add new user to database.
        
        Returns:
            True if successful, False if user already exists
        """
        try:
            with self._get_connection() as conn:
                conn.execute(
                    """INSERT INTO users (id, name, embedding, num_samples)
                       VALUES (?, ?, ?, ?)""",
                    (user_id, name, self._serialize_embedding(embedding), num_samples)
                )
                conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False
    
    def update_user_embedding(
        self,
        user_id: str,
        embedding: np.ndarray,
        num_samples: int
    ) -> bool:
        """Update user's embedding."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                """UPDATE users 
                   SET embedding = ?, num_samples = ?, updated_at = CURRENT_TIMESTAMP
                   WHERE id = ?""",
                (self._serialize_embedding(embedding), num_samples, user_id)
            )
            conn.commit()
            return cursor.rowcount > 0
    
    def get_user(self, user_id: str) -> Optional[User]:
        """Get user by ID."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM users WHERE id = ?", (user_id,)
            ).fetchone()
            
            if row is None:
                return None
            
            return User(
                id=row['id'],
                name=row['name'],
                embedding=self._deserialize_embedding(row['embedding']),
                num_samples=row['num_samples'],
                created_at=datetime.fromisoformat(row['created_at']),
                updated_at=datetime.fromisoformat(row['updated_at'])
            )
    
    def get_user_embedding(self, user_id: str) -> Optional[np.ndarray]:
        """Get user's embedding vector."""
        user = self.get_user(user_id)
        return user.embedding if user else None
    
    def list_users(self) -> List[Dict[str, Any]]:
        """List all users (without embeddings)."""
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT id, name, num_samples, created_at, updated_at FROM users"
            ).fetchall()
            
            return [dict(row) for row in rows]
    
    def delete_user(self, user_id: str) -> bool:
        """Delete user from database."""
        with self._get_connection() as conn:
            cursor = conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
            conn.commit()
            return cursor.rowcount > 0
    
    def user_exists(self, user_id: str) -> bool:
        """Check if user exists."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT 1 FROM users WHERE id = ?", (user_id,)
            ).fetchone()
            return row is not None
    
    # ==================== Access Log Operations ====================
    
    def log_access(
        self,
        user_id: str,
        decision: str,
        score: float,
        threshold: float,
        challenge_id: Optional[str] = None,
        transcription: Optional[str] = None,
        sentence_match: Optional[bool] = None,
        liveness_passed: Optional[bool] = None
    ) -> None:
        """Log access attempt."""
        with self._get_connection() as conn:
            conn.execute(
                """INSERT INTO access_logs 
                   (user_id, decision, score, threshold, challenge_id, transcription, sentence_match, liveness_passed)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (user_id, decision, score, threshold, challenge_id, transcription, sentence_match, liveness_passed)
            )
            conn.commit()
    
    def get_access_logs(
        self,
        user_id: str = None,
        limit: int = 100
    ) -> List[AccessLog]:
        """Get access logs, optionally filtered by user."""
        with self._get_connection() as conn:
            if user_id:
                rows = conn.execute(
                    """SELECT * FROM access_logs 
                       WHERE user_id = ? 
                       ORDER BY timestamp DESC LIMIT ?""",
                    (user_id, limit)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM access_logs ORDER BY timestamp DESC LIMIT ?",
                    (limit,)
                ).fetchall()
            
            return [
                AccessLog(
                    id=row['id'],
                    user_id=row['user_id'],
                    timestamp=datetime.fromisoformat(row['timestamp']),
                    decision=row['decision'],
                    score=row['score'],
                    threshold=row['threshold'],
                    challenge_id=dict(row).get('challenge_id'),
                    transcription=dict(row).get('transcription'),
                    sentence_match=bool(row['sentence_match']) if dict(row).get('sentence_match') is not None else None,
                    liveness_passed=bool(row['liveness_passed']) if dict(row).get('liveness_passed') is not None else None
                )
                for row in rows
            ]
    
    # ==================== Challenge Operations ====================
    
    def create_challenge(
        self,
        challenge_id: str,
        sentence: str,
        expires_at: datetime,
        user_id: Optional[str] = None
    ) -> bool:
        """
        Create a new liveness challenge.
        
        Returns:
            True if successful, False if challenge ID already exists
        """
        try:
            with self._get_connection() as conn:
                conn.execute(
                    """INSERT INTO challenges (id, user_id, sentence, expires_at)
                       VALUES (?, ?, ?, ?)""",
                    (challenge_id, user_id, sentence, expires_at.isoformat())
                )
                conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False
    
    def get_challenge(self, challenge_id: str) -> Optional[Challenge]:
        """Get challenge by ID."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM challenges WHERE id = ?", (challenge_id,)
            ).fetchone()
            
            if row is None:
                return None
            
            return Challenge(
                id=row['id'],
                user_id=row['user_id'],
                sentence=row['sentence'],
                created_at=datetime.fromisoformat(row['created_at']),
                expires_at=datetime.fromisoformat(row['expires_at']),
                used=bool(row['used'])
            )
    
    def mark_challenge_used(self, challenge_id: str) -> bool:
        """Mark challenge as used."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "UPDATE challenges SET used = 1 WHERE id = ?",
                (challenge_id,)
            )
            conn.commit()
            return cursor.rowcount > 0
    
    def delete_expired_challenges(self) -> int:
        """Delete expired challenges."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM challenges WHERE expires_at < ?",
                (datetime.now().isoformat(),)
            )
            conn.commit()
            return cursor.rowcount



# Global database instance
_db_instance: Optional[Database] = None


def get_database() -> Database:
    """Get singleton database instance."""
    global _db_instance
    if _db_instance is None:
        _db_instance = Database()
    return _db_instance
