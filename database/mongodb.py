"""
MongoDB Database Connection Module

Provides database connectivity for the Text Summarizer application.
Handles user data, summaries, and session management.
"""

import os
from datetime import datetime
from typing import Optional, Dict, List, Any
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.database import Database
from pymongo.collection import Collection
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

# Load environment variables
load_dotenv()


class MongoDB:
    """
    MongoDB connection manager for Text Summarizer.
    
    Collections:
    - users: User accounts and authentication
    - summaries: Generated summaries history
    - sessions: User session tracking
    """
    
    _instance = None
    _client: Optional[MongoClient] = None
    _db: Optional[Database] = None
    
    def __new__(cls):
        """Singleton pattern for database connection."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize MongoDB connection from environment variables."""
        if self._client is None:
            self.connect()
    
    def connect(self) -> bool:
        """
        Establish connection to MongoDB.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017")
            db_name = os.getenv("MONGO_DB_NAME", "text_summarizer")
            
            self._client = MongoClient(
                mongo_uri,
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=5000
            )
            
            # Test connection
            self._client.admin.command('ping')
            self._db = self._client[db_name]
            
            # Create indexes
            self._create_indexes()
            
            print(f"✅ Connected to MongoDB: {db_name}")
            return True
            
        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            print(f"❌ MongoDB connection failed: {e}")
            self._client = None
            self._db = None
            return False
    
    def _create_indexes(self):
        """Create database indexes for better performance."""
        if self._db is not None:
            # Users collection indexes
            self._db.users.create_index("username", unique=True)
            self._db.users.create_index("email", unique=True)
            
            # Summaries collection indexes
            self._db.summaries.create_index("user_id")
            self._db.summaries.create_index("created_at")
    
    @property
    def is_connected(self) -> bool:
        """Check if database is connected."""
        if self._client is None:
            return False
        try:
            self._client.admin.command('ping')
            return True
        except:
            return False
    
    @property
    def db(self) -> Optional[Database]:
        """Get database instance."""
        return self._db
    
    @property
    def users(self) -> Optional[Collection]:
        """Get users collection."""
        return self._db.users if self._db is not None else None
    
    @property
    def summaries(self) -> Optional[Collection]:
        """Get summaries collection."""
        return self._db.summaries if self._db is not None else None
    
    @property
    def sessions(self) -> Optional[Collection]:
        """Get sessions collection."""
        return self._db.sessions if self._db is not None else None
    
    def close(self):
        """Close database connection."""
        if self._client:
            self._client.close()
            self._client = None
            self._db = None


class UserRepository:
    """
    User data repository for MongoDB.
    
    Handles all user-related database operations.
    """
    
    def __init__(self, db: MongoDB):
        """Initialize with database connection."""
        self.db = db
    
    def create_user(
        self,
        username: str,
        email: str,
        password_hash: str,
        full_name: Optional[str] = None
    ) -> tuple[bool, str, Optional[str]]:
        """
        Create a new user in the database.
        
        Returns:
            Tuple of (success, message, user_id)
        """
        if not self.db.is_connected:
            return False, "Database not connected", None
        
        try:
            user_doc = {
                "username": username,
                "email": email.lower(),
                "password_hash": password_hash,
                "full_name": full_name,
                "created_at": datetime.now(),
                "last_login": None,
                "is_active": True
            }
            
            result = self.db.users.insert_one(user_doc)
            return True, "Registration successful!", str(result.inserted_id)
            
        except Exception as e:
            if "duplicate key" in str(e):
                if "username" in str(e):
                    return False, "Username already exists", None
                if "email" in str(e):
                    return False, "Email already registered", None
            return False, f"Registration failed: {str(e)}", None
    
    def find_by_username(self, username: str) -> Optional[Dict]:
        """Find user by username."""
        if not self.db.is_connected:
            return None
        return self.db.users.find_one({"username": {"$regex": f"^{username}$", "$options": "i"}})
    
    def find_by_email(self, email: str) -> Optional[Dict]:
        """Find user by email."""
        if not self.db.is_connected:
            return None
        return self.db.users.find_one({"email": email.lower()})
    
    def find_by_username_or_email(self, identifier: str) -> Optional[Dict]:
        """Find user by username or email."""
        if not self.db.is_connected:
            return None
        return self.db.users.find_one({
            "$or": [
                {"username": {"$regex": f"^{identifier}$", "$options": "i"}},
                {"email": identifier.lower()}
            ]
        })
    
    def update_last_login(self, user_id: str):
        """Update user's last login timestamp."""
        if self.db.is_connected:
            from bson import ObjectId
            self.db.users.update_one(
                {"_id": ObjectId(user_id)},
                {"$set": {"last_login": datetime.now()}}
            )
    
    def email_exists(self, email: str) -> bool:
        """Check if email is registered."""
        return self.find_by_email(email) is not None
    
    def username_exists(self, username: str) -> bool:
        """Check if username is taken."""
        return self.find_by_username(username) is not None


class SummaryRepository:
    """
    Summary history repository for MongoDB.
    
    Stores generated summaries for users.
    """
    
    def __init__(self, db: MongoDB):
        """Initialize with database connection."""
        self.db = db
    
    def save_summary(
        self,
        user_id: str,
        original_text: str,
        summary: str,
        method: str,
        compression_ratio: float,
        word_count_original: int,
        word_count_summary: int,
        rouge_scores: Optional[Dict] = None
    ) -> Optional[str]:
        """
        Save a generated summary to history.
        
        Returns:
            Summary document ID if successful
        """
        if not self.db.is_connected:
            return None
        
        try:
            from bson import ObjectId
            summary_doc = {
                "user_id": ObjectId(user_id),
                "original_text": original_text[:1000],  # Store first 1000 chars
                "summary": summary,
                "method": method,
                "compression_ratio": compression_ratio,
                "word_count_original": word_count_original,
                "word_count_summary": word_count_summary,
                "rouge_scores": rouge_scores,
                "created_at": datetime.now()
            }
            
            result = self.db.summaries.insert_one(summary_doc)
            return str(result.inserted_id)
        except Exception as e:
            print(f"Error saving summary: {e}")
            return None
    
    def get_user_summaries(
        self,
        user_id: str,
        limit: int = 10,
        skip: int = 0
    ) -> List[Dict]:
        """Get user's summary history."""
        if not self.db.is_connected:
            return []
        
        try:
            from bson import ObjectId
            cursor = self.db.summaries.find(
                {"user_id": ObjectId(user_id)}
            ).sort("created_at", -1).skip(skip).limit(limit)
            
            return list(cursor)
        except:
            return []
    
    def get_summary_count(self, user_id: str) -> int:
        """Get total summary count for user."""
        if not self.db.is_connected:
            return 0
        
        try:
            from bson import ObjectId
            return self.db.summaries.count_documents({"user_id": ObjectId(user_id)})
        except:
            return 0


# Global database instance
db = MongoDB()
user_repo = UserRepository(db)
summary_repo = SummaryRepository(db)


def get_db() -> MongoDB:
    """Get the database instance."""
    return db


def get_user_repository() -> UserRepository:
    """Get the user repository."""
    return user_repo


def get_summary_repository() -> SummaryRepository:
    """Get the summary repository."""
    return summary_repo
