"""
Authentication Manager for Text Summarizer Application

Provides user authentication with:
- User registration
- Login/Logout
- Password hashing
- Session management
- MongoDB storage (with JSON fallback)
"""

import hashlib
import json
import os
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional, Dict
from pathlib import Path

# Try to import MongoDB
try:
    from database import get_db, get_user_repository
    MONGODB_AVAILABLE = True
except ImportError:
    MONGODB_AVAILABLE = False


@dataclass
class User:
    """User data class."""
    username: str
    email: str
    password_hash: str
    created_at: str
    last_login: Optional[str] = None
    full_name: Optional[str] = None
    user_id: Optional[str] = None  # MongoDB _id


class AuthManager:
    """
    Authentication manager for user login/signup.
    
    Uses MongoDB for storage with JSON file fallback.
    """
    
    def __init__(self, storage_path: str = "data/users.json", use_mongodb: bool = True):
        """
        Initialize the auth manager.
        
        Args:
            storage_path: Path to store user data (fallback)
            use_mongodb: Whether to use MongoDB (default True)
        """
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._users: Dict[str, User] = {}
        
        # Try MongoDB first
        self.use_mongodb = use_mongodb and MONGODB_AVAILABLE
        self.db = None
        self.user_repo = None
        
        if self.use_mongodb:
            try:
                self.db = get_db()
                if self.db.is_connected:
                    self.user_repo = get_user_repository()
                    print("✅ AuthManager using MongoDB")
                else:
                    self.use_mongodb = False
                    print("⚠️ MongoDB not connected, using JSON fallback")
            except Exception as e:
                self.use_mongodb = False
                print(f"⚠️ MongoDB error: {e}, using JSON fallback")
        
        # Load from JSON as fallback
        if not self.use_mongodb:
            self._load_users()
    
    def _load_users(self):
        """Load users from JSON storage file."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)
                    for username, user_data in data.items():
                        self._users[username] = User(**user_data)
            except (json.JSONDecodeError, TypeError):
                self._users = {}
    
    def _save_users(self):
        """Save users to JSON storage file."""
        data = {username: asdict(user) for username, user in self._users.items()}
        with open(self.storage_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    @staticmethod
    def _hash_password(password: str) -> str:
        """Hash password using SHA-256."""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def register(
        self,
        username: str,
        email: str,
        password: str,
        full_name: Optional[str] = None
    ) -> tuple[bool, str]:
        """
        Register a new user.
        
        Args:
            username: Unique username
            email: User email
            password: User password
            full_name: Optional full name
            
        Returns:
            Tuple of (success, message)
        """
        # Validation
        if not username or len(username) < 3:
            return False, "Username must be at least 3 characters"
        
        if not email or '@' not in email:
            return False, "Please enter a valid email"
        
        if not password or len(password) < 6:
            return False, "Password must be at least 6 characters"
        
        password_hash = self._hash_password(password)
        
        # Use MongoDB if available
        if self.use_mongodb and self.user_repo:
            success, message, user_id = self.user_repo.create_user(
                username=username,
                email=email,
                password_hash=password_hash,
                full_name=full_name
            )
            return success, message
        
        # JSON fallback
        if username.lower() in [u.lower() for u in self._users.keys()]:
            return False, "Username already exists"
        
        # Check if email already exists
        for user in self._users.values():
            if user.email.lower() == email.lower():
                return False, "Email already registered"
        
        # Create user
        user = User(
            username=username,
            email=email,
            password_hash=password_hash,
            created_at=datetime.now().isoformat(),
            full_name=full_name
        )
        
        self._users[username] = user
        self._save_users()
        
        return True, "Registration successful! Please login."
    
    def login(self, username: str, password: str) -> tuple[bool, str, Optional[User]]:
        """
        Authenticate user login.
        
        Args:
            username: Username or email
            password: User password
            
        Returns:
            Tuple of (success, message, user_object)
        """
        if not username or not password:
            return False, "Please enter username and password", None
        
        password_hash = self._hash_password(password)
        
        # Use MongoDB if available
        if self.use_mongodb and self.user_repo:
            user_doc = self.user_repo.find_by_username_or_email(username)
            
            if not user_doc:
                return False, "User not found", None
            
            if user_doc.get("password_hash") != password_hash:
                return False, "Incorrect password", None
            
            # Update last login
            self.user_repo.update_last_login(str(user_doc["_id"]))
            
            user = User(
                username=user_doc["username"],
                email=user_doc["email"],
                password_hash=user_doc["password_hash"],
                created_at=user_doc["created_at"].isoformat() if hasattr(user_doc["created_at"], 'isoformat') else str(user_doc["created_at"]),
                last_login=datetime.now().isoformat(),
                full_name=user_doc.get("full_name"),
                user_id=str(user_doc["_id"])
            )
            
            return True, f"Welcome back, {user.full_name or user.username}!", user
        
        # JSON fallback
        user = None
        for u in self._users.values():
            if u.username.lower() == username.lower() or u.email.lower() == username.lower():
                user = u
                break
        
        if not user:
            return False, "User not found", None
        
        if user.password_hash != password_hash:
            return False, "Incorrect password", None
        
        # Update last login
        user.last_login = datetime.now().isoformat()
        self._save_users()
        
        return True, f"Welcome back, {user.full_name or user.username}!", user
    
    def get_user(self, username: str) -> Optional[User]:
        """Get user by username."""
        # Use MongoDB if available
        if self.use_mongodb and self.user_repo:
            user_doc = self.user_repo.find_by_username(username)
            if user_doc:
                return User(
                    username=user_doc["username"],
                    email=user_doc["email"],
                    password_hash=user_doc["password_hash"],
                    created_at=user_doc["created_at"].isoformat() if hasattr(user_doc["created_at"], 'isoformat') else str(user_doc["created_at"]),
                    last_login=user_doc.get("last_login"),
                    full_name=user_doc.get("full_name"),
                    user_id=str(user_doc["_id"])
                )
            return None
        
        return self._users.get(username)
    
    def user_exists(self, username: str) -> bool:
        """Check if username exists."""
        # Use MongoDB if available
        if self.use_mongodb and self.user_repo:
            return self.user_repo.username_exists(username)
        
        return username.lower() in [u.lower() for u in self._users.keys()]
    
    def update_password(self, username: str, old_password: str, new_password: str) -> tuple[bool, str]:
        """
        Update user password.
        
        Args:
            username: Username
            old_password: Current password
            new_password: New password
            
        Returns:
            Tuple of (success, message)
        """
        user = self._users.get(username)
        if not user:
            return False, "User not found"
        
        if user.password_hash != self._hash_password(old_password):
            return False, "Current password is incorrect"
        
        if len(new_password) < 6:
            return False, "New password must be at least 6 characters"
        
        user.password_hash = self._hash_password(new_password)
        self._save_users()
        
        return True, "Password updated successfully"
    
    def email_exists(self, email: str) -> bool:
        """
        Check if email is registered.
        
        Args:
            email: Email address to check
            
        Returns:
            True if email exists, False otherwise
        """
        # Use MongoDB if available
        if self.use_mongodb and self.user_repo:
            return self.user_repo.email_exists(email)
        
        # JSON fallback
        email_lower = email.lower()
        for user in self._users.values():
            if user.email.lower() == email_lower:
                return True
        return False
    
    def get_user_by_email(self, email: str) -> Optional[User]:
        """
        Get user by email address.
        
        Args:
            email: Email address
            
        Returns:
            User object if found, None otherwise
        """
        # Use MongoDB if available
        if self.use_mongodb and self.user_repo:
            user_doc = self.user_repo.find_by_email(email)
            if user_doc:
                self.user_repo.update_last_login(str(user_doc["_id"]))
                return User(
                    username=user_doc["username"],
                    email=user_doc["email"],
                    password_hash=user_doc["password_hash"],
                    created_at=user_doc["created_at"].isoformat() if hasattr(user_doc["created_at"], 'isoformat') else str(user_doc["created_at"]),
                    last_login=datetime.now().isoformat(),
                    full_name=user_doc.get("full_name"),
                    user_id=str(user_doc["_id"])
                )
            return None
        
        # JSON fallback
        email_lower = email.lower()
        for user in self._users.values():
            if user.email.lower() == email_lower:
                # Update last login
                user.last_login = datetime.now().isoformat()
                self._save_users()
                return user
        return None
