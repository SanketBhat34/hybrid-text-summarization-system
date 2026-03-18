"""
OTP (One-Time Password) Manager

Provides OTP generation and verification for secure authentication.
Supports email-based OTP delivery with configurable expiry.
"""

import random
import string
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
from typing import Optional, Dict, Tuple
from dataclasses import dataclass
import hashlib


@dataclass
class OTPRecord:
    """OTP record with metadata."""
    otp_hash: str
    email: str
    created_at: datetime
    expires_at: datetime
    attempts: int = 0
    verified: bool = False


class OTPManager:
    """
    Manages OTP generation, storage, and verification.
    
    Features:
    - 6-digit numeric OTP
    - Configurable expiry (default 5 minutes)
    - Rate limiting (max 3 attempts)
    - Secure hashing for storage
    """
    
    def __init__(
        self,
        otp_length: int = 6,
        expiry_minutes: int = 5,
        max_attempts: int = 3
    ):
        """
        Initialize OTP Manager.
        
        Args:
            otp_length: Length of OTP (default 6)
            expiry_minutes: OTP validity in minutes (default 5)
            max_attempts: Maximum verification attempts (default 3)
        """
        self.otp_length = otp_length
        self.expiry_minutes = expiry_minutes
        self.max_attempts = max_attempts
        self._otp_store: Dict[str, OTPRecord] = {}
    
    @staticmethod
    def _hash_otp(otp: str) -> str:
        """Hash OTP for secure storage."""
        return hashlib.sha256(otp.encode()).hexdigest()
    
    def generate_otp(self, email: str) -> Tuple[str, datetime]:
        """
        Generate a new OTP for the given email.
        
        Args:
            email: User email address
            
        Returns:
            Tuple of (OTP, expiry_time)
        """
        # Generate numeric OTP
        otp = ''.join(random.choices(string.digits, k=self.otp_length))
        
        # Calculate expiry
        now = datetime.now()
        expires_at = now + timedelta(minutes=self.expiry_minutes)
        
        # Store OTP record (hashed)
        self._otp_store[email.lower()] = OTPRecord(
            otp_hash=self._hash_otp(otp),
            email=email.lower(),
            created_at=now,
            expires_at=expires_at,
            attempts=0,
            verified=False
        )
        
        return otp, expires_at
    
    def verify_otp(self, email: str, otp: str) -> Tuple[bool, str]:
        """
        Verify OTP for the given email.
        
        Args:
            email: User email address
            otp: OTP to verify
            
        Returns:
            Tuple of (success, message)
        """
        email = email.lower()
        
        # Check if OTP exists
        if email not in self._otp_store:
            return False, "No OTP found. Please request a new one."
        
        record = self._otp_store[email]
        
        # Check if already verified
        if record.verified:
            return False, "OTP already used. Please request a new one."
        
        # Check if expired
        if datetime.now() > record.expires_at:
            del self._otp_store[email]
            return False, "OTP expired. Please request a new one."
        
        # Check attempts
        if record.attempts >= self.max_attempts:
            del self._otp_store[email]
            return False, "Too many failed attempts. Please request a new OTP."
        
        # Verify OTP
        if self._hash_otp(otp) == record.otp_hash:
            record.verified = True
            return True, "OTP verified successfully!"
        else:
            record.attempts += 1
            remaining = self.max_attempts - record.attempts
            return False, f"Invalid OTP. {remaining} attempts remaining."
    
    def is_otp_valid(self, email: str) -> bool:
        """Check if there's a valid (non-expired) OTP for email."""
        email = email.lower()
        if email not in self._otp_store:
            return False
        
        record = self._otp_store[email]
        return datetime.now() <= record.expires_at and not record.verified
    
    def get_time_remaining(self, email: str) -> Optional[int]:
        """Get seconds remaining for OTP validity."""
        email = email.lower()
        if email not in self._otp_store:
            return None
        
        record = self._otp_store[email]
        remaining = (record.expires_at - datetime.now()).total_seconds()
        return max(0, int(remaining))
    
    def clear_otp(self, email: str):
        """Clear OTP for email (after successful login)."""
        email = email.lower()
        if email in self._otp_store:
            del self._otp_store[email]


class EmailOTPSender:
    """
    Send OTP via email using SMTP.
    
    Requires SMTP configuration for email delivery.
    """
    
    def __init__(
        self,
        smtp_server: str,
        smtp_port: int = 587,
        sender_email: str = "",
        sender_password: str = ""
    ):
        """
        Initialize email sender.
        
        Args:
            smtp_server: SMTP server address (required)
            smtp_port: SMTP port (default 587 for TLS)
            sender_email: Sender email address
            sender_password: Sender email password/app password
        """
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.sender_email = sender_email
        self.sender_password = sender_password
    
    def send_otp_email(self, recipient_email: str, otp: str, expiry_minutes: int = 5) -> Tuple[bool, str]:
        """
        Send OTP via email.
        
        Args:
            recipient_email: Recipient's email address
            otp: The OTP to send
            expiry_minutes: OTP validity in minutes
            
        Returns:
            Tuple of (success, message)
        """
        if not self.smtp_server or not self.sender_email:
            return False, "Email service not configured. Contact administrator."
        
        try:
            # Create email message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = '🔐 Your Text Summarizer Login OTP'
            msg['From'] = self.sender_email
            msg['To'] = recipient_email
            
            # HTML email content
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <style>
                    body {{ font-family: 'Segoe UI', Arial, sans-serif; background: #f5f5f5; padding: 20px; }}
                    .container {{ max-width: 500px; margin: 0 auto; background: white; border-radius: 15px; padding: 40px; box-shadow: 0 10px 30px rgba(0,0,0,0.1); }}
                    .header {{ text-align: center; margin-bottom: 30px; }}
                    .logo {{ font-size: 3rem; }}
                    .title {{ color: #667eea; font-size: 1.5rem; font-weight: bold; }}
                    .otp-box {{ background: linear-gradient(135deg, #667eea, #764ba2); color: white; font-size: 2.5rem; letter-spacing: 10px; padding: 20px; border-radius: 10px; text-align: center; font-weight: bold; margin: 20px 0; }}
                    .info {{ color: #666; font-size: 0.9rem; text-align: center; }}
                    .warning {{ color: #e74c3c; font-size: 0.85rem; text-align: center; margin-top: 15px; }}
                    .footer {{ text-align: center; margin-top: 30px; color: #999; font-size: 0.8rem; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="header">
                        <div class="logo">📝</div>
                        <div class="title">Text Summarizer</div>
                    </div>
                    <p style="text-align: center; color: #333;">Your One-Time Password for login:</p>
                    <div class="otp-box">{otp}</div>
                    <p class="info">This OTP is valid for <strong>{expiry_minutes} minutes</strong></p>
                    <p class="warning">⚠️ Do not share this OTP with anyone</p>
                    <div class="footer">
                        <p>If you didn't request this, please ignore this email.</p>
                    </div>
                </div>
            </body>
            </html>
            """
            
            text_content = f"Your Text Summarizer OTP is: {otp}\nValid for {expiry_minutes} minutes.\nDo not share this OTP."
            
            msg.attach(MIMEText(text_content, 'plain'))
            msg.attach(MIMEText(html_content, 'html'))
            
            # Send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port, timeout=30) as server:
                server.starttls()
                server.login(self.sender_email, self.sender_password)
                server.send_message(msg)
            
            return True, f"OTP sent to {recipient_email}"
            
        except smtplib.SMTPAuthenticationError:
            return False, "Email authentication failed. Check credentials."
        except smtplib.SMTPException as e:
            return False, f"Failed to send email: {str(e)}"
        except OSError as e:
            return False, f"Network error: {str(e)}. Check internet connection."
        except Exception as e:
            return False, f"Error: {str(e)}"


def format_time_remaining(seconds: int) -> str:
    """Format remaining seconds as MM:SS."""
    minutes = seconds // 60
    secs = seconds % 60
    return f"{minutes:02d}:{secs:02d}"
