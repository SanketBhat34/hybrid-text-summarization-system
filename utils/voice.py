"""
Voice Input/Output Module
Speech-to-Text and Text-to-Speech functionality
"""

import io
import tempfile
import os
from typing import Dict, Optional
from pathlib import Path


def speech_to_text(audio_data: bytes, language: str = 'en') -> Dict:
    """
    Convert speech audio to text.
    
    Args:
        audio_data: Audio data in bytes (WAV format preferred)
        language: Language code for recognition (en, hi, es, fr, de)
        
    Returns:
        Dictionary with:
        - success: Boolean indicating if transcription was successful
        - text: Transcribed text
        - error: Error message if failed
    """
    result = {
        'success': False,
        'text': '',
        'error': None
    }
    
    if not audio_data:
        result['error'] = "No audio data provided"
        return result
    
    try:
        import speech_recognition as sr
        
        recognizer = sr.Recognizer()
        
        # Save audio to temporary file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_file.write(audio_data)
            temp_path = temp_file.name
        
        try:
            # Load audio file
            with sr.AudioFile(temp_path) as source:
                audio = recognizer.record(source)
            
            # Map language code to Google Speech API format
            language_map = {
                'en': 'en-US',
                'hi': 'hi-IN',
                'es': 'es-ES',
                'fr': 'fr-FR',
                'de': 'de-DE'
            }
            lang_code = language_map.get(language, 'en-US')
            
            # Transcribe using Google Speech Recognition (free tier)
            text = recognizer.recognize_google(audio, language=lang_code)
            
            result['success'] = True
            result['text'] = text
            
        finally:
            # Clean up temp file
            try:
                os.unlink(temp_path)
            except:
                pass
                
    except sr.UnknownValueError:
        result['error'] = "Could not understand the audio. Please speak clearly and try again."
    except sr.RequestError as e:
        result['error'] = f"Speech recognition service error: {str(e)}"
    except ImportError:
        result['error'] = "Speech recognition library not installed. Run: pip install SpeechRecognition"
    except Exception as e:
        result['error'] = f"Error processing audio: {str(e)}"
    
    return result


def text_to_speech(text: str, language: str = 'en') -> Dict:
    """
    Convert text to speech audio.
    
    Args:
        text: Text to convert to speech
        language: Language code (en, hi, es, fr, de)
        
    Returns:
        Dictionary with:
        - success: Boolean indicating if conversion was successful
        - audio_data: Audio data in bytes (MP3 format)
        - error: Error message if failed
    """
    result = {
        'success': False,
        'audio_data': None,
        'error': None
    }
    
    if not text or not text.strip():
        result['error'] = "No text provided for speech synthesis"
        return result
    
    # Limit text length for TTS
    max_chars = 5000
    if len(text) > max_chars:
        text = text[:max_chars]
    
    try:
        from gtts import gTTS
        
        # Map language code to gTTS format
        language_map = {
            'en': 'en',
            'hi': 'hi',
            'es': 'es',
            'fr': 'fr',
            'de': 'de'
        }
        lang_code = language_map.get(language, 'en')
        
        # Generate speech
        tts = gTTS(text=text, lang=lang_code, slow=False)
        
        # Save to bytes buffer
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        
        result['success'] = True
        result['audio_data'] = audio_buffer.read()
        
    except ImportError:
        result['error'] = "gTTS library not installed. Run: pip install gTTS"
    except Exception as e:
        error_msg = str(e).lower()
        if "no text to send" in error_msg:
            result['error'] = "Text is too short or contains only special characters"
        elif "connection" in error_msg or "network" in error_msg:
            result['error'] = "Network error. Please check your internet connection."
        else:
            result['error'] = f"Text-to-speech error: {str(e)}"
    
    return result


def convert_audio_format(audio_data: bytes, from_format: str = 'webm', to_format: str = 'wav') -> bytes:
    """
    Convert audio from one format to another.
    Requires ffmpeg installed for most conversions.
    
    Args:
        audio_data: Input audio data in bytes
        from_format: Source format (webm, mp3, etc.)
        to_format: Target format (wav, mp3, etc.)
        
    Returns:
        Converted audio data in bytes
    """
    try:
        from pydub import AudioSegment
        
        # Create buffer for input
        input_buffer = io.BytesIO(audio_data)
        
        # Load audio
        audio = AudioSegment.from_file(input_buffer, format=from_format)
        
        # Export to new format
        output_buffer = io.BytesIO()
        audio.export(output_buffer, format=to_format)
        output_buffer.seek(0)
        
        return output_buffer.read()
        
    except Exception as e:
        # Return original if conversion fails
        return audio_data


class VoiceProcessor:
    """Class for voice processing with utilities."""
    
    def __init__(self, default_language: str = 'en'):
        self.default_language = default_language
    
    def transcribe(self, audio_data: bytes, language: Optional[str] = None) -> Dict:
        """Transcribe audio to text."""
        lang = language or self.default_language
        return speech_to_text(audio_data, lang)
    
    def synthesize(self, text: str, language: Optional[str] = None) -> Dict:
        """Synthesize text to speech."""
        lang = language or self.default_language
        return text_to_speech(text, lang)
    
    def convert_format(self, audio_data: bytes, from_fmt: str, to_fmt: str) -> bytes:
        """Convert audio format."""
        return convert_audio_format(audio_data, from_fmt, to_fmt)


# Convenience functions
def get_supported_tts_languages() -> Dict[str, str]:
    """Get supported TTS languages."""
    return {
        'en': 'English',
        'hi': 'Hindi',
        'es': 'Spanish',
        'fr': 'French',
        'de': 'German'
    }


def get_supported_stt_languages() -> Dict[str, str]:
    """Get supported STT languages."""
    return {
        'en': 'English',
        'hi': 'Hindi',
        'es': 'Spanish',
        'fr': 'French',
        'de': 'German'
    }
