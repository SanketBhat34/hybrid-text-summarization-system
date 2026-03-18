"""
Language Detection and Multi-Language Support Module
Supports: English, Hindi, Spanish, French, German
"""

from typing import Dict, Optional, List
from langdetect import detect, detect_langs, LangDetectException


# Supported languages configuration
SUPPORTED_LANGUAGES = {
    'en': {
        'name': 'English',
        'flag': '🇬🇧',
        'nltk_name': 'english',
        'tts_code': 'en'
    },
    'hi': {
        'name': 'Hindi',
        'flag': '🇮🇳',
        'nltk_name': 'hindi',  # Limited support
        'tts_code': 'hi'
    },
    'es': {
        'name': 'Spanish',
        'flag': '🇪🇸',
        'nltk_name': 'spanish',
        'tts_code': 'es'
    },
    'fr': {
        'name': 'French',
        'flag': '🇫🇷',
        'nltk_name': 'french',
        'tts_code': 'fr'
    },
    'de': {
        'name': 'German',
        'flag': '🇩🇪',
        'nltk_name': 'german',
        'tts_code': 'de'
    }
}


def detect_language(text: str) -> Dict:
    """
    Detect the language of the given text.
    
    Args:
        text: Input text to detect language
        
    Returns:
        Dictionary with:
        - code: ISO 639-1 language code
        - name: Language name
        - flag: Language flag emoji
        - confidence: Detection confidence (0-1)
        - supported: Whether the language is supported
    """
    result = {
        'code': 'en',
        'name': 'English',
        'flag': '🇬🇧',
        'confidence': 0.0,
        'supported': True,
        'error': None
    }
    
    if not text or len(text.strip()) < 10:
        result['error'] = "Text too short for accurate language detection"
        return result
    
    try:
        # Get detailed detection with probabilities
        langs = detect_langs(text)
        
        if langs:
            top_lang = langs[0]
            lang_code = top_lang.lang
            confidence = top_lang.prob
            
            result['code'] = lang_code
            result['confidence'] = confidence
            
            if lang_code in SUPPORTED_LANGUAGES:
                lang_info = SUPPORTED_LANGUAGES[lang_code]
                result['name'] = lang_info['name']
                result['flag'] = lang_info['flag']
                result['supported'] = True
            else:
                result['name'] = lang_code.upper()
                result['flag'] = '🌍'
                result['supported'] = False
                
    except LangDetectException as e:
        result['error'] = f"Language detection failed: {str(e)}"
    except Exception as e:
        result['error'] = f"Error detecting language: {str(e)}"
    
    return result


def get_stopwords(language_code: str) -> List[str]:
    """
    Get stopwords for the specified language.
    
    Args:
        language_code: ISO 639-1 language code
        
    Returns:
        List of stopwords
    """
    try:
        import nltk
        from nltk.corpus import stopwords
        
        # Download stopwords if not available
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords', quiet=True)
        
        # Map language code to NLTK name
        nltk_name = SUPPORTED_LANGUAGES.get(language_code, {}).get('nltk_name', 'english')
        
        try:
            return stopwords.words(nltk_name)
        except OSError:
            # Fallback to English if language not available
            return stopwords.words('english')
            
    except Exception:
        # Return empty list if NLTK fails
        return []


def get_language_info(language_code: str) -> Dict:
    """
    Get information about a language.
    
    Args:
        language_code: ISO 639-1 language code
        
    Returns:
        Dictionary with language information
    """
    if language_code in SUPPORTED_LANGUAGES:
        return SUPPORTED_LANGUAGES[language_code].copy()
    return {
        'name': language_code.upper(),
        'flag': '🌍',
        'nltk_name': 'english',
        'tts_code': 'en'
    }


def get_supported_languages() -> Dict:
    """Get all supported languages."""
    return SUPPORTED_LANGUAGES.copy()


def is_language_supported(language_code: str) -> bool:
    """Check if a language is supported."""
    return language_code in SUPPORTED_LANGUAGES


class LanguageProcessor:
    """Class for language processing with caching."""
    
    def __init__(self):
        self.cached_language = None
        self.cached_text_hash = None
    
    def detect(self, text: str, use_cache: bool = True) -> Dict:
        """Detect language with optional caching."""
        text_hash = hash(text[:500])  # Hash first 500 chars for performance
        
        if use_cache and text_hash == self.cached_text_hash and self.cached_language:
            return self.cached_language
        
        result = detect_language(text)
        
        self.cached_text_hash = text_hash
        self.cached_language = result
        
        return result
    
    def get_tts_code(self, language_code: str) -> str:
        """Get TTS language code."""
        lang_info = get_language_info(language_code)
        return lang_info.get('tts_code', 'en')
