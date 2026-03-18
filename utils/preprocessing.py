"""
Text Preprocessing Module
Implements various text preprocessing algorithms for text summarization.

Algorithms Implemented:
1. Sentence Segmentation - Splits text into sentences
2. Tokenization - Splits sentences into words/tokens
3. Stop-word Removal - Removes common words with low semantic value
4. Lemmatization - Reduces words to their base/dictionary form
5. Stemming - Reduces words to their root form (alternative to lemmatization)
"""

import re
import string
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from typing import List, Dict, Tuple, Optional


class TextPreprocessor:
    """
    Comprehensive text preprocessing pipeline.
    
    Provides methods for:
    - Sentence segmentation
    - Tokenization
    - Stop-word removal
    - Lemmatization / Stemming
    - Text cleaning
    """
    
    # Language code to NLTK language name mapping
    LANGUAGE_MAP = {
        'en': 'english',
        'hi': 'english',  # Hindi fallback to English stopwords (limited NLTK support)
        'es': 'spanish',
        'fr': 'french',
        'de': 'german'
    }
    
    def __init__(self, use_lemmatization: bool = True, language: str = 'en'):
        """
        Initialize the preprocessor.
        
        Args:
            use_lemmatization: If True, use lemmatization; if False, use stemming
            language: Language code (en, hi, es, fr, de)
        """
        self._download_nltk_data()
        self.language = language
        self.nltk_language = self.LANGUAGE_MAP.get(language, 'english')
        
        # Load stopwords for the specified language
        try:
            self.stop_words = set(stopwords.words(self.nltk_language))
        except OSError:
            # Fallback to English if language not available
            self.stop_words = set(stopwords.words('english'))
        
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        self.use_lemmatization = use_lemmatization
        
        # Add custom stop words for summarization (English-specific)
        if language == 'en':
            self.custom_stop_words = {
                'said', 'also', 'would', 'could', 'may', 'might',
                'one', 'two', 'three', 'first', 'second', 'third',
                'however', 'therefore', 'thus', 'hence', 'moreover'
            }
            self.stop_words.update(self.custom_stop_words)
    
    def set_language(self, language: str):
        """
        Change the language for preprocessing.
        
        Args:
            language: Language code (en, hi, es, fr, de)
        """
        self.language = language
        self.nltk_language = self.LANGUAGE_MAP.get(language, 'english')
        
        try:
            self.stop_words = set(stopwords.words(self.nltk_language))
        except OSError:
            self.stop_words = set(stopwords.words('english'))
        
        # Add English custom stopwords only for English
        if language == 'en':
            self.custom_stop_words = {
                'said', 'also', 'would', 'could', 'may', 'might',
                'one', 'two', 'three', 'first', 'second', 'third',
                'however', 'therefore', 'thus', 'hence', 'moreover'
            }
            self.stop_words.update(self.custom_stop_words)
    
    def _download_nltk_data(self):
        """Download required NLTK data packages."""
        required_packages = [
            ('tokenizers/punkt', 'punkt'),
            ('tokenizers/punkt_tab', 'punkt_tab'),
            ('corpora/stopwords', 'stopwords'),
            ('corpora/wordnet', 'wordnet'),
            ('taggers/averaged_perceptron_tagger', 'averaged_perceptron_tagger'),
            ('taggers/averaged_perceptron_tagger_eng', 'averaged_perceptron_tagger_eng')
        ]
        
        for path, package in required_packages:
            try:
                nltk.data.find(path)
            except LookupError:
                try:
                    nltk.download(package, quiet=True)
                except Exception:
                    pass
    
    def segment_sentences(self, text: str) -> List[str]:
        """
        Segment text into sentences.
        
        Algorithm: Uses NLTK's Punkt sentence tokenizer which uses
        unsupervised learning to build a model for abbreviation words,
        collocations, and sentence starters.
        
        Args:
            text: Input text
            
        Returns:
            List of sentences
        """
        if not text or not text.strip():
            return []
        
        # Clean text first
        text = self._clean_text(text)
        
        # Use NLTK sentence tokenizer
        sentences = sent_tokenize(text)
        
        # Additional cleanup for each sentence
        cleaned_sentences = []
        for sent in sentences:
            sent = sent.strip()
            if sent and len(sent) > 3:  # Filter very short fragments
                cleaned_sentences.append(sent)
        
        return cleaned_sentences
    
    def tokenize(self, text: str, remove_punctuation: bool = True) -> List[str]:
        """
        Tokenize text into words.
        
        Algorithm: Uses NLTK's word_tokenize which uses TreebankWordTokenizer
        along with PunktSentenceTokenizer for better handling of contractions
        and punctuation.
        
        Args:
            text: Input text or sentence
            remove_punctuation: Whether to remove punctuation tokens
            
        Returns:
            List of tokens
        """
        if not text:
            return []
        
        tokens = word_tokenize(text.lower())
        
        if remove_punctuation:
            tokens = [t for t in tokens if t not in string.punctuation and t.isalnum()]
        
        return tokens
    
    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """
        Remove stop words from token list.
        
        Algorithm: Filters tokens against a comprehensive stop word list
        that includes NLTK's English stop words plus custom additions
        for summarization context.
        
        Args:
            tokens: List of word tokens
            
        Returns:
            Filtered token list
        """
        return [t for t in tokens if t.lower() not in self.stop_words]
    
    def lemmatize(self, tokens: List[str]) -> List[str]:
        """
        Lemmatize tokens to their base dictionary form.
        
        Algorithm: Uses WordNet Lemmatizer which uses the WordNet Database
        to find the morphological root of words.
        
        Example: "running" -> "run", "better" -> "good", "cats" -> "cat"
        
        Args:
            tokens: List of word tokens
            
        Returns:
            List of lemmatized tokens
        """
        lemmatized = []
        
        # Get POS tags for better lemmatization
        try:
            pos_tags = nltk.pos_tag(tokens)
            
            for token, tag in pos_tags:
                # Map POS tag to WordNet tag
                wn_tag = self._get_wordnet_pos(tag)
                if wn_tag:
                    lemmatized.append(self.lemmatizer.lemmatize(token, wn_tag))
                else:
                    lemmatized.append(self.lemmatizer.lemmatize(token))
        except Exception:
            # Fallback without POS tagging
            lemmatized = [self.lemmatizer.lemmatize(t) for t in tokens]
        
        return lemmatized
    
    def stem(self, tokens: List[str]) -> List[str]:
        """
        Stem tokens to their root form using Porter Stemmer.
        
        Algorithm: Porter Stemmer uses a series of rules to iteratively
        strip suffixes from words.
        
        Example: "running" -> "run", "happiness" -> "happi"
        
        Note: Lemmatization is preferred as it produces valid words.
        
        Args:
            tokens: List of word tokens
            
        Returns:
            List of stemmed tokens
        """
        return [self.stemmer.stem(t) for t in tokens]
    
    def normalize(self, tokens: List[str]) -> List[str]:
        """
        Normalize tokens using configured method (lemmatization or stemming).
        
        Args:
            tokens: List of word tokens
            
        Returns:
            List of normalized tokens
        """
        if self.use_lemmatization:
            return self.lemmatize(tokens)
        else:
            return self.stem(tokens)
    
    def preprocess(self, text: str, full_pipeline: bool = True) -> Dict:
        """
        Run complete preprocessing pipeline on text.
        
        Args:
            text: Input text
            full_pipeline: If True, runs all preprocessing steps
            
        Returns:
            Dictionary with preprocessing results
        """
        result = {
            "original_text": text,
            "sentences": [],
            "tokens": [],
            "tokens_no_stopwords": [],
            "normalized_tokens": [],
            "processed_sentences": []
        }
        
        if not text:
            return result
        
        # Step 1: Sentence segmentation
        sentences = self.segment_sentences(text)
        result["sentences"] = sentences
        
        # Step 2-5: Process each sentence
        processed_sentences = []
        all_tokens = []
        all_tokens_no_sw = []
        all_normalized = []
        
        for sent in sentences:
            # Tokenize
            tokens = self.tokenize(sent)
            all_tokens.extend(tokens)
            
            # Remove stopwords
            tokens_no_sw = self.remove_stopwords(tokens)
            all_tokens_no_sw.extend(tokens_no_sw)
            
            # Normalize (lemmatize/stem)
            if full_pipeline:
                normalized = self.normalize(tokens_no_sw)
                all_normalized.extend(normalized)
            else:
                normalized = tokens_no_sw
            
            processed_sentences.append({
                "original": sent,
                "tokens": tokens,
                "filtered_tokens": tokens_no_sw,
                "normalized_tokens": normalized,
                "processed_text": ' '.join(normalized)
            })
        
        result["tokens"] = all_tokens
        result["tokens_no_stopwords"] = all_tokens_no_sw
        result["normalized_tokens"] = all_normalized
        result["processed_sentences"] = processed_sentences
        
        return result
    
    def preprocess_sentence(self, sentence: str) -> str:
        """
        Preprocess a single sentence and return cleaned text.
        
        Args:
            sentence: Input sentence
            
        Returns:
            Preprocessed sentence as string
        """
        tokens = self.tokenize(sentence)
        tokens = self.remove_stopwords(tokens)
        tokens = self.normalize(tokens)
        return ' '.join(tokens)
    
    def _clean_text(self, text: str) -> str:
        """Clean text by removing extra whitespace and special characters."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special unicode characters
        text = text.encode('ascii', 'ignore').decode('ascii')
        return text.strip()
    
    def _get_wordnet_pos(self, treebank_tag: str) -> Optional[str]:
        """Convert treebank POS tag to WordNet POS tag."""
        from nltk.corpus import wordnet
        
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            return None


# Convenience functions
def preprocess_text(text: str, use_lemmatization: bool = True) -> Dict:
    """Quick preprocessing of text."""
    preprocessor = TextPreprocessor(use_lemmatization=use_lemmatization)
    return preprocessor.preprocess(text)


def get_sentences(text: str) -> List[str]:
    """Quick sentence segmentation."""
    preprocessor = TextPreprocessor()
    return preprocessor.segment_sentences(text)


def get_clean_tokens(text: str, lemmatize: bool = True) -> List[str]:
    """Get cleaned and normalized tokens from text."""
    preprocessor = TextPreprocessor(use_lemmatization=lemmatize)
    tokens = preprocessor.tokenize(text)
    tokens = preprocessor.remove_stopwords(tokens)
    tokens = preprocessor.normalize(tokens)
    return tokens
