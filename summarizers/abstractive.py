"""
Abstractive Summarization Module
Uses transformer models (BART/T5) to generate new summary text.
"""

import os
# Set environment variable for Keras 3 compatibility BEFORE importing transformers
os.environ['TF_USE_LEGACY_KERAS'] = '1'

from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from typing import Optional


class AbstractiveSummarizer:
    """
    Abstractive summarizer using transformer models.
    Generates new summary text that may contain words not in the original.
    """
    
    # Available models
    MODELS = {
        "bart": "facebook/bart-large-cnn",
        "t5": "t5-small",
        "t5-base": "t5-base",
        "pegasus": "google/pegasus-xsum",
        "distilbart": "sshleifer/distilbart-cnn-12-6"  # Faster, smaller model
    }
    
    def __init__(self, model_name: str = "distilbart"):
        """
        Initialize the summarizer with specified model.
        
        Args:
            model_name: Name of the model to use (bart, t5, t5-base, pegasus, distilbart)
        """
        self.model_name = model_name
        self.model_path = self.MODELS.get(model_name, self.MODELS["distilbart"])
        self.summarizer = None
        self.tokenizer = None
        self.model = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def _load_model(self):
        """Lazy load the model when first needed."""
        if self.summarizer is None:
            print(f"Loading model: {self.model_path}...")
            try:
                self.summarizer = pipeline(
                    "summarization",
                    model=self.model_path,
                    device=0 if self._device == "cuda" else -1,
                    framework="pt"
                )
                print("Model loaded successfully!")
            except Exception as e:
                print(f"Error loading model: {e}")
                # Fallback to a smaller model
                print("Falling back to t5-small model...")
                self.summarizer = pipeline(
                    "summarization",
                    model="t5-small",
                    device=0 if self._device == "cuda" else -1,
                    framework="pt"
                )
    
    def _chunk_text(self, text: str, max_length: int = 1024) -> list:
        """
        Split text into chunks that fit within model's max input length.
        
        Args:
            text: Input text
            max_length: Maximum token length per chunk
            
        Returns:
            List of text chunks
        """
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0
        
        # Approximate tokens as words (rough estimate)
        for word in words:
            current_chunk.append(word)
            current_length += 1
            
            if current_length >= max_length - 100:  # Leave buffer
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_length = 0
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks if chunks else [text]
    
    def summarize(
        self, 
        text: str, 
        max_length: int = 150,
        min_length: int = 30,
        do_sample: bool = False,
        num_beams: int = 4
    ) -> dict:
        """
        Generate abstractive summary of the text.
        
        Args:
            text: Input text to summarize
            max_length: Maximum length of summary in tokens
            min_length: Minimum length of summary in tokens
            do_sample: Whether to use sampling (adds randomness)
            num_beams: Number of beams for beam search
            
        Returns:
            Dictionary containing summary and metadata
        """
        if not text or not text.strip():
            return {
                "summary": "",
                "model_used": self.model_name,
                "original_length": 0,
                "summary_length": 0
            }
        
        # Load model if not already loaded
        self._load_model()
        
        original_length = len(text.split())
        
        # Handle long texts by chunking
        chunks = self._chunk_text(text)
        
        summaries = []
        for chunk in chunks:
            if len(chunk.split()) < 20:
                # Skip very short chunks
                summaries.append(chunk)
                continue
            
            try:
                # Adjust lengths based on input
                chunk_words = len(chunk.split())
                adj_max = min(max_length, max(50, chunk_words // 2))
                adj_min = min(min_length, adj_max - 10)
                
                # For T5 models, add prefix
                if "t5" in self.model_path.lower():
                    chunk = "summarize: " + chunk
                
                result = self.summarizer(
                    chunk,
                    max_length=adj_max,
                    min_length=adj_min,
                    do_sample=do_sample,
                    num_beams=num_beams,
                    early_stopping=True
                )
                summaries.append(result[0]['summary_text'])
            except Exception as e:
                print(f"Error summarizing chunk: {e}")
                summaries.append(chunk[:500] + "...")
        
        # Combine chunk summaries
        final_summary = ' '.join(summaries)
        
        # If combined summary is still long, summarize again
        if len(final_summary.split()) > max_length * 1.5 and len(chunks) > 1:
            try:
                if "t5" in self.model_path.lower():
                    final_summary = "summarize: " + final_summary
                    
                result = self.summarizer(
                    final_summary,
                    max_length=max_length,
                    min_length=min_length,
                    do_sample=do_sample,
                    num_beams=num_beams
                )
                final_summary = result[0]['summary_text']
            except:
                pass
        
        return {
            "summary": final_summary,
            "model_used": self.model_name,
            "original_length": original_length,
            "summary_length": len(final_summary.split()),
            "compression_ratio": len(final_summary.split()) / original_length if original_length > 0 else 0
        }


class HybridSummarizer:
    """
    Combines extractive and abstractive summarization for better results.
    """
    
    def __init__(self, abstractive_model: str = "distilbart"):
        """
        Initialize hybrid summarizer.
        
        Args:
            abstractive_model: Model to use for abstractive part
        """
        from .extractive import ExtractiveSummarizer
        self.extractive = ExtractiveSummarizer()
        self.abstractive = AbstractiveSummarizer(model_name=abstractive_model)
    
    def summarize(
        self, 
        text: str, 
        extractive_ratio: float = 0.5,
        final_max_length: int = 150
    ) -> dict:
        """
        Generate hybrid summary: first extract key sentences, then paraphrase.
        
        Args:
            text: Input text
            extractive_ratio: Ratio of sentences to extract first
            final_max_length: Maximum length of final summary
            
        Returns:
            Dictionary with summary and metadata
        """
        # Step 1: Extract key sentences
        extractive_result = self.extractive.summarize(text, ratio=extractive_ratio)
        extracted_text = extractive_result["summary"]
        
        # Step 2: Generate abstractive summary from extracted sentences
        abstractive_result = self.abstractive.summarize(
            extracted_text, 
            max_length=final_max_length
        )
        
        return {
            "summary": abstractive_result["summary"],
            "method": "hybrid",
            "extractive_intermediate": extracted_text,
            "original_sentences": extractive_result["original_sentences"],
            "compression_ratio": abstractive_result.get("compression_ratio", 0)
        }


# Convenience function
def summarize_text(
    text: str, 
    model: str = "distilbart",
    max_length: int = 150
) -> str:
    """
    Quick function to summarize text using abstractive method.
    
    Args:
        text: Input text to summarize
        model: Model to use
        max_length: Maximum summary length
        
    Returns:
        Summary string
    """
    summarizer = AbstractiveSummarizer(model_name=model)
    result = summarizer.summarize(text, max_length=max_length)
    return result["summary"]
