"""
Sentence-BERT (SBERT) Semantic Similarity Module

This module provides deep learning-based semantic sentence embeddings
for improved text summarization through better similarity computation.

Key Features:
1. Semantic sentence embeddings using pre-trained SBERT models
2. Cosine similarity based on meaning (not just word overlap)
3. Clustering with semantic embeddings
4. Maximal Marginal Relevance (MMR) for diverse sentence selection
"""

import os
# Set environment variable for Keras 3 compatibility BEFORE importing transformers
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import numpy as np
from typing import List, Dict, Optional, Tuple, Union
from nltk.tokenize import sent_tokenize
import warnings

# Lazy loading for sentence-transformers
_sbert_model = None
_model_name = None


def _load_sbert_model(model_name: str = 'all-MiniLM-L6-v2'):
    """
    Lazy load the SBERT model to avoid slow startup.
    
    Args:
        model_name: Name of the sentence-transformers model
                   Options: 'all-MiniLM-L6-v2' (fast), 
                           'all-mpnet-base-v2' (accurate),
                           'paraphrase-MiniLM-L6-v2' (paraphrase detection)
    """
    global _sbert_model, _model_name
    
    if _sbert_model is None or _model_name != model_name:
        try:
            from sentence_transformers import SentenceTransformer
            _sbert_model = SentenceTransformer(model_name)
            _model_name = model_name
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. "
                "Run: pip install sentence-transformers"
            )
    
    return _sbert_model


class SBERTSummarizer:
    """
    Sentence-BERT based extractive summarizer using semantic embeddings.
    
    This summarizer uses deep learning embeddings to understand sentence
    meaning, providing better quality summaries than TF-IDF based methods.
    
    Advantages over TF-IDF:
    - Understands synonyms and paraphrases
    - Captures semantic meaning, not just word frequency
    - Better redundancy detection
    - More coherent sentence selection
    """
    
    def __init__(
        self,
        model_name: str = 'all-MiniLM-L6-v2',
        use_mmr: bool = True,
        diversity: float = 0.3,
        similarity_threshold: float = 0.85
    ):
        """
        Initialize the SBERT summarizer.
        
        Args:
            model_name: SBERT model to use
                       - 'all-MiniLM-L6-v2': Fast, good quality (recommended)
                       - 'all-mpnet-base-v2': Slower, highest quality
                       - 'paraphrase-MiniLM-L6-v2': Good for detecting similar sentences
            use_mmr: Whether to use Maximal Marginal Relevance for diversity
            diversity: MMR diversity factor (0=similarity only, 1=diversity only)
            similarity_threshold: Threshold for redundancy removal
        """
        self.model_name = model_name
        self.use_mmr = use_mmr
        self.diversity = diversity
        self.similarity_threshold = similarity_threshold
        self._model = None
        self._embeddings_cache = {}
    
    def _get_model(self):
        """Lazy load the model."""
        if self._model is None:
            self._model = _load_sbert_model(self.model_name)
        return self._model
    
    def encode_sentences(self, sentences: List[str]) -> np.ndarray:
        """
        Encode sentences into dense vector embeddings.
        
        Args:
            sentences: List of sentences to encode
            
        Returns:
            NumPy array of shape (n_sentences, embedding_dim)
        """
        model = self._get_model()
        embeddings = model.encode(
            sentences, 
            convert_to_numpy=True,
            show_progress_bar=False
        )
        return embeddings
    
    def compute_similarity_matrix(self, sentences: List[str]) -> np.ndarray:
        """
        Compute pairwise semantic similarity between sentences.
        
        Args:
            sentences: List of sentences
            
        Returns:
            Similarity matrix of shape (n, n)
        """
        embeddings = self.encode_sentences(sentences)
        
        # Normalize embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized = embeddings / (norms + 1e-10)
        
        # Compute cosine similarity
        similarity_matrix = np.dot(normalized, normalized.T)
        
        return similarity_matrix
    
    def compute_document_similarity(self, sentences: List[str]) -> np.ndarray:
        """
        Compute similarity of each sentence to the document centroid.
        
        Args:
            sentences: List of sentences
            
        Returns:
            Array of similarity scores
        """
        embeddings = self.encode_sentences(sentences)
        
        # Compute document centroid
        centroid = np.mean(embeddings, axis=0)
        
        # Compute similarity to centroid
        centroid_norm = centroid / (np.linalg.norm(centroid) + 1e-10)
        embedding_norms = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10)
        
        similarities = np.dot(embedding_norms, centroid_norm)
        
        return similarities
    
    def mmr_selection(
        self,
        sentences: List[str],
        num_sentences: int,
        embeddings: Optional[np.ndarray] = None
    ) -> List[int]:
        """
        Select sentences using Maximal Marginal Relevance (MMR).
        
        MMR balances relevance to the document with diversity among
        selected sentences, avoiding redundancy.
        
        Args:
            sentences: List of sentences
            num_sentences: Number to select
            embeddings: Pre-computed embeddings (optional)
            
        Returns:
            Indices of selected sentences
        """
        if embeddings is None:
            embeddings = self.encode_sentences(sentences)
        
        # Compute document centroid (relevance target)
        doc_centroid = np.mean(embeddings, axis=0)
        
        # Normalize everything
        doc_norm = doc_centroid / (np.linalg.norm(doc_centroid) + 1e-10)
        emb_norms = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10)
        
        # Compute relevance scores (similarity to document)
        relevance = np.dot(emb_norms, doc_norm)
        
        # Compute pairwise similarities
        similarity_matrix = np.dot(emb_norms, emb_norms.T)
        
        # MMR selection
        selected = []
        candidates = list(range(len(sentences)))
        
        for _ in range(min(num_sentences, len(sentences))):
            if not candidates:
                break
            
            mmr_scores = []
            for idx in candidates:
                rel_score = relevance[idx]
                
                if selected:
                    # Max similarity to already selected
                    max_sim = max(similarity_matrix[idx][s] for s in selected)
                else:
                    max_sim = 0
                
                # MMR score: balance relevance and diversity
                mmr = (1 - self.diversity) * rel_score - self.diversity * max_sim
                mmr_scores.append((idx, mmr))
            
            # Select highest MMR score
            best_idx = max(mmr_scores, key=lambda x: x[1])[0]
            selected.append(best_idx)
            candidates.remove(best_idx)
        
        return selected
    
    def greedy_selection(
        self,
        sentences: List[str],
        num_sentences: int
    ) -> List[int]:
        """
        Simple greedy selection based on document similarity.
        Removes redundant sentences above similarity threshold.
        
        Args:
            sentences: List of sentences
            num_sentences: Number to select
            
        Returns:
            Indices of selected sentences
        """
        embeddings = self.encode_sentences(sentences)
        doc_similarities = self.compute_document_similarity(sentences)
        
        # Sort by document similarity
        ranked_indices = np.argsort(doc_similarities)[::-1]
        
        # Normalize embeddings for similarity computation
        emb_norms = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10)
        
        selected = []
        for idx in ranked_indices:
            if len(selected) >= num_sentences:
                break
            
            # Check redundancy
            is_redundant = False
            for sel_idx in selected:
                sim = np.dot(emb_norms[idx], emb_norms[sel_idx])
                if sim > self.similarity_threshold:
                    is_redundant = True
                    break
            
            if not is_redundant:
                selected.append(idx)
        
        return selected
    
    def summarize(
        self,
        text: str,
        num_sentences: int = 5,
        return_scores: bool = False
    ) -> Union[str, Tuple[str, Dict]]:
        """
        Generate extractive summary using SBERT embeddings.
        
        Args:
            text: Input text to summarize
            num_sentences: Number of sentences in summary
            return_scores: Whether to return scoring details
            
        Returns:
            Summary string, or tuple of (summary, details) if return_scores=True
        """
        sentences = sent_tokenize(text)
        
        if len(sentences) <= num_sentences:
            if return_scores:
                return text, {'method': 'sbert', 'selected_all': True}
            return text
        
        # Select sentences using MMR or greedy
        if self.use_mmr:
            selected_indices = self.mmr_selection(sentences, num_sentences)
        else:
            selected_indices = self.greedy_selection(sentences, num_sentences)
        
        # Sort by original order for coherence
        selected_indices.sort()
        
        summary_sentences = [sentences[i] for i in selected_indices]
        summary = ' '.join(summary_sentences)
        
        if return_scores:
            doc_sims = self.compute_document_similarity(sentences)
            
            return summary, {
                'method': 'sbert',
                'model': self.model_name,
                'selected_indices': selected_indices,
                'document_similarities': {i: float(doc_sims[i]) for i in selected_indices},
                'use_mmr': self.use_mmr,
                'diversity': self.diversity
            }
        
        return summary
    
    def find_similar_sentences(
        self,
        query: str,
        sentences: List[str],
        top_k: int = 5
    ) -> List[Tuple[int, str, float]]:
        """
        Find sentences most similar to a query.
        
        Args:
            query: Query sentence
            sentences: List of sentences to search
            top_k: Number of results to return
            
        Returns:
            List of (index, sentence, similarity_score) tuples
        """
        model = self._get_model()
        
        # Encode query and sentences
        query_embedding = model.encode([query], convert_to_numpy=True)[0]
        sentence_embeddings = self.encode_sentences(sentences)
        
        # Normalize
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)
        sent_norms = sentence_embeddings / (np.linalg.norm(sentence_embeddings, axis=1, keepdims=True) + 1e-10)
        
        # Compute similarities
        similarities = np.dot(sent_norms, query_norm)
        
        # Get top-k
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = [(int(i), sentences[i], float(similarities[i])) for i in top_indices]
        return results


class SemanticRedundancyRemover:
    """
    Remove redundant sentences using SBERT semantic similarity.
    
    More accurate than TF-IDF based redundancy removal because it
    understands paraphrases and semantically similar sentences.
    """
    
    def __init__(
        self,
        model_name: str = 'all-MiniLM-L6-v2',
        threshold: float = 0.85
    ):
        """
        Initialize the redundancy remover.
        
        Args:
            model_name: SBERT model name
            threshold: Similarity threshold for considering sentences redundant
        """
        self.model_name = model_name
        self.threshold = threshold
        self._model = None
    
    def _get_model(self):
        if self._model is None:
            self._model = _load_sbert_model(self.model_name)
        return self._model
    
    def remove_redundancy(
        self,
        sentences: List[str],
        scores: Optional[List[float]] = None
    ) -> Tuple[List[str], List[int]]:
        """
        Remove redundant sentences, keeping highest scored ones.
        
        Args:
            sentences: List of sentences
            scores: Optional relevance scores (keeps highest scored when redundant)
            
        Returns:
            Tuple of (filtered_sentences, kept_indices)
        """
        if len(sentences) <= 1:
            return sentences, list(range(len(sentences)))
        
        model = self._get_model()
        embeddings = model.encode(sentences, convert_to_numpy=True, show_progress_bar=False)
        
        # Normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized = embeddings / (norms + 1e-10)
        
        # Compute similarity matrix
        sim_matrix = np.dot(normalized, normalized.T)
        
        # If scores provided, process in score order
        if scores is not None:
            order = np.argsort(scores)[::-1]
        else:
            order = list(range(len(sentences)))
        
        kept_indices = []
        for idx in order:
            is_redundant = False
            for kept_idx in kept_indices:
                if sim_matrix[idx][kept_idx] > self.threshold:
                    is_redundant = True
                    break
            
            if not is_redundant:
                kept_indices.append(idx)
        
        # Sort by original order
        kept_indices.sort()
        filtered = [sentences[i] for i in kept_indices]
        
        return filtered, kept_indices


def get_semantic_similarity(text1: str, text2: str, model_name: str = 'all-MiniLM-L6-v2') -> float:
    """
    Compute semantic similarity between two texts.
    
    Args:
        text1: First text
        text2: Second text
        model_name: SBERT model name
        
    Returns:
        Similarity score between 0 and 1
    """
    model = _load_sbert_model(model_name)
    
    embeddings = model.encode([text1, text2], convert_to_numpy=True)
    
    # Normalize and compute similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / (norms + 1e-10)
    
    similarity = float(np.dot(normalized[0], normalized[1]))
    
    return similarity
