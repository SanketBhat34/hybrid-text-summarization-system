"""
Sentence Scoring and Heuristics Module
Implements various scoring algorithms for extractive summarization.

Algorithms Implemented:
1. Position-based Weighting - Scores based on sentence position
2. Sentence Length Filtering - Filters sentences by length
3. Keyword Boosting - Boosts scores for keyword-rich sentences
4. Redundancy Removal - Removes similar/duplicate sentences
5. Cosine Similarity - Measures sentence similarity
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from collections import Counter
import math
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity


class SentenceScorer:
    """
    Comprehensive sentence scoring using multiple heuristics.
    
    This scorer combines various algorithms to produce optimal
    sentence rankings for extractive summarization.
    """
    
    def __init__(
        self,
        position_weight: float = 0.15,
        length_weight: float = 0.10,
        keyword_weight: float = 0.20,
        tfidf_weight: float = 0.55,
        min_sentence_words: int = 5,
        max_sentence_words: int = 50,
        redundancy_threshold: float = 0.7
    ):
        """
        Initialize the scorer with configurable weights.
        
        Args:
            position_weight: Weight for position-based scoring (0-1)
            length_weight: Weight for length-based scoring (0-1)
            keyword_weight: Weight for keyword boosting (0-1)
            tfidf_weight: Weight for TF-IDF scoring (0-1)
            min_sentence_words: Minimum words for a valid sentence
            max_sentence_words: Maximum words for a valid sentence
            redundancy_threshold: Similarity threshold for redundancy (0-1)
        """
        self.position_weight = position_weight
        self.length_weight = length_weight
        self.keyword_weight = keyword_weight
        self.tfidf_weight = tfidf_weight
        self.min_sentence_words = min_sentence_words
        self.max_sentence_words = max_sentence_words
        self.redundancy_threshold = redundancy_threshold
        
        # Validate weights sum to ~1
        total = position_weight + length_weight + keyword_weight + tfidf_weight
        if abs(total - 1.0) > 0.01:
            print(f"Warning: Weights sum to {total}, not 1.0. Results may be skewed.")
    
    # =========================================
    # 1. POSITION-BASED WEIGHTING ALGORITHM
    # =========================================
    
    def calculate_position_scores(self, sentences: List[str]) -> List[float]:
        """
        Calculate position-based scores for sentences.
        
        Algorithm:
        - First 20% of sentences: High weight (1.0 - 0.8)
        - Last 10% of sentences: Medium-high weight (0.7)
        - Middle sentences: Lower weight with gradual decay
        
        Rationale: Important information typically appears at the
        beginning (introduction) and end (conclusion) of documents.
        
        Args:
            sentences: List of sentences
            
        Returns:
            List of position scores (0-1)
        """
        n = len(sentences)
        if n == 0:
            return []
        
        scores = []
        first_section = max(1, int(n * 0.2))  # First 20%
        last_section = max(1, int(n * 0.1))   # Last 10%
        
        for i, _ in enumerate(sentences):
            if i == 0:
                # First sentence is most important
                score = 1.0
            elif i < first_section:
                # First 20%: High importance, gradual decay
                score = 1.0 - (i / first_section) * 0.2  # 1.0 to 0.8
            elif i >= n - last_section:
                # Last 10%: Conclusion sentences
                score = 0.7
            else:
                # Middle sentences: Lower importance with decay
                middle_pos = (i - first_section) / max(1, (n - first_section - last_section))
                score = 0.6 - (middle_pos * 0.3)  # 0.6 to 0.3
            
            scores.append(max(0.1, score))  # Minimum score of 0.1
        
        return scores
    
    # =========================================
    # 2. SENTENCE LENGTH FILTERING ALGORITHM
    # =========================================
    
    def calculate_length_scores(self, sentences: List[str]) -> List[float]:
        """
        Calculate length-based scores for sentences.
        
        Algorithm:
        - Optimal length: 15-30 words (score = 1.0)
        - Too short (<5 words): Penalized heavily
        - Too long (>50 words): Penalized moderately
        - Uses bell curve scoring around optimal length
        
        Rationale: Very short sentences lack information,
        very long sentences are hard to read in summaries.
        
        Args:
            sentences: List of sentences
            
        Returns:
            List of length scores (0-1)
        """
        scores = []
        optimal_min = 15
        optimal_max = 30
        
        for sent in sentences:
            word_count = len(sent.split())
            
            if word_count < self.min_sentence_words:
                # Too short - heavily penalized
                score = 0.1
            elif word_count > self.max_sentence_words:
                # Too long - moderately penalized
                excess = word_count - self.max_sentence_words
                score = max(0.3, 1.0 - (excess * 0.02))
            elif optimal_min <= word_count <= optimal_max:
                # Optimal range
                score = 1.0
            elif word_count < optimal_min:
                # Slightly short
                score = 0.6 + (word_count - self.min_sentence_words) / (optimal_min - self.min_sentence_words) * 0.4
            else:
                # Slightly long (between optimal_max and max_sentence_words)
                score = 0.8 - (word_count - optimal_max) / (self.max_sentence_words - optimal_max) * 0.4
            
            scores.append(max(0.1, score))
        
        return scores
    
    def filter_by_length(self, sentences: List[str]) -> Tuple[List[str], List[int]]:
        """
        Filter sentences by length criteria.
        
        Args:
            sentences: List of sentences
            
        Returns:
            Tuple of (filtered sentences, original indices)
        """
        filtered = []
        indices = []
        
        for i, sent in enumerate(sentences):
            word_count = len(sent.split())
            if self.min_sentence_words <= word_count <= self.max_sentence_words * 1.5:
                filtered.append(sent)
                indices.append(i)
        
        return filtered, indices
    
    # =========================================
    # 3. KEYWORD BOOSTING ALGORITHM
    # =========================================
    
    def extract_keywords(self, sentences: List[str], top_n: int = 10) -> Set[str]:
        """
        Extract top keywords from the document using TF-IDF.
        
        Algorithm:
        1. Build TF-IDF matrix for all sentences
        2. Sum TF-IDF scores across all sentences for each term
        3. Select top N terms as keywords
        
        Args:
            sentences: List of sentences
            top_n: Number of keywords to extract
            
        Returns:
            Set of keyword strings
        """
        if not sentences or len(sentences) < 2:
            return set()
        
        try:
            vectorizer = TfidfVectorizer(
                stop_words='english',
                max_features=100,
                ngram_range=(1, 2)
            )
            tfidf_matrix = vectorizer.fit_transform(sentences)
            
            # Sum TF-IDF scores for each term
            term_scores = np.asarray(tfidf_matrix.sum(axis=0)).flatten()
            feature_names = vectorizer.get_feature_names_out()
            
            # Get top N terms
            top_indices = term_scores.argsort()[-top_n:][::-1]
            keywords = {feature_names[i] for i in top_indices}
            
            return keywords
        except Exception:
            return set()
    
    def calculate_keyword_scores(
        self, 
        sentences: List[str], 
        keywords: Optional[Set[str]] = None,
        custom_keywords: Optional[Set[str]] = None
    ) -> List[float]:
        """
        Calculate keyword-based scores for sentences.
        
        Algorithm:
        1. Extract document keywords if not provided
        2. For each sentence, count keyword occurrences
        3. Score = keyword_count / max_keyword_count (normalized)
        4. Boost for custom/domain-specific keywords
        
        Args:
            sentences: List of sentences
            keywords: Pre-extracted keywords (optional)
            custom_keywords: User-defined important keywords (optional)
            
        Returns:
            List of keyword scores (0-1)
        """
        if not sentences:
            return []
        
        # Extract keywords if not provided
        if keywords is None:
            keywords = self.extract_keywords(sentences)
        
        # Merge with custom keywords
        all_keywords = keywords.copy()
        if custom_keywords:
            all_keywords.update(custom_keywords)
        
        if not all_keywords:
            return [0.5] * len(sentences)  # Neutral score if no keywords
        
        scores = []
        max_count = 0
        keyword_counts = []
        
        # Count keywords in each sentence
        for sent in sentences:
            sent_lower = sent.lower()
            count = sum(1 for kw in all_keywords if kw.lower() in sent_lower)
            
            # Extra boost for custom keywords
            if custom_keywords:
                custom_count = sum(1 for kw in custom_keywords if kw.lower() in sent_lower)
                count += custom_count * 0.5  # 50% bonus for custom keywords
            
            keyword_counts.append(count)
            max_count = max(max_count, count)
        
        # Normalize scores
        for count in keyword_counts:
            if max_count > 0:
                score = 0.3 + (count / max_count) * 0.7  # Range: 0.3 to 1.0
            else:
                score = 0.5
            scores.append(score)
        
        return scores
    
    # =========================================
    # 4. COSINE SIMILARITY ALGORITHM
    # =========================================
    
    def calculate_cosine_similarity(self, sentences: List[str]) -> np.ndarray:
        """
        Calculate pairwise cosine similarity between all sentences.
        
        Algorithm:
        1. Convert sentences to TF-IDF vectors
        2. Calculate cosine similarity: cos(θ) = (A·B) / (||A|| × ||B||)
        3. Return similarity matrix
        
        Used for: Redundancy removal and TextRank
        
        Args:
            sentences: List of sentences
            
        Returns:
            N×N similarity matrix
        """
        if len(sentences) < 2:
            return np.array([[1.0]])
        
        try:
            vectorizer = TfidfVectorizer(stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(sentences)
            similarity_matrix = sklearn_cosine_similarity(tfidf_matrix, tfidf_matrix)
            return similarity_matrix
        except Exception:
            # Fallback: return identity matrix
            n = len(sentences)
            return np.eye(n)
    
    def get_sentence_similarity(self, sent1: str, sent2: str) -> float:
        """
        Calculate cosine similarity between two sentences.
        
        Args:
            sent1: First sentence
            sent2: Second sentence
            
        Returns:
            Similarity score (0-1)
        """
        try:
            vectorizer = TfidfVectorizer(stop_words='english')
            tfidf_matrix = vectorizer.fit_transform([sent1, sent2])
            similarity = sklearn_cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
            return float(similarity[0][0])
        except Exception:
            return 0.0
    
    # =========================================
    # 5. REDUNDANCY REMOVAL ALGORITHM
    # =========================================
    
    def remove_redundancy(
        self, 
        sentences: List[str], 
        scores: List[float],
        threshold: Optional[float] = None
    ) -> Tuple[List[str], List[float], List[int]]:
        """
        Remove redundant sentences based on similarity threshold.
        
        Algorithm:
        1. Sort sentences by score (descending)
        2. For each sentence (highest score first):
           - Check similarity with already selected sentences
           - If similarity > threshold with any selected, skip (redundant)
           - Otherwise, add to selected set
        3. Return non-redundant sentences in original order
        
        Rationale: Ensures summary diversity and information coverage.
        
        Args:
            sentences: List of sentences
            scores: Corresponding scores for each sentence
            threshold: Similarity threshold (uses instance default if None)
            
        Returns:
            Tuple of (filtered sentences, filtered scores, original indices)
        """
        if not sentences:
            return [], [], []
        
        if threshold is None:
            threshold = self.redundancy_threshold
        
        n = len(sentences)
        
        # Calculate similarity matrix
        similarity_matrix = self.calculate_cosine_similarity(sentences)
        
        # Create list of (score, index, sentence)
        scored = [(scores[i], i, sentences[i]) for i in range(n)]
        scored.sort(reverse=True, key=lambda x: x[0])
        
        # Select non-redundant sentences
        selected_indices = []
        
        for score, idx, sent in scored:
            is_redundant = False
            
            for selected_idx in selected_indices:
                similarity = similarity_matrix[idx][selected_idx]
                if similarity > threshold:
                    is_redundant = True
                    break
            
            if not is_redundant:
                selected_indices.append(idx)
        
        # Sort by original position to maintain flow
        selected_indices.sort()
        
        # Build output
        filtered_sentences = [sentences[i] for i in selected_indices]
        filtered_scores = [scores[i] for i in selected_indices]
        
        return filtered_sentences, filtered_scores, selected_indices
    
    # =========================================
    # COMBINED SCORING
    # =========================================
    
    def calculate_combined_scores(
        self,
        sentences: List[str],
        tfidf_scores: Optional[List[float]] = None,
        custom_keywords: Optional[Set[str]] = None
    ) -> List[float]:
        """
        Calculate combined scores using all algorithms.
        
        Formula:
        final_score = (position_weight × position_score +
                      length_weight × length_score +
                      keyword_weight × keyword_score +
                      tfidf_weight × tfidf_score)
        
        Args:
            sentences: List of sentences
            tfidf_scores: Pre-calculated TF-IDF scores (optional)
            custom_keywords: User-defined keywords for boosting (optional)
            
        Returns:
            List of combined scores
        """
        if not sentences:
            return []
        
        n = len(sentences)
        
        # Calculate individual scores
        position_scores = self.calculate_position_scores(sentences)
        length_scores = self.calculate_length_scores(sentences)
        keyword_scores = self.calculate_keyword_scores(sentences, custom_keywords=custom_keywords)
        
        # Use provided TF-IDF scores or calculate
        if tfidf_scores is None:
            # Simple TF-IDF scoring
            try:
                vectorizer = TfidfVectorizer(stop_words='english')
                tfidf_matrix = vectorizer.fit_transform(sentences)
                tfidf_scores = np.asarray(tfidf_matrix.sum(axis=1)).flatten()
                # Normalize
                max_score = max(tfidf_scores) if max(tfidf_scores) > 0 else 1
                tfidf_scores = [s / max_score for s in tfidf_scores]
            except Exception:
                tfidf_scores = [0.5] * n
        
        # Combine scores
        combined = []
        for i in range(n):
            score = (
                self.position_weight * position_scores[i] +
                self.length_weight * length_scores[i] +
                self.keyword_weight * keyword_scores[i] +
                self.tfidf_weight * tfidf_scores[i]
            )
            combined.append(score)
        
        return combined


class RedundancyRemover:
    """
    Standalone redundancy removal using threshold-based similarity filtering.
    """
    
    def __init__(self, threshold: float = 0.7):
        """
        Initialize redundancy remover.
        
        Args:
            threshold: Similarity threshold above which sentences are considered redundant
        """
        self.threshold = threshold
    
    def remove(
        self, 
        sentences: List[str], 
        scores: List[float]
    ) -> Tuple[List[str], List[float], List[int]]:
        """
        Remove redundant sentences.
        
        Args:
            sentences: List of sentences
            scores: Importance scores for sentences
            
        Returns:
            Tuple of (non-redundant sentences, scores, original indices)
        """
        scorer = SentenceScorer(redundancy_threshold=self.threshold)
        return scorer.remove_redundancy(sentences, scores)


# Convenience functions
def score_sentences(
    sentences: List[str],
    custom_keywords: Optional[Set[str]] = None
) -> List[float]:
    """Quick sentence scoring with all heuristics."""
    scorer = SentenceScorer()
    return scorer.calculate_combined_scores(sentences, custom_keywords=custom_keywords)


def remove_redundant_sentences(
    sentences: List[str],
    scores: List[float],
    threshold: float = 0.7
) -> Tuple[List[str], List[float]]:
    """Quick redundancy removal."""
    remover = RedundancyRemover(threshold=threshold)
    filtered_sents, filtered_scores, _ = remover.remove(sentences, scores)
    return filtered_sents, filtered_scores
