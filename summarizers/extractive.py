"""
Extractive Summarization Module
Comprehensive implementation of text summarization algorithms.

Algorithms Implemented:
1. Text Preprocessing (Tokenization, Stop-word Removal, Lemmatization)
2. TF-IDF (Term Frequency-Inverse Document Frequency)
3. Cosine Similarity (for sentence comparison)
4. TextRank (Graph-based ranking with PageRank)
5. Position-based Weighting (sentence position heuristics)
6. Sentence Length Filtering (quality filtering)
7. Keyword Boosting (domain relevance)
8. Redundancy Removal (duplicate filtering)
9. K-Means Clustering (topic coverage and diversity)
"""

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.cluster.util import cosine_distance
from nltk.stem import WordNetLemmatizer
import numpy as np
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from collections import Counter
from typing import List, Dict, Set, Optional, Tuple
import math


class TFIDFSummarizer:
    """
    TF-IDF based extractive summarizer.
    Scores sentences based on the TF-IDF weights of their constituent words.
    
    TF-IDF Formula:
    - TF (Term Frequency) = (Number of times term t appears in document) / (Total terms in document)
    - IDF (Inverse Document Frequency) = log(Total documents / Documents containing term t)
    - TF-IDF = TF × IDF
    """
    
    def __init__(self):
        """Initialize the TF-IDF summarizer."""
        self._download_nltk_data()
        self.stop_words = set(stopwords.words('english'))
        self.vectorizer = None
    
    def _download_nltk_data(self):
        """Download required NLTK data packages."""
        required_packages = ['punkt', 'stopwords', 'punkt_tab']
        for package in required_packages:
            try:
                nltk.data.find(f'tokenizers/{package}')
            except LookupError:
                try:
                    nltk.download(package, quiet=True)
                except Exception:
                    pass
    
    def _preprocess_sentence(self, sentence: str) -> str:
        """
        Preprocess a sentence by removing stopwords and non-alphanumeric characters.
        
        Args:
            sentence: Input sentence
            
        Returns:
            Preprocessed sentence
        """
        words = word_tokenize(sentence.lower())
        filtered_words = [word for word in words if word.isalnum() and word not in self.stop_words]
        return ' '.join(filtered_words)
    
    def _calculate_tfidf_scores(self, sentences: list) -> np.ndarray:
        """
        Calculate TF-IDF matrix for sentences.
        
        Args:
            sentences: List of sentences
            
        Returns:
            TF-IDF matrix
        """
        # Preprocess sentences
        processed_sentences = [self._preprocess_sentence(sent) for sent in sentences]
        
        # Filter out empty sentences
        valid_indices = [i for i, sent in enumerate(processed_sentences) if sent.strip()]
        
        if len(valid_indices) < 2:
            return np.zeros(len(sentences))
        
        # Create TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=5000,
            ngram_range=(1, 2)  # Include bigrams for better context
        )
        
        # Fit and transform
        try:
            tfidf_matrix = self.vectorizer.fit_transform(processed_sentences)
        except ValueError:
            # Handle case where all sentences are empty after preprocessing
            return np.zeros(len(sentences))
        
        return tfidf_matrix
    
    def _score_sentences_by_tfidf(self, sentences: list) -> list:
        """
        Score each sentence based on average TF-IDF of its words.
        
        Args:
            sentences: List of sentences
            
        Returns:
            List of (score, index, sentence) tuples
        """
        tfidf_matrix = self._calculate_tfidf_scores(sentences)
        
        if isinstance(tfidf_matrix, np.ndarray) and tfidf_matrix.sum() == 0:
            # Fallback: score by sentence length (simple heuristic)
            return [(len(sent.split()), i, sent) for i, sent in enumerate(sentences)]
        
        # Calculate sentence scores as sum of TF-IDF values
        sentence_scores = np.asarray(tfidf_matrix.sum(axis=1)).flatten()
        
        # Normalize by sentence length to avoid bias towards longer sentences
        for i, sent in enumerate(sentences):
            word_count = len(sent.split())
            if word_count > 0:
                sentence_scores[i] = sentence_scores[i] / math.sqrt(word_count)
        
        return [(sentence_scores[i], i, sent) for i, sent in enumerate(sentences)]
    
    def summarize(self, text: str, num_sentences: int = 3, ratio: float = None) -> dict:
        """
        Generate extractive summary using TF-IDF scoring.
        
        Args:
            text: Input text to summarize
            num_sentences: Number of sentences in summary (default: 3)
            ratio: Alternative way to specify summary length as ratio of original
            
        Returns:
            Dictionary containing summary and metadata
        """
        if not text or not text.strip():
            return {
                "summary": "",
                "original_sentences": 0,
                "summary_sentences": 0,
                "compression_ratio": 0,
                "method": "tfidf"
            }
        
        sentences = sent_tokenize(text)
        
        if len(sentences) <= 2:
            return {
                "summary": text,
                "original_sentences": len(sentences),
                "summary_sentences": len(sentences),
                "compression_ratio": 1.0,
                "method": "tfidf"
            }
        
        # Calculate number of sentences based on ratio if provided
        if ratio is not None:
            num_sentences = max(1, int(len(sentences) * ratio))
        
        # Ensure we don't request more sentences than available
        num_sentences = min(num_sentences, len(sentences))
        
        # Score sentences using TF-IDF
        scored_sentences = self._score_sentences_by_tfidf(sentences)
        
        # Rank sentences by score
        ranked_sentences = sorted(scored_sentences, key=lambda x: x[0], reverse=True)
        
        # Select top sentences and sort by original position to maintain flow
        selected = sorted(ranked_sentences[:num_sentences], key=lambda x: x[1])
        summary = ' '.join([s for _, _, s in selected])
        
        # Create score dictionary
        sentence_scores = {sent: score for score, _, sent in scored_sentences}
        
        return {
            "summary": summary,
            "original_sentences": len(sentences),
            "summary_sentences": num_sentences,
            "compression_ratio": num_sentences / len(sentences),
            "method": "tfidf",
            "sentence_scores": sentence_scores,
            "top_terms": self._get_top_terms(5) if self.vectorizer else []
        }
    
    def _get_top_terms(self, n: int = 5) -> list:
        """Get top N terms by TF-IDF weight."""
        if self.vectorizer is None:
            return []
        
        feature_names = self.vectorizer.get_feature_names_out()
        return list(feature_names[:n])


class KMeansClusterSummarizer:
    """
    K-Means Clustering-based Summarizer for topic coverage and diversity.
    
    This algorithm:
    1. Converts sentences to TF-IDF vectors
    2. Clusters sentences using K-Means
    3. Selects representative sentences from each cluster
    4. Ensures diverse topic coverage in the summary
    
    Benefits:
    - Ensures all major topics are covered
    - Avoids redundancy by selecting from different clusters
    - Provides a balanced summary across document themes
    """
    
    def __init__(
        self,
        n_clusters: Optional[int] = None,
        cluster_ratio: float = 0.3,
        selection_method: str = 'centroid',
        random_state: int = 42
    ):
        """
        Initialize the K-Means clustering summarizer.
        
        Args:
            n_clusters: Number of clusters (auto-calculated if None)
            cluster_ratio: Ratio to determine clusters from sentence count
            selection_method: 'centroid' (closest to center) or 'highest_tfidf'
            random_state: Random seed for reproducibility
        """
        self.n_clusters = n_clusters
        self.cluster_ratio = cluster_ratio
        self.selection_method = selection_method
        self.random_state = random_state
        self.vectorizer = TfidfVectorizer(stop_words='english')
        
    def _calculate_optimal_clusters(self, n_sentences: int) -> int:
        """Calculate optimal number of clusters based on sentence count."""
        if self.n_clusters is not None:
            return min(self.n_clusters, n_sentences)
        
        # Use cluster_ratio to determine number of clusters
        calculated = max(2, int(n_sentences * self.cluster_ratio))
        return min(calculated, n_sentences)
    
    def _select_from_cluster(
        self,
        cluster_sentences: List[Tuple[int, str]],
        cluster_vectors: np.ndarray,
        centroid: np.ndarray
    ) -> int:
        """Select the best sentence from a cluster."""
        if len(cluster_sentences) == 0:
            return -1
            
        if self.selection_method == 'centroid':
            # Select sentence closest to cluster centroid
            distances = np.linalg.norm(cluster_vectors - centroid, axis=1)
            best_idx = np.argmin(distances)
        else:
            # Select sentence with highest TF-IDF sum
            tfidf_sums = np.sum(cluster_vectors, axis=1)
            best_idx = np.argmax(tfidf_sums)
            
        return cluster_sentences[best_idx][0]
    
    def summarize(
        self,
        text: str,
        num_sentences: int = 5,
        return_details: bool = False
    ):
        """
        Summarize text using K-Means clustering.
        
        Args:
            text: Input text to summarize
            num_sentences: Number of sentences in summary
            return_details: If True, return clustering details
            
        Returns:
            str: Summarized text if return_details is False
            Tuple[str, Dict]: Summary and details if return_details is True
        """
        # Tokenize into sentences
        sentences = sent_tokenize(text)
        
        if len(sentences) <= num_sentences:
            if return_details:
                return text, {'clusters': 1, 'sentences_per_cluster': len(sentences)}
            return text
        
        # Create TF-IDF matrix
        try:
            tfidf_matrix = self.vectorizer.fit_transform(sentences)
        except ValueError:
            # Handle empty vocabulary
            if return_details:
                return ' '.join(sentences[:num_sentences]), {'error': 'Empty vocabulary'}
            return ' '.join(sentences[:num_sentences])
        
        # Determine number of clusters
        n_clusters = self._calculate_optimal_clusters(len(sentences))
        n_clusters = min(n_clusters, num_sentences * 2)  # Don't over-cluster
        
        # Perform K-Means clustering
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=self.random_state,
            n_init=10
        )
        cluster_labels = kmeans.fit_predict(tfidf_matrix.toarray())
        
        # Group sentences by cluster
        clusters: Dict[int, List[Tuple[int, str]]] = {i: [] for i in range(n_clusters)}
        for idx, (sentence, label) in enumerate(zip(sentences, cluster_labels)):
            clusters[label].append((idx, sentence))
        
        # Select representatives from each cluster
        selected_indices = []
        tfidf_array = tfidf_matrix.toarray()
        
        # Sort clusters by size (larger clusters get priority)
        sorted_clusters = sorted(clusters.items(), key=lambda x: len(x[1]), reverse=True)
        
        for cluster_id, cluster_sentences in sorted_clusters:
            if len(selected_indices) >= num_sentences:
                break
                
            if len(cluster_sentences) == 0:
                continue
                
            # Get vectors for this cluster
            cluster_indices = [s[0] for s in cluster_sentences]
            cluster_vectors = tfidf_array[cluster_indices]
            centroid = kmeans.cluster_centers_[cluster_id]
            
            # Select best sentence from cluster
            selected_idx = self._select_from_cluster(
                cluster_sentences, cluster_vectors, centroid
            )
            
            if selected_idx not in selected_indices:
                selected_indices.append(selected_idx)
        
        # If we need more sentences, add from largest clusters
        while len(selected_indices) < num_sentences:
            for cluster_id, cluster_sentences in sorted_clusters:
                for idx, sentence in cluster_sentences:
                    if idx not in selected_indices:
                        selected_indices.append(idx)
                        break
                if len(selected_indices) >= num_sentences:
                    break
        
        # Sort by original position for coherent output
        selected_indices.sort()
        summary_sentences = [sentences[i] for i in selected_indices[:num_sentences]]
        
        if return_details:
            return ' '.join(summary_sentences), {
                'n_clusters': n_clusters,
                'cluster_sizes': {k: len(v) for k, v in clusters.items()},
                'selected_indices': selected_indices
            }
        
        return ' '.join(summary_sentences)
    
    def get_cluster_info(self, text: str) -> Dict:
        """
        Get clustering information for analysis.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with cluster information
        """
        sentences = sent_tokenize(text)
        
        if len(sentences) < 2:
            return {'error': 'Not enough sentences for clustering'}
        
        tfidf_matrix = self.vectorizer.fit_transform(sentences)
        n_clusters = self._calculate_optimal_clusters(len(sentences))
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)
        labels = kmeans.fit_predict(tfidf_matrix.toarray())
        
        # Get top terms per cluster
        feature_names = self.vectorizer.get_feature_names_out()
        cluster_info = {}
        
        for i in range(n_clusters):
            centroid = kmeans.cluster_centers_[i]
            top_indices = centroid.argsort()[-5:][::-1]
            top_terms = [feature_names[idx] for idx in top_indices]
            
            cluster_sentences = [sentences[j] for j, label in enumerate(labels) if label == i]
            
            cluster_info[f'cluster_{i}'] = {
                'size': len(cluster_sentences),
                'top_terms': top_terms,
                'sample_sentence': cluster_sentences[0] if cluster_sentences else None
            }
        
        return cluster_info


class AdvancedSummarizer:
    """
    Advanced Extractive Summarizer using ALL algorithms.
    
    Combines multiple techniques for optimal summarization:
    1. Preprocessing with lemmatization
    2. TF-IDF for term importance
    3. Position-based weighting
    4. Sentence length filtering
    5. Keyword boosting
    6. Redundancy removal with cosine similarity
    7. TextRank for graph-based ranking
    8. K-Means Clustering for topic diversity (optional)
    
    This is the recommended summarizer for best results.
    """
    
    def __init__(
        self,
        use_lemmatization: bool = True,
        position_weight: float = 0.15,
        length_weight: float = 0.10,
        keyword_weight: float = 0.15,
        tfidf_weight: float = 0.30,
        textrank_weight: float = 0.30,
        min_sentence_words: int = 5,
        max_sentence_words: int = 50,
        redundancy_threshold: float = 0.7,
        custom_keywords: Optional[Set[str]] = None,
        use_clustering: bool = False,
        cluster_weight: float = 0.15
    ):
        """
        Initialize the advanced summarizer.
        
        Args:
            use_lemmatization: Whether to use lemmatization in preprocessing
            position_weight: Weight for position-based scoring
            length_weight: Weight for sentence length scoring
            keyword_weight: Weight for keyword boosting
            tfidf_weight: Weight for TF-IDF scoring
            textrank_weight: Weight for TextRank scoring
            min_sentence_words: Minimum words for valid sentence
            max_sentence_words: Maximum words for valid sentence
            redundancy_threshold: Similarity threshold for redundancy removal
            custom_keywords: User-defined important keywords
            use_clustering: Whether to use K-Means clustering for diversity
            cluster_weight: Weight for cluster diversity scoring
        """
        self._download_nltk_data()
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.use_lemmatization = use_lemmatization
        
        # Scoring weights
        self.position_weight = position_weight
        self.length_weight = length_weight
        self.keyword_weight = keyword_weight
        self.tfidf_weight = tfidf_weight
        self.textrank_weight = textrank_weight
        self.cluster_weight = cluster_weight
        
        # Filtering parameters
        self.min_sentence_words = min_sentence_words
        self.max_sentence_words = max_sentence_words
        self.redundancy_threshold = redundancy_threshold
        
        # Custom keywords for boosting
        self.custom_keywords = custom_keywords or set()
        
        # Clustering option
        self.use_clustering = use_clustering
    
    def _download_nltk_data(self):
        """Download required NLTK data."""
        packages = ['punkt', 'stopwords', 'punkt_tab', 'wordnet', 
                   'averaged_perceptron_tagger', 'averaged_perceptron_tagger_eng']
        for package in packages:
            try:
                if 'taggers' in package or 'averaged' in package:
                    nltk.data.find(f'taggers/{package}')
                elif package == 'wordnet':
                    nltk.data.find(f'corpora/{package}')
                else:
                    nltk.data.find(f'tokenizers/{package}')
            except LookupError:
                try:
                    nltk.download(package, quiet=True)
                except Exception:
                    pass
    
    # =========================================
    # PREPROCESSING ALGORITHMS
    # =========================================
    
    def _preprocess_text(self, text: str) -> List[str]:
        """Segment text into sentences."""
        return sent_tokenize(text)
    
    def _tokenize(self, sentence: str) -> List[str]:
        """Tokenize sentence into words."""
        return [w.lower() for w in word_tokenize(sentence) if w.isalnum()]
    
    def _remove_stopwords(self, tokens: List[str]) -> List[str]:
        """Remove stop words from tokens."""
        return [t for t in tokens if t not in self.stop_words]
    
    def _lemmatize(self, tokens: List[str]) -> List[str]:
        """Lemmatize tokens to base form."""
        if not self.use_lemmatization:
            return tokens
        return [self.lemmatizer.lemmatize(t) for t in tokens]
    
    def _preprocess_sentence(self, sentence: str) -> str:
        """Full preprocessing pipeline for a sentence."""
        tokens = self._tokenize(sentence)
        tokens = self._remove_stopwords(tokens)
        tokens = self._lemmatize(tokens)
        return ' '.join(tokens)
    
    # =========================================
    # POSITION-BASED WEIGHTING
    # =========================================
    
    def _calculate_position_scores(self, sentences: List[str]) -> List[float]:
        """
        Calculate position-based scores.
        First and last sentences get higher scores.
        """
        n = len(sentences)
        if n == 0:
            return []
        
        scores = []
        first_section = max(1, int(n * 0.2))
        last_section = max(1, int(n * 0.1))
        
        for i in range(n):
            if i == 0:
                score = 1.0
            elif i < first_section:
                score = 1.0 - (i / first_section) * 0.2
            elif i >= n - last_section:
                score = 0.7
            else:
                middle_pos = (i - first_section) / max(1, (n - first_section - last_section))
                score = 0.6 - (middle_pos * 0.3)
            scores.append(max(0.1, score))
        
        return scores
    
    # =========================================
    # SENTENCE LENGTH FILTERING
    # =========================================
    
    def _calculate_length_scores(self, sentences: List[str]) -> List[float]:
        """
        Calculate length-based scores.
        Optimal length: 15-30 words.
        """
        scores = []
        optimal_min, optimal_max = 15, 30
        
        for sent in sentences:
            word_count = len(sent.split())
            
            if word_count < self.min_sentence_words:
                score = 0.1
            elif word_count > self.max_sentence_words:
                excess = word_count - self.max_sentence_words
                score = max(0.3, 1.0 - (excess * 0.02))
            elif optimal_min <= word_count <= optimal_max:
                score = 1.0
            elif word_count < optimal_min:
                score = 0.6 + (word_count - self.min_sentence_words) / (optimal_min - self.min_sentence_words) * 0.4
            else:
                score = 0.8 - (word_count - optimal_max) / (self.max_sentence_words - optimal_max) * 0.4
            
            scores.append(max(0.1, score))
        
        return scores
    
    def _filter_by_length(self, sentences: List[str]) -> Tuple[List[str], List[int]]:
        """Filter sentences by length criteria."""
        filtered, indices = [], []
        for i, sent in enumerate(sentences):
            word_count = len(sent.split())
            if self.min_sentence_words <= word_count <= self.max_sentence_words * 1.5:
                filtered.append(sent)
                indices.append(i)
        return filtered, indices
    
    # =========================================
    # KEYWORD BOOSTING
    # =========================================
    
    def _extract_keywords(self, sentences: List[str], top_n: int = 10) -> Set[str]:
        """Extract top keywords using TF-IDF."""
        if len(sentences) < 2:
            return set()
        
        try:
            vectorizer = TfidfVectorizer(stop_words='english', max_features=100, ngram_range=(1, 2))
            tfidf_matrix = vectorizer.fit_transform(sentences)
            term_scores = np.asarray(tfidf_matrix.sum(axis=0)).flatten()
            feature_names = vectorizer.get_feature_names_out()
            top_indices = term_scores.argsort()[-top_n:][::-1]
            return {feature_names[i] for i in top_indices}
        except Exception:
            return set()
    
    def _calculate_keyword_scores(self, sentences: List[str], keywords: Set[str]) -> List[float]:
        """Calculate keyword-based scores."""
        if not keywords:
            return [0.5] * len(sentences)
        
        all_keywords = keywords | self.custom_keywords
        scores = []
        max_count = 0
        counts = []
        
        for sent in sentences:
            sent_lower = sent.lower()
            count = sum(1 for kw in all_keywords if kw.lower() in sent_lower)
            if self.custom_keywords:
                custom_count = sum(1 for kw in self.custom_keywords if kw.lower() in sent_lower)
                count += custom_count * 0.5
            counts.append(count)
            max_count = max(max_count, count)
        
        for count in counts:
            score = 0.3 + (count / max_count) * 0.7 if max_count > 0 else 0.5
            scores.append(score)
        
        return scores
    
    # =========================================
    # TF-IDF SCORING
    # =========================================
    
    def _calculate_tfidf_scores(self, sentences: List[str]) -> List[float]:
        """Calculate TF-IDF based scores."""
        if len(sentences) < 2:
            return [1.0] * len(sentences)
        
        try:
            processed = [self._preprocess_sentence(s) for s in sentences]
            vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
            tfidf_matrix = vectorizer.fit_transform(processed)
            scores = np.asarray(tfidf_matrix.sum(axis=1)).flatten()
            
            # Normalize by sentence length
            for i, sent in enumerate(sentences):
                word_count = len(sent.split())
                if word_count > 0:
                    scores[i] = scores[i] / math.sqrt(word_count)
            
            # Normalize to 0-1 range
            max_score = max(scores) if max(scores) > 0 else 1
            return [s / max_score for s in scores]
        except Exception:
            return [0.5] * len(sentences)
    
    # =========================================
    # TEXTRANK ALGORITHM
    # =========================================
    
    def _calculate_textrank_scores(self, sentences: List[str]) -> List[float]:
        """Calculate TextRank scores using PageRank."""
        n = len(sentences)
        if n < 2:
            return [1.0] * n
        
        # Build similarity matrix using cosine similarity
        try:
            processed = [self._preprocess_sentence(s) for s in sentences]
            vectorizer = TfidfVectorizer(stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(processed)
            similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
            
            # Zero out diagonal
            np.fill_diagonal(similarity_matrix, 0)
            
            # Apply PageRank
            nx_graph = nx.from_numpy_array(similarity_matrix)
            scores = nx.pagerank(nx_graph, max_iter=100)
            
            # Normalize to 0-1 range
            max_score = max(scores.values()) if scores else 1
            return [scores.get(i, 0) / max_score for i in range(n)]
        except Exception:
            return [0.5] * n
    
    # =========================================
    # K-MEANS CLUSTERING SCORES
    # =========================================
    
    def _calculate_cluster_scores(self, sentences: List[str]) -> List[float]:
        """
        Calculate cluster-based diversity scores using K-Means.
        
        Sentences closer to cluster centroids get higher scores,
        promoting topic diversity in the final summary.
        """
        n = len(sentences)
        if n < 3:
            return [0.5] * n
        
        try:
            processed = [self._preprocess_sentence(s) for s in sentences]
            vectorizer = TfidfVectorizer(stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(processed)
            
            # Determine optimal clusters (roughly sqrt of sentences, min 2)
            n_clusters = max(2, min(int(np.sqrt(n)), n // 2))
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(tfidf_matrix.toarray())
            
            tfidf_array = tfidf_matrix.toarray()
            scores = []
            
            for i, (label, vector) in enumerate(zip(labels, tfidf_array)):
                # Calculate distance to centroid
                centroid = kmeans.cluster_centers_[label]
                distance = np.linalg.norm(vector - centroid)
                
                # Inverse distance (closer = higher score)
                # Add small epsilon to avoid division by zero
                scores.append(1.0 / (1.0 + distance))
            
            # Normalize to 0-1 range
            max_score = max(scores) if scores else 1
            min_score = min(scores) if scores else 0
            score_range = max_score - min_score if max_score != min_score else 1
            
            return [(s - min_score) / score_range for s in scores]
        except Exception:
            return [0.5] * n

    # =========================================
    # COSINE SIMILARITY
    # =========================================
    
    def _calculate_similarity_matrix(self, sentences: List[str]) -> np.ndarray:
        """Calculate pairwise cosine similarity between sentences."""
        if len(sentences) < 2:
            return np.array([[1.0]])
        
        try:
            processed = [self._preprocess_sentence(s) for s in sentences]
            vectorizer = TfidfVectorizer(stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(processed)
            return cosine_similarity(tfidf_matrix, tfidf_matrix)
        except Exception:
            return np.eye(len(sentences))
    
    # =========================================
    # REDUNDANCY REMOVAL
    # =========================================
    
    def _remove_redundancy(
        self, 
        sentences: List[str], 
        scores: List[float],
        indices: List[int]
    ) -> Tuple[List[str], List[float], List[int]]:
        """
        Remove redundant sentences based on similarity threshold.
        Keeps highest-scored sentence when duplicates found.
        """
        if not sentences:
            return [], [], []
        
        similarity_matrix = self._calculate_similarity_matrix(sentences)
        
        # Sort by score descending
        scored = list(zip(scores, range(len(sentences)), sentences, indices))
        scored.sort(reverse=True, key=lambda x: x[0])
        
        selected_local_indices = []
        
        for score, local_idx, sent, orig_idx in scored:
            is_redundant = False
            for sel_local_idx in selected_local_indices:
                if similarity_matrix[local_idx][sel_local_idx] > self.redundancy_threshold:
                    is_redundant = True
                    break
            
            if not is_redundant:
                selected_local_indices.append(local_idx)
        
        # Sort by original position
        selected_local_indices.sort()
        
        filtered_sentences = [sentences[i] for i in selected_local_indices]
        filtered_scores = [scores[i] for i in selected_local_indices]
        filtered_orig_indices = [indices[i] for i in selected_local_indices]
        
        return filtered_sentences, filtered_scores, filtered_orig_indices
    
    # =========================================
    # COMBINED SCORING
    # =========================================
    
    def _calculate_combined_scores(self, sentences: List[str]) -> List[float]:
        """Calculate combined scores using all algorithms."""
        if not sentences:
            return []
        
        # Calculate individual scores
        position_scores = self._calculate_position_scores(sentences)
        length_scores = self._calculate_length_scores(sentences)
        
        keywords = self._extract_keywords(sentences)
        keyword_scores = self._calculate_keyword_scores(sentences, keywords)
        
        tfidf_scores = self._calculate_tfidf_scores(sentences)
        textrank_scores = self._calculate_textrank_scores(sentences)
        
        # Calculate cluster scores if enabled
        if self.use_clustering:
            cluster_scores = self._calculate_cluster_scores(sentences)
        else:
            cluster_scores = [0.0] * len(sentences)
        
        # Combine with weights
        combined = []
        for i in range(len(sentences)):
            score = (
                self.position_weight * position_scores[i] +
                self.length_weight * length_scores[i] +
                self.keyword_weight * keyword_scores[i] +
                self.tfidf_weight * tfidf_scores[i] +
                self.textrank_weight * textrank_scores[i] +
                self.cluster_weight * cluster_scores[i]
            )
            combined.append(score)
        
        return combined
    
    # =========================================
    # MAIN SUMMARIZE METHOD
    # =========================================
    
    def summarize(
        self, 
        text: str, 
        num_sentences: int = 3, 
        ratio: float = None,
        remove_redundancy: bool = True
    ) -> dict:
        """
        Generate extractive summary using all algorithms.
        
        Args:
            text: Input text to summarize
            num_sentences: Number of sentences in summary
            ratio: Summary length as ratio of original
            remove_redundancy: Whether to remove redundant sentences
            
        Returns:
            Dictionary containing summary and detailed metadata
        """
        if not text or not text.strip():
            return {
                "summary": "",
                "original_sentences": 0,
                "summary_sentences": 0,
                "compression_ratio": 0,
                "method": "advanced",
                "algorithms_used": []
            }
        
        # Step 1: Sentence segmentation
        sentences = self._preprocess_text(text)
        original_count = len(sentences)
        
        if len(sentences) <= 2:
            return {
                "summary": text,
                "original_sentences": original_count,
                "summary_sentences": len(sentences),
                "compression_ratio": 1.0,
                "method": "advanced",
                "algorithms_used": ["sentence_segmentation"]
            }
        
        # Step 2: Sentence length filtering
        filtered_sentences, original_indices = self._filter_by_length(sentences)
        
        if len(filtered_sentences) == 0:
            filtered_sentences = sentences
            original_indices = list(range(len(sentences)))
        
        # Step 3: Calculate combined scores
        combined_scores = self._calculate_combined_scores(filtered_sentences)
        
        # Step 4: Redundancy removal
        if remove_redundancy and len(filtered_sentences) > 3:
            filtered_sentences, combined_scores, original_indices = self._remove_redundancy(
                filtered_sentences, combined_scores, original_indices
            )
        
        # Step 5: Calculate final number of sentences
        if ratio is not None:
            num_sentences = max(1, int(original_count * ratio))
        num_sentences = min(num_sentences, len(filtered_sentences))
        
        # Step 6: Select top sentences
        scored = list(zip(combined_scores, original_indices, filtered_sentences))
        scored.sort(reverse=True, key=lambda x: x[0])
        
        # Select top N and sort by original position
        selected = sorted(scored[:num_sentences], key=lambda x: x[1])
        summary = ' '.join([s for _, _, s in selected])
        
        # Build detailed result
        return {
            "summary": summary,
            "original_sentences": original_count,
            "summary_sentences": num_sentences,
            "compression_ratio": num_sentences / original_count,
            "method": "advanced",
            "algorithms_used": [
                "sentence_segmentation",
                "tokenization",
                "stopword_removal",
                "lemmatization" if self.use_lemmatization else "none",
                "tfidf",
                "textrank",
                "position_weighting",
                "length_filtering",
                "keyword_boosting",
                "redundancy_removal" if remove_redundancy else "none",
                "cosine_similarity"
            ],
            "sentence_scores": {filtered_sentences[i]: combined_scores[i] 
                              for i in range(len(filtered_sentences))},
            "keywords": list(self._extract_keywords(filtered_sentences))
        }


class ExtractiveSummarizer:
    """
    Extractive summarizer supporting multiple algorithms.
    
    Supported Methods:
    - 'textrank': Graph-based ranking using sentence similarity
    - 'tfidf': TF-IDF based sentence scoring  
    - 'advanced': All algorithms combined (RECOMMENDED)
    
    Extracts key sentences from the original text.
    """
    
    METHODS = ['textrank', 'tfidf', 'advanced']
    
    def __init__(self, method: str = 'advanced'):
        """
        Initialize the summarizer.
        
        Args:
            method: Summarization method ('textrank', 'tfidf', or 'advanced')
        """
        self._download_nltk_data()
        self.stop_words = set(stopwords.words('english'))
        self.method = method.lower()
        
        if self.method not in self.METHODS:
            print(f"Warning: Unknown method '{method}'. Using 'advanced'.")
            self.method = 'advanced'
        
        # Initialize appropriate summarizer
        if self.method == 'tfidf':
            self._summarizer = TFIDFSummarizer()
        elif self.method == 'advanced':
            self._summarizer = AdvancedSummarizer()
        else:
            self._summarizer = None  # Use internal TextRank
    
    def _download_nltk_data(self):
        """Download required NLTK data packages."""
        required_packages = ['punkt', 'stopwords', 'punkt_tab']
        for package in required_packages:
            try:
                nltk.data.find(f'tokenizers/{package}')
            except LookupError:
                try:
                    nltk.download(package, quiet=True)
                except Exception:
                    pass
    
    def _preprocess_text(self, text: str) -> list:
        """Preprocess text by tokenizing into sentences."""
        return sent_tokenize(text)
    
    def _sentence_similarity(self, sent1: str, sent2: str) -> float:
        """Calculate cosine similarity between two sentences."""
        words1 = [word.lower() for word in word_tokenize(sent1) 
                  if word.isalnum() and word.lower() not in self.stop_words]
        words2 = [word.lower() for word in word_tokenize(sent2) 
                  if word.isalnum() and word.lower() not in self.stop_words]
        
        all_words = list(set(words1 + words2))
        
        if len(all_words) == 0:
            return 0.0
        
        vector1 = [1 if word in words1 else 0 for word in all_words]
        vector2 = [1 if word in words2 else 0 for word in all_words]
        
        if sum(vector1) == 0 or sum(vector2) == 0:
            return 0.0
        
        return 1 - cosine_distance(vector1, vector2)
    
    def _build_similarity_matrix(self, sentences: list) -> np.ndarray:
        """Build similarity matrix for all sentences."""
        n = len(sentences)
        similarity_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    similarity_matrix[i][j] = self._sentence_similarity(
                        sentences[i], sentences[j]
                    )
        
        return similarity_matrix
    
    def summarize(self, text: str, num_sentences: int = 3, ratio: float = None, method: str = None) -> dict:
        """
        Generate extractive summary of the text.
        
        Args:
            text: Input text to summarize
            num_sentences: Number of sentences in summary (default: 3)
            ratio: Summary length as ratio of original
            method: Override the default method
            
        Returns:
            Dictionary containing summary and metadata
        """
        active_method = method.lower() if method else self.method
        
        # Use TF-IDF summarizer
        if active_method == 'tfidf':
            if not hasattr(self, '_summarizer') or not isinstance(self._summarizer, TFIDFSummarizer):
                self._summarizer = TFIDFSummarizer()
            return self._summarizer.summarize(text, num_sentences, ratio)
        
        # Use Advanced summarizer (RECOMMENDED)
        if active_method == 'advanced':
            if not hasattr(self, '_summarizer') or not isinstance(self._summarizer, AdvancedSummarizer):
                self._summarizer = AdvancedSummarizer()
            return self._summarizer.summarize(text, num_sentences, ratio)
        
        # TextRank method (basic)
        if not text or not text.strip():
            return {
                "summary": "",
                "original_sentences": 0,
                "summary_sentences": 0,
                "compression_ratio": 0,
                "method": "textrank"
            }
        
        sentences = self._preprocess_text(text)
        
        if len(sentences) <= 2:
            return {
                "summary": text,
                "original_sentences": len(sentences),
                "summary_sentences": len(sentences),
                "compression_ratio": 1.0,
                "method": "textrank"
            }
        
        if ratio is not None:
            num_sentences = max(1, int(len(sentences) * ratio))
        num_sentences = min(num_sentences, len(sentences))
        
        similarity_matrix = self._build_similarity_matrix(sentences)
        
        try:
            nx_graph = nx.from_numpy_array(similarity_matrix)
            scores = nx.pagerank(nx_graph, max_iter=100)
        except Exception:
            summary = ' '.join(sentences[:num_sentences])
            return {
                "summary": summary,
                "original_sentences": len(sentences),
                "summary_sentences": num_sentences,
                "compression_ratio": num_sentences / len(sentences),
                "method": "textrank"
            }
        
        ranked_sentences = sorted(
            ((scores[i], i, s) for i, s in enumerate(sentences)),
            reverse=True
        )
        
        selected = sorted(ranked_sentences[:num_sentences], key=lambda x: x[1])
        summary = ' '.join([s for _, _, s in selected])
        
        return {
            "summary": summary,
            "original_sentences": len(sentences),
            "summary_sentences": num_sentences,
            "compression_ratio": num_sentences / len(sentences),
            "method": "textrank",
            "sentence_scores": {sentences[i]: scores[i] for i in range(len(sentences))}
        }


# Convenience functions
def summarize_text(text: str, num_sentences: int = 3, method: str = 'advanced') -> str:
    """Quick function to summarize text."""
    summarizer = ExtractiveSummarizer(method=method)
    result = summarizer.summarize(text, num_sentences=num_sentences)
    return result["summary"]


def summarize_with_tfidf(text: str, num_sentences: int = 3) -> str:
    """Quick function to summarize using TF-IDF."""
    summarizer = TFIDFSummarizer()
    result = summarizer.summarize(text, num_sentences=num_sentences)
    return result["summary"]


def summarize_advanced(text: str, num_sentences: int = 3, **kwargs) -> str:
    """Quick function to summarize using all algorithms."""
    summarizer = AdvancedSummarizer(**kwargs)
    result = summarizer.summarize(text, num_sentences=num_sentences)
    return result["summary"]

