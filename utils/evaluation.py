"""
Evaluation Metrics Module for Text Summarization

This module provides metrics to evaluate summary quality:
1. ROUGE-1: Unigram overlap (word-level)
2. ROUGE-2: Bigram overlap (phrase-level)
3. ROUGE-L: Longest Common Subsequence (sentence-level structure)
4. ROUGE-Lsum: ROUGE-L for multi-sentence summaries

Each metric provides:
- Precision: What fraction of the summary is relevant
- Recall: What fraction of the reference is captured
- F1-Score: Harmonic mean of precision and recall
"""

from typing import Dict, List, Optional, Union, Tuple
import numpy as np
from collections import Counter

# Lazy loading for rouge_score
_rouge_scorer = None


def _load_rouge_scorer():
    """Lazy load ROUGE scorer."""
    global _rouge_scorer
    if _rouge_scorer is None:
        try:
            from rouge_score import rouge_scorer
            _rouge_scorer = rouge_scorer.RougeScorer(
                ['rouge1', 'rouge2', 'rougeL', 'rougeLsum'],
                use_stemmer=True
            )
        except ImportError:
            raise ImportError(
                "rouge-score not installed. Run: pip install rouge-score"
            )
    return _rouge_scorer


class SummaryEvaluator:
    """
    Comprehensive summary evaluation using ROUGE metrics.
    
    ROUGE (Recall-Oriented Understudy for Gisting Evaluation) measures
    the overlap between generated summaries and reference summaries.
    
    Usage:
        evaluator = SummaryEvaluator()
        scores = evaluator.evaluate(generated_summary, reference_summary)
    """
    
    def __init__(self, use_stemmer: bool = True):
        """
        Initialize the evaluator.
        
        Args:
            use_stemmer: Whether to use Porter stemmer for word matching
        """
        self.use_stemmer = use_stemmer
        self._scorer = None
    
    def _get_scorer(self):
        """Get or create ROUGE scorer."""
        if self._scorer is None:
            try:
                from rouge_score import rouge_scorer
                self._scorer = rouge_scorer.RougeScorer(
                    ['rouge1', 'rouge2', 'rougeL', 'rougeLsum'],
                    use_stemmer=self.use_stemmer
                )
            except ImportError:
                raise ImportError(
                    "rouge-score not installed. Run: pip install rouge-score"
                )
        return self._scorer
    
    def evaluate(
        self,
        generated_summary: str,
        reference_summary: str,
        return_all: bool = True
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate a generated summary against a reference.
        
        Args:
            generated_summary: The system-generated summary
            reference_summary: The human-written reference summary
            return_all: If True, return all metrics; if False, return only F1
            
        Returns:
            Dictionary with ROUGE scores:
            {
                'rouge1': {'precision': 0.8, 'recall': 0.7, 'f1': 0.75},
                'rouge2': {...},
                'rougeL': {...},
                'rougeLsum': {...}
            }
        """
        if not generated_summary or not reference_summary:
            return self._empty_scores()
        
        scorer = self._get_scorer()
        scores = scorer.score(reference_summary, generated_summary)
        
        result = {}
        for metric_name, score in scores.items():
            if return_all:
                result[metric_name] = {
                    'precision': round(score.precision, 4),
                    'recall': round(score.recall, 4),
                    'f1': round(score.fmeasure, 4)
                }
            else:
                result[metric_name] = round(score.fmeasure, 4)
        
        return result
    
    def evaluate_batch(
        self,
        generated_summaries: List[str],
        reference_summaries: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate multiple summaries and return average scores.
        
        Args:
            generated_summaries: List of generated summaries
            reference_summaries: List of reference summaries
            
        Returns:
            Dictionary with average ROUGE scores
        """
        if len(generated_summaries) != len(reference_summaries):
            raise ValueError("Number of generated and reference summaries must match")
        
        all_scores = []
        for gen, ref in zip(generated_summaries, reference_summaries):
            scores = self.evaluate(gen, ref)
            all_scores.append(scores)
        
        # Average the scores
        avg_scores = {}
        metrics = ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']
        
        for metric in metrics:
            avg_scores[metric] = {
                'precision': round(np.mean([s[metric]['precision'] for s in all_scores]), 4),
                'recall': round(np.mean([s[metric]['recall'] for s in all_scores]), 4),
                'f1': round(np.mean([s[metric]['f1'] for s in all_scores]), 4)
            }
        
        return avg_scores
    
    def _empty_scores(self) -> Dict[str, Dict[str, float]]:
        """Return empty scores structure."""
        return {
            'rouge1': {'precision': 0.0, 'recall': 0.0, 'f1': 0.0},
            'rouge2': {'precision': 0.0, 'recall': 0.0, 'f1': 0.0},
            'rougeL': {'precision': 0.0, 'recall': 0.0, 'f1': 0.0},
            'rougeLsum': {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        }
    
    def compare_methods(
        self,
        original_text: str,
        reference_summary: str,
        method_summaries: Dict[str, str]
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare multiple summarization methods against a reference.
        
        Args:
            original_text: The original document
            reference_summary: Human-written reference summary
            method_summaries: Dict mapping method names to their summaries
            
        Returns:
            Dictionary mapping method names to their ROUGE scores
        """
        results = {}
        for method_name, summary in method_summaries.items():
            results[method_name] = self.evaluate(summary, reference_summary)
        
        return results
    
    def get_best_method(
        self,
        method_scores: Dict[str, Dict[str, Dict[str, float]]],
        metric: str = 'rouge1',
        score_type: str = 'f1'
    ) -> Tuple[str, float]:
        """
        Determine the best performing method.
        
        Args:
            method_scores: Output from compare_methods()
            metric: Which ROUGE metric to use ('rouge1', 'rouge2', 'rougeL')
            score_type: Which score to compare ('precision', 'recall', 'f1')
            
        Returns:
            Tuple of (best_method_name, score)
        """
        best_method = None
        best_score = -1
        
        for method, scores in method_scores.items():
            score = scores[metric][score_type]
            if score > best_score:
                best_score = score
                best_method = method
        
        return best_method, best_score


class BasicMetrics:
    """
    Basic evaluation metrics that don't require reference summaries.
    
    Useful for intrinsic evaluation when no reference is available.
    """
    
    @staticmethod
    def compression_ratio(original: str, summary: str) -> float:
        """
        Calculate compression ratio.
        
        Returns:
            Ratio of summary length to original length (0-1)
        """
        orig_words = len(original.split())
        summ_words = len(summary.split())
        
        if orig_words == 0:
            return 0.0
        
        return round(summ_words / orig_words, 4)
    
    @staticmethod
    def compression_percentage(original: str, summary: str) -> float:
        """
        Calculate percentage of text compressed away.
        
        Returns:
            Percentage of original text removed (0-100)
        """
        orig_words = len(original.split())
        summ_words = len(summary.split())
        
        if orig_words == 0:
            return 0.0
        
        return round((1 - summ_words / orig_words) * 100, 2)
    
    @staticmethod
    def retention_ratio(original: str, summary: str) -> float:
        """
        Calculate word retention ratio (how many original words appear in summary).
        
        Returns:
            Ratio of retained words (0-1)
        """
        orig_words = set(original.lower().split())
        summ_words = set(summary.lower().split())
        
        if not orig_words:
            return 0.0
        
        retained = len(orig_words.intersection(summ_words))
        return round(retained / len(orig_words), 4)
    
    @staticmethod
    def novelty_ratio(original: str, summary: str) -> float:
        """
        Calculate novelty ratio (words in summary not in original).
        For extractive methods, this should be 0 or very low.
        For abstractive methods, this shows how much new phrasing was used.
        
        Returns:
            Ratio of novel words (0-1)
        """
        orig_words = set(original.lower().split())
        summ_words = set(summary.lower().split())
        
        if not summ_words:
            return 0.0
        
        novel = len(summ_words - orig_words)
        return round(novel / len(summ_words), 4)
    
    @staticmethod
    def sentence_count(text: str) -> int:
        """Count sentences in text."""
        from nltk.tokenize import sent_tokenize
        return len(sent_tokenize(text))
    
    @staticmethod
    def word_count(text: str) -> int:
        """Count words in text."""
        return len(text.split())
    
    @staticmethod
    def avg_sentence_length(text: str) -> float:
        """Calculate average sentence length in words."""
        from nltk.tokenize import sent_tokenize
        sentences = sent_tokenize(text)
        if not sentences:
            return 0.0
        
        total_words = sum(len(s.split()) for s in sentences)
        return round(total_words / len(sentences), 2)


def calculate_rouge_scores(
    generated: str,
    reference: str
) -> Dict[str, float]:
    """
    Quick function to calculate ROUGE F1 scores.
    
    Args:
        generated: Generated summary
        reference: Reference summary
        
    Returns:
        Dictionary with F1 scores for each ROUGE metric
    """
    evaluator = SummaryEvaluator()
    scores = evaluator.evaluate(generated, reference, return_all=False)
    return scores


def format_rouge_scores(scores: Dict[str, Dict[str, float]]) -> str:
    """
    Format ROUGE scores for display.
    
    Args:
        scores: ROUGE scores dictionary
        
    Returns:
        Formatted string for display
    """
    lines = []
    for metric, values in scores.items():
        if isinstance(values, dict):
            lines.append(f"{metric.upper()}:")
            lines.append(f"  Precision: {values['precision']:.2%}")
            lines.append(f"  Recall: {values['recall']:.2%}")
            lines.append(f"  F1-Score: {values['f1']:.2%}")
        else:
            lines.append(f"{metric.upper()}: {values:.2%}")
    
    return '\n'.join(lines)


def get_evaluation_report(
    original: str,
    summary: str,
    reference: Optional[str] = None,
    method_name: str = "Summary"
) -> Dict:
    """
    Generate a comprehensive evaluation report.
    
    Args:
        original: Original text
        summary: Generated summary
        reference: Reference summary (optional)
        method_name: Name of the summarization method
        
    Returns:
        Dictionary with all evaluation metrics
    """
    basic = BasicMetrics()
    
    report = {
        'method': method_name,
        'basic_metrics': {
            'original_words': basic.word_count(original),
            'summary_words': basic.word_count(summary),
            'original_sentences': basic.sentence_count(original),
            'summary_sentences': basic.sentence_count(summary),
            'compression_ratio': basic.compression_ratio(original, summary),
            'compression_percentage': basic.compression_percentage(original, summary),
            'retention_ratio': basic.retention_ratio(original, summary),
            'novelty_ratio': basic.novelty_ratio(original, summary),
            'avg_sentence_length': basic.avg_sentence_length(summary)
        }
    }
    
    if reference:
        evaluator = SummaryEvaluator()
        report['rouge_scores'] = evaluator.evaluate(summary, reference)
    
    return report
