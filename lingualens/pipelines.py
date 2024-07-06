from typing import Dict, Any, Optional
from .analyzer import TextAnalyzer
from .metrics import Metrics

def check_similarity(text1: str, text2: str, spacy_model: str = "en_core_web_sm", sentence_model: Optional[str] = None) -> Dict[str, Any]:
    analyzer = TextAnalyzer(spacy_model, sentence_model)
    metrics = Metrics(analyzer)
    
    temporal_shift = metrics.detect_temporal_shift(text1, text2)
    semantic_sim = metrics.semantic_similarity(text1, text2)
    
    overall_similarity = 0.6 * semantic_sim + 0.4 * temporal_shift
    
    return {
        "temporal_shift": temporal_shift,
        "semantic_similarity": semantic_sim,
        "overall_similarity": overall_similarity
    }

def check_summary(original: str, summary: str, spacy_model: str = "en_core_web_sm", sentence_model: Optional[str] = None) -> Dict[str, Any]:
    analyzer = TextAnalyzer(spacy_model, sentence_model)
    metrics = Metrics(analyzer)
    
    semantic_sim = metrics.semantic_similarity(original, summary)
    temporal_consistency = metrics.detect_temporal_shift(original, summary)
    
    overall_score = 0.7 * semantic_sim + 0.3 * temporal_consistency
    
    return {
        "semantic_similarity": semantic_sim,
        "temporal_consistency": temporal_consistency,
        "overall_score": overall_score
    }

def check_paraphrase(original: str, paraphrase: str, spacy_model: str = "en_core_web_sm", sentence_model: Optional[str] = None) -> Dict[str, Any]:
    analyzer = TextAnalyzer(spacy_model, sentence_model)
    metrics = Metrics(analyzer)
    
    semantic_sim = metrics.semantic_similarity(original, paraphrase)
    temporal_consistency = metrics.detect_temporal_shift(original, paraphrase)
    
    overall_score = 0.8 * semantic_sim + 0.2 * temporal_consistency
    
    return {
        "semantic_similarity": semantic_sim,
        "temporal_consistency": temporal_consistency,
        "overall_score": overall_score
    }