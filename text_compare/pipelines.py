from typing import Dict, Any
from .analyzer import TextAnalyzer
from .metrics import Metrics

def check_summary(original: str, summary: str, spacy_model: str = "en_core_web_sm") -> Dict[str, Any]:
    analyzer = TextAnalyzer(spacy_model)
    metrics = Metrics(analyzer)
    # Implementation here
    pass

def check_paraphrase(original: str, paraphrased: str, spacy_model: str = "en_core_web_sm") -> Dict[str, Any]:
    analyzer = TextAnalyzer(spacy_model)
    metrics = Metrics(analyzer)
    # Implementation here
    pass

def check_similarity(text1: str, text2: str, spacy_model: str = "en_core_web_sm") -> Dict[str, Any]:
    analyzer = TextAnalyzer(spacy_model)
    metrics = Metrics(analyzer)
    
    temporal_shift = metrics.detect_temporal_shift(text1, text2)
    
    return {
        "temporal_shift": temporal_shift,
        "overall_similarity": temporal_shift  # For now, overall similarity is just temporal shift
    }