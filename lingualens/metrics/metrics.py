from typing import Optional
from .temporal_shift import detect_temporal_shift
from .semantic_similarity import semantic_similarity

class Metrics:
    def __init__(self, analyzer):
        self.analyzer = analyzer

    def detect_temporal_shift(self, text1: str, text2: str, print_info: bool = False, llm: str = None, llm_apikey: str = None, parse_dependence: bool = False, sentence_model: Optional[str] = None) -> float:
        return detect_temporal_shift(self.analyzer, text1, text2, print_info,llm, llm_apikey, parse_dependence, sentence_model)

    def semantic_similarity(self, text1: str, text2: str, print_info: bool = False, sentence_model: Optional[str] = None) -> float:
        return semantic_similarity(self.analyzer, text1, text2, print_info, sentence_model)