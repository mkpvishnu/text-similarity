import spacy
from typing import List, Dict, Any
from .utils import ensure_model

class TextAnalyzer:
    def __init__(self, spacy_model: str = "en_core_web_sm"):
        self.spacy_model = ensure_model(spacy_model)
        self.nlp = spacy.load(self.spacy_model)
        # Initialize other necessary components

    def process(self, text: str) -> Any:
        return self.nlp(text)

    def split_into_paragraphs(self, text: str) -> List[str]:
        return [p.strip() for p in text.split('\n') if p.strip()]

    # Add other analysis methods here