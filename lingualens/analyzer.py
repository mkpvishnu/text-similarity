import spacy
from typing import List, Dict, Any, Optional
from .utils import ensure_model
from sentence_transformers import SentenceTransformer

class TextAnalyzer:
    def __init__(self, spacy_model: str = "en_core_web_sm", sentence_model: Optional[str] = None):
        self.spacy_model = ensure_model(spacy_model)
        self.nlp = spacy.load(self.spacy_model)
        #self.sentence_model = SentenceTransformer(sentence_model) if sentence_model else None

    def process(self, text: str) -> Any:
        return self.nlp(text)

    def split_into_paragraphs(self, text: str) -> List[str]:
        return [p.strip() for p in text.split('\n') if p.strip()]
    
    def analyze(self, text: str) -> Dict[str, Any]:
        doc = self.process(text)
        analysis = {
            'doc': doc,
            'tokens': [token.text for token in doc],
            'lemmas': [token.lemma_ for token in doc],
            'pos_tags': [token.pos_ for token in doc],
            'entities': [(ent.text, ent.label_) for ent in doc.ents],
        }
        return analysis

    # Add other analysis methods here