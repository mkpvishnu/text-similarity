import spacy
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import Dict, Any
from utils import detect_role_reversal, detect_negation_mismatch, detect_temporal_shift, detect_tense_change, calculate_entity_preservation
from scoring_functions import score_summarization, score_similarity, score_paraphrase, score_contradiction

class TextComparisonSystem:
    def __init__(self, nlp_model="en_core_web_lg", sentence_model='sentence-transformers/all-MiniLM-L6-v2'):
        self.nlp = spacy.load(nlp_model)
        self.model = SentenceTransformer(sentence_model)
        self.scoring_functions = {
            "summarization": score_summarization,
            "similarity": score_similarity,
            "paraphrase": score_paraphrase,
            "contradiction": score_contradiction
        }

    def compare_texts(self, text1: str, text2: str, task: str, custom_penalties: Dict[str, float] = None) -> Dict[str, Any]:
        doc1 = self.nlp(text1)
        doc2 = self.nlp(text2)

        embeddings = self.model.encode([text1, text2])
        base_score = np.dot(embeddings[0], embeddings[1]) / (np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1]))

        metrics = {
            "role_reversal": detect_role_reversal(doc1, doc2),
            "negation_mismatch": detect_negation_mismatch(doc1, doc2),
            "temporal_shift": detect_temporal_shift(doc1, doc2),
            "tense_change": detect_tense_change(doc1, doc2),
            "entity_preservation": calculate_entity_preservation(doc1, doc2),
            "length_ratio": len(doc2) / len(doc1)
        }

        if task not in self.scoring_functions:
            raise ValueError(f"Unknown task: {task}")

        scoring_function = self.scoring_functions[task]
        final_score = scoring_function(base_score, metrics, custom_penalties)

        return {"base_score": base_score, **metrics, "final_score": final_score}
