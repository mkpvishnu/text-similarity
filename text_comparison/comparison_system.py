# text_comparison/comparison_system.py

import spacy
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import Dict, Any
from utils import detect_role_reversal, detect_negation_mismatch, detect_temporal_shift, detect_tense_change, calculate_entity_preservation
from scoring_functions import score_summarization, score_similarity, score_paraphrase, score_contradiction
from semantic_analysis import SemanticAnalyzer, calculate_semantic_similarity, calculate_sense_preservation
from transformers import AutoTokenizer, AutoModel
import torch

class TextComparisonSystem:
    def __init__(self, nlp_model="en_core_web_lg", sentence_model='sentence-transformers/all-MiniLM-L6-v2', bert_model='sentence-transformers/all-MiniLM-L12-v2'):
        self.nlp = spacy.load(nlp_model)
        self.sentence_model = SentenceTransformer(sentence_model)
        self.bert_tokenizer = AutoTokenizer.from_pretrained(bert_model)
        self.bert_model = AutoModel.from_pretrained(bert_model)
        self.semantic_analyzer = SemanticAnalyzer(bert_model)
        self.scoring_functions = {
            "summarization": score_summarization,
            "similarity": score_similarity,
            "paraphrase": score_paraphrase,
            "contradiction": score_contradiction
        }

    def get_bert_embedding(self, text):
        inputs = self.bert_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

    def compare_texts(self, text1: str, text2: str, task: str, custom_penalties: Dict[str, float] = None) -> Dict[str, Any]:
        doc1 = self.nlp(text1)
        doc2 = self.nlp(text2)

        embeddings = self.sentence_model.encode([text1, text2])
        base_score = np.dot(embeddings[0], embeddings[1]) / (np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1]))

        bert_emb1 = self.get_bert_embedding(text1)
        bert_emb2 = self.get_bert_embedding(text2)
        bert_score = np.dot(bert_emb1, bert_emb2) / (np.linalg.norm(bert_emb1) * np.linalg.norm(bert_emb2))

        metrics = {
            "role_reversal": detect_role_reversal(doc1, doc2),
            "negation_mismatch": detect_negation_mismatch(doc1, doc2),
            "temporal_shift": detect_temporal_shift(doc1, doc2),
            "tense_change": detect_tense_change(doc1, doc2),
            "entity_preservation": calculate_entity_preservation(doc1, doc2),
            "length_ratio": len(doc2) / len(doc1),
            "key_info_preservation": self.calculate_key_info_preservation(doc1, doc2),
            "abstractiveness": self.calculate_abstractiveness(doc1, doc2),
            "structural_similarity": self.calculate_structural_similarity(doc1, doc2),
            "semantic_similarity": calculate_semantic_similarity(doc1, doc2, self.semantic_analyzer),
            "sense_preservation": calculate_sense_preservation(doc1, doc2, self.semantic_analyzer)
        }

        if task not in self.scoring_functions:
            raise ValueError(f"Unknown task: {task}")

        scoring_function = self.scoring_functions[task]
        final_score = scoring_function(base_score, bert_score, metrics, custom_penalties)

        return {"base_score": base_score, "bert_score": bert_score, **metrics, "final_score": final_score}

    def calculate_key_info_preservation(self, doc1, doc2):
        # Extract key elements (subjects, objects, main verbs) from both documents
        key_elements1 = set(self._extract_key_elements(doc1))
        key_elements2 = set(self._extract_key_elements(doc2))
        
        # Calculate the overlap
        overlap = len(key_elements1.intersection(key_elements2))
        total = len(key_elements1)
        
        return overlap / total if total > 0 else 1.0

    def _extract_key_elements(self, doc):
        elements = []
        for token in doc:
            if token.dep_ in ["nsubj", "dobj", "pobj"] or (token.pos_ == "VERB" and token.dep_ == "ROOT"):
                elements.append(token.lemma_)
        return elements

    def calculate_abstractiveness(self, doc1, doc2):
        # Calculate n-gram overlap
        n_gram_overlap = self._calculate_n_gram_overlap(doc1, doc2)
        
        # Calculate the ratio of new words in doc2
        words1 = set(token.lower_ for token in doc1)
        words2 = set(token.lower_ for token in doc2)
        new_words_ratio = len(words2 - words1) / len(words2) if len(words2) > 0 else 0
        
        # Combine these metrics (you can adjust the weights)
        abstractiveness = (1 - n_gram_overlap) * 0.7 + new_words_ratio * 0.3
        
        return abstractiveness

    def _calculate_n_gram_overlap(self, doc1, doc2, n=3):
        def get_n_grams(doc, n):
            return set(' '.join(token.lower_ for token in doc[i:i+n]) for i in range(len(doc)-n+1))
        
        n_grams1 = get_n_grams(doc1, n)
        n_grams2 = get_n_grams(doc2, n)
        
        overlap = len(n_grams1.intersection(n_grams2))
        total = len(n_grams2)
        
        return overlap / total if total > 0 else 0

    def calculate_structural_similarity(self, doc1, doc2):
        # Compare dependency structures
        dep_sim = self._compare_dependency_structures(doc1, doc2)
        
        # Compare POS tag sequences
        pos_sim = self._compare_pos_sequences(doc1, doc2)
        
        # Combine these metrics (you can adjust the weights)
        structural_similarity = dep_sim * 0.6 + pos_sim * 0.4
        
        return structural_similarity

    def _compare_dependency_structures(self, doc1, doc2):
        def get_dep_structure(doc):
            return set((token.dep_, token.head.dep_) for token in doc)
        
        struct1 = get_dep_structure(doc1)
        struct2 = get_dep_structure(doc2)
        
        similarity = len(struct1.intersection(struct2)) / max(len(struct1), len(struct2))
        return similarity

    def _compare_pos_sequences(self, doc1, doc2):
        def get_pos_sequence(doc):
            return ' '.join(token.pos_ for token in doc)
        
        seq1 = get_pos_sequence(doc1)
        seq2 = get_pos_sequence(doc2)
        
        # Use Levenshtein distance to compare sequences
        distance = self._levenshtein_distance(seq1, seq2)
        max_length = max(len(seq1), len(seq2))
        similarity = 1 - (distance / max_length)
        
        return similarity

    def _levenshtein_distance(self, s1, s2):
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]