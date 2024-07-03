# text_comparison/scoring_functions.py

from typing import Dict
import numpy as np

def score_summarization(base_score: float, bert_score: float, metrics: Dict[str, float], custom_penalties: Dict[str, float] = None) -> float:
    penalties = {
        "role_reversal": 0.4,
        "negation_mismatch": 0.5,
        "temporal_shift": 0.1,
        "tense_change": 0.1,
        "entity_preservation": 0.2,
        "length_ratio": 0.1,
        "key_info_preservation": 0.3,
        "abstractiveness": 0.1,
        "semantic_similarity": 0.2,
        "sense_preservation": 0.2
    }
    if custom_penalties:
        penalties.update(custom_penalties)

    semantic_score = (base_score + bert_score) / 2
    
    adjustments = [
        1 - (metrics["role_reversal"] * penalties["role_reversal"]),
        0.5 if metrics["negation_mismatch"] else 1,
        0.9 if metrics["temporal_shift"] else 1,
        1 - (metrics["tense_change"] * penalties["tense_change"]),
        metrics["entity_preservation"] ** penalties["entity_preservation"],
        np.clip(1 - abs(1 - metrics["length_ratio"]), 0.5, 1) ** penalties["length_ratio"],
        metrics["key_info_preservation"] ** penalties["key_info_preservation"],
        (1 + metrics["abstractiveness"]) ** penalties["abstractiveness"],
        metrics["semantic_similarity"] ** penalties["semantic_similarity"],
        metrics["sense_preservation"] ** penalties["sense_preservation"]
    ]

    adjustment_score = np.prod(adjustments)
    
    final_score = semantic_score * 0.6 + adjustment_score * 0.4
    
    return max(0, min(1, final_score))

def score_similarity(base_score: float, bert_score: float, metrics: Dict[str, float], custom_penalties: Dict[str, float] = None) -> float:
    penalties = {
        "role_reversal": 0.5,
        "negation_mismatch": 0.6,
        "temporal_shift": 0.2,
        "tense_change": 0.2,
        "entity_preservation": 0.2,
        "structural_similarity": 0.2,
        "semantic_similarity": 0.3,
        "sense_preservation": 0.3
    }
    if custom_penalties:
        penalties.update(custom_penalties)

    semantic_score = (base_score + bert_score) / 2
    
    adjustments = [
        1 - (metrics["role_reversal"] * penalties["role_reversal"]),
        0.4 if metrics["negation_mismatch"] else 1,
        0.8 if metrics["temporal_shift"] else 1,
        1 - (metrics["tense_change"] * penalties["tense_change"]),
        metrics["entity_preservation"] ** penalties["entity_preservation"],
        metrics["structural_similarity"] ** penalties["structural_similarity"],
        metrics["semantic_similarity"] ** penalties["semantic_similarity"],
        metrics["sense_preservation"] ** penalties["sense_preservation"]
    ]

    adjustment_score = np.prod(adjustments)
    
    final_score = semantic_score * 0.7 + adjustment_score * 0.3
    
    return max(0, min(1, final_score))

def score_paraphrase(base_score: float, bert_score: float, metrics: Dict[str, float], custom_penalties: Dict[str, float] = None) -> float:
    penalties = {
        "role_reversal": 0.3,
        "negation_mismatch": 0.5,
        "temporal_shift": 0.2,
        "tense_change": 0.2,
        "entity_preservation": 0.2,
        "length_ratio": 0.1,
        "structural_similarity": 0.2,
        "key_info_preservation": 0.3,
        "semantic_similarity": 0.4,
        "sense_preservation": 0.4
    }
    if custom_penalties:
        penalties.update(custom_penalties)

    semantic_score = (base_score + bert_score) / 2
    
    adjustments = [
        1 - (metrics["role_reversal"] * penalties["role_reversal"]),
        0.5 if metrics["negation_mismatch"] else 1,
        0.8 if metrics["temporal_shift"] else 1,
        1 - (metrics["tense_change"] * penalties["tense_change"]),
        metrics["entity_preservation"] ** penalties["entity_preservation"],
        np.clip(1 - abs(1 - metrics["length_ratio"]), 0.5, 1) ** penalties["length_ratio"],
        metrics["structural_similarity"] ** penalties["structural_similarity"],
        metrics["key_info_preservation"] ** penalties["key_info_preservation"],
        metrics["semantic_similarity"] ** penalties["semantic_similarity"],
        metrics["sense_preservation"] ** penalties["sense_preservation"]
    ]

    adjustment_score = np.prod(adjustments)
    
    final_score = semantic_score * 0.5 + adjustment_score * 0.5
    
    return max(0, min(1, final_score))

def score_contradiction(base_score: float, bert_score: float, metrics: Dict[str, float], custom_penalties: Dict[str, float] = None) -> float:
    penalties = {
        "role_reversal": 0.3,
        "negation_mismatch": 0.7,
        "temporal_shift": 0.2,
        "tense_change": 0.2,
        "entity_preservation": 0.2,
        "key_info_preservation": 0.3,
        "semantic_similarity": 0.2,
        "sense_preservation": 0.2
    }
    if custom_penalties:
        penalties.update(custom_penalties)

    semantic_score = 1 - (base_score + bert_score) / 2  # Invert the semantic score for contradiction
    
    adjustments = [
        1 + (metrics["role_reversal"] * penalties["role_reversal"]),
        1.7 if metrics["negation_mismatch"] else 1,
        1.2 if metrics["temporal_shift"] else 1,
        1 + (metrics["tense_change"] * penalties["tense_change"]),
        (2 - metrics["entity_preservation"]) ** penalties["entity_preservation"],
        (2 - metrics["key_info_preservation"]) ** penalties["key_info_preservation"],
        (2 - metrics["semantic_similarity"]) ** penalties["semantic_similarity"],
        (2 - metrics["sense_preservation"]) ** penalties["sense_preservation"]
    ]

    adjustment_score = np.prod(adjustments)
    
    final_score = semantic_score * 0.4 + adjustment_score * 0.6
    
    return max(0, min(1, final_score))