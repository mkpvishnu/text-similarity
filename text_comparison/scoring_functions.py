# text_comparison/scoring_functions.py

from typing import Dict

def score_summarization(base_score: float, metrics: Dict[str, float], custom_penalties: Dict[str, float] = None) -> float:
    penalties = {
        "role_reversal": 0.3,
        "negation_mismatch": 0.5,
        "temporal_shift": 0.1,
        "tense_change": 0.1,
        "entity_preservation": 1.0,
        "length_ratio": 0.5
    }
    if custom_penalties:
        penalties.update(custom_penalties)

    score = base_score
    score *= (1 - metrics["role_reversal"] * penalties["role_reversal"])
    score *= 0.5 if metrics["negation_mismatch"] else 1
    score *= 0.9 if metrics["temporal_shift"] else 1
    score -= metrics["tense_change"] * penalties["tense_change"]
    score *= metrics["entity_preservation"] ** penalties["entity_preservation"]
    score *= min(1, 2 - metrics["length_ratio"]) ** penalties["length_ratio"]

    return max(0, min(1, score))

def score_similarity(base_score: float, metrics: Dict[str, float], custom_penalties: Dict[str, float] = None) -> float:
    penalties = {
        "role_reversal": 0.5,
        "negation_mismatch": 0.8,
        "temporal_shift": 0.3,
        "tense_change": 0.3,
        "entity_preservation": 0.5
    }
    if custom_penalties:
        penalties.update(custom_penalties)

    score = base_score
    score *= (1 - metrics["role_reversal"] * penalties["role_reversal"])
    score *= 0.2 if metrics["negation_mismatch"] else 1
    score *= 0.7 if metrics["temporal_shift"] else 1
    score *= (1 - metrics["tense_change"] * penalties["tense_change"])
    score *= (metrics["entity_preservation"] * penalties["entity_preservation"] + (1 - penalties["entity_preservation"]))

    return max(0, min(1, score))

def score_paraphrase(base_score: float, metrics: Dict[str, float], custom_penalties: Dict[str, float] = None) -> float:
    penalties = {
        "role_reversal": 0.2,
        "negation_mismatch": 0.7,
        "temporal_shift": 0.2,
        "tense_change": 0.2,
        "entity_preservation": 0.7,
        "length_ratio": 0.2
    }
    if custom_penalties:
        penalties.update(custom_penalties)

    score = base_score
    score *= (1 - metrics["role_reversal"] * penalties["role_reversal"])
    score *= 0.3 if metrics["negation_mismatch"] else 1
    score *= 0.8 if metrics["temporal_shift"] else 1
    score *= (1 - metrics["tense_change"] * penalties["tense_change"])
    score *= (metrics["entity_preservation"] * penalties["entity_preservation"] + (1 - penalties["entity_preservation"]))
    score *= 1 - abs(1 - metrics["length_ratio"]) * penalties["length_ratio"]

    return max(0, min(1, score))

def score_contradiction(base_score: float, metrics: Dict[str, float], custom_penalties: Dict[str, float] = None) -> float:
    penalties = {
        "role_reversal": 0.3,
        "negation_mismatch": 0.5,
        "temporal_shift": 0.1,
        "tense_change": 0.1,
        "entity_preservation": 0.5
    }
    if custom_penalties:
        penalties.update(custom_penalties)

    score = 1 - base_score  # Invert the base score for contradiction
    score *= (1 + metrics["role_reversal"] * penalties["role_reversal"])
    score *= 1.5 if metrics["negation_mismatch"] else 1
    score *= 1.1 if metrics["temporal_shift"] else 1
    score *= (1 + metrics["tense_change"] * penalties["tense_change"])
    score *= (2 - metrics["entity_preservation"]) ** penalties["entity_preservation"]

    return max(0, min(1, score))