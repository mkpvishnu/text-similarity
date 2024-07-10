import numpy as np
from typing import Optional
from sentence_transformers import SentenceTransformer

def semantic_similarity(analyzer, text1: str, text2: str, print_info: bool = False, sentence_model: Optional[str] = None) -> float:
    if sentence_model:
        return _semantic_similarity_advanced(analyzer, text1, text2, print_info, sentence_model)
    else:
        return _semantic_similarity_simple(analyzer, text1, text2, print_info)

def _semantic_similarity_advanced(analyzer, text1: str, text2: str, print_info: bool,sentence_model: Optional[str] = None) -> float:
    model = SentenceTransformer(sentence_model)
    embedding1 = model.encode(text1)
    embedding2 = model.encode(text2)

    similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))

    if print_info:
        print(f"Semantic similarity with {sentence_model}: {similarity}")

    return similarity

def _semantic_similarity_simple(analyzer, text1: str, text2: str, print_info: bool) -> float:
    doc1 = analyzer.process(text1)
    doc2 = analyzer.process(text2)

    similarity = doc1.similarity(doc2)

    if print_info:
        print(f"Semantic similarity with Spacy: {similarity}")

    return similarity