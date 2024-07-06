from typing import Dict, Any, List, Optional
from .semantic_similarity import semantic_similarity

def detect_temporal_shift(analyzer, text1: str, text2: str, print_info: bool = False, parse_dependence: bool = False, sentence_model: Optional[str] = None) -> float:
    if text1 == text2:
        print("Sentences are Identical. No parsing required")
        return 1.0
    analysis1 = analyzer.analyze(text1)
    analysis2 = analyzer.analyze(text2)
    
    if sentence_model:
        return _detect_temporal_shift_hybrid(analyzer, analysis1, analysis2, print_info, sentence_model)
    else:
        if parse_dependence:
            return _detect_temporal_shift_advanced(analysis1, analysis2, print_info)
        else:
            return _detect_temporal_shift_simple(analysis1, analysis2, print_info)

def _detect_temporal_shift_simple(analysis1: Dict[str, Any], analysis2: Dict[str, Any], print_info: bool) -> float:
    temporal_words = {'yesterday', 'today', 'tomorrow', 'now', 'then', 'soon', 'later'}
    
    def extract_temporal_info(doc):
        return set(token.text.lower() for token in doc if token.text.lower() in temporal_words or token.ent_type_ == 'DATE')
    
    temp1 = extract_temporal_info(analysis1['doc'])
    temp2 = extract_temporal_info(analysis2['doc'])
    
    similarity = len(temp1.intersection(temp2)) / len(temp1.union(temp2)) if temp1 or temp2 else 1.0
    
    if print_info:
        print(f"Temporal info in text 1: {temp1}")
        print(f"Temporal info in text 2: {temp2}")
        print(f"Temporal shift similarity: {similarity}")
    
    return similarity

def _detect_temporal_shift_advanced(analysis1: Dict[str, Any], analysis2: Dict[str, Any], print_info: bool) -> float:
    temporal_info1 = _extract_temporal_info(analysis1['doc'])
    temporal_info2 = _extract_temporal_info(analysis2['doc'])
    
    similarity = _compare_temporal_info(temporal_info1, temporal_info2)
    
    if print_info:
        print(f"Temporal info in text 1: {temporal_info1}")
        print(f"Temporal info in text 2: {temporal_info2}")
        print(f"Temporal shift similarity: {similarity}")
    
    return similarity
def _detect_temporal_shift_hybrid(analyzer, analysis1: Dict[str, Any], analysis2: Dict[str, Any], print_info: bool, sentence_model: Optional[str] = None) -> float:
    # Extract temporal information
    temporal_info1 = _extract_temporal_info(analysis1['doc'])
    temporal_info2 = _extract_temporal_info(analysis2['doc'])
    print(f"Temporal info in text 1: {temporal_info1}")
    print(f"Temporal info in text 2: {temporal_info2}")
    
    # Calculate structural similarity
    structural_sim = _compare_temporal_info(temporal_info1, temporal_info2)
    print(f"Structural similarity: {structural_sim}")
    
    # Calculate semantic similarity of temporal contexts
    semantic_sim = _calculate_semantic_similarity(analyzer, temporal_info1, temporal_info2, sentence_model)
    
    # Combine structural and semantic similarities
    hybrid_sim = 0.6 * structural_sim + 0.4 * semantic_sim
    
    if print_info:
        print(f"Semantic similarity: {semantic_sim}")
        print(f"Hybrid temporal shift similarity: {hybrid_sim}")
    
    return hybrid_sim

def _extract_temporal_info(doc: Any) -> List[Dict[str, str]]:
    temporal_info = []
    for sent in doc.sents:
        info = {'temporal': None, 'subject': None, 'verb': None, 'object': None, 'context': sent.text}
        for token in sent:
            if token.ent_type_ == 'DATE' or token.text.lower() in ['yesterday', 'today', 'tomorrow']:
                info['temporal'] = token.text
            elif token.dep_ == 'nsubj':
                info['subject'] = token.text
            elif token.pos_ == 'VERB':
                info['verb'] = token.lemma_
            elif token.dep_ in ['dobj', 'pobj']:
                info['object'] = token.text
        if info['temporal'] and info['verb']:
            temporal_info.append(info)
    return temporal_info

def _compare_temporal_info(info1: List[Dict[str, str]], info2: List[Dict[str, str]]) -> float:
    if not info1 and not info2:
        return 1.0
    if not info1 or not info2:
        return 0.0
    
    matches = sum(1 for item1 in info1 for item2 in info2 
                  if item1['temporal'] == item2['temporal'] and 
                     item1['verb'] == item2['verb'] and 
                     item1['subject'] == item2['subject'] and 
                     item1['object'] == item2['object'])
    return matches / max(len(info1), len(info2))

def _calculate_semantic_similarity(analyzer, info1: List[Dict[str, Any]], info2: List[Dict[str, Any]], sentence_model: Optional[str] = None) -> float:
    if not info1 or not info2:
        return 0.0
    
    contexts1 = " ".join([item['context'] for item in info1])
    contexts2 = " ".join([item['context'] for item in info2])
    
    return semantic_similarity(analyzer, contexts1, contexts2, sentence_model=sentence_model)