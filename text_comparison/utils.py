# text_comparison/utils.py

import spacy
from spacy.tokens import Doc

def get_semantic_roles(doc: Doc):
    agent, action, patient = None, None, None
    for token in doc:
        if token.dep_ == "ROOT" and token.pos_ == "VERB":
            action = token.lemma_
        elif token.dep_ in ["nsubj", "nsubjpass"] and not agent:
            agent = token.text.lower()
        elif token.dep_ in ["dobj", "pobj"] and not patient:
            patient = token.text.lower()
    return agent, action, patient

def detect_role_reversal(doc1: Doc, doc2: Doc) -> float:
    agent1, action1, patient1 = get_semantic_roles(doc1)
    agent2, action2, patient2 = get_semantic_roles(doc2)
    
    if not all([agent1, action1, patient1, agent2, action2, patient2]):
        return 0
    
    if action1 != action2:
        return 0
    
    if agent1 == patient2 and patient1 == agent2:
        return 1
    elif agent1 == patient2 or patient1 == agent2:
        return 0.5
    
    return 0

def detect_negation_mismatch(doc1: Doc, doc2: Doc) -> bool:
    def has_negation(doc):
        for token in doc:
            if token.dep_ == "neg" or token.lower_ in ["no", "not", "never", "neither", "nor", "without", "n't", "nothing" ]:
                return True
        return False

    return has_negation(doc1) != has_negation(doc2)

def is_temporal(word: str) -> bool:
    temporal_indicators = ["year", "month", "week", "day", "tomorrow", "yesterday", "next", "last", "future", "past", "now", "then", "soon", "later", "earlier", "after", "before", "when", "while", "during", "until", "since", "ago", "today", "tonight", "morning", "evening", "night", "hour", "minute", "second", "moment", "period", "time", "season", "spring", "summer", "fall", "autumn", "winter", "january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december", "dawn", "dusk", "noon", "midnight", "early", "late", "nowadays", "currently", "recently", "previously", "formerly", "suddenly", "immediately", "finally", "eventually", "always", "never", "sometimes", "often", "rarely", "seldom", "usually", "frequently", "occasionally", "constantly", "continuously", "periodically", "regularly", "daily", "weekly", "monthly", "annually", "hourly", "minutely", "secondly", "momentarily", "temporarily", "permanently", "briefly", "shortly", "long", "forever", "eternally"]
    return any(indicator in word.lower() for indicator in temporal_indicators)

def detect_temporal_shift(doc1: Doc, doc2: Doc) -> bool:
    temp1 = next((token.text for token in doc1 if is_temporal(token.text)), None)
    temp2 = next((token.text for token in doc2 if is_temporal(token.text)), None)
    return temp1 != temp2

def get_verb_tense(token):
    if token.tag_ in ["VB", "VBP"]:
        return "present"
    elif token.tag_ == "VBD":
        return "past"
    elif token.tag_ == "VBG":
        return "present_participle"
    elif token.tag_ == "VBN":
        return "past_participle"
    elif token.tag_ == "VBZ":
        return "present_3rd_person"
    elif token.tag_ == "MD":
        return "modal"
    else:
        return "other"

def detect_tense_change(doc1: Doc, doc2: Doc) -> float:
    verb1 = next((token for token in doc1 if token.pos_ == "VERB"), None)
    verb2 = next((token for token in doc2 if token.pos_ == "VERB"), None)
    
    if not (verb1 and verb2):
        return 0
    
    tense1 = get_verb_tense(verb1)
    tense2 = get_verb_tense(verb2)
    
    if tense1 == tense2:
        return 0
    elif (tense1 in ["present", "present_3rd_person"] and tense2 in ["present", "present_3rd_person"]) or \
         (tense1 in ["past", "past_participle"] and tense2 in ["past", "past_participle"]):
        return 0.1
    else:
        return 0.3

def calculate_entity_preservation(doc1: Doc, doc2: Doc) -> float:
    ents1 = set(ent.text.lower() for ent in doc1.ents)
    ents2 = set(ent.text.lower() for ent in doc2.ents)
    
    if not ents1:
        return 1.0
    
    return len(ents1.intersection(ents2)) / len(ents1)