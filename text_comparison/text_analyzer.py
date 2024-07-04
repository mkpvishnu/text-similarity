# text_comparison/text_analyzer.py

import spacy
from spacy.tokens import Doc, Span
from typing import Dict, Any, List, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
from nltk.corpus import wordnet, stopwords
import nltk
from dateutil.parser import parse as dateparse
from collections import Counter
import re
import textstat
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.stats import pearsonr
import string
from fuzzywuzzy import fuzz
import math
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline


nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('stopwords')

class TextAnalyzer:
    def __init__(self):
        self.nlp = spacy.load('en_core_web_trf')
        self.sentence_model = SentenceTransformer('all-mpnet-base-v2')  # More powerful than 'all-MiniLM-L6-v2'
        self.zero_shot_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        self.ner_model = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")

        try:
            import neuralcoref
            neuralcoref.add_to_pipe(self.nlp)
            self.coref_available = True
        except ImportError:
            print("Neuralcoref not available. Coreference resolution will be skipped.")
            self.coref_available = False

    def analyze(self, text1: str, text2: str) -> Dict[str, Any]:
        doc1 = self.safe_process(text1)
        doc2 = self.safe_process(text2)

        paragraphs1 = self.split_into_paragraphs(text1)
        paragraphs2 = self.split_into_paragraphs(text2)

        return {
            'voice_change': self.detect_voice_change(doc1, doc2),
            'role_reversal': self.detect_role_reversal(doc1, doc2),
            'negation': self.detect_negation_change(doc1, doc2),
            'number_change': self.detect_number_change(doc1, doc2),
            'synonym_usage': self.detect_synonym_usage(doc1, doc2),
            'anecdote_detection': self.detect_anecdote(doc1, doc2),
            'temporal_shift': self.detect_temporal_shift(doc1, doc2),
            'paragraph_structure': self.compare_paragraph_structure(paragraphs1, paragraphs2),
            'entity_recognition': self.compare_entities(doc1, doc2),
            'sentiment_consistency': self.compare_sentiment(doc1, doc2),
            'semantic_similarity': self.semantic_similarity(text1, text2),
            'syntactic_similarity': self.syntactic_similarity(doc1, doc2),
            'dependency_similarity': self.dependency_similarity(doc1, doc2),
            'readability_comparison': self.compare_readability(text1, text2),
            'topic_consistency': self.compare_topics(text1, text2),
            'style_similarity': self.compare_writing_style(text1, text2),
            'coherence_analysis': self.analyze_coherence(paragraphs1, paragraphs2),
            'information_density': self.compare_information_density(doc1, doc2),
            'argument_structure': self.compare_argument_structure(doc1, doc2),
            'contextual_consistency': self.analyze_contextual_consistency(doc1, doc2),
            'factual_consistency': self.compare_factual_consistency(doc1, doc2),
            'figurative_language': self.compare_figurative_language(doc1, doc2),
            'discourse_markers': self.compare_discourse_markers(doc1, doc2),
            'lexical_chain_similarity': self.compare_lexical_chains(doc1, doc2),
            'coreference_consistency': self.compare_coreference(doc1, doc2),
            'hedging_language': self.compare_hedging(doc1, doc2),
            'rhetorical_structure': self.compare_rhetorical_structure(doc1, doc2),
            'subjectivity_analysis': self.compare_subjectivity(doc1, doc2),
            'named_entity_consistency': self.compare_named_entities(doc1, doc2)
        }

    def safe_process(self, text: str) -> Doc:
        try:
            return self.nlp(text[:1000000])  # Limit to 1M chars to prevent memory issues
        except Exception as e:
            print(f"Error processing text: {e}")
            return Doc(self.nlp.vocab, words=text.split())

    def split_into_paragraphs(self, text: str) -> List[str]:
        return [p.strip() for p in re.split(r'\n\s*\n|\r\n\s*\r\n|\r\s*\r', text) if p.strip()]

    def detect_voice_change(self, doc1: Doc, doc2: Doc) -> float:
        def passive_ratio(doc):
            passive_count = sum(1 for sent in doc.sents if any(token.dep_ == "auxpass" for token in sent))
            total_sents = len(list(doc.sents))
            return passive_count / total_sents if total_sents > 0 else 0

        ratio1 = passive_ratio(doc1)
        ratio2 = passive_ratio(doc2)
        return abs(ratio1 - ratio2)

    def detect_role_reversal(self, doc1: Doc, doc2: Doc) -> float:
        def get_svo_triples(doc):
            triples = []
            for sent in doc.sents:
                subject = [token for token in sent if token.dep_ in ("nsubj", "nsubjpass")]
                verb = [token for token in sent if token.pos_ == "VERB"]
                obj = [token for token in sent if token.dep_ in ("dobj", "pobj")]
                if subject and verb and obj:
                    triples.append((subject[0].lemma_, verb[0].lemma_, obj[0].lemma_))
            return set(triples)

        triples1 = get_svo_triples(doc1)
        triples2 = get_svo_triples(doc2)
        
        reversed_count = sum(1 for (s1, v1, o1) in triples1 if (o1, v1, s1) in triples2)
        total_triples = len(triples1)
        
        return reversed_count / total_triples if total_triples > 0 else 0

    def detect_negation_change(self, doc1: Doc, doc2: Doc) -> float:
        neg_words = set(["not", "no", "never", "neither", "nor", "none", "nobody", "nowhere", "nothing"])
        
        def negation_ratio(doc):
            neg_count = sum(1 for token in doc if token.text.lower() in neg_words or token.dep_ == "neg")
            return neg_count / len(doc)

        ratio1 = negation_ratio(doc1)
        ratio2 = negation_ratio(doc2)
        return abs(ratio1 - ratio2)

    def detect_number_change(self, doc1: Doc, doc2: Doc) -> float:
        def number_ratio(doc):
            number_count = sum(1 for token in doc if token.pos_ == "NUM")
            return number_count / len(doc)

        ratio1 = number_ratio(doc1)
        ratio2 = number_ratio(doc2)
        return abs(ratio1 - ratio2)

    def detect_synonym_usage(self, doc1: Doc, doc2: Doc) -> float:
        words1 = [token.text.lower() for token in doc1 if not token.is_stop and token.is_alpha]
        words2 = [token.text.lower() for token in doc2 if not token.is_stop and token.is_alpha]
        
        embeddings1 = self.sentence_model.encode(words1)
        embeddings2 = self.sentence_model.encode(words2)
        
        similarity_matrix = cosine_similarity(embeddings1, embeddings2)
        max_similarities = np.max(similarity_matrix, axis=1)
        
        return np.mean(max_similarities)

    def detect_anecdote(self, doc1: Doc, doc2: Doc) -> float:
        personal_pronouns = set(['I', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours'])
        narrative_verbs = set(['said', 'told', 'recalled', 'remembered', 'experienced'])
        
        def anecdote_score(doc):
            pronoun_count = sum(1 for token in doc if token.text in personal_pronouns)
            verb_count = sum(1 for token in doc if token.lemma_ in narrative_verbs)
            return (pronoun_count + verb_count) / len(doc)

        score1 = anecdote_score(doc1)
        score2 = anecdote_score(doc2)
        return abs(score1 - score2)

    def detect_temporal_shift(self, doc1: Doc, doc2: Doc) -> float:
        def extract_temporal_info(doc):
            temporal_words = set(['yesterday', 'today', 'tomorrow', 'now', 'then', 'soon', 'later'])
            dates = [ent.text for ent in doc.ents if ent.label_ == 'DATE']
            temporal = [token.text for token in doc if token.text.lower() in temporal_words]
            return set(dates + temporal)

        temp1 = extract_temporal_info(doc1)
        temp2 = extract_temporal_info(doc2)
        
        total_temporal = len(temp1.union(temp2))
        if total_temporal == 0:
            return 0
        return 1 - len(temp1.intersection(temp2)) / total_temporal

    def compare_paragraph_structure(self, paragraphs1: List[str], paragraphs2: List[str]) -> float:
        if not paragraphs1 or not paragraphs2:
            return 0.0

        avg_len1 = sum(len(p.split()) for p in paragraphs1) / len(paragraphs1)
        avg_len2 = sum(len(p.split()) for p in paragraphs2) / len(paragraphs2)

        len_similarity = min(avg_len1, avg_len2) / max(avg_len1, avg_len2)
        count_similarity = min(len(paragraphs1), len(paragraphs2)) / max(len(paragraphs1), len(paragraphs2))

        return (len_similarity + count_similarity) / 2

    def compare_entities(self, doc1: Doc, doc2: Doc) -> float:
        text1 = doc1.text
        text2 = doc2.text
        
        if not text1 or not text2:
            return 0.0
        
        ents1 = self.ner_model(text1)
        ents2 = self.ner_model(text2)
        
        def get_entity_types(ents):
            return set(ent['entity'] for ent in ents)
        
        types1 = get_entity_types(ents1)
        types2 = get_entity_types(ents2)
        
        return len(types1.intersection(types2)) / max(len(types1), len(types2)) if max(len(types1), len(types2)) > 0 else 1.0

    def compare_sentiment(self, doc1: Doc, doc2: Doc) -> float:
        def get_sentiment(doc):
            return sum(token.sentiment for token in doc)
        
        sent1 = get_sentiment(doc1) / len(doc1)
        sent2 = get_sentiment(doc2) / len(doc2)
        
        return 1 - abs(sent1 - sent2)

    def semantic_similarity(self, text1: str, text2: str) -> float:
        embeddings = self.sentence_model.encode([text1, text2])
        return cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

    def syntactic_similarity(self, doc1: Doc, doc2: Doc) -> float:
        pos1 = [token.pos_ for token in doc1 if token.pos_ != ""]
        pos2 = [token.pos_ for token in doc2 if token.pos_ != ""]
        
        if not pos1 or not pos2:
            return 0.0
        
        return sum(a == b for a, b in zip(pos1, pos2)) / max(len(pos1), len(pos2))

    def dependency_similarity(self, doc1: Doc, doc2: Doc) -> float:
        deps1 = Counter(token.dep_ for token in doc1 if token.dep_ != "")
        deps2 = Counter(token.dep_ for token in doc2 if token.dep_ != "")
        
        if not deps1 or not deps2:
            return 0.0
        
        all_deps = set(deps1.keys()).union(deps2.keys())
        return sum(min(deps1.get(dep, 0), deps2.get(dep, 0)) for dep in all_deps) / max(sum(deps1.values()), sum(deps2.values()))

    def compare_readability(self, text1: str, text2: str) -> float:
        score1 = textstat.flesch_reading_ease(text1)
        score2 = textstat.flesch_reading_ease(text2)
        return 1 - abs(score1 - score2) / 100  # Normalize to [0, 1]

    def compare_topics(self, text1: str, text2: str) -> float:
        candidate_labels = ["science", "nature", "discovery", "research", "technology", "environment", "health", "medicine", "biology", "physics", "chemistry", "astronomy", "geology", "ecology", "climate", "energy", "sustainability", "innovation", "engineering", "mathematics", "education", "society", "culture", "politics", "economics", "history", "philosophy", "religion", "psychology", "sociology", "anthropology", "linguistics", "literature", "art", "music", "film", "theater", "television", "sports", "games", "fashion", "food", "travel", "lifestyle", "health", "fitness", "wellness", "beauty", "parenting", "relationships", "dating", "marriage", "family", "children", "pets", "home", "garden", "decor", "cooking", "baking", "crafts", "DIY", "furniture", "fashion", "style", "shopping", "beauty", "hair", "makeup", "skincare", "nails", "fitness", "exercise", "nutrition", "diet", "wellness", "mental health", "meditation", "yoga", "self-care", "self-improvement", "motivation", "inspiration", "productivity", "success", "leadership", "management", "entrepreneurship", "business", "marketing", "sales", "finance", "investing", "economics", "technology", "innovation", "startups", "work", "career", "jobs", "interviews", "resumes", "networking", "education", "learning", "teaching", "students", "schools", "college", "university", "online learning", "e-learning", "edtech", "classroom", "homework", "exams", "studying", "research", "science", "technology", "engineering", "mathematics", "STEM", "coding", "programming", "software", "hardware", "data", "analytics", "machine learning", "artificial intelligence", "robotics", "automation", "cybersecurity", "networking", "cloud computing", "blockchain", "fintech", "healthtech", "biotech", "agritech", "cleantech", "energy", "sustainability", "environment", "climate", "conservation", "recycling", "renewables", "green", "organic", "natural", "wildlife", "biodiversity", "ecosystems", "pollution", "waste", "plastic", "oceans", "forests", "deserts", "mountains", "rivers", "lakes", "parks", "gardens", "urban", "rural", "global", "local", "community", "society", "culture", "politics", "government", "economy", "business", "industry", "commerce", "trade", "markets", "finance", "banking", "investing", "insurance", "taxes", "budgets", "debt", "wealth", "poverty", "inequality", "justice", "law", "crime", "punishment", "rights", "freedom", "democracy", "dictatorship", "monarchy", "republic", "federalism", "parliament", "presidency", "congress", "courts", "lawyers", "judges", "police", "military", "security", "defense", "diplomacy", "war", "peace"]
        result1 = self.zero_shot_classifier(text1, candidate_labels)
        result2 = self.zero_shot_classifier(text2, candidate_labels)
        
        scores1 = dict(zip(result1['labels'], result1['scores']))
        scores2 = dict(zip(result2['labels'], result2['scores']))
        
        similarity = sum(scores1[label] * scores2[label] for label in candidate_labels)
        return similarity / (sum(scores1.values()) * sum(scores2.values())) ** 0.5


    def compare_writing_style(self, text1: str, text2: str) -> float:
        def get_style_features(text):
            sentences = nltk.sent_tokenize(text)
            words = nltk.word_tokenize(text)
            return {
                'avg_sentence_length': np.mean([len(nltk.word_tokenize(sent)) for sent in sentences]) if sentences else 0,
                'avg_word_length': np.mean([len(word) for word in words]) if words else 0,
                'lexical_diversity': len(set(words)) / len(words) if words else 0,
                'punctuation_ratio': sum(1 for char in text if char in string.punctuation) / len(text) if text else 0
            }
        
        features1 = get_style_features(text1)
        features2 = get_style_features(text2)
        
        differences = []
        for k in features1:
            if features1[k] == 0 and features2[k] == 0:
                differences.append(0)  # Both are zero, consider them identical
            elif features1[k] == 0 or features2[k] == 0:
                differences.append(1)  # One is zero and the other isn't, consider them completely different
            else:
                differences.append(abs(features1[k] - features2[k]) / max(features1[k], features2[k]))
        
        return 1 - (sum(differences) / len(differences)) if differences else 1

    def analyze_coherence(self, paragraphs1: List[str], paragraphs2: List[str]) -> float:
        def get_coherence_score(paragraphs):
            if len(paragraphs) < 2:
                return 1.0
            scores = []
            for i in range(len(paragraphs) - 1):
                score = self.semantic_similarity(paragraphs[i], paragraphs[i+1])
                scores.append(score)
            return sum(scores) / len(scores)
        
        score1 = get_coherence_score(paragraphs1)
        score2 = get_coherence_score(paragraphs2)
        
        return 1 - abs(score1 - score2)

    def compare_information_density(self, doc1: Doc, doc2: Doc) -> float:
        def get_density(doc):
            content_words = [token for token in doc if token.pos_ in ['NOUN', 'VERB', 'ADJ', 'ADV'] and not token.is_stop]
            return len(content_words) / len(doc)
        
        density1 = get_density(doc1)
        density2 = get_density(doc2)
        
        return 1 - abs(density1 - density2)

    def compare_argument_structure(self, doc1: Doc, doc2: Doc) -> float:
        def get_argument_structure(doc):
            conjunctions = [token for token in doc if token.dep_ in ['cc', 'conj']]
            causal_markers = [token for token in doc if token.text.lower() in ['because', 'therefore', 'thus', 'hence', 'so']]
            return len(conjunctions) + len(causal_markers)
        
        struct1 = get_argument_structure(doc1)
        struct2 = get_argument_structure(doc2)
        
        return 1 - abs(struct1 - struct2) / max(struct1, struct2) if max(struct1, struct2) > 0 else 1

    def analyze_contextual_consistency(self, doc1: Doc, doc2: Doc) -> float:
        if doc1.text == doc2.text:
            return 1.0
        return self.semantic_similarity(doc1.text, doc2.text)

    def compare_factual_consistency(self, doc1: Doc, doc2: Doc) -> float:
        facts1 = set((ent.text, ent.label_) for ent in doc1.ents)
        facts2 = set((ent.text, ent.label_) for ent in doc2.ents)
        
        if not facts1 and not facts2:
            return 1.0
        elif not facts1 or not facts2:
            return 0.0
        
        return len(facts1.intersection(facts2)) / max(len(facts1), len(facts2))

    def compare_figurative_language(self, doc1: Doc, doc2: Doc) -> float:
        if doc1.text == doc2.text:
            return 1.0
        def detect_figurative(doc):
            potential_figurative = []
            for sent in doc.sents:
                for token in sent:
                    if token.pos_ in ['NOUN', 'VERB'] and len(wordnet.synsets(token.text)) > 1:
                        potential_figurative.append(token.text)
            return set(potential_figurative)
        
        fig1 = detect_figurative(doc1)
        fig2 = detect_figurative(doc2)
        
        return len(fig1.symmetric_difference(fig2)) / (len(fig1) + len(fig2)) if (len(fig1) + len(fig2)) > 0 else 1

    def compare_discourse_markers(self, doc1: Doc, doc2: Doc) -> float:
        discourse_markers = set(['however', 'moreover', 'therefore', 'thus', 'consequently', 'nevertheless'])
        
        def count_markers(doc):
            return sum(1 for token in doc if token.text.lower() in discourse_markers)
        
        count1 = count_markers(doc1)
        count2 = count_markers(doc2)
        
        return 1 - abs(count1 - count2) / max(count1, count2) if max(count1, count2) > 0 else 1

    def compare_lexical_chains(self, doc1: Doc, doc2: Doc) -> float:
        def build_chain(doc):
            return [token.lemma_ for token in doc if token.pos_ in ['NOUN', 'VERB', 'ADJ'] and not token.is_stop]
        
        chain1 = build_chain(doc1)
        chain2 = build_chain(doc2)
        
        embeddings1 = self.sentence_model.encode(chain1)
        embeddings2 = self.sentence_model.encode(chain2)
        
        similarity_matrix = cosine_similarity(embeddings1, embeddings2)
        return np.mean(np.max(similarity_matrix, axis=1))
    
    def compare_coreference(self, doc1: Doc, doc2: Doc) -> float:
        if not self.coref_available:
            return 1.0  # Return perfect score if coreference resolution is not available
        
        def get_coreference_chains(doc):
            return [len(chain) for chain in doc._.coref_chains]
        
        chains1 = get_coreference_chains(doc1)
        chains2 = get_coreference_chains(doc2)
        
        if not chains1 and not chains2:
            return 1.0
        elif not chains1 or not chains2:
            return 0.0
        
        return 1 - abs(np.mean(chains1) - np.mean(chains2)) / max(np.mean(chains1), np.mean(chains2))

    def compare_hedging(self, doc1: Doc, doc2: Doc) -> float:
        hedging_words = set(['may', 'might', 'could', 'perhaps', 'possibly', 'probably', 'seemingly'])
        
        def count_hedging(doc):
            return sum(1 for token in doc if token.text.lower() in hedging_words)
        
        count1 = count_hedging(doc1)
        count2 = count_hedging(doc2)
        
        return 1 - abs(count1 - count2) / max(count1, count2) if max(count1, count2) > 0 else 1

    def compare_rhetorical_structure(self, doc1: Doc, doc2: Doc) -> float:
        def get_rhetorical_structure(doc):
            structure = []
            for sent in doc.sents:
                if any(token.text.lower() in ['if', 'when', 'because'] for token in sent):
                    structure.append('condition')
                elif any(token.text.lower() in ['but', 'however', 'nevertheless'] for token in sent):
                    structure.append('contrast')
                elif any(token.text.lower() in ['therefore', 'thus', 'consequently'] for token in sent):
                    structure.append('result')
                else:
                    structure.append('statement')
            return structure
        
        struct1 = get_rhetorical_structure(doc1)
        struct2 = get_rhetorical_structure(doc2)
        
        return fuzz.token_sort_ratio(struct1, struct2) / 100

    def compare_subjectivity(self, doc1: Doc, doc2: Doc) -> float:
        subjective_words = set(['think', 'believe', 'feel', 'suggest', 'assume', 'consider'])
        
        def subjectivity_score(doc):
            return sum(1 for token in doc if token.lemma_.lower() in subjective_words) / len(doc)
        
        score1 = subjectivity_score(doc1)
        score2 = subjectivity_score(doc2)
        
        return 1 - abs(score1 - score2)

    def compare_named_entities(self, doc1: Doc, doc2: Doc) -> float:
        def get_named_entities(doc):
            return set((ent.text, ent.label_) for ent in doc.ents)
        
        ents1 = get_named_entities(doc1)
        ents2 = get_named_entities(doc2)
        
        return len(ents1.intersection(ents2)) / max(len(ents1), len(ents2)) if max(len(ents1), len(ents2)) > 0 else 1

def compare_texts(text1: str, text2: str, analyzer: TextAnalyzer) -> Dict[str, Any]:
    return analyzer.analyze(text1, text2)