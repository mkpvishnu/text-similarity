# text_comparison/comparison_system.py

from typing import Dict, Any
from text_analyzer import TextAnalyzer
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import numpy as np

class TextComparisonSystem:
    def __init__(self):
        self.analyzer = TextAnalyzer()
        self.weights = {
            'semantic_similarity': 0.2,
            'syntactic_similarity': 0.1,
            'entity_recognition': 0.1,
            'topic_consistency': 0.1,
            'style_similarity': 0.1,
            'coherence_analysis': 0.1,
            'factual_consistency': 0.1,
            'contextual_consistency': 0.1,
            'named_entity_consistency': 0.1
        }

    def compare_texts(self, text1: str, text2: str) -> Dict[str, Any]:
        if text1 == text2:
            return {metric: 1.0 for metric in [
                'voice_change', 'role_reversal', 'negation', 'number_change', 'synonym_usage',
                'anecdote_detection', 'temporal_shift', 'paragraph_structure', 'entity_recognition',
                'sentiment_consistency', 'semantic_similarity', 'syntactic_similarity',
                'dependency_similarity', 'readability_comparison', 'topic_consistency',
                'style_similarity', 'coherence_analysis', 'information_density',
                'argument_structure', 'contextual_consistency', 'factual_consistency',
                'figurative_language', 'discourse_markers', 'lexical_chain_similarity',
                'coreference_consistency', 'hedging_language', 'rhetorical_structure',
                'subjectivity_analysis', 'named_entity_consistency', 'overall_similarity'
            ]}

        doc1 = self.analyzer.safe_process(text1)
        doc2 = self.analyzer.safe_process(text2)

        paragraphs1 = self.analyzer.split_into_paragraphs(text1)
        paragraphs2 = self.analyzer.split_into_paragraphs(text2)

        results = {
            'voice_change': self.analyzer.detect_voice_change(doc1, doc2),
            'role_reversal': self.analyzer.detect_role_reversal(doc1, doc2),
            'negation': self.analyzer.detect_negation_change(doc1, doc2),
            'number_change': self.analyzer.detect_number_change(doc1, doc2),
            'synonym_usage': self.analyzer.detect_synonym_usage(doc1, doc2),
            'anecdote_detection': self.analyzer.detect_anecdote(doc1, doc2),
            'temporal_shift': self.analyzer.detect_temporal_shift(doc1, doc2),
            'paragraph_structure': self.analyzer.compare_paragraph_structure(paragraphs1, paragraphs2),
            'entity_recognition': self.analyzer.compare_entities(doc1, doc2),
            'sentiment_consistency': self.analyzer.compare_sentiment(doc1, doc2),
            'semantic_similarity': self.analyzer.semantic_similarity(text1, text2),
            'syntactic_similarity': self.analyzer.syntactic_similarity(doc1, doc2),
            'dependency_similarity': self.analyzer.dependency_similarity(doc1, doc2),
            'readability_comparison': self.analyzer.compare_readability(text1, text2),
            'topic_consistency': self.analyzer.compare_topics(text1, text2),
            'style_similarity': self.analyzer.compare_writing_style(text1, text2),
            'coherence_analysis': self.analyzer.analyze_coherence(paragraphs1, paragraphs2),
            'information_density': self.analyzer.compare_information_density(doc1, doc2),
            'argument_structure': self.analyzer.compare_argument_structure(doc1, doc2),
            'contextual_consistency': self.analyzer.analyze_contextual_consistency(doc1, doc2),
            'factual_consistency': self.analyzer.compare_factual_consistency(doc1, doc2),
            'figurative_language': self.analyzer.compare_figurative_language(doc1, doc2),
            'discourse_markers': self.analyzer.compare_discourse_markers(doc1, doc2),
            'lexical_chain_similarity': self.analyzer.compare_lexical_chains(doc1, doc2),
            'hedging_language': self.analyzer.compare_hedging(doc1, doc2),
            'rhetorical_structure': self.analyzer.compare_rhetorical_structure(doc1, doc2),
            'subjectivity_analysis': self.analyzer.compare_subjectivity(doc1, doc2),
            'named_entity_consistency': self.analyzer.compare_named_entities(doc1, doc2)
        }

        if hasattr(self.analyzer, 'coref_available') and self.analyzer.coref_available:
            results['coreference_consistency'] = self.analyzer.compare_coreference(doc1, doc2)
        else:
            results['coreference_consistency'] = None

        # Handle potential NaN or inf values
        for key, value in results.items():
            if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
                results[key] = 0.0  # or another appropriate default value


        # Calculate overall similarity score using weighted average
        weighted_scores = [
            (results[metric], weight) 
            for metric, weight in self.weights.items() 
            if metric in results and isinstance(results[metric], (int, float)) and not np.isnan(results[metric]) and not np.isinf(results[metric])
        ]
        
        if weighted_scores:
            results['overall_similarity'] = sum(score * weight for score, weight in weighted_scores) / sum(weight for _, weight in weighted_scores)
        else:
            results['overall_similarity'] = 0

        return results

    def interpret_results(self, results: Dict[str, Any]) -> str:
        """
        Interpret the comparison results and provide a human-readable summary.
        
        Args:
            results (Dict[str, Any]): The dictionary of comparison results
        
        Returns:
            str: A human-readable interpretation of the results
        """
        interpretation = "Text Comparison Results:\n\n"
        
        for key, value in results.items():
            if isinstance(value, float):
                interpretation += f"{key.replace('_', ' ').title()}: {value:.2f}\n"
            else:
                interpretation += f"{key.replace('_', ' ').title()}: {value}\n"
        
        interpretation += f"\nOverall Similarity: {results['overall_similarity']:.2f}\n"
        
        if results['overall_similarity'] > 0.8:
            interpretation += "\nInterpretation: The texts are highly similar."
        elif results['overall_similarity'] > 0.6:
            interpretation += "\nInterpretation: The texts have moderate similarity."
        elif results['overall_similarity'] > 0.4:
            interpretation += "\nInterpretation: The texts have some similarities but significant differences."
        else:
            interpretation += "\nInterpretation: The texts are largely dissimilar."
        
        return interpretation

# Usage example
if __name__ == "__main__":
    comparator = TextComparisonSystem()
    
    text1 = "The quick brown fox jumps over the lazy dog. It was a sunny day in the forest."
    text2 = "The quick brown fox jumps over the lazy cat. It was a sunny day in the forest."
    print(TextAnalyzer.compare_topics(text1, text2))
    
    # results = comparator.compare_texts(text1, text2)
    # interpretation = comparator.interpret_results(results)
    
    # print(interpretation)
    
    # #default pipelines
    # analyzer.check_summary(original, summary)
    # analyzer.check_paraphrase(original, paraphrased)
    # analyzer.check_similarity(text1, text2)
    
    # #single metrics
    # analyzer.metric.
    
    