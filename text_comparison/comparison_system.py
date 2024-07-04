# text_comparison/comparison_system.py

from typing import Dict, Any
from text_analyzer import TextAnalyzer

class TextComparisonSystem:
    def __init__(self):
        self.analyzer = TextAnalyzer()
        self.weights = {
            'semantic_similarity': 0.25,
            'entity_recognition': 0.15,
            'topic_consistency': 0.15,
            'contextual_consistency': 0.15,
            'factual_consistency': 0.15,
            'synonym_usage': 0.15
        }

    def compare_texts(self, text1: str, text2: str, task: str) -> Dict[str, Any]:
        doc1 = self.analyzer.safe_process(text1)
        doc2 = self.analyzer.safe_process(text2)

        results = {
            'semantic_similarity': self.analyzer.semantic_similarity(text1, text2),
            'entity_recognition': self.analyzer.compare_entities(doc1, doc2),
            'topic_consistency': self.analyzer.compare_topics(doc1, doc2),
            'contextual_consistency': self.analyzer.analyze_contextual_consistency(doc1, doc2),
            'factual_consistency': self.analyzer.compare_factual_consistency(doc1, doc2),
            'synonym_usage': self.analyzer.detect_synonym_usage(doc1, doc2)
        }

        # Calculate overall similarity score using weighted average
        weighted_scores = [
            (results[metric], weight) 
            for metric, weight in self.weights.items() 
            if metric in results and isinstance(results[metric], (int, float))
        ]
        
        if weighted_scores:
            results['overall_similarity'] = sum(score * weight for score, weight in weighted_scores) / sum(weight for _, weight in weighted_scores)
        else:
            results['overall_similarity'] = 0

        results['interpretation'] = self.interpret_results(results, task)

        return results

    def interpret_results(self, results: Dict[str, float], task: str) -> str:
        if task == 'summarization':
            return self.interpret_summarization(results)
        elif task == 'paraphrasing':
            return self.interpret_paraphrasing(results)
        elif task == 'similarity':
            return self.interpret_similarity(results)
        else:
            return "Unknown task type."

    def interpret_summarization(self, results: Dict[str, float]) -> str:
        overall_score = results['overall_similarity']
        interpretation = f"Overall Summary Quality: {overall_score:.2f}\n\n"

        if results['semantic_similarity'] < 0.5:
            interpretation += "The summary may not capture the main ideas of the original text effectively.\n"
        elif results['semantic_similarity'] > 0.8:
            interpretation += "The summary captures the main ideas of the original text very well.\n"

        if results['factual_consistency'] < 0.7:
            interpretation += "There might be factual inconsistencies between the summary and the original text.\n"

        if results['entity_recognition'] < 0.6:
            interpretation += "Some important entities from the original text may be missing in the summary.\n"

        if overall_score < 0.5:
            interpretation += "Overall, the summary needs significant improvement.\n"
        elif overall_score < 0.7:
            interpretation += "The summary is adequate but could be improved.\n"
        else:
            interpretation += "Overall, this is a good quality summary.\n"

        return interpretation

    def interpret_paraphrasing(self, results: Dict[str, float]) -> str:
        overall_score = results['overall_similarity']
        interpretation = f"Overall Paraphrase Quality: {overall_score:.2f}\n\n"

        if results['semantic_similarity'] < 0.7:
            interpretation += "The paraphrase may not maintain the original meaning effectively.\n"
        elif results['semantic_similarity'] > 0.95:
            interpretation += "The paraphrase might be too similar to the original text.\n"

        if results['synonym_usage'] < 0.3:
            interpretation += "The paraphrase could use more synonym substitutions.\n"
        elif results['synonym_usage'] > 0.8:
            interpretation += "Good use of synonyms in the paraphrase.\n"

        if results['factual_consistency'] < 0.9:
            interpretation += "Ensure all facts from the original text are preserved in the paraphrase.\n"

        if overall_score < 0.6:
            interpretation += "Overall, the paraphrase needs significant improvement.\n"
        elif overall_score < 0.8:
            interpretation += "The paraphrase is adequate but could be improved.\n"
        else:
            interpretation += "Overall, this is a good quality paraphrase.\n"

        return interpretation

    def interpret_similarity(self, results: Dict[str, float]) -> str:
        overall_score = results['overall_similarity']
        interpretation = f"Overall Similarity: {overall_score:.2f}\n\n"

        if overall_score < 0.3:
            interpretation += "The texts are highly dissimilar.\n"
        elif overall_score < 0.6:
            interpretation += "The texts have some similarities but significant differences.\n"
        elif overall_score < 0.8:
            interpretation += "The texts are moderately similar.\n"
        else:
            interpretation += "The texts are highly similar.\n"

        if results['semantic_similarity'] > 0.9 and results['factual_consistency'] < 0.7:
            interpretation += "The texts are semantically similar but may differ in specific facts or details.\n"

        if results['topic_consistency'] < 0.5:
            interpretation += "The texts may be discussing different topics.\n"

        return interpretation

# Usage example
if __name__ == "__main__":
    comparator = TextComparisonSystem()
    
    text1 = "The quick brown fox jumps over the lazy dog. It was a sunny day in the forest."
    text2 = "The slow brown fox jumps over the lazy dog. It was a sunny day in the forest."
    
    results = comparator.compare_texts(text1, text2, "summarization")
    interpretation = comparator.interpret_results(results)
    
    print(interpretation)