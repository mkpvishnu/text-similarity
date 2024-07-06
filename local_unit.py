from lingualens.analyzer import TextAnalyzer 
from lingualens.metrics import Metrics
from lingualens.pipelines import check_similarity
from lingualens.utils import ensure_model

# Ensure the desired Spacy model is downloaded
spacy_model = ensure_model("en_core_web_sm")

# Using individual metrics
analyzer = TextAnalyzer(spacy_model)
metrics = Metrics(analyzer)

def run_temporal_shift_test(text1, text2, description, analyzer, metrics):
    print(f"\n--- Test Case: {description} ---")
    print(f"Text 1: {text1}")
    print(f"Text 2: {text2}")
    result = metrics.detect_temporal_shift(text1, text2, print_info=True, sentence_model='all-MiniLM-L12-v2')
    print(f"Temporal Shift Score: {result}")
    print("-" * 50)

# Test cases
test_cases = [
    # Positive case (high similarity)
    {
        "description": "Positive Case - High Similarity",
        "text1": "Yesterday, I went to the store. Today, I'm staying home.",
        "text2": "I visited the store yesterday. I am at home today."
    },
    
    # Negative case (low similarity)
    {
        "description": "Negative Case - Low Similarity",
        "text1": "Last week, we had a picnic in the park. Tomorrow, we're going to the beach.",
        "text2": "Next month, I'm starting a new job. Last year, I graduated from college."
    },
    
    # Complicated scenario
    {
        "description": "Complicated Scenario",
        "text1": "In 2010, the company was founded. By 2015, it had expanded to three countries. Now, in 2023, it's a global leader.",
        "text2": "The company's journey began 13 years ago. Five years later, it had a presence in multiple nations. Currently, it dominates the global market."
    },
    
    # Summary
    {
        "description": "Summary",
        "text1": "On Monday, John went to work early. Tuesday, he had a long meeting. Wednesday was a day off. Thursday and Friday were busy with project deadlines.",
        "text2": "John had a busy work week, with only Wednesday off."
    },
    
    # Paraphrase
    {
        "description": "Paraphrase",
        "text1": "The ancient civilization flourished 3000 years ago. It suddenly collapsed within a decade, leaving behind mysterious ruins.",
        "text2": "Three millennia in the past, a great society thrived. Its abrupt downfall over just ten years left enigmatic remnants for future generations to ponder."
    },
    
    # Mixed tenses
    {
        "description": "Mixed Tenses",
        "text1": "I am going to the party tomorrow. I went to the store yesterday. I am cooking dinner now.",
        "text2": "Yesterday, shopping was done. Currently, meal preparation is underway. The upcoming day holds party plans."
    }
]

# Run tests
for case in test_cases:
    run_temporal_shift_test(case["text1"], case["text2"], case["description"], analyzer, metrics)
    
# # Using pipeline
# similarity_results = check_similarity(text1, text2, spacy_model="en_core_web_sm")
# print("Similarity check results:", similarity_results)