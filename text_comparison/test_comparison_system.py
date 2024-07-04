# test_comparison_system.py

import spacy
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import BertTokenizer, BertModel
import torch
from comparison_system import TextComparisonSystem
from scoring_functions import score_summarization, score_similarity, score_paraphrase, score_contradiction

def minilm_similarity(model, text1, text2):
    embeddings = model.encode([text1, text2])
    return np.dot(embeddings[0], embeddings[1]) / (np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1]))

# def bert_similarity(tokenizer, model, text1, text2):
#     inputs1 = tokenizer(text1, return_tensors='pt', padding=True, truncation=True)
#     inputs2 = tokenizer(text2, return_tensors='pt', padding=True, truncation=True)
    
#     with torch.no_grad():
#         outputs1 = model(**inputs1)
#         outputs2 = model(**inputs2)
    
#     embeddings1 = outputs1.last_hidden_state.mean(dim=1)
#     embeddings2 = outputs2.last_hidden_state.mean(dim=1)
    
#     similarity = torch.nn.functional.cosine_similarity(embeddings1, embeddings2)
#     return similarity.item()

def run_tests():
    comparison_system = TextComparisonSystem()
    minilm_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    # bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # bert_model = BertModel.from_pretrained('bert-base-uncased')

    test_cases = [
        ("summarization", 
         "The cat chased the mouse around the house, jumping over furniture and knocking over a vase in the process.", 
         "A cat pursued a mouse in a house, causing some chaos."),
        ("similarity", 
         "The cat chased the mouse around the house", 
         "The mouse chased the cat around the house"),
        ("similarity", 
         "The cat chased the mouse around the house", 
         "The cat did not chase the mouse around the house"),
        ("summarization",
         "Despite heavy rain, the outdoor concert attracted a large crowd of enthusiastic fans who danced and sang along to every song.",
         "Many people attended a rainy concert."),
        ("similarity",
         "The old clock tower in the town square chimed twelve times at midnight.",
         "At midnight, the ancient timepiece in the village center rang out a dozen times."),
        ("similarity",
         "The stock market reached record highs yesterday.",
         "Yesterday, the financial markets achieved unprecedented peaks."),
        ("paraphrase",
         "The scientist discovered a new species of butterfly in the Amazon rainforest.",
         "A researcher found a previously unknown type of lepidopteran in the Amazonian jungle."),
        ("paraphrase",
         "She couldn't fall asleep because of the loud noise from the street.",
         "The racket from outside prevented her from drifting off to slumber."),
        ("contradiction",
         "The restaurant is open every day from 9 AM to 10 PM.",
         "The establishment is closed on Sundays and only operates until 8 PM on weekdays."),
        ("contradiction",
         "All students must complete the assignment by Friday.",
         "The homework is optional and can be submitted anytime next week."),
    ]

    for task, text1, text2 in test_cases:
        print(f"\nTask: {task}")
        print(f"Text 1: {text1}")
        print(f"Text 2: {text2}")
        comparator = TextComparisonSystem()

        results = comparator.compare_texts(text1, text2)
        interpretation = comparator.interpret_results(results)
    
        print(interpretation)

        minilm_score = minilm_similarity(minilm_model, text1, text2)
        #bert_score = bert_similarity(bert_tokenizer, bert_model, text1, text2)
        print(f"MiniLM score: {minilm_score:.4f}")
        #print(f"BERT score: {bert_score:.4f}")

        # if task == "contradiction":
        #     print("Better than baselines:", result['final_score'] > minilm_score and result['final_score'] > bert_score)
        # else:
        #     print("Close to or better than baselines:", 
        #           abs(result['final_score'] - minilm_score) < 0.1 or result['final_score'] < minilm_score,
        #           abs(result['final_score'] - bert_score) < 0.1 or result['final_score'] < bert_score)

    # Test custom penalties
    # text1 = "The cat chased the mouse."
    # text2 = "The mouse was chased by the cat."
    
    # print("\nTesting custom penalties:")
    # print(f"Text 1: {text1}")
    # print(f"Text 2: {text2}")

    # default_result = comparison_system.compare_texts(text1, text2, "similarity")
    # custom_result = comparison_system.compare_texts(text1, text2, "similarity", 
    #                                                 custom_penalties={"role_reversal": 0.1})
    
    # print(f"Default score: {default_result['final_score']:.4f}")
    # print(f"Custom penalties score: {custom_result['final_score']:.4f}")
    # print("Custom penalties improved score:", custom_result['final_score'] > default_result['final_score'])

    # # Test invalid task
    # try:
    #     comparison_system.compare_texts("Text 1", "Text 2", "invalid_task")
    # except ValueError as e:
    #     print("\nTesting invalid task:")
    #     print("Correctly raised ValueError:", str(e))

if __name__ == "__main__":
    run_tests()