import pytest
from text_comparison.comparison_system import TextComparisonSystem
from sentence_transformers import SentenceTransformer
from transformers import BertTokenizer, BertModel
import torch
import numpy as np

@pytest.fixture
def comparison_system():
    return TextComparisonSystem()

@pytest.fixture
def minilm_model():
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

@pytest.fixture
def bert_model():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    return tokenizer, model

def minilm_similarity(model, text1, text2):
    embeddings = model.encode([text1, text2])
    return np.dot(embeddings[0], embeddings[1]) / (np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1]))

def bert_similarity(tokenizer, model, text1, text2):
    inputs1 = tokenizer(text1, return_tensors='pt', padding=True, truncation=True)
    inputs2 = tokenizer(text2, return_tensors='pt', padding=True, truncation=True)
    
    with torch.no_grad():
        outputs1 = model(**inputs1)
        outputs2 = model(**inputs2)
    
    embeddings1 = outputs1.last_hidden_state.mean(dim=1)
    embeddings2 = outputs2.last_hidden_state.mean(dim=1)
    
    similarity = torch.nn.functional.cosine_similarity(embeddings1, embeddings2)
    return similarity.item()

@pytest.mark.parametrize("task,text1,text2,expected_better", [
    ("summarization", 
     "The cat chased the mouse around the house, jumping over furniture and knocking over a vase in the process.", 
     "A cat pursued a mouse in a house, causing some chaos.", 
     True),
    ("summarization",
     "Despite heavy rain, the outdoor concert attracted a large crowd of enthusiastic fans who danced and sang along to every song.",
     "Many people attended a rainy concert.",
     True),
    ("similarity",
     "The old clock tower in the town square chimed twelve times at midnight.",
     "At midnight, the ancient timepiece in the village center rang out a dozen times.",
     True),
    ("similarity",
     "The stock market reached record highs yesterday.",
     "Yesterday, the financial markets achieved unprecedented peaks.",
     True),
    ("paraphrase",
     "The scientist discovered a new species of butterfly in the Amazon rainforest.",
     "A researcher found a previously unknown type of lepidopteran in the Amazonian jungle.",
     True),
    ("paraphrase",
     "She couldn't fall asleep because of the loud noise from the street.",
     "The racket from outside prevented her from drifting off to slumber.",
     True),
    ("contradiction",
     "The restaurant is open every day from 9 AM to 10 PM.",
     "The establishment is closed on Sundays and only operates until 8 PM on weekdays.",
     True),
    ("contradiction",
     "All students must complete the assignment by Friday.",
     "The homework is optional and can be submitted anytime next week.",
     True),
])
def test_comparison_system(comparison_system, minilm_model, bert_model, task, text1, text2, expected_better):
    result = comparison_system.compare_texts(text1, text2, task)
    minilm_score = minilm_similarity(minilm_model, text1, text2)
    bert_score = bert_similarity(bert_model[0], bert_model[1], text1, text2)
    
    print(f"\nTask: {task}")
    print(f"Text 1: {text1}")
    print(f"Text 2: {text2}")
    print(f"Our model score: {result['final_score']:.4f}")
    print(f"MiniLM score: {minilm_score:.4f}")
    print(f"BERT score: {bert_score:.4f}")
    
    if task == "contradiction":
        assert result['final_score'] > minilm_score and result['final_score'] > bert_score
    else:
        if expected_better:
            assert abs(result['final_score'] - minilm_score) < 0.1 or result['final_score'] < minilm_score
            assert abs(result['final_score'] - bert_score) < 0.1 or result['final_score'] < bert_score
        else:
            assert abs(result['final_score'] - minilm_score) < 0.1 or result['final_score'] > minilm_score
            assert abs(result['final_score'] - bert_score) < 0.1 or result['final_score'] > bert_score

def test_custom_penalties(comparison_system):
    text1 = "The cat chased the mouse."
    text2 = "The mouse was chased by the cat."
    
    default_result = comparison_system.compare_texts(text1, text2, "similarity")
    custom_result = comparison_system.compare_texts(text1, text2, "similarity", 
                                                    custom_penalties={"role_reversal": 0.1})
    
    assert custom_result['final_score'] > default_result['final_score']

def test_invalid_task(comparison_system):
    with pytest.raises(ValueError):
        comparison_system.compare_texts("Text 1", "Text 2", "invalid_task")

if __name__ == "__main__":
    pytest.main([__file__])