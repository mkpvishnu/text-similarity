# text_comparison/semantic_analysis.py

import torch
from transformers import BertTokenizer, BertForMaskedLM
import numpy as np
from scipy.spatial.distance import cosine

class SemanticAnalyzer:
    def __init__(self, model_name='bert-base-uncased'):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForMaskedLM.from_pretrained(model_name)
        self.model.eval()

    def get_word_embedding(self, word, context):
        # Tokenize the context
        tokens = self.tokenizer.tokenize(context)
        # Find the start and end position of the word in the tokenized context
        word_tokens = self.tokenizer.tokenize(word)
        start_idx = -1
        for i in range(len(tokens) - len(word_tokens) + 1):
            if tokens[i:i+len(word_tokens)] == word_tokens:
                start_idx = i
                break
        if start_idx == -1:
            return None  # Word not found in context

        end_idx = start_idx + len(word_tokens)

        # Encode the context
        input_ids = self.tokenizer.encode(context, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.model(input_ids, output_hidden_states=True)
        
        # Use the hidden state from the last layer
        last_hidden_state = outputs.hidden_states[-1].squeeze(0)
        
        # Get the embeddings for the word tokens
        word_embeddings = last_hidden_state[start_idx:end_idx]
        return word_embeddings.mean(dim=0).numpy()

    def calculate_word_similarity(self, word1, context1, word2, context2):
        emb1 = self.get_word_embedding(word1, context1)
        emb2 = self.get_word_embedding(word2, context2)
        if emb1 is None or emb2 is None:
            return 0  # Return 0 similarity if either word is not found in its context
        return 1 - cosine(emb1, emb2)

    def disambiguate_word(self, word, context):
        # Tokenize the context
        tokens = self.tokenizer.tokenize(context)
        # Find the position of the word in the tokenized context
        word_tokens = self.tokenizer.tokenize(word)
        start_idx = -1
        for i in range(len(tokens) - len(word_tokens) + 1):
            if tokens[i:i+len(word_tokens)] == word_tokens:
                start_idx = i
                break
        if start_idx == -1:
            return []  # Word not found in context

        # Prepare input by replacing the target word with [MASK] token
        masked_tokens = tokens.copy()
        masked_tokens[start_idx:start_idx+len(word_tokens)] = ['[MASK]'] * len(word_tokens)
        masked_text = self.tokenizer.convert_tokens_to_string(masked_tokens)

        input_ids = self.tokenizer.encode(masked_text, return_tensors="pt")
        mask_token_index = torch.where(input_ids == self.tokenizer.mask_token_id)[1]

        with torch.no_grad():
            outputs = self.model(input_ids)

        predictions = outputs[0][0, mask_token_index].topk(10)
        
        return [self.tokenizer.decode([pred_id]) for pred_id in predictions.indices[0]]

def calculate_semantic_similarity(doc1, doc2, analyzer):
    similarities = []
    for token1 in doc1:
        if token1.pos_ in ['NOUN', 'VERB', 'ADJ', 'ADV']:
            best_similarity = 0
            for token2 in doc2:
                if token2.pos_ == token1.pos_:
                    similarity = analyzer.calculate_word_similarity(
                        token1.text, doc1.text,
                        token2.text, doc2.text
                    )
                    best_similarity = max(best_similarity, similarity)
            similarities.append(best_similarity)
    return np.mean(similarities) if similarities else 0

def calculate_sense_preservation(doc1, doc2, analyzer):
    sense_matches = 0
    total_words = 0
    for token1 in doc1:
        if token1.pos_ in ['NOUN', 'VERB', 'ADJ', 'ADV']:
            total_words += 1
            senses1 = set(analyzer.disambiguate_word(token1.text, doc1.text))
            for token2 in doc2:
                if token2.pos_ == token1.pos_:
                    senses2 = set(analyzer.disambiguate_word(token2.text, doc2.text))
                    if senses1.intersection(senses2):
                        sense_matches += 1
                        break
    return sense_matches / total_words if total_words > 0 else 1