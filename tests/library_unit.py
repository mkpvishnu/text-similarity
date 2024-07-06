from lingualens import TextAnalyzer, Metrics, check_similarity, ensure_model

# Ensure the desired Spacy model is downloaded
spacy_model = ensure_model("en_core_web_sm")

# Using individual metrics
analyzer = TextAnalyzer(spacy_model)
metrics = Metrics(analyzer)

text1 = "Yesterday, I went to the store. Today, I'm staying home."
text2 = "Today, I went to the store. Yesterday I stayed home."

temporal_shift = metrics.detect_temporal_shift(text1, text2, print_info=True)
print(f"Temporal shift similarity: {temporal_shift}")

# # Using pipeline
# similarity_results = check_similarity(text1, text2, spacy_model="en_core_web_sm")
# print("Similarity check results:", similarity_results)