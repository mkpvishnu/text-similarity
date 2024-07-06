from transformers import pipeline

# Load the NER pipeline with the model
ner = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")

text = "Hugging Face Inc. is a company based in New York City started 3000 years ago. 3 millennia ago Its headquarters are in DUMBO, therefore very close to the Manhattan Bridge."
results = ner(text)

for entity in results:
    print(f"Entity: {entity['word']}, Label: {entity['entity']}")
