from transformers import pipeline

# Model 1: Sentiment Analysis
print("=== SENTIMENT ANALYSIS ===\n")
classifier = pipeline("sentiment-analysis")

sentences = [
    "I love data science, it is amazing!",
    "Hyderabad has the best biryani!",
    "I am excited about my new journey.",
    "This is really hard and confusing.",
]

for s in sentences:
    result = classifier(s)[0]
    print(f"Text: {s}")
    print(f"→ {result['label']} ({result['score']:.2f})\n")

# Model 2: Named Entity Recognition
print("=== NAMED ENTITY RECOGNITION ===\n")
ner = pipeline("ner", aggregation_strategy="simple")

text = "Sundar Pichai is the CEO of Google and studied at IIT Kharagpur in Hyderabad, India."
entities = ner(text)

for e in entities:
    print(f"{e['word']} → {e['entity_group']} ({e['score']:.2f})")