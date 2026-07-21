from transformers import pipeline
import pandas as pd

# NLP WITH HUGGINGFACE PIPELINE FOR SENTIMENT ANALYSIS

classifier = pipeline("sentiment-analysis")

reviews = [
    "This movie was absolutely fantastic!",
    "Worst experience ever, totally disappointed.",
    "The food was okay, nothing special.",
    "I loved every moment of this trip!",
    "Never going back to this restaurant again.",
 ]

for review in reviews:
    result = classifier(review)[0]
    print(f"Text: {review }")
    print(f"→ {result['label']} ({result['score']*100:.1f}%)\n")


#2 Name entity recognition
print("\n2. NAMED ENTITY RECOGNITION")
print("-" * 40)
ner = pipeline("ner", aggregation_strategy="simple")

text = "Narendra Modi met Sundar Pichai at Google headquarters in Hyderabad, India."
entities = ner(text)
for e in entities:
    print(f"{e['word']} → {e['entity_group']} ({e['score']*100:.1f}%)")

print("\n3. ZERO SHOT CLASSIFICATION")
print("-" * 40)
zero_shot = pipeline("zero-shot-classification")

texts = [
    "The stock market crashed today losing 500 points",
    "Virat Kohli scored a century in the final match",
    "New AI model beats human performance on coding tasks",
]

labels = ["finance", "sports", "technology", "politics"]

for text in texts:
    result = zero_shot(text, candidate_labels=labels)
    top_label = result['labels'][0]
    top_score = result['scores'][0]
    print(f"Text: {text}")
    print(f"→ {top_label} ({top_score*100:.1f}%)\n")

print("NLP analysis complete!")