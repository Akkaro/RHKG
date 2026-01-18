from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification

# 1. Load your trained model
MODEL_PATH = "./hipe-ajmc-model-final"
print(f"Loading model from {MODEL_PATH}...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)

# 2. Create the Inference Pipeline
# aggregation_strategy="simple" merges "Na", "##po", "##leon" back into "Napoleon"
ner_pipeline = pipeline("token-classification", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

# 3. Test on a sample sentence (Historical context)
# text = "In 1802, Napoleon Bonaparte visited the city of Paris to discuss the Treaty of Amiens."
text = "See Matthew 5:2 for a discussion on the Sermon on the Mount."

print(f"\nInput: {text}\n")
results = ner_pipeline(text)

# 4. Print Results
print(f"{'Entity':<20} {'Label':<10} {'Score':<5}")
print("-" * 40)
for entity in results:
    print(f"{entity['word']:<20} {entity['entity_group']:<10} {entity['score']:.4f}")