import csv
import os
import re
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification

MODEL_PATH = "./models/final/hipe-ajmc-multilingual-model-v4"
CORPUS_FILE = "napoleonic_wars.txt"

def split_into_sentences(text):
    """
    A standalone sentence splitter using regex. 
    It splits text at . ! or ? followed by whitespace, accounting for common abbreviations.
    """
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s', text)
    return [s.strip() for s in sentences if s.strip()]

def aggregate_for_kg(raw_results, threshold=0.7):
    """
    Manually aggregates tokens for Knowledge Graph ingestion.
    Merges subwords and consecutive identical labels even if both are 'B-'.
    """
    if not raw_results:
        return []

    aggregated_entities = []
    
    # Start with the first entity
    current_entity = {
        "word": raw_results[0]['word'],
        "label": raw_results[0]['entity'].split('-')[-1], # Strip B- or I-
        "scores": [raw_results[0]['score']],
        "start": raw_results[0]['start'],
        "end": raw_results[0]['end']
    }

    for i in range(1, len(raw_results)):
        token = raw_results[i]
        token_label = token['entity'].split('-')[-1]
        
        is_subword = token['word'].startswith("##")
        is_same_label = (token_label == current_entity['label'])
        is_consecutive = (token['start'] == current_entity['end'])
        
        if is_subword or (is_same_label and is_consecutive):
            # Merge token into current entity
            current_entity["word"] += token['word'].replace("##", "")
            current_entity["scores"].append(token['score'])
            current_entity["end"] = token['end']
        else:            
            avg_score = sum(current_entity["scores"]) / len(current_entity["scores"])
            if avg_score >= threshold:
                aggregated_entities.append({
                    "entity": current_entity["word"].strip().replace(" - ", "-"),
                    "label": current_entity["label"],
                    "confidence": round(avg_score, 4)
                })
                
            current_entity = {
                "word": token['word'],
                "label": token_label,
                "scores": [token['score']],
                "start": token['start'],
                "end": token['end']
            }

    # Finalize the last entity
    avg_score = sum(current_entity["scores"]) / len(current_entity["scores"])
    if avg_score >= threshold:
        aggregated_entities.append({
            "entity": current_entity["word"].strip().replace(" - ", "-"),
            "label": current_entity["label"],
            "confidence": round(avg_score, 4)
        })
        
    return aggregated_entities

def save_entities_to_csv(entities, filename="ner_results.csv"):
    """
    Saves the cleaned entities to a CSV file.
    """
    if not entities:
        return

    fields = ["entity", "label", "confidence"]
    file_exists = os.path.isfile(filename)
    
    with open(filename, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not file_exists:
            writer.writeheader()
        writer.writerows(entities)
        
def main():
    # 1. Check for the Model
    if (MODEL_PATH.startswith("./") or MODEL_PATH.startswith("/")) and not os.path.exists(MODEL_PATH):
        print(f"Error: Could not find trained model at {MODEL_PATH}.")
        print("Please ensure you have run 'python ner/run_training.py' first.")
        return
    
    # 2. Check for the Corpus File
    if not os.path.exists(CORPUS_FILE):
        print(f"Error: Could not find corpus file at {CORPUS_FILE}.")
        print(f"Please create a file named '{CORPUS_FILE}' in the project root.")
        return

    print(f"--- Loading Model from {MODEL_PATH} ---")
    
    # Initialize components using classes from the project
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)
    
    # Create the inference pipeline with aggregation to handle subwords
    ner_pipeline = pipeline(
        "token-classification", 
        model=model, 
        tokenizer=tokenizer, 
    )

    # 3. Read the Corpus from File
    print(f"Reading text from {CORPUS_FILE}...")
    with open(CORPUS_FILE, "r", encoding="utf-8") as f:
        text_corpus = f.read()

    # 4. Presplit text into sentences
    print("Splitting corpus into sentences...")
    sentences = split_into_sentences(text_corpus)
    print(f"Processing {len(sentences)} sentences...\n")

    # 5. Clean, Print and Aggregate Results
    if os.path.exists("ner_results.csv"):
        os.remove("ner_results.csv")

    for idx, sentence in enumerate(sentences):
        print(f"\n{'='*80}")
        print(f"SENTENCE {idx+1}: {sentence}")
        print(f"{'='*80}")
        
        # 1. Get Raw Predictions from the pipeline
        raw_results = ner_pipeline(sentence)
        
        print("\n[STAGE 1: RAW PREDICTIONS]")
        print(f"{'Token':<25} {'Raw Label':<15} {'Score':<10}")
        print("-" * 55)
        for res in raw_results:
            print(f"{res['word']:<25} {res['entity']:<15} {res['score']:.4f}")

        # 2. Get Aggregated & Thresholded Results
        clean_entities = aggregate_for_kg(raw_results, threshold=0.7)
        
        print("\n[STAGE 2: CLEANED GRAPH NODES]")
        print(f"{'Entity Name':<25} {'Graph Label':<15} {'Avg Score':<10}")
        print("-" * 55)
        
        if not clean_entities:
            print("No entities met the 0.85 confidence threshold.")
        else:
            for entity in clean_entities:
                print(f"{entity['entity']:<25} {entity['label']:<15} {entity['confidence']:.4f}")

        # 3. Save to CSV
        save_entities_to_csv(clean_entities, filename="ner_results.csv")
        
if __name__ == "__main__":
    main()