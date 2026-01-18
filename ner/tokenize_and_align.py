from transformers import AutoTokenizer

# 1. hmBERT was pre-trained on historical text
MODEL_CHECKPOINT = "dbmdz/bert-base-historic-multilingual-cased"

def get_label_list(dataset):
    unique_labels = set()
    for split in dataset.keys():
        for tags in dataset[split]["ne_coarse_lit"]:
            unique_labels.update(tags)
    # Remove metadata artifacts if they slipped in
    unique_labels.discard("NE-COARSE-LIT") 
    return sorted(list(unique_labels))

def tokenize_and_format(dataset):
    print("--- Loading Tokenizer (Standard Mode) ---")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
    label_list = get_label_list(dataset)
    
    # Create Mappings
    label2id = {l: i for i, l in enumerate(label_list)}
    id2label = {i: l for i, l in enumerate(label_list)}
    
    print(f"Labels: {label_list}")

    def tokenize_function(examples):
        # 1. Tokenize with SLIDING WINDOW: cut long docs into chunks
        tokenized_inputs = tokenizer(
            examples["tokens"], 
            truncation=True, 
            is_split_into_words=True,
            max_length=512  # Increased from 128 to cover your long sentences
        )
        
        all_labels = examples["ne_coarse_lit"]
        new_labels = []
        
        for i, labels in enumerate(all_labels):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            
            aligned_labels = []
            previous_word_idx = None
            
            for word_idx in word_ids:
                if word_idx is None:
                    # Special tokens ([CLS], [SEP]) -> Ignore
                    aligned_labels.append(-100)
                elif word_idx != previous_word_idx:
                    # New word start
                    label_str = labels[word_idx]
                    
                    # Handle null/empty annotations by masking them
                    if label_str in ["-", "_", ""]: 
                        aligned_labels.append(-100) # Ignore null annotations
                    else:
                        # Map valid label to ID, default to O if unknown
                        aligned_labels.append(label2id.get(label_str, label2id.get("O", 0)))
                        
                else:
                    # Subword parts -> Ignore
                    aligned_labels.append(-100)
                
                previous_word_idx = word_idx
            
            new_labels.append(aligned_labels)
            
        tokenized_inputs["labels"] = new_labels
        return tokenized_inputs

    # Apply to dataset
    # Note: batched=True is essential for sliding window
    tokenized_datasets = dataset.map(
        tokenize_function, 
        batched=True, 
        remove_columns=dataset["train"].column_names
    )
    
    return tokenized_datasets, label2id, id2label