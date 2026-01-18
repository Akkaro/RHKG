from transformers import AutoTokenizer

# 1. hmBERT was pre-trained on historical text
MODEL_CHECKPOINT = "dbmdz/bert-base-historic-multilingual-cased"

def get_label_list(dataset):
    """
    Extract all unique named-entity labels from a dataset.
    
    :param dataset: A dictionary containing dataset splits. Each split is
                    expected to contain a key `ne_coarse_lit`, which is a
                    list of label sequences.
    :return: A sorted list of unique labels.
    """
    unique_labels = set()
    excluded_categories = ["scope", "work", "object"]
    for split in dataset.keys():
        for tags in dataset[split]["ne_coarse_lit"]:
            for tag in tags:
                if not any(cat in tag.lower() for cat in excluded_categories):
                    unique_labels.add(tag)
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
        # 1. Tokenize with SLIDING WINDOW
        tokenized_inputs = tokenizer(
            examples["tokens"], 
            truncation=True, 
            is_split_into_words=True, # Input is already provided as a list of words
            max_length=512            # Max length for BERT-based models
        )
        
        all_labels = examples["ne_coarse_lit"]
        new_labels = []
        
        for i, labels in enumerate(all_labels):
            word_ids = tokenized_inputs.word_ids(batch_index=i) # Map tokens to their respective word
            
            aligned_labels = []
            previous_word_idx = None
            
            for word_idx in word_ids:
                if word_idx is None:
                    aligned_labels.append(-100)         # Special tokens ([CLS], [SEP]) -> Ignore
                elif word_idx != previous_word_idx:
                    label_str = labels[word_idx]        # Original label for this word
                    if label_str in ["-", "_", ""]:
                        aligned_labels.append(-100)     # Ignore null annotations
                    else:
                        # Map valid label to ID, default to O if unknown
                        aligned_labels.append(label2id.get(label_str, label2id.get("O", 0)))      
                else:
                    # Alternative: For subword tokens, we can choose to assign the same label. This proved less effective.
                    # label_str = labels[word_idx]
                    # if label_str.startswith("B-"):
                    #     label_str = "I-" + label_str[2:]
                    # aligned_labels.append(label2id.get(label_str, label2id.get("O", 0)))
                    
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