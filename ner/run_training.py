import torch
import torch.nn as nn
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
)
from datasets import DatasetDict, Dataset, concatenate_datasets
import os
import numpy as np
from collections import Counter

from hipe_loader import hipe_generator, HIPE_FEATURES
from tokenize_and_align import tokenize_and_format, MODEL_CHECKPOINT
from metrics import compute_metrics


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def calculate_class_weights(tokenized_dataset, num_labels):
    """
    Calculates weights inversely proportional to class frequencies.
    Formula: weight = total_tokens / (num_classes * class_count)
    """
    # Flatten all labels and filter out the -100 (ignored) tokens
    all_labels = [
        label
        for sublist in tokenized_dataset["labels"]
        for label in sublist
        if label != -100
    ]

    # Count occurrences of each ID
    counts = Counter(all_labels)
    total_tokens = len(all_labels)

    # Initialize weights with 1.0
    weights = np.ones(num_labels, dtype=np.float32)

    for label_id in range(num_labels):
        count = counts.get(label_id, 0)
        if count > 0:
            # Standard Inverse Frequency formula
            weights[label_id] = total_tokens / (num_labels * count)
        else:
            weights[label_id] = 1.0  # Fallback for labels not in training set

    return torch.tensor(weights)


class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        # Move weights to the correct device (CUDA/MPS/CPU)
        if class_weights is not None:
            self.class_weights = class_weights.to(self.args.device)
        else:
            self.class_weights = None

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        num_labels = self.model.config.num_labels

        # Use the dynamic weights if provided
        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights, ignore_index=-100)

        loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def main():
    print(f"--- Starting Training Pipeline on {get_device().upper()} ---")
    # 1. Load Data
    dataset_names = ["ajmc", "letemps", "newseye", "sonar", "topres19th"]
    languages = ["en", "de", "fr", "fi", "sv"]
    
    train_sets = []
    val_sets = []

    data_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "HIPE-2022-data", "data", "v2.1"))
    
    for ds in dataset_names:
        for lang in languages:
            folder_path = os.path.join(data_root, ds, lang)
            
            train_file = os.path.join(folder_path, f"HIPE-2022-v2.1-{ds}-train-{lang}.tsv")
            test_file = os.path.join(folder_path, f"HIPE-2022-v2.1-{ds}-test-{lang}.tsv")
            dev_file = os.path.join(folder_path, f"HIPE-2022-v2.1-{ds}-dev-{lang}.tsv")

            # Try to load training data
            if os.path.exists(train_file):
                print(f"  [FOUND] Train: {ds}-{lang}")
                train_sets.append(Dataset.from_generator(
                    hipe_generator, features=HIPE_FEATURES, gen_kwargs={"filepath": train_file}
                ))
            
            # Try to load validation data (prefer 'test' if it exists, else 'dev')
            eval_path = test_file if os.path.exists(test_file) else (dev_file if os.path.exists(dev_file) else None)
            if eval_path:
                print(f"  [FOUND] Eval:  {ds}-{lang}")
                val_sets.append(Dataset.from_generator(
                    hipe_generator, features=HIPE_FEATURES, gen_kwargs={"filepath": eval_path}
                ))

    
    print(f"\nSuccessfully loaded {len(train_sets)} training files and {len(val_sets)} validation files.")

    # Combine into a single DatasetDict
    raw_datasets = DatasetDict({
        "train": concatenate_datasets(train_sets),
        "validation": concatenate_datasets(val_sets) if val_sets else train_sets[0] # Fallback if no val data
    })
    # 2. Tokenize & Align
    tokenized_datasets, label2id, id2label = tokenize_and_format(raw_datasets, column="ne_coarse_lit")

    print("\n" + "=" * 40)
    print(f"RAW Documents:       {len(raw_datasets['train'])}")
    print(f"EXPANDED Examples:   {len(tokenized_datasets['train'])}")
    print("=" * 40 + "\n")

    # 3. Calculate Weights Dynamically
    num_labels = len(label2id)
    class_weights = calculate_class_weights(tokenized_datasets["train"], num_labels)

    print("\n--- Dynamic Class Weights ---")
    for label, idx in label2id.items():
        print(f"{label:<10}: {class_weights[idx]:.4f}")
    print("-----------------------------\n")

    # 4. Model Initialization
    print(f"Initializing Model: {MODEL_CHECKPOINT} with {len(label2id)} labels.")
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_CHECKPOINT, 
        num_labels=len(label2id), 
        id2label=id2label, 
        label2id=label2id,
        ignore_mismatched_sizes=True # add this if not hmBERT is the starting point
    )

    # 5. Tokenizer & Data Collator
    # We must explicitly load the tokenizer again for the collator
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer, pad_to_multiple_of=8
    )

    # 6. Training Arguments
    args = TrainingArguments(
        output_dir="models/checkpoints/hipe-ajmc-multilingual-model-v5",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=5,
        weight_decay=0.01,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        use_mps_device=(get_device() == "mps"),
    )

    # 7. Initialize Trainer
    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=lambda p: compute_metrics(p, id2label),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        class_weights=class_weights,
    )

    # 8. Train
    print("--- Starting Fine-Tuning ---")
    trainer.train()

    print("--- Saving Final Model ---")
    trainer.save_model("models/final/hipe-ajmc-multilingual-model-v5")


if __name__ == "__main__":
    main()
