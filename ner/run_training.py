import torch
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)
from datasets import DatasetDict, Dataset
import os

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

def main():
    print(f"--- Starting Training Pipeline on {get_device().upper()} ---")
    # 1. Load Data
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "HIPE-2022-data", "data", "v2.1", "ajmc", "en"))
    
    files = {
        "train": os.path.join(base_path, "HIPE-2022-v2.1-ajmc-train-en.tsv"),
        "validation": os.path.join(base_path, "HIPE-2022-v2.1-ajmc-test-en.tsv"),
    }

    raw_datasets = DatasetDict()
    for split, path in files.items():
        # Clean loading using the generator
        raw_datasets[split] = Dataset.from_generator(
            hipe_generator, 
            features=HIPE_FEATURES, 
            gen_kwargs={"filepath": path}
        )

    # 2. Tokenize & Align
    tokenized_datasets, label2id, id2label = tokenize_and_format(raw_datasets)
    
    print("\n" + "="*40)
    print(f"RAW Documents:       {len(raw_datasets['train'])}")
    print(f"EXPANDED Examples:   {len(tokenized_datasets['train'])}")
    print("="*40 + "\n")
    
    # 3. Model Initialization
    print(f"Initializing Model: {MODEL_CHECKPOINT} with {len(label2id)} labels.")
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_CHECKPOINT, 
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id
    )
    
    # 4. Tokenizer & Data Collator
    # We must explicitly load the tokenizer again for the collator
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
    
    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer,
        pad_to_multiple_of=8
    )

    # 5. Training Arguments
    args = TrainingArguments(
        output_dir="hipe-ajmc-model",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=5,
        weight_decay=0.01,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        use_mps_device=(get_device() == "mps")
    )

    # 6. Initialize Trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=lambda p: compute_metrics(p, id2label),
    )

    # 7. Train!
    print("--- Starting Fine-Tuning ---")
    trainer.train()
    
    print("--- Saving Final Model ---")
    trainer.save_model("hipe-ajmc-model-final")

if __name__ == "__main__":
    main()