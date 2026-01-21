import os
import torch
from transformers import (
    AutoModelForTokenClassification, 
    AutoTokenizer, 
    Trainer, 
    TrainingArguments, 
    DataCollatorForTokenClassification
)
from datasets import DatasetDict, Dataset, concatenate_datasets

# Import project modules
from hipe_loader import hipe_generator, HIPE_FEATURES
from metrics import compute_metrics
from tokenize_and_align import tokenize_and_format

def evaluate_version(model_version):
    # 1. Setup Paths
    model_path = os.path.join("models", "final", model_version)
    
    if not os.path.exists(model_path):
        print(f"Error: Model path '{model_path}' does not exist.")
        return

    print(f"\n--- Initializing Evaluation for Version: {model_version} ---")

    # 2. Load Model and Tokenizer first to get the correct Label Mapping
    # This ensures we use the labels the model was actually trained on.
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Extract the original label list from the model config
    id2label = model.config.id2label
    trained_labels = [id2label[i] for i in range(len(id2label))]
    
    print(f"Model Labels: {trained_labels}")

    # 3. Load Multilingual Test Data
    dataset_names = ["ajmc", "letemps", "newseye", "sonar", "topres19th"]
    languages = ["en", "de", "fr", "fi", "sv"]
    test_sets = []

    data_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "HIPE-2022-data", "data", "v2.1"))
    
    for ds in dataset_names:
        for lang in languages:
            folder_path = os.path.join(data_root, ds, lang)
            
            test_file = os.path.join(folder_path, f"HIPE-2022-v2.1-{ds}-test-{lang}.tsv")

            if os.path.exists(test_file):
                print(f"Loading {lang.upper()} test data...")
                test_sets.append(Dataset.from_generator(
                    hipe_generator, features=HIPE_FEATURES, gen_kwargs={"filepath": test_file}
                ))
            else:
                print(f"Warning: Test file missing for {lang.upper()} at {test_file}")

    if not test_sets:
        print("Error: No test data found. Check your HIPE-2022-data directory.")
        return

    # Combine all languages into one validation split
    raw_datasets = DatasetDict({"validation": concatenate_datasets(test_sets)})

    # 4. Tokenize and Format 
    # We pass the 'trained_labels' to ensure the IDs match the model's weights
    print("Preprocessing datasets...")
    tokenized_datasets, _, _ = tokenize_and_format(raw_datasets, labels=trained_labels)

    # 5. Initialize Trainer for Evaluation
    args = TrainingArguments(
        output_dir="./temp_eval",
        per_device_eval_batch_size=16,
        use_mps_device=torch.backends.mps.is_available(),
        logging_dir="./logs",
    )

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer, pad_to_multiple_of=8)

    trainer = Trainer(
        model=model,
        args=args,
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=lambda p: compute_metrics(p, id2label),
    )

    # 6. Run Evaluation
    print("--- Running Inference ---")
    results = trainer.evaluate()

    # 7. Print Final Metrics
    print("\n" + "="*40)
    print(f"FINAL METRICS: {model_version}")
    print("-" * 40)
    print(f"F1 Score:  {results['eval_f1']:.4f}")
    print(f"Precision: {results['eval_precision']:.4f}")
    print(f"Recall:    {results['eval_recall']:.4f}")
    print(f"Accuracy:  {results['eval_accuracy']:.4f}")
    print("="*40 + "\n")

if __name__ == "__main__":
    # Change this variable to the name of the folder you want to evaluate
    MY_MODEL_VERSION = "hipe-ajmc-multilingual-model-v4" 
    evaluate_version(MY_MODEL_VERSION)