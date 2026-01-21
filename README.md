# Historical NER to Knowledge Graph Pipeline

## 1. Project Overview
This project implements a fine-tuned Named Entity Recognition (NER) pipeline specialized for **historical multilingual documents** (specifically the HIPE-2022 AJMC dataset). It transforms raw historical text into structured entities ready for ingestion into a **Knowledge Graph**.


**Key Technologies:**
* **Model:** `dbmdz/bert-base-historic-multilingual-cased` (hmBERT).
* **Framework:** Hugging Face `transformers` & `datasets`.
* **Metric:** Span-Level F1 Score (via `seqeval`).

---

## 2. Directory Structure

```text
thesis/
├── .venv/                      # Virtual Environment
├── HIPE-2022-data/             # Raw Data (External Repo)
│   └── data/
│       └── v2.1/
│           └── hipe2020/       # Target Dataset (Newspapers)
├── napoleonic_wars.txt         # Test data
├── ner/                        # Source Code
│   ├── corpus_inference.py     # Inference on longer corpus
│   ├── hipe_loader.py          # Custom Data Ingestion
│   ├── inference.py            # Inference Demo
│   ├── metrics.py              # Evaluation Logic
│   ├── run_training.py         # Main Training Loop
│   └── tokenize_and_align.py   # Preprocessing & Sliding Window
└── requirements.txt
```

---
## 3. Component Details

### A. Data Ingestion (`hipe_loader.py`)

**Role:** Parses the non-standard HIPE-2022 TSV format.

**Key Features:**

* **Header Skipping:** Explicitly detects and skips metadata (`#`) and header rows (`TOKEN`) to prevent data corruption.
* **Sentence Aggregation:** Reconstructs sentences from vertical token lists using empty-line delimiters.
* **Schema Definition:** Maps coarse and fine-grained labels (Literal vs. Metonymic) for flexible training configurations.

### B. Preprocessing (`tokenize_and_align.py`)

**Role:** Prepares text for BERT and handles label alignment.

**Key Features:**

* **Sliding Window:** Automatically segments long documents (>512 tokens) into overlapping 128-token chunks to prevent truncation and maximize context.
* **Label Filtering:** Dynamically eliminates noisy or irrelevant classes (e.g., `scope`, `work`, `object`) to focus the model on high-value entities.
* **-100 Masking Strategy:** Aligns labels to the first subword of a token (e.g., `Ulm` -> `Ul`(B-LOC), `##m`(-100)). Masking subwords and special tokens ensures the loss function ignores structural artifacts.

### C. Evaluation (`metrics.py`)

**Role:** Computes rigorous Span-Level F1 scores.

**Key Features:**

* **Seqeval Integration:** Uses standard CoNLL-style evaluation to assess the model's ability to identify full entity spans rather than just individual tokens.
* **Validation Filtering:** Automatically removes `-100` masks and padding before calculation to provide an intellectually honest assessment of model performance.

### D. Training Orchestrator (`run_training.py`)

**Role:** Fine-tunes the model using specialized historical NLP strategies.

**Key Features:**

* **Weighted Categorical Cross-Entropy:** Implements a custom `WeightedTrainer` to address class imbalance.
* **Dynamic IDF Weighting:** Calculates class weights inversely proportional to their frequency in the training data. This ensures rare entities (like `pers` or `loc`) have a higher impact on the loss than the dominant `O` (Outside) class.
* **Weight Sharing:** Groups  and  tags by category during weight calculation to ensure semantic unity and prevent fragmented entity detection.

### E. Post-Processing & KG Aggregation (`corpus_inference.py`)

**Role:** Transforms raw model predictions into "Graph-Ready" nodes.

**Key Features:**

* **Knowledge Graph Aggregator:** A custom multi-stage algorithm that reassembles subwords (e.g., `178` + `##9` → `1789`) and merges consecutive tokens with identical labels into single spans.
* **Confidence Thresholding:** Implements a strict probability filter (default: **0.85**) to eliminate "noisy" entities from the final Knowledge Graph.
* **CSV Export:** Generates `ner_results.csv`, a structured list of cleaned entities, labels, and average confidence scores optimized for graph database ingestion.

---

## 4. Execution Commands

### Prerequisites

```bash
# Create Environment
python3 -m venv .venv
source .venv/bin/activate

# Install Dependencies
pip install torch transformers datasets evaluate seqeval pandas accelerate -U

```

### Running the Pipeline

1. **Clear Cache** (Required if changing datasets or loader logic):
```bash
rm -rf ~/.cache/huggingface/datasets

```


2. **Train the Model**:
```bash
python ner/run_training.py

```


3. **Run Inference**:
```bash
python ner/inference.py

```

## 5. Execution Commands

Models:
### hipe-ajmc-model-final
- base model: dbmdz/bert-base-historic-multilingual-cased
- trained only on the English corpora of the AJMC dataset
- "scope", "work", "object" classes are excluded
- batch size of 16, 10 fine-tuning epochs, learning rate of 3e-5
========================================
FINAL METRICS: hipe-ajmc-model-final
----------------------------------------
F1 Score:  0.8688
Precision: 0.8067
Recall:    0.9412
Accuracy:  0.9949
========================================



### hipe-ajmc-multilingual-model
- base model: dbmdz/bert-base-historic-multilingual-cased
- trained on the entire corpora of the AJMC dataset, concatenating the English, German and Franch data for train and test respectively
- "scope", "work", "object" classes are excluded
- batch size of 16, 10 fine-tuning epochs, learning rate of 3e-5
========================================
FINAL METRICS: hipe-ajmc-multilingual-model
----------------------------------------
F1 Score:  0.8433
Precision: 0.8052
Recall:    0.8851
Accuracy:  0.9914
========================================



### hipe-ajmc-multilingual-model-v1
- base model: dbmdz/bert-base-historic-multilingual-cased
- trained on the entire corpora of the AJMC dataset, concatenating the English, German and Franch data for train and test respectively
- "scope", "object" classes are excluded
- batch size of 16, 10 fine-tuning epochs, learning rate of 3e-5
========================================
FINAL METRICS: hipe-ajmc-multilingual-model-v1
----------------------------------------
F1 Score:  0.8188
Precision: 0.7633
Recall:    0.8829
Accuracy:  0.9823
========================================



### hipe-ajmc-multilingual-model-v2
- base model: dbmdz/bert-base-historic-multilingual-cased
- trained on the entire corpora of the AJMC dataset, concatenating the English, German and Franch data for train and test respectively
- "scope", "object" classes are excluded
- used fine grained tags
- batch size of 16, 10 fine-tuning epochs, learning rate of 3e-5
========================================
FINAL METRICS: hipe-ajmc-multilingual-model-v2
----------------------------------------
F1 Score:  0.6840
Precision: 0.6024
Recall:    0.7911
Accuracy:  0.9716
========================================



### hipe-ajmc-multilingual-model-v3
- base model: Babelscape/wikineural-multilingual-ner
- trained on the entire corpora of the AJMC dataset, concatenating the English, German and Franch data for train and test respectively
- "scope", "object" classes are excluded
- used fine grained tags
- batch size of 16, 10 fine-tuning epochs, learning rate of 3e-5
========================================
FINAL METRICS: hipe-ajmc-multilingual-model-v3
----------------------------------------
F1 Score:  0.7014
Precision: 0.6458
Recall:    0.7674
Accuracy:  0.9767
========================================



### hipe-ajmc-multilingual-model-v4
- base model: dbmdz/bert-base-historic-multilingual-cased
- trained on all dataset ("ajmc", "letemps", "newseye", "sonar", "topres19th"), all languages ("en", "de", "fr", "fi", "sv")
- Tag Set: 'B-date', 'B-loc', 'B-org', 'B-pers', 'B-work', 'I-date', 'I-loc', 'I-org', 'I-pers', 'I-work', 'O'
- batch size of 4, 5 fine-tuning epochs, learning rate of 5e-5
- PROBLEM: cannot measure with
========================================
FINAL METRICS: hipe-ajmc-multilingual-model-v4
----------------------------------------
F1 Score:  0.5499
Precision: 0.4858
Recall:    0.6333
Accuracy:  0.9566
========================================