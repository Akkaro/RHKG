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