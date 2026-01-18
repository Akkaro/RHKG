# Verifiable Historical NER Pipeline

## 1. Project Overview

This project implements an End-to-End Named Entity Recognition (NER) pipeline for historical documents. It is designed to handle the specific challenges of historical data, including OCR noise, archaic language, and complex document structures .

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
├── ner/                        # Source Code
│   ├── hipe_loader.py          # Custom Data Ingestion
│   ├── tokenize_and_align.py   # Preprocessing & Sliding Window
│   ├── metrics.py              # Evaluation Logic
│   ├── run_training.py         # Main Training Loop
│   └── inference.py            # Inference Demo
└── requirements.txt

```

---

## 3. Component Details

### A. Data Ingestion (`hipe_loader.py`)

**Role:** Parses the non-standard HIPE-2022 TSV format.

**Key Features:**
* **Header Skipping:** Explicitly detects and skips metadata (`#`) and header rows (`TOKEN`) to prevent data corruption.
* **Sentence Aggregation:** Reconstructs sentences from vertical token lists using empty-line delimiters .
* **Schema Definition:** Maps all 10+ columns (Literal vs. Metonymic) for future "Vertical Stem" experiments .


### B. Preprocessing (`tokenize_and_align.py`)

**Role:** Prepares text for BERT.

**Key Features:**
* **Sliding Window:** Cuts long documents (often >512 tokens) into overlapping 128-token chunks to solve data scarcity.
* **Subword Alignment:** Aligns labels to the first subword of a token (e.g., `Paris` -> `Pa`(B-LOC), `##ris`(-100)) .
* **Masking:** Assigns `-100` to special tokens and null annotations so they don't affect the loss.

### C. Evaluation (`metrics.py`)

**Role:** Computes rigorous Span-Level F1 scores.

**Key Features:**
* **Seqeval:** Uses standard CoNLL-style evaluation.
* **Filtering:** Removes all `-100` masks before calculation to ensure accuracy .

### D. Training Orchestrator (`run_training.py`)

**Role:** Integrates all components and fine-tunes the model.

**Key Features:**
* **Dynamic Padding:** Uses `DataCollatorForTokenClassification` for efficiency.
* **Hyperparameters:** Tuned for small historical datasets (LR: 3e-5, Epochs: 5) .

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