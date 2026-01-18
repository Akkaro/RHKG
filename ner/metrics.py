import numpy as np
import evaluate

# 'seqeval' is the standard library for NER evaluation.
# It calculates F1 based on full entities (e.g., "New York"), not just individual words.
seqeval = evaluate.load("seqeval")

def compute_metrics(p, id2label):
    """
    Computes Span-Level F1 scores, Precision, and Recall ignoring -100 (masked subwords).
    """
    predictions, labels = p
    # Convert logits (probabilities) to the single mostly likely class ID
    predictions = np.argmax(predictions, axis=2)

    # Convert predictions and labels from integers to strings
    # filter out the -100s to align with the original words
    true_predictions = [
        [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [id2label[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    # Calculate Precision, Recall, and F1
    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }