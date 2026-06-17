"""
evaluate_bertscore.py

Cleaned BERTScore evaluation artifact for the adaptive Indonesian
multi-hop RAG system.

Original Colab workflow:
- Load full-pipeline answer results.
- Compare predicted answers with reference answers.
- Compute BERTScore Precision, Recall, and F1.
- Use F1 threshold = 0.85 to determine semantic correctness.
- Report confidence distribution and discrepancy analysis.

Sensitive paths, Google Drive mounting, Colab-specific commands, and
long output logs are removed from this public artifact.
"""

import argparse
import json
import os
import time
from typing import Dict, List, Tuple

import numpy as np
from bert_score import score as bert_score


BERTSCORE_MODEL = "bert-base-multilingual-cased"
BERTSCORE_THRESHOLD = 0.85


def load_results(file_path: str) -> List[Dict]:
    """Load pipeline results from a JSON file."""
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    if isinstance(data, dict) and "results" in data:
        return data["results"]

    if isinstance(data, list):
        return data

    raise ValueError("Unsupported result file format.")


def prepare_predictions_and_references(results: List[Dict]) -> Tuple[List[str], List[str]]:
    """Extract predicted answers and reference answers from pipeline results."""
    predictions = []
    references = []

    for result in results:
        predicted = result.get("predicted_answer", "").strip()
        reference = result.get("answer", result.get("reference_answer", "")).strip()

        if not predicted:
            predicted = "[NO ANSWER]"
        if not reference:
            reference = "[NO ANSWER]"

        predictions.append(predicted)
        references.append(reference)

    return predictions, references


def compute_bertscore_batches(
    predictions: List[str],
    references: List[str],
    model_type: str = BERTSCORE_MODEL,
    batch_size: int = 64,
) -> Tuple[List[float], List[float], List[float]]:
    """Compute BERTScore in batches."""
    all_precision = []
    all_recall = []
    all_f1 = []

    for start_idx in range(0, len(predictions), batch_size):
        batch_predictions = predictions[start_idx:start_idx + batch_size]
        batch_references = references[start_idx:start_idx + batch_size]

        precision, recall, f1 = bert_score(
            batch_predictions,
            batch_references,
            lang="multilingual",
            model_type=model_type,
            verbose=False,
        )

        all_precision.extend(precision.tolist())
        all_recall.extend(recall.tolist())
        all_f1.extend(f1.tolist())

    return all_precision, all_recall, all_f1


def categorize_confidence(f1_value: float) -> str:
    """Categorize BERTScore confidence based on F1 value."""
    if f1_value >= 0.90:
        return "high"
    if f1_value >= 0.80:
        return "medium"
    return "low"


def summarize_bertscore(
    results: List[Dict],
    precision_scores: List[float],
    recall_scores: List[float],
    f1_scores: List[float],
    threshold: float = BERTSCORE_THRESHOLD,
) -> Dict:
    """Summarize BERTScore evaluation results."""
    total = len(results)
    correct = sum(1 for value in f1_scores if value >= threshold)

    confidence_distribution = {
        "high": sum(1 for value in f1_scores if value >= 0.90),
        "medium": sum(1 for value in f1_scores if 0.80 <= value < 0.90),
        "low": sum(1 for value in f1_scores if value < 0.80),
    }

    em_0_bert_1 = 0
    em_1_bert_0 = 0

    detailed_results = []

    for idx, result in enumerate(results):
        original_em = result.get("em", 0)
        is_correct = f1_scores[idx] >= threshold
        confidence = categorize_confidence(f1_scores[idx])

        if original_em == 0 and is_correct:
            em_0_bert_1 += 1
        elif original_em == 1 and not is_correct:
            em_1_bert_0 += 1

        detailed_results.append({
            **result,
            "bertscore": {
                "precision": float(precision_scores[idx]),
                "recall": float(recall_scores[idx]),
                "f1": float(f1_scores[idx]),
                "is_correct": bool(is_correct),
                "confidence": confidence,
            },
        })

    return {
        "metrics": {
            "total_samples": total,
            "threshold": threshold,
            "accuracy": correct / total * 100 if total else 0,
            "mean_f1": float(np.mean(f1_scores)) if total else 0,
            "median_f1": float(np.median(f1_scores)) if total else 0,
            "std_f1": float(np.std(f1_scores)) if total else 0,
            "min_f1": float(np.min(f1_scores)) if total else 0,
            "max_f1": float(np.max(f1_scores)) if total else 0,
            "confidence_distribution": confidence_distribution,
            "discrepancy_analysis": {
                "em_0_bert_1": em_0_bert_1,
                "em_1_bert_0": em_1_bert_0,
            },
        },
        "results": detailed_results,
    }


def save_output(output: Dict, output_file: str) -> None:
    """Save BERTScore evaluation output."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(output, file, ensure_ascii=False, indent=2)


def main() -> None:
    """Run BERTScore evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate generated answers using BERTScore.")
    parser.add_argument("--input", required=True, help="Path to pipeline result JSON file.")
    parser.add_argument("--output", default="outputs/bertscore_results.json", help="Output JSON path.")
    parser.add_argument("--model", default=BERTSCORE_MODEL, help="BERTScore model type.")
    parser.add_argument("--threshold", type=float, default=BERTSCORE_THRESHOLD, help="F1 threshold.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for BERTScore.")

    args = parser.parse_args()

    start_time = time.time()

    results = load_results(args.input)
    predictions, references = prepare_predictions_and_references(results)

    precision_scores, recall_scores, f1_scores = compute_bertscore_batches(
        predictions=predictions,
        references=references,
        model_type=args.model,
        batch_size=args.batch_size,
    )

    summary = summarize_bertscore(
        results=results,
        precision_scores=precision_scores,
        recall_scores=recall_scores,
        f1_scores=f1_scores,
        threshold=args.threshold,
    )

    summary["metadata"] = {
        "model": args.model,
        "threshold": args.threshold,
        "batch_size": args.batch_size,
        "runtime_seconds": time.time() - start_time,
    }

    save_output(summary, args.output)

    metrics = summary["metrics"]

    print("BERTScore Evaluation Complete")
    print("=============================")
    print(f"Total samples: {metrics['total_samples']}")
    print(f"Accuracy: {metrics['accuracy']:.2f}%")
    print(f"Mean F1: {metrics['mean_f1']:.4f}")
    print(f"Median F1: {metrics['median_f1']:.4f}")
    print(f"High confidence: {metrics['confidence_distribution']['high']}")
    print(f"Medium confidence: {metrics['confidence_distribution']['medium']}")
    print(f"Low confidence: {metrics['confidence_distribution']['low']}")
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
