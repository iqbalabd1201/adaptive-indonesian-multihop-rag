"""
run_full_pipeline.py

Cleaned code artifact for the adaptive Indonesian multi-hop
Retrieval-Augmented Generation (RAG) pipeline.

Original experiment:
Classifier → Adaptive Routing → Retrieval → Answer Generation → Evaluation

This version is prepared for thesis documentation and GitHub publication.
Sensitive information such as API keys, local paths, checkpoint files, and
full datasets are not included.
"""

import re
import string
from collections import Counter
from typing import Dict, List


# ============================================================
# Answer Evaluation Utilities
# ============================================================

def normalize_answer(text: str) -> str:
    """Normalize answer text for Exact Match and F1 evaluation."""

    def remove_articles(s: str) -> str:
        return re.sub(r"\b(a|an|the|yang|sebuah|satu)\b", " ", s)

    def remove_punctuation(s: str) -> str:
        return "".join(ch for ch in s if ch not in set(string.punctuation))

    def white_space_fix(s: str) -> str:
        return " ".join(s.split())

    return white_space_fix(remove_articles(remove_punctuation(text.lower())))


def exact_match_score(prediction: str, ground_truth: str) -> int:
    """Calculate Exact Match score."""
    return int(normalize_answer(prediction) == normalize_answer(ground_truth))


def f1_score(prediction: str, ground_truth: str) -> float:
    """Calculate token-level F1 score."""
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()

    if len(prediction_tokens) == 0 or len(ground_truth_tokens) == 0:
        return float(prediction_tokens == ground_truth_tokens)

    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(prediction_tokens)
    recall = num_same / len(ground_truth_tokens)

    return 2 * precision * recall / (precision + recall)


def clean_answer(prediction: str) -> str:
    """Clean generated answer before evaluation."""
    prediction = prediction.replace("Jawaban:", "").replace("jawaban:", "").strip()
    prediction = prediction.strip('"').strip("'").strip()
    prediction = prediction.rstrip(".,!?;:")

    return prediction.strip()


# ============================================================
# Prompt Construction
# ============================================================

def create_2hop_prompt(question: str, doc1: str, doc2: str) -> str:
    """Create prompt for 2-hop answer generation."""
    return f"""
Berdasarkan pertanyaan dan dua dokumen pendukung, berikan jawaban yang singkat dan tepat.

Pertanyaan:
{question}

Dokumen 1:
{doc1}

Dokumen 2:
{doc2}

Instruksi:
1. Jawab hanya dalam bahasa Indonesia.
2. Jawaban harus sangat singkat.
3. Jangan tambahkan penjelasan.
4. Gunakan informasi dari dokumen pendukung.

Jawaban:
""".strip()


def create_4hop_prompt(
    question: str,
    doc1: str,
    doc2: str,
    doc3: str,
    doc4: str
) -> str:
    """Create prompt for 4-hop answer generation."""
    return f"""
Berdasarkan pertanyaan dan empat dokumen pendukung, berikan jawaban yang singkat dan tepat.

Pertanyaan:
{question}

Dokumen 1:
{doc1}

Dokumen 2:
{doc2}

Dokumen 3:
{doc3}

Dokumen 4:
{doc4}

Instruksi:
1. Jawab hanya dalam bahasa Indonesia.
2. Jawaban harus sangat singkat.
3. Jangan tambahkan penjelasan.
4. Gunakan informasi dari seluruh dokumen pendukung.

Jawaban:
""".strip()


# ============================================================
# Adaptive Pipeline Placeholders
# ============================================================

def classify_query(question: str) -> str:
    """
    Classify query complexity as 2-hop or 4-hop.

    In the original experiment, this component used a BERT-based classifier
    with bert-base-multilingual-cased as the backbone.
    """
    raise NotImplementedError(
        "Load the trained query complexity classifier before running this function."
    )


def retrieve_2hop(question: str, candidate_documents: List[str]) -> List[str]:
    """
    Retrieve supporting documents for 2-hop questions.

    In the original experiment, the system used two-stage sequential retrieval:
    Stage 1 retrieves the first supporting document, and Stage 2 retrieves the
    second supporting document using contextual input.
    """
    raise NotImplementedError(
        "Load the trained 2-hop retrieval models before running this function."
    )


def retrieve_4hop(question: str, candidate_documents: List[str]) -> List[str]:
    """
    Retrieve supporting documents for 4-hop questions.

    In the original experiment, the system used ranking-based retrieval with
    transfer learning from the 2-hop retrieval model.
    """
    raise NotImplementedError(
        "Load the trained 4-hop ranking model before running this function."
    )


def generate_answer(question: str, retrieved_documents: List[str]) -> str:
    """
    Generate final answer from retrieved documents.

    In the original experiment, answer generation used few-shot prompting with
    a large language model through an external API. API credentials are not
    included in this public artifact.
    """
    raise NotImplementedError(
        "Configure the answer generation model before running this function."
    )


# ============================================================
# Full Pipeline
# ============================================================

def run_pipeline(sample: Dict) -> Dict:
    """
    Run one sample through the adaptive RAG pipeline.

    Expected sample format:
    {
        "question": "...",
        "candidate_documents": [...],
        "answer": "..."
    }
    """
    question = sample["question"]
    candidate_documents = sample["candidate_documents"]
    reference_answer = sample["answer"]

    predicted_complexity = classify_query(question)

    if predicted_complexity == "2-hop":
        retrieved_documents = retrieve_2hop(question, candidate_documents)
    else:
        retrieved_documents = retrieve_4hop(question, candidate_documents)

    predicted_answer = generate_answer(question, retrieved_documents)
    predicted_answer = clean_answer(predicted_answer)

    em = exact_match_score(predicted_answer, reference_answer)
    f1 = f1_score(predicted_answer, reference_answer)

    return {
        "question": question,
        "predicted_complexity": predicted_complexity,
        "retrieved_documents": retrieved_documents,
        "predicted_answer": predicted_answer,
        "reference_answer": reference_answer,
        "exact_match": em,
        "f1_score": f1,
    }


if __name__ == "__main__":
    print("Adaptive Indonesian Multi-Hop RAG pipeline artifact.")
    print("This cleaned version documents the structure of the full pipeline.")
