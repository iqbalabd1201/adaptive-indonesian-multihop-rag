"""
preprocess_hotpot.py

Cleaned preprocessing artifact for Indonesian HotpotQA.

Original workflow:
- Load translated Indonesian HotpotQA train and validation files.
- Apply Indonesian stemming using Sastrawi.
- Preserve original question and context for reference.
- Save stemmed train and validation data.
- Save preprocessing metadata.

Colab-specific commands, Google Drive paths, installation commands,
visualization logs, and manual confirmation prompts are removed.
"""

import argparse
import json
import os
import re
from typing import Dict, List

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from tqdm import tqdm


def create_stemmer():
    """Create Indonesian stemmer using Sastrawi."""
    stemmer_factory = StemmerFactory()
    return stemmer_factory.create_stemmer()


def stem_text(text: str, stemmer) -> str:
    """Stem Indonesian text using Sastrawi."""
    if not text or not str(text).strip():
        return "empty"

    try:
        stemmed = stemmer.stem(str(text).lower())
        stemmed = re.sub(r"\s+", " ", stemmed).strip()
        return stemmed

    except Exception:
        return str(text).lower()


def preprocess_sample(sample: Dict, stemmer) -> Dict:
    """Preprocess one HotpotQA sample using stemming only."""
    processed = {}

    processed["_original_question"] = sample["question"]
    processed["_original_context"] = sample["context"]

    processed["question"] = stem_text(sample["question"], stemmer)

    processed_context = []

    for doc_title, doc_sentences in sample["context"]:
        stemmed_title = stem_text(doc_title, stemmer)

        stemmed_sentences = [
            stem_text(sentence, stemmer)
            for sentence in doc_sentences
        ]

        processed_context.append([stemmed_title, stemmed_sentences])

    processed["context"] = processed_context

    metadata_keys = [
        "_id",
        "answer",
        "supporting_facts",
        "type",
        "level",
        "doc_labels",
        "supporting_titles",
    ]

    for key in metadata_keys:
        if key in sample:
            processed[key] = sample[key]

    return processed


def process_dataset(data: List[Dict], stemmer, description: str) -> List[Dict]:
    """Preprocess a full dataset."""
    processed_data = []
    errors = 0

    for sample in tqdm(data, desc=description):
        try:
            processed_data.append(preprocess_sample(sample, stemmer))
        except Exception:
            errors += 1
            continue

    print(f"{description}: {len(processed_data)}/{len(data)} samples processed")

    if errors > 0:
        print(f"{description}: {errors} samples skipped because of preprocessing errors")

    return processed_data


def load_json(file_path: str) -> List[Dict]:
    """Load JSON data."""
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def save_json(data, file_path: str) -> None:
    """Save JSON data."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def main() -> None:
    """Run HotpotQA preprocessing."""
    parser = argparse.ArgumentParser(
        description="Preprocess Indonesian HotpotQA using Sastrawi stemming."
    )

    parser.add_argument("--train-input", required=True, help="Path to translated train JSON.")
    parser.add_argument("--val-input", required=True, help="Path to translated validation JSON.")
    parser.add_argument("--output-dir", required=True, help="Directory for preprocessed output.")

    args = parser.parse_args()

    stemmer = create_stemmer()

    train_data = load_json(args.train_input)
    val_data = load_json(args.val_input)

    train_processed = process_dataset(
        data=train_data,
        stemmer=stemmer,
        description="HotpotQA train",
    )

    val_processed = process_dataset(
        data=val_data,
        stemmer=stemmer,
        description="HotpotQA validation",
    )

    train_output = os.path.join(args.output_dir, "train_16k_stemmed.json")
    val_output = os.path.join(args.output_dir, "val_4k_stemmed.json")
    metadata_output = os.path.join(args.output_dir, "preprocessing_metadata.json")

    save_json(train_processed, train_output)
    save_json(val_processed, val_output)

    metadata = {
        "dataset": "HotpotQA",
        "language": "Indonesian",
        "preprocessing": "stemming_only",
        "stemmer": "Sastrawi",
        "train_samples": len(train_processed),
        "validation_samples": len(val_processed),
        "notes": "Original question and context are preserved in _original_question and _original_context.",
    }

    save_json(metadata, metadata_output)

    print("HotpotQA preprocessing complete")
    print(f"Train output: {train_output}")
    print(f"Validation output: {val_output}")
    print(f"Metadata output: {metadata_output}")


if __name__ == "__main__":
    main()
