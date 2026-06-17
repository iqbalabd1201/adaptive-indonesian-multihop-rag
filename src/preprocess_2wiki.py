"""
preprocess_2wiki.py

Cleaned preprocessing artifact for Indonesian 2WikiMultihopQA.

Original workflow:
- Load translated Indonesian 2WikiMultihopQA train and test files.
- Apply Indonesian stemming using Sastrawi.
- Preserve original question and context for reference.
- Map doc_labels from original English data to Indonesian stemmed data by _id.
- Save stemmed and mapped data.

Colab-specific commands, Google Drive paths, installation commands,
visualization logs, and manual prompts are removed.
"""

import argparse
import json
import os
import re
from collections import Counter
from typing import Dict, List, Tuple

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
    """Preprocess one 2WikiMultihopQA sample using stemming only."""
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
        "evidences",
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


def derive_doc_labels_from_english(sample: Dict) -> Tuple[List[int], List[str]]:
    """
    Derive document labels from original English 2Wiki data.

    This uses English supporting_facts and English context titles,
    then later applies the labels to Indonesian data by matching _id.
    """
    supporting_facts = sample.get("supporting_facts", [])
    context = sample.get("context", [])

    gold_titles = set()

    for supporting_fact in supporting_facts:
        if isinstance(supporting_fact, (list, tuple)) and len(supporting_fact) >= 1:
            gold_titles.add(supporting_fact[0])

    doc_labels = []

    for title, _ in context:
        doc_labels.append(1 if title in gold_titles else 0)

    return doc_labels, list(gold_titles)


def build_doc_label_mapping(english_data: List[Dict]) -> Dict[str, List[int]]:
    """Build _id to doc_labels mapping from English 2Wiki data."""
    mapping = {}
    gold_counts = []

    for sample in tqdm(english_data, desc="Building doc label mapping"):
        sample_id = sample.get("_id")

        if not sample_id:
            continue

        doc_labels, _ = derive_doc_labels_from_english(sample)

        mapping[sample_id] = doc_labels
        gold_counts.append(sum(doc_labels))

    distribution = Counter(gold_counts)

    print("Gold document distribution:")
    for num_gold, count in sorted(distribution.items()):
        print(f"  {num_gold} gold docs: {count}")

    return mapping


def apply_doc_label_mapping(
    indonesian_data: List[Dict],
    mapping: Dict[str, List[int]],
) -> Tuple[List[Dict], Dict]:
    """Apply doc_labels mapping to Indonesian data using _id."""
    mapped = 0
    missing = 0

    for sample in tqdm(indonesian_data, desc="Applying doc label mapping"):
        sample_id = sample.get("_id")

        if sample_id in mapping:
            sample["doc_labels"] = mapping[sample_id]
            mapped += 1
        else:
            sample["doc_labels"] = [0] * len(sample.get("context", []))
            missing += 1

    metadata = {
        "mapped": mapped,
        "missing": missing,
        "total": len(indonesian_data),
    }

    return indonesian_data, metadata


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
    """Run 2Wiki preprocessing and optional doc label mapping."""
    parser = argparse.ArgumentParser(
        description="Preprocess Indonesian 2WikiMultihopQA using Sastrawi stemming."
    )

    parser.add_argument("--train-input", required=True, help="Path to translated 2Wiki train JSON.")
    parser.add_argument("--test-input", required=True, help="Path to translated 2Wiki test JSON.")
    parser.add_argument("--output-dir", required=True, help="Directory for preprocessed output.")

    parser.add_argument(
        "--english-train",
        default=None,
        help="Optional path to original English 2Wiki train JSON for doc_labels mapping.",
    )
    parser.add_argument(
        "--english-test",
        default=None,
        help="Optional path to original English 2Wiki test JSON for doc_labels mapping.",
    )

    args = parser.parse_args()

    stemmer = create_stemmer()

    train_data = load_json(args.train_input)
    test_data = load_json(args.test_input)

    train_processed = process_dataset(
        data=train_data,
        stemmer=stemmer,
        description="2Wiki train",
    )

    test_processed = process_dataset(
        data=test_data,
        stemmer=stemmer,
        description="2Wiki test",
    )

    mapping_metadata = {}

    if args.english_train:
        english_train = load_json(args.english_train)
        train_mapping = build_doc_label_mapping(english_train)
        train_processed, train_mapping_metadata = apply_doc_label_mapping(
            train_processed,
            train_mapping,
        )
        mapping_metadata["train"] = train_mapping_metadata

    if args.english_test:
        english_test = load_json(args.english_test)
        test_mapping = build_doc_label_mapping(english_test)
        test_processed, test_mapping_metadata = apply_doc_label_mapping(
            test_processed,
            test_mapping,
        )
        mapping_metadata["test"] = test_mapping_metadata

    train_output = os.path.join(args.output_dir, "train_16k_stemmed_mapped.json")
    test_output = os.path.join(args.output_dir, "test_4k_stemmed_mapped.json")
    metadata_output = os.path.join(args.output_dir, "preprocessing_metadata.json")

    save_json(train_processed, train_output)
    save_json(test_processed, test_output)

    metadata = {
        "dataset": "2WikiMultihopQA",
        "language": "Indonesian",
        "preprocessing": "stemming_only",
        "stemmer": "Sastrawi",
        "train_samples": len(train_processed),
        "test_samples": len(test_processed),
        "doc_label_mapping": mapping_metadata,
        "notes": "doc_labels are mapped from original English data by _id when English data is provided.",
    }

    save_json(metadata, metadata_output)

    print("2Wiki preprocessing complete")
    print(f"Train output: {train_output}")
    print(f"Test output: {test_output}")
    print(f"Metadata output: {metadata_output}")


if __name__ == "__main__":
    main()
