"""
translate_2wiki.py

Cleaned translation artifact for 2WikiMultihopQA to Indonesian using
NLLB-200 1.3B.

Original workflow:
- Load 2WikiMultihopQA data from parquet files.
- Parse context and supporting facts.
- Translate question, answer, document titles, and context sentences.
- Use batch processing, translation cache, and checkpointing.

Colab-specific commands, Google Drive paths, test-only blocks, and long
output logs are removed from this public artifact.
"""

import argparse
import hashlib
import json
import os
import time
from typing import Dict, List, Tuple

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


MODEL_NAME = "facebook/nllb-200-1.3B"
SRC_LANG = "eng_Latn"
TGT_LANG = "ind_Latn"
BATCH_SIZE = 16
MAX_LENGTH = 512


translation_cache = {}
cache_hits = 0
cache_misses = 0


def get_cache_key(text: str) -> str:
    """Generate cache key for translated text."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def load_model(model_name: str = MODEL_NAME):
    """Load NLLB tokenizer and translation model."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    model.eval()
    return tokenizer, model


def get_forced_bos_token_id(tokenizer, target_language: str = TGT_LANG) -> int:
    """Get forced BOS token ID for NLLB target language."""
    try:
        return tokenizer.lang_code_to_id.get(target_language)
    except AttributeError:
        return tokenizer.convert_tokens_to_ids(target_language)


def translate_batch(
    texts: List[str],
    tokenizer,
    model,
    target_language: str = TGT_LANG,
    max_length: int = MAX_LENGTH,
    use_cache: bool = True,
) -> List[str]:
    """Translate a batch of texts with optional caching."""
    global cache_hits, cache_misses

    if not texts:
        return []

    results = []
    texts_to_translate = []
    text_indices = []

    for index, text in enumerate(texts):
        if not text or not str(text).strip():
            results.append("")
            continue

        text = str(text)
        cache_key = get_cache_key(text)

        if use_cache and cache_key in translation_cache:
            results.append(translation_cache[cache_key])
            cache_hits += 1
        else:
            results.append(None)
            texts_to_translate.append(text)
            text_indices.append(index)
            cache_misses += 1

    if texts_to_translate:
        inputs = tokenizer(
            texts_to_translate,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )

        if torch.cuda.is_available():
            inputs = {key: value.cuda() for key, value in inputs.items()}

        forced_bos_token_id = get_forced_bos_token_id(tokenizer, target_language)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                forced_bos_token_id=forced_bos_token_id,
                max_length=max_length,
                num_beams=4,
                early_stopping=True,
            )

        translations = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        for original_text, translation, index in zip(
            texts_to_translate,
            translations,
            text_indices,
        ):
            cache_key = get_cache_key(original_text)
            translation_cache[cache_key] = translation
            results[index] = translation

    return results


def parse_context_and_supporting_facts(row) -> Tuple[List, List]:
    """Parse context and supporting facts from parquet row."""
    context = row["context"]

    if isinstance(context, str):
        context = json.loads(context)

    supporting_facts = row["supporting_facts"]

    if isinstance(supporting_facts, str):
        supporting_facts = json.loads(supporting_facts)

    return context, supporting_facts


def dataframe_to_samples(df: pd.DataFrame) -> List[Dict]:
    """Convert 2Wiki dataframe rows into sample dictionaries."""
    samples = []

    for _, row in df.iterrows():
        context, supporting_facts = parse_context_and_supporting_facts(row)

        sample = {
            "_id": row["_id"],
            "question": row["question"],
            "answer": row["answer"],
            "type": row["type"],
            "context": context,
            "supporting_facts": supporting_facts,
        }

        if "evidences" in row:
            sample["evidences"] = row["evidences"]

        samples.append(sample)

    return samples


def translate_sample(sample: Dict, tokenizer, model, batch_size: int = BATCH_SIZE) -> Dict:
    """Translate one 2WikiMultihopQA sample into Indonesian."""
    translated = {
        "_id": sample.get("_id"),
        "type": sample.get("type"),
        "supporting_facts": sample.get("supporting_facts"),
    }

    texts_to_translate = []
    text_map = []

    texts_to_translate.append(sample["question"])
    text_map.append(("question", None, None))

    texts_to_translate.append(sample["answer"])
    text_map.append(("answer", None, None))

    for doc_index, (title, sentences) in enumerate(sample["context"]):
        texts_to_translate.append(title)
        text_map.append(("context_title", doc_index, None))

        for sentence_index, sentence in enumerate(sentences):
            texts_to_translate.append(sentence)
            text_map.append(("context_sentence", doc_index, sentence_index))

    all_translations = []

    for start_index in range(0, len(texts_to_translate), batch_size):
        batch = texts_to_translate[start_index:start_index + batch_size]
        translations = translate_batch(batch, tokenizer, model)
        all_translations.extend(translations)

    translated_context = []
    current_doc_index = -1
    current_title = ""
    current_sentences = []

    for translation, (field_type, doc_index, sentence_index) in zip(
        all_translations,
        text_map,
    ):
        if field_type == "question":
            translated["question"] = translation

        elif field_type == "answer":
            translated["answer"] = translation

        elif field_type == "context_title":
            if current_doc_index >= 0:
                translated_context.append([current_title, current_sentences])

            current_doc_index = doc_index
            current_title = translation
            current_sentences = []

        elif field_type == "context_sentence":
            current_sentences.append(translation)

    if current_doc_index >= 0:
        translated_context.append([current_title, current_sentences])

    translated["context"] = translated_context

    if "evidences" in sample:
        translated["evidences"] = sample["evidences"]

    return translated


def load_checkpoint(checkpoint_file: str):
    """Load translation checkpoint if available."""
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r", encoding="utf-8") as file:
            return json.load(file)

    return None


def save_checkpoint(checkpoint_file: str, translated_data: List[Dict]) -> None:
    """Save translation checkpoint."""
    os.makedirs(os.path.dirname(checkpoint_file), exist_ok=True)

    with open(checkpoint_file, "w", encoding="utf-8") as file:
        json.dump(translated_data, file, ensure_ascii=False, indent=2)


def translate_parquet_file(
    input_file: str,
    output_file: str,
    checkpoint_file: str,
    tokenizer,
    model,
) -> None:
    """Translate one 2WikiMultihopQA parquet file."""
    dataframe = pd.read_parquet(input_file)
    data = dataframe_to_samples(dataframe)

    checkpoint = load_checkpoint(checkpoint_file)
    translated_data = checkpoint if checkpoint else []
    start_index = len(translated_data)

    start_time = time.time()

    for index in tqdm(range(start_index, len(data)), desc=f"Translating {input_file}"):
        try:
            translated_sample = translate_sample(data[index], tokenizer, model)
            translated_data.append(translated_sample)

            if (index + 1) % 100 == 0:
                save_checkpoint(checkpoint_file, translated_data)

        except Exception as error:
            print(f"Error translating sample {index}: {error}")
            save_checkpoint(checkpoint_file, translated_data)
            continue

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(translated_data, file, ensure_ascii=False, indent=2)

    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)

    total_time = time.time() - start_time

    print("Translation complete")
    print(f"Input: {input_file}")
    print(f"Output: {output_file}")
    print(f"Samples translated: {len(translated_data)}")
    print(f"Time: {total_time / 60:.2f} minutes")

    if cache_hits + cache_misses > 0:
        cache_rate = cache_hits / (cache_hits + cache_misses) * 100
        print(f"Cache hit rate: {cache_rate:.2f}%")


def main() -> None:
    """Run 2WikiMultihopQA translation."""
    parser = argparse.ArgumentParser(
        description="Translate 2WikiMultihopQA data to Indonesian."
    )

    parser.add_argument("--test-input", required=True, help="Path to 2Wiki test parquet.")
    parser.add_argument("--test-output", required=True, help="Path to translated test JSON.")
    parser.add_argument("--train-input", required=True, help="Path to 2Wiki train parquet.")
    parser.add_argument("--train-output", required=True, help="Path to translated train JSON.")
    parser.add_argument(
        "--checkpoint-dir",
        default="checkpoints/2wiki_translation",
        help="Directory for translation checkpoints.",
    )

    args = parser.parse_args()

    tokenizer, model = load_model()

    translate_parquet_file(
        input_file=args.test_input,
        output_file=args.test_output,
        checkpoint_file=os.path.join(args.checkpoint_dir, "test_checkpoint.json"),
        tokenizer=tokenizer,
        model=model,
    )

    translate_parquet_file(
        input_file=args.train_input,
        output_file=args.train_output,
        checkpoint_file=os.path.join(args.checkpoint_dir, "train_checkpoint.json"),
        tokenizer=tokenizer,
        model=model,
    )


if __name__ == "__main__":
    main()
