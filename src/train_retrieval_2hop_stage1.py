"""
train_retrieval_2hop_stage1.py

Cleaned training artifact for Stage 1 retrieval on Indonesian HotpotQA.

Task:
Given a question and candidate documents, train a retrieval model to identify
the first supporting document for 2-hop questions.

Original workflow:
- Load stemmed Indonesian HotpotQA data.
- Build question-document pairs.
- Train XLM-RoBERTa retrieval model.
- Evaluate whether at least one supporting document appears in top-ranked results.

Colab-specific commands, Google Drive paths, installation commands,
large logs, and manual experiment outputs are removed.
"""

import argparse
import json
import os
import random
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup


SEED = 42


def set_seed(seed: int = SEED) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class RetrievalConfig:
    """Configuration for Stage 1 retrieval."""

    model_name = "xlm-roberta-base"
    max_seq_length = 384
    batch_size = 8
    gradient_accumulation_steps = 4
    learning_rate = 2e-5
    weight_decay = 0.01
    num_epochs = 3
    warmup_ratio = 0.1
    gradient_clip = 1.0


class BinaryRetrievalDataset(Dataset):
    """
    Binary retrieval dataset.

    Each sample is a question-document pair:
    - label 1: supporting document
    - label 0: distractor document
    """

    def __init__(self, data: List[Dict], tokenizer, max_length: int = 384):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        for item in data:
            question = item.get("question", "")
            contexts = item.get("context", [])
            doc_labels = item.get("doc_labels", [])

            if len(contexts) != len(doc_labels):
                continue

            for index, (doc_title, doc_sentences) in enumerate(contexts):
                doc_text = doc_title + " " + " ".join(doc_sentences)
                label = doc_labels[index]

                self.samples.append({
                    "question": question,
                    "document": doc_text,
                    "label": float(label),
                })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict:
        sample = self.samples[index]

        encoding = self.tokenizer(
            str(sample["question"]),
            str(sample["document"]),
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            return_token_type_ids=False,
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(sample["label"], dtype=torch.float),
        }


class RetrievalModel(nn.Module):
    """XLM-RoBERTa-based binary retrieval model."""

    def __init__(self, model_name: str = "xlm-roberta-base"):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.scorer = nn.Linear(self.encoder.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        pooled = outputs.last_hidden_state[:, 0, :]
        pooled = self.dropout(pooled)

        logits = self.scorer(pooled).squeeze(-1)

        return logits


def load_json(file_path: str) -> List[Dict]:
    """Load JSON data."""
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def binary_loss(logits, labels):
    """Binary cross entropy loss with logits."""
    return F.binary_cross_entropy_with_logits(logits, labels)


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    scheduler,
    device,
    config: RetrievalConfig,
) -> float:
    """Train retrieval model for one epoch."""
    model.train()

    total_loss = 0.0
    optimizer.zero_grad()

    for step, batch in enumerate(tqdm(dataloader, desc="Training")):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        logits = model(input_ids, attention_mask)
        loss = binary_loss(logits, labels)
        loss = loss / config.gradient_accumulation_steps

        loss.backward()

        if (step + 1) % config.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * config.gradient_accumulation_steps

    return total_loss / len(dataloader)


@torch.no_grad()
def score_documents(
    question: str,
    contexts: List,
    model,
    tokenizer,
    device,
    max_length: int,
) -> List[Tuple[int, float]]:
    """Score all candidate documents for one question."""
    model.eval()

    scores = []

    for index, (doc_title, doc_sentences) in enumerate(contexts):
        doc_text = doc_title + " " + " ".join(doc_sentences)

        encoding = tokenizer(
            str(question),
            str(doc_text),
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            return_token_type_ids=False,
        )

        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        logits = model(input_ids, attention_mask)
        score = torch.sigmoid(logits).item()

        scores.append((index, score))

    scores.sort(key=lambda item: item[1], reverse=True)

    return scores


@torch.no_grad()
def evaluate_topk(
    data: List[Dict],
    model,
    tokenizer,
    device,
    max_length: int,
    k_values: List[int],
) -> Dict:
    """Evaluate retrieval accuracy using Top-K."""
    total = 0
    correct_at_k = {k: 0 for k in k_values}

    for sample in tqdm(data, desc="Evaluating"):
        question = sample.get("question", "")
        contexts = sample.get("context", [])
        doc_labels = sample.get("doc_labels", [])

        if len(contexts) != len(doc_labels):
            continue

        gold_indices = {
            index
            for index, label in enumerate(doc_labels)
            if label == 1
        }

        if not gold_indices:
            continue

        ranked_docs = score_documents(
            question=question,
            contexts=contexts,
            model=model,
            tokenizer=tokenizer,
            device=device,
            max_length=max_length,
        )

        ranked_indices = [index for index, _ in ranked_docs]

        for k in k_values:
            top_k = set(ranked_indices[:k])

            if gold_indices.intersection(top_k):
                correct_at_k[k] += 1

        total += 1

    return {
        f"top_{k}_accuracy": correct_at_k[k] / total * 100 if total else 0
        for k in k_values
    }


def save_model(model, tokenizer, output_dir: str) -> None:
    """Save trained model and tokenizer."""
    os.makedirs(output_dir, exist_ok=True)

    torch.save(model.state_dict(), os.path.join(output_dir, "model.pt"))
    tokenizer.save_pretrained(output_dir)


def main() -> None:
    """Train Stage 1 retrieval model."""
    parser = argparse.ArgumentParser(
        description="Train Stage 1 retrieval model for Indonesian HotpotQA."
    )

    parser.add_argument("--train-file", required=True, help="Path to stemmed HotpotQA train JSON.")
    parser.add_argument("--val-file", required=True, help="Path to stemmed HotpotQA validation JSON.")
    parser.add_argument("--output-dir", required=True, help="Directory to save model.")

    args = parser.parse_args()

    set_seed()

    config = RetrievalConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_data = load_json(args.train_file)
    val_data = load_json(args.val_file)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    train_dataset = BinaryRetrievalDataset(
        data=train_data,
        tokenizer=tokenizer,
        max_length=config.max_seq_length,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
    )

    model = RetrievalModel(config.model_name).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    total_training_steps = len(train_loader) * config.num_epochs
    warmup_steps = int(total_training_steps * config.warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )

    best_top1 = 0.0

    for epoch in range(1, config.num_epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            config=config,
        )

        metrics = evaluate_topk(
            data=val_data,
            model=model,
            tokenizer=tokenizer,
            device=device,
            max_length=config.max_seq_length,
            k_values=[1, 2, 3, 4],
        )

        print(f"Epoch {epoch}/{config.num_epochs}")
        print(f"Training loss: {train_loss:.4f}")
        print(metrics)

        if metrics["top_1_accuracy"] > best_top1:
            best_top1 = metrics["top_1_accuracy"]
            save_model(model, tokenizer, args.output_dir)

    print("Stage 1 retrieval training complete")
    print(f"Best Top-1 accuracy: {best_top1:.2f}%")
    print(f"Model saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
