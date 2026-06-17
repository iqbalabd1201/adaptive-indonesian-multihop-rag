"""
train_retrieval_4hop.py

Cleaned training artifact for 4-hop retrieval on Indonesian 2WikiMultihopQA.

Task:
Given a 4-hop question and candidate documents, train a ranking-based
retrieval model to select four supporting documents.

Original workflow:
- Load stemmed and mapped Indonesian 2WikiMultihopQA data.
- Use doc_labels to identify four gold supporting documents.
- Train XLM-RoBERTa retrieval model with ranking objective.
- Evaluate Top-4 retrieval accuracy.

Colab-specific commands, Google Drive paths, installation commands,
debug output, and long experiment logs are removed.
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


class Retrieval4HopConfig:
    """Configuration for 4-hop retrieval."""

    model_name = "xlm-roberta-base"
    max_seq_length = 512
    batch_size = 2
    gradient_accumulation_steps = 4
    learning_rate = 5e-6
    weight_decay = 0.01
    num_epochs = 2
    warmup_ratio = 0.1
    margin = 0.5
    gradient_clip = 1.0


class Ranking4HopDataset(Dataset):
    """
    Ranking dataset for 4-hop retrieval.

    Each item contains one question and all candidate documents.
    The model learns to assign higher scores to four supporting documents.
    """

    def __init__(self, data: List[Dict], tokenizer, max_length: int = 512):
        self.data = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        for item in data:
            contexts = item.get("context", [])
            doc_labels = item.get("doc_labels", [])

            if len(contexts) != len(doc_labels):
                continue

            if sum(doc_labels) != 4:
                continue

            self.data.append(item)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Dict:
        sample = self.data[index]

        question = sample.get("question", "")
        contexts = sample.get("context", [])
        doc_labels = sample.get("doc_labels", [])

        input_ids = []
        attention_masks = []

        for doc_title, doc_sentences in contexts:
            doc_text = doc_title + " " + " ".join(doc_sentences)

            encoding = self.tokenizer(
                str(question),
                str(doc_text),
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
                return_token_type_ids=False,
            )

            input_ids.append(encoding["input_ids"].squeeze(0))
            attention_masks.append(encoding["attention_mask"].squeeze(0))

        return {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attention_masks),
            "labels": torch.tensor(doc_labels, dtype=torch.float),
        }


class RetrievalModel(nn.Module):
    """XLM-RoBERTa-based document scoring model."""

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


def ranking_loss(scores, labels, margin: float = 0.5):
    """
    Margin ranking loss.

    Positive documents should receive higher scores than negative documents.
    """
    batch_size = scores.size(0)

    losses = []

    for batch_index in range(batch_size):
        sample_scores = scores[batch_index]
        sample_labels = labels[batch_index]

        positive_scores = sample_scores[sample_labels == 1]
        negative_scores = sample_scores[sample_labels == 0]

        if len(positive_scores) == 0 or len(negative_scores) == 0:
            continue

        differences = positive_scores.unsqueeze(1) - negative_scores.unsqueeze(0)
        sample_loss = F.relu(margin - differences).mean()
        losses.append(sample_loss)

    if not losses:
        return torch.tensor(0.0, device=scores.device, requires_grad=True)

    return torch.stack(losses).mean()


def load_json(file_path: str) -> List[Dict]:
    """Load JSON data."""
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    scheduler,
    device,
    config: Retrieval4HopConfig,
) -> float:
    """Train 4-hop ranking model for one epoch."""
    model.train()

    total_loss = 0.0
    optimizer.zero_grad()

    for step, batch in enumerate(tqdm(dataloader, desc="Training")):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        batch_size, num_docs, seq_length = input_ids.shape

        flat_input_ids = input_ids.view(batch_size * num_docs, seq_length)
        flat_attention_mask = attention_mask.view(batch_size * num_docs, seq_length)

        flat_scores = model(flat_input_ids, flat_attention_mask)
        scores = flat_scores.view(batch_size, num_docs)

        loss = ranking_loss(scores, labels, margin=config.margin)
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

        score = model(input_ids, attention_mask).item()
        scores.append((index, score))

    scores.sort(key=lambda item: item[1], reverse=True)

    return scores


@torch.no_grad()
def evaluate_top4(
    data: List[Dict],
    model,
    tokenizer,
    device,
    max_length: int,
) -> float:
    """
    Evaluate Top-4 retrieval accuracy.

    A sample is correct if all four gold documents are retrieved
    within the top four ranked documents.
    """
    total = 0
    correct = 0

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

        if len(gold_indices) != 4:
            continue

        ranked_documents = score_documents(
            question=question,
            contexts=contexts,
            model=model,
            tokenizer=tokenizer,
            device=device,
            max_length=max_length,
        )

        top4_indices = {
            index
            for index, _ in ranked_documents[:4]
        }

        if gold_indices == top4_indices:
            correct += 1

        total += 1

    return correct / total * 100 if total else 0.0


def save_model(model, tokenizer, output_dir: str) -> None:
    """Save trained model and tokenizer."""
    os.makedirs(output_dir, exist_ok=True)

    torch.save(model.state_dict(), os.path.join(output_dir, "model.pt"))
    tokenizer.save_pretrained(output_dir)


def main() -> None:
    """Train 4-hop retrieval model."""
    parser = argparse.ArgumentParser(
        description="Train 4-hop ranking retrieval model for Indonesian 2WikiMultihopQA."
    )

    parser.add_argument("--train-file", required=True, help="Path to stemmed and mapped 2Wiki train JSON.")
    parser.add_argument("--test-file", required=True, help="Path to stemmed and mapped 2Wiki test JSON.")
    parser.add_argument("--output-dir", required=True, help="Directory to save model.")
    parser.add_argument(
        "--init-checkpoint",
        default=None,
        help="Optional checkpoint from 2-hop retrieval model for transfer learning.",
    )

    args = parser.parse_args()

    set_seed()

    config = Retrieval4HopConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_data = load_json(args.train_file)
    test_data = load_json(args.test_file)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    train_dataset = Ranking4HopDataset(
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

    if args.init_checkpoint:
        model.load_state_dict(torch.load(args.init_checkpoint, map_location=device))

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

    best_top4 = 0.0

    for epoch in range(1, config.num_epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            config=config,
        )

        top4_accuracy = evaluate_top4(
            data=test_data,
            model=model,
            tokenizer=tokenizer,
            device=device,
            max_length=config.max_seq_length,
        )

        print(f"Epoch {epoch}/{config.num_epochs}")
        print(f"Training loss: {train_loss:.4f}")
        print(f"Validation Top-4 accuracy: {top4_accuracy:.2f}%")

        if top4_accuracy > best_top4:
            best_top4 = top4_accuracy
            save_model(model, tokenizer, args.output_dir)

    print("4-hop retrieval training complete")
    print(f"Best Top-4 accuracy: {best_top4:.2f}%")
    print(f"Model saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
