"""
train_retrieval_2hop_stage2.py

Cleaned training artifact for Stage 2 retrieval on Indonesian HotpotQA.

Task:
Given a question, the first retrieved supporting document, and remaining
candidate documents, train a model to retrieve the second supporting document.

Original workflow:
- Load stemmed Indonesian HotpotQA data.
- Use first-hop document information.
- Build contextual second-hop examples.
- Train XLM-RoBERTa retrieval model.
- Evaluate whether the second supporting document is retrieved correctly.

Colab-specific commands, Google Drive paths, installation commands,
debug blocks, and long logs are removed.
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


class SecondHopConfig:
    """Configuration for Stage 2 retrieval."""

    model_name = "xlm-roberta-base"
    max_seq_length = 512
    batch_size = 4
    gradient_accumulation_steps = 4
    learning_rate = 2e-5
    weight_decay = 0.01
    num_epochs = 3
    warmup_ratio = 0.1
    gradient_clip = 1.0


class RetrievalModel(nn.Module):
    """XLM-RoBERTa-based retrieval model."""

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


class SecondHopDataset(Dataset):
    """
    Contextual second-hop dataset.

    For each 2-hop question:
    - choose one gold document as the first-hop context
    - train the model to identify the other gold document
    - use remaining documents as negative candidates
    """

    def __init__(
        self,
        data: List[Dict],
        tokenizer,
        max_length: int = 512,
        error_rate: float = 0.05,
    ):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        for item in data:
            question = item.get("question", "")
            contexts = item.get("context", [])
            doc_labels = item.get("doc_labels", [])

            if len(contexts) != len(doc_labels):
                continue

            gold_indices = [
                index
                for index, label in enumerate(doc_labels)
                if label == 1
            ]

            if len(gold_indices) != 2:
                continue

            if random.random() < (1.0 - error_rate):
                first_index = random.choice(gold_indices)
            else:
                non_gold_indices = [
                    index
                    for index, label in enumerate(doc_labels)
                    if label == 0
                ]
                first_index = random.choice(non_gold_indices) if non_gold_indices else gold_indices[0]

            remaining_gold_indices = [
                index
                for index in gold_indices
                if index != first_index
            ]

            if len(remaining_gold_indices) != 1:
                continue

            second_gold_index = remaining_gold_indices[0]

            first_title, first_sentences = contexts[first_index]
            first_doc_text = first_title + " " + " ".join(first_sentences)

            for candidate_index, (candidate_title, candidate_sentences) in enumerate(contexts):
                if candidate_index == first_index:
                    continue

                candidate_text = candidate_title + " " + " ".join(candidate_sentences)

                label = 1.0 if candidate_index == second_gold_index else 0.0

                self.samples.append({
                    "question": question,
                    "first_document": first_doc_text,
                    "candidate_document": candidate_text,
                    "label": label,
                })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict:
        sample = self.samples[index]

        context_input = sample["first_document"] + " " + sample["candidate_document"]

        encoding = self.tokenizer(
            str(sample["question"]),
            str(context_input),
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
    config: SecondHopConfig,
) -> float:
    """Train second-hop model for one epoch."""
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
def score_second_hop_candidates(
    question: str,
    first_doc_text: str,
    contexts: List,
    first_index: int,
    model,
    tokenizer,
    device,
    max_length: int,
) -> List[Tuple[int, float]]:
    """Score candidate documents for the second hop."""
    model.eval()

    scores = []

    for candidate_index, (candidate_title, candidate_sentences) in enumerate(contexts):
        if candidate_index == first_index:
            continue

        candidate_text = candidate_title + " " + " ".join(candidate_sentences)
        context_input = first_doc_text + " " + candidate_text

        encoding = tokenizer(
            str(question),
            str(context_input),
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

        scores.append((candidate_index, score))

    scores.sort(key=lambda item: item[1], reverse=True)

    return scores


@torch.no_grad()
def evaluate_second_hop(
    data: List[Dict],
    model,
    tokenizer,
    device,
    max_length: int,
) -> float:
    """
    Evaluate second-hop retrieval.

    This evaluation uses a gold first document and checks whether the model
    selects the other gold document as the top-ranked second document.
    """
    total = 0
    correct = 0

    for sample in tqdm(data, desc="Evaluating"):
        question = sample.get("question", "")
        contexts = sample.get("context", [])
        doc_labels = sample.get("doc_labels", [])

        if len(contexts) != len(doc_labels):
            continue

        gold_indices = [
            index
            for index, label in enumerate(doc_labels)
            if label == 1
        ]

        if len(gold_indices) != 2:
            continue

        first_index = gold_indices[0]
        second_gold_index = gold_indices[1]

        first_title, first_sentences = contexts[first_index]
        first_doc_text = first_title + " " + " ".join(first_sentences)

        ranked_candidates = score_second_hop_candidates(
            question=question,
            first_doc_text=first_doc_text,
            contexts=contexts,
            first_index=first_index,
            model=model,
            tokenizer=tokenizer,
            device=device,
            max_length=max_length,
        )

        if not ranked_candidates:
            continue

        top_candidate_index = ranked_candidates[0][0]

        if top_candidate_index == second_gold_index:
            correct += 1

        total += 1

    return correct / total * 100 if total else 0.0


def save_model(model, tokenizer, output_dir: str) -> None:
    """Save trained model and tokenizer."""
    os.makedirs(output_dir, exist_ok=True)

    torch.save(model.state_dict(), os.path.join(output_dir, "model.pt"))
    tokenizer.save_pretrained(output_dir)


def main() -> None:
    """Train Stage 2 retrieval model."""
    parser = argparse.ArgumentParser(
        description="Train Stage 2 retrieval model for Indonesian HotpotQA."
    )

    parser.add_argument("--train-file", required=True, help="Path to stemmed HotpotQA train JSON.")
    parser.add_argument("--val-file", required=True, help="Path to stemmed HotpotQA validation JSON.")
    parser.add_argument("--output-dir", required=True, help="Directory to save model.")
    parser.add_argument("--error-rate", type=float, default=0.05, help="Simulated first-hop error rate.")

    args = parser.parse_args()

    set_seed()

    config = SecondHopConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_data = load_json(args.train_file)
    val_data = load_json(args.val_file)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    train_dataset = SecondHopDataset(
        data=train_data,
        tokenizer=tokenizer,
        max_length=config.max_seq_length,
        error_rate=args.error_rate,
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

    best_accuracy = 0.0

    for epoch in range(1, config.num_epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            config=config,
        )

        accuracy = evaluate_second_hop(
            data=val_data,
            model=model,
            tokenizer=tokenizer,
            device=device,
            max_length=config.max_seq_length,
        )

        print(f"Epoch {epoch}/{config.num_epochs}")
        print(f"Training loss: {train_loss:.4f}")
        print(f"Validation second-hop accuracy: {accuracy:.2f}%")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            save_model(model, tokenizer, args.output_dir)

    print("Stage 2 retrieval training complete")
    print(f"Best validation accuracy: {best_accuracy:.2f}%")
    print(f"Model saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
