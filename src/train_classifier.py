"""
train_classifier.py

Cleaned training artifact for query complexity classification.

Task:
Classify Indonesian multi-hop questions into:
- 0: 2-hop question
- 1: 4-hop question

Original workflow:
- Load Indonesian HotpotQA and 2WikiMultihopQA datasets.
- Train a BERT-based classifier using bert-base-multilingual-cased.
- Evaluate classification accuracy, precision, recall, F1-score,
  and confusion matrix.

Colab-specific paths, Google Drive mounting, and output logs are removed.
"""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup


SEED = 42
LABEL_2HOP = 0
LABEL_4HOP = 1
LABEL_NAMES = ["2-hop", "4-hop"]


def set_seed(seed: int = SEED) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ClassifierConfig:
    """Configuration for the BERT-based query complexity classifier."""

    bert_model = "bert-base-multilingual-cased"
    max_length = 128
    hidden_dim = 768
    num_heads = 12
    num_classes = 2

    batch_size = 32
    val_batch_size = 64
    learning_rate = 2e-5
    weight_decay = 0.01
    num_epochs = 5
    warmup_ratio = 0.1
    gradient_clip = 1.0

    dropout_1 = 0.3
    dropout_2 = 0.2


def load_json_questions(file_path: str, label: int) -> List[Dict]:
    """Load questions from a JSON dataset and assign a class label."""
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    if isinstance(data, dict):
        if "data" in data:
            data = data["data"]
        elif "questions" in data:
            data = data["questions"]

    samples = []

    for item in data:
        if "question" in item:
            samples.append({
                "question": item["question"],
                "label": label,
            })

    return samples


def load_datasets(
    hotpot_train: str,
    wiki_train: str,
    hotpot_val: str,
    wiki_val: str,
) -> Tuple[List[Dict], List[Dict]]:
    """Load training and validation datasets."""
    train_data = (
        load_json_questions(hotpot_train, LABEL_2HOP)
        + load_json_questions(wiki_train, LABEL_4HOP)
    )

    val_data = (
        load_json_questions(hotpot_val, LABEL_2HOP)
        + load_json_questions(wiki_val, LABEL_4HOP)
    )

    random.shuffle(train_data)
    random.shuffle(val_data)

    return train_data, val_data


class BERTHierarchicalModel(nn.Module):
    """
    BERT-based classifier for query complexity classification.

    Architecture:
    1. BERT encoder
    2. Multi-head decomposition attention
    3. Complexity-aware pooling
    4. Feed-forward classification head
    """

    def __init__(self, config: ClassifierConfig):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(config.bert_model)

        for param in self.encoder.embeddings.parameters():
            param.requires_grad = False

        for layer_index in range(8):
            for param in self.encoder.encoder.layer[layer_index].parameters():
                param.requires_grad = False

        self.num_heads = config.num_heads
        self.head_dim = config.hidden_dim // config.num_heads

        self.query_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.key_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.value_proj = nn.Linear(config.hidden_dim, config.hidden_dim)

        self.complexity_attention = nn.Sequential(
            nn.Linear(config.hidden_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 1),
        )

        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_dim, 512),
            nn.ReLU(),
            nn.Dropout(config.dropout_1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(config.dropout_2),
            nn.Linear(256, config.num_classes),
        )

    def multi_head_decomposition(self, hidden_states, attention_mask):
        """Apply multi-head decomposition attention."""
        batch_size, seq_len, hidden_dim = hidden_states.shape

        query = self.query_proj(hidden_states).view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim,
        )
        key = self.key_proj(hidden_states).view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim,
        )
        value = self.value_proj(hidden_states).view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim,
        )

        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        attention_scores = torch.matmul(query, key.transpose(-2, -1))
        attention_scores = attention_scores / (self.head_dim ** 0.5)

        attention_scores = attention_scores.masked_fill(
            attention_mask.unsqueeze(1).unsqueeze(2) == 0,
            -1e9,
        )

        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_output = torch.matmul(attention_weights, value)

        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.view(batch_size, seq_len, hidden_dim)

        return attention_output

    def forward(self, input_ids, attention_mask):
        """Forward pass."""
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        hidden_states = encoder_outputs.last_hidden_state

        decomposed = self.multi_head_decomposition(hidden_states, attention_mask)

        complexity_scores = self.complexity_attention(decomposed)
        complexity_scores = complexity_scores.masked_fill(
            attention_mask.unsqueeze(-1) == 0,
            -1e9,
        )

        complexity_weights = F.softmax(complexity_scores, dim=1)
        pooled_output = (decomposed * complexity_weights).sum(dim=1)

        logits = self.classifier(pooled_output)

        return logits


def prepare_dataloader(
    data: List[Dict],
    tokenizer,
    max_length: int,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    """Tokenize questions and create DataLoader."""
    questions = [item["question"] for item in data]
    labels = [item["label"] for item in data]

    encodings = tokenizer(
        questions,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    dataset = TensorDataset(
        encodings["input_ids"],
        encodings["attention_mask"],
        torch.tensor(labels),
    )

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    scheduler,
    criterion,
    device,
    gradient_clip: float,
) -> Tuple[float, float]:
    """Train model for one epoch."""
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for input_ids, attention_mask, labels in tqdm(dataloader, desc="Training"):
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

        predictions = torch.argmax(logits, dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    return total_loss / len(dataloader), correct / total


@torch.no_grad()
def evaluate(model, dataloader, device) -> Tuple[float, List[int], List[int]]:
    """Evaluate classifier."""
    model.eval()

    all_predictions = []
    all_labels = []

    for input_ids, attention_mask, labels in tqdm(dataloader, desc="Evaluating"):
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        logits = model(input_ids, attention_mask)
        predictions = torch.argmax(logits, dim=1)

        all_predictions.extend(predictions.cpu().tolist())
        all_labels.extend(labels.tolist())

    accuracy = accuracy_score(all_labels, all_predictions)

    return accuracy, all_predictions, all_labels


def print_detailed_metrics(labels: List[int], predictions: List[int]) -> None:
    """Print classification report, confusion matrix, and per-class metrics."""
    print("\nClassification Report")
    print("=====================")
    print(
        classification_report(
            labels,
            predictions,
            target_names=LABEL_NAMES,
            digits=4,
        )
    )

    matrix = confusion_matrix(labels, predictions)

    print("\nConfusion Matrix")
    print("================")
    print(matrix)

    precision, recall, f1, support = precision_recall_fscore_support(
        labels,
        predictions,
        average=None,
    )

    print("\nPer-Class Metrics")
    print("=================")

    for index, label_name in enumerate(LABEL_NAMES):
        print(f"{label_name}:")
        print(f"  Precision: {precision[index]:.4f}")
        print(f"  Recall:    {recall[index]:.4f}")
        print(f"  F1-score:  {f1[index]:.4f}")
        print(f"  Support:   {support[index]}")


def main() -> None:
    """Train and evaluate the query complexity classifier."""
    parser = argparse.ArgumentParser(
        description="Train BERT classifier for 2-hop vs 4-hop question classification."
    )

    parser.add_argument("--hotpot-train", required=True, help="Path to HotpotQA train JSON.")
    parser.add_argument("--wiki-train", required=True, help="Path to 2Wiki train JSON.")
    parser.add_argument("--hotpot-val", required=True, help="Path to HotpotQA validation JSON.")
    parser.add_argument("--wiki-val", required=True, help="Path to 2Wiki validation JSON.")
    parser.add_argument("--output-model", required=True, help="Path to save trained model.")

    args = parser.parse_args()

    set_seed()

    config = ClassifierConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_data, val_data = load_datasets(
        hotpot_train=args.hotpot_train,
        wiki_train=args.wiki_train,
        hotpot_val=args.hotpot_val,
        wiki_val=args.wiki_val,
    )

    tokenizer = AutoTokenizer.from_pretrained(config.bert_model)

    train_loader = prepare_dataloader(
        data=train_data,
        tokenizer=tokenizer,
        max_length=config.max_length,
        batch_size=config.batch_size,
        shuffle=True,
    )

    val_loader = prepare_dataloader(
        data=val_data,
        tokenizer=tokenizer,
        max_length=config.max_length,
        batch_size=config.val_batch_size,
        shuffle=False,
    )

    model = BERTHierarchicalModel(config).to(device)

    criterion = nn.CrossEntropyLoss()

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

    best_val_accuracy = 0.0

    for epoch in range(1, config.num_epochs + 1):
        train_loss, train_accuracy = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            device=device,
            gradient_clip=config.gradient_clip,
        )

        val_accuracy, predictions, labels = evaluate(model, val_loader, device)

        print(
            f"Epoch {epoch}/{config.num_epochs} | "
            f"Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy:.4f} | "
            f"Val Acc: {val_accuracy:.4f}"
        )

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy

            output_dir = Path(args.output_model).parent
            output_dir.mkdir(parents=True, exist_ok=True)

            torch.save(model.state_dict(), args.output_model)

    model.load_state_dict(torch.load(args.output_model, map_location=device))

    val_accuracy, predictions, labels = evaluate(model, val_loader, device)

    print(f"\nBest validation accuracy: {best_val_accuracy:.4f}")
    print_detailed_metrics(labels, predictions)


if __name__ == "__main__":
    main()
