# Adaptive Indonesian Multi-Hop RAG

This repository contains cleaned code artifacts for an adaptive Retrieval-Augmented Generation (RAG) system for Indonesian multi-hop question answering.

The system handles different reasoning complexities using adaptive routing:

* **2-hop questions** use two-stage sequential retrieval.
* **4-hop questions** use ranking-based retrieval with transfer learning.
* Retrieved documents are used for answer generation and evaluated with lexical and semantic metrics.

## Research Overview

This project focuses on Indonesian multi-hop question answering using translated versions of HotpotQA and 2WikiMultihopQA.

The main pipeline consists of:

1. Dataset translation to Indonesian.
2. Indonesian text preprocessing with stemming.
3. Query complexity classification.
4. Adaptive retrieval based on predicted complexity.
5. Answer generation using retrieved supporting documents.
6. Evaluation using Exact Match, F1-score, BERTScore, and LLM-as-Judge.

## Repository Structure

```text
adaptive-indonesian-multihop-rag/
├── README.md
├── requirements.txt
├── configs/
│   └── config.example.yaml
├── src/
│   ├── translate_hotpot.py
│   ├── translate_2wiki.py
│   ├── preprocess_hotpot.py
│   ├── preprocess_2wiki.py
│   ├── train_classifier.py
│   ├── train_retrieval_2hop_stage1.py
│   ├── train_retrieval_2hop_stage2.py
│   ├── train_retrieval_4hop.py
│   ├── run_full_pipeline.py
│   ├── evaluate_bertscore.py
│   └── evaluate_llm_judge.py
├── examples/
│   ├── sample_prompt.txt
│   ├── sample_output.json
│   ├── sample_bertscore_output.json
│   └── sample_llm_judge_output.json
└── docs/
    └── experiment_summary.md
```

## Pipeline Steps

### 1. Dataset Translation

The original English datasets are translated into Indonesian using NLLB-200 1.3B.

Files:

```text
src/translate_hotpot.py
src/translate_2wiki.py
```

Purpose:

* Translate HotpotQA 2-hop data into Indonesian.
* Translate 2WikiMultihopQA 4-hop data into Indonesian.
* Preserve question, answer, context, supporting facts, and document structure.
* Use batch processing, caching, and checkpointing.

Example:

```bash
python src/translate_hotpot.py \
  --train-input data/hotpot/train_16k.json \
  --train-output data/hotpot/train_16k_nllb13b_id.json \
  --val-input data/hotpot/val_4k.json \
  --val-output data/hotpot/val_4k_nllb13b_id.json
```

```bash
python src/translate_2wiki.py \
  --test-input data/2wiki/test_4hop_4k.parquet \
  --test-output data/2wiki/test_4hop_4k_nllb13b_id.json \
  --train-input data/2wiki/training_backup_4hop_16k.parquet \
  --train-output data/2wiki/training_backup_4hop_16k_nllb13b_id.json
```

## 2. Text Preprocessing

Indonesian text preprocessing is performed using Sastrawi stemming.

Files:

```text
src/preprocess_hotpot.py
src/preprocess_2wiki.py
```

Purpose:

* Stem Indonesian questions and context documents.
* Preserve original question and context for reference.
* Prepare stemmed data for retrieval training.
* Map 2Wiki document labels from original English data to Indonesian data using sample IDs.

Example:

```bash
python src/preprocess_hotpot.py \
  --train-input data/hotpot/train_16k_nllb13b_id.json \
  --val-input data/hotpot/val_4k_nllb13b_id.json \
  --output-dir data/hotpot_stemmed
```

```bash
python src/preprocess_2wiki.py \
  --train-input data/2wiki/training_backup_4hop_16k_nllb13b_id.json \
  --test-input data/2wiki/test_4hop_4k_nllb13b_id.json \
  --english-train data/2wiki/training_backup_4hop_16k.json \
  --english-test data/2wiki/test_4hop_4k.json \
  --output-dir data/2wiki_stemmed
```

## 3. Query Complexity Classification

A BERT-based classifier predicts whether a question is 2-hop or 4-hop.

File:

```text
src/train_classifier.py
```

Purpose:

* Train a query complexity classifier.
* Use Indonesian HotpotQA as 2-hop data.
* Use Indonesian 2WikiMultihopQA as 4-hop data.
* Predict routing labels for the adaptive retrieval pipeline.

Labels:

```text
0 = 2-hop
1 = 4-hop
```

Example:

```bash
python src/train_classifier.py \
  --hotpot-train data/hotpot/train_16k_nllb13b_id.json \
  --wiki-train data/2wiki/training_backup_4hop_16k_nllb13b_id.json \
  --hotpot-val data/hotpot/val_4k_nllb13b_id.json \
  --wiki-val data/2wiki/test_4hop_4k_nllb13b_id.json \
  --output-model models/bert_hierarchical_indonesian_only.pt
```

## 4. 2-Hop Retrieval Training

The 2-hop retrieval path uses two-stage sequential retrieval.

Files:

```text
src/train_retrieval_2hop_stage1.py
src/train_retrieval_2hop_stage2.py
```

### Stage 1: First-Hop Retrieval

The first model retrieves one supporting document from the candidate documents.

Example:

```bash
python src/train_retrieval_2hop_stage1.py \
  --train-file data/hotpot_stemmed/train_16k_stemmed.json \
  --val-file data/hotpot_stemmed/val_4k_stemmed.json \
  --output-dir models/first_hop_binary
```

### Stage 2: Second-Hop Retrieval

The second model retrieves the next supporting document using the question and the first retrieved document as context.

Example:

```bash
python src/train_retrieval_2hop_stage2.py \
  --train-file data/hotpot_stemmed/train_16k_stemmed.json \
  --val-file data/hotpot_stemmed/val_4k_stemmed.json \
  --output-dir models/second_hop_contextual
```

## 5. 4-Hop Retrieval Training

The 4-hop path uses ranking-based retrieval to select four supporting documents.

File:

```text
src/train_retrieval_4hop.py
```

Purpose:

* Train an XLM-RoBERTa ranking model.
* Use stemmed and mapped 2WikiMultihopQA data.
* Select four supporting documents from the candidate set.
* Support transfer learning from the 2-hop retrieval model.

Example:

```bash
python src/train_retrieval_4hop.py \
  --train-file data/2wiki_stemmed/train_16k_stemmed_mapped.json \
  --test-file data/2wiki_stemmed/test_4k_stemmed_mapped.json \
  --output-dir models/2wiki_finetuned_ranking \
  --init-checkpoint models/first_hop_binary/model.pt
```

## 6. Full Pipeline Evaluation

The full pipeline combines classification, adaptive routing, retrieval, answer generation, and evaluation.

File:

```text
src/run_full_pipeline.py
```

Pipeline flow:

```text
Question
→ Query Complexity Classifier
→ Adaptive Routing
→ 2-hop or 4-hop Retrieval
→ Answer Generation
→ Evaluation
```

Retrieval strategy:

```text
2-hop → Two-stage sequential retrieval
4-hop → Ranking-based retrieval
```

Example:

```bash
python src/run_full_pipeline.py
```

Note: The public artifact provides the cleaned pipeline structure. Local paths, checkpoints, datasets, and API keys are intentionally excluded.

## 7. Semantic Evaluation with BERTScore

BERTScore is used to evaluate semantic similarity between predicted answers and reference answers.

File:

```text
src/evaluate_bertscore.py
```

Purpose:

* Compare predicted answers with reference answers.
* Compute BERTScore precision, recall, and F1.
* Use F1 threshold to determine semantic correctness.
* Analyze confidence distribution and discrepancy against Exact Match.

Example:

```bash
python src/evaluate_bertscore.py \
  --input outputs/pipeline_with_answers.json \
  --output outputs/bertscore_results.json
```

Example output:

```text
examples/sample_bertscore_output.json
```

## 8. Semantic Evaluation with LLM-as-Judge

LLM-as-Judge is used to evaluate whether generated answers are semantically correct.

File:

```text
src/evaluate_llm_judge.py
```

Purpose:

* Compare predicted answers with reference answers.
* Use an LLM judge to assess semantic correctness.
* Return judgment as JSON with correctness, reasoning, and confidence.
* Analyze cases where Exact Match is too strict.

Example:

```bash
export OPENAI_API_KEY="your_api_key_here"

python src/evaluate_llm_judge.py \
  --input outputs/pipeline_with_answers.json \
  --output outputs/llm_judge_results.json
```

Example output:

```text
examples/sample_llm_judge_output.json
```

## Example Files

The `examples/` directory provides small documentation-only samples:

| File                           | Description                             |
| ------------------------------ | --------------------------------------- |
| `sample_prompt.txt`            | Example answer generation prompt format |
| `sample_output.json`           | Example full-pipeline output            |
| `sample_bertscore_output.json` | Example BERTScore evaluation output     |
| `sample_llm_judge_output.json` | Example LLM-as-Judge evaluation output  |

Full experiment outputs are not included because they contain 8,000 samples and may include long retrieved contexts.

## Main Experimental Results

| Component                                | Result |
| ---------------------------------------- | -----: |
| Query complexity classification accuracy | 99.98% |
| Full-pipeline retrieval accuracy         | 93.16% |
| 2-hop retrieval accuracy                 | 89.60% |
| 4-hop retrieval accuracy                 | 96.73% |
| Answer generation Exact Match            | 63.11% |
| Answer generation F1-score               | 75.67% |
| BERTScore accuracy                       | 69.03% |
| LLM-as-Judge accuracy                    | 88.61% |

## Baseline Comparison

| Method                   | Overall Retrieval Accuracy |
| ------------------------ | -------------------------: |
| TF-IDF K=3               |                     64.14% |
| BM25 K=3                 |                     66.46% |
| DPR K=3                  |                     82.90% |
| DPR K=4                  |                     87.31% |
| Proposed adaptive method |                     93.16% |

The proposed adaptive method outperformed the best baseline, DPR K=4, by **5.85 percentage points**.

## Notes

This repository contains cleaned code artifacts for research documentation. The following files are not included:

* Full datasets
* Model checkpoints
* Google Drive paths
* API keys
* Full 8,000-sample experiment outputs
* Long experiment logs
* Temporary checkpoint files

To reproduce the full experiment, users need to prepare the translated datasets, trained checkpoints, and local configuration paths.

## Security Notice

Do not commit API keys, private Google Drive paths, or raw Colab notebooks containing credentials. Use environment variables such as:

```bash
export OPENAI_API_KEY="your_api_key_here"
```

## Citation

If you use this repository, please cite the related thesis or research work:

```text
Iqbal Abdul Rahman. Adaptive Retrieval-Augmented Generation for Indonesian Multi-Hop Question Answering.
Master Thesis, 2026.
```
