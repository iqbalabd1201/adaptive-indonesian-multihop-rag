# Experiment Summary

This document summarizes the main experimental setup and results of the adaptive Indonesian multi-hop Retrieval-Augmented Generation (RAG) system.

## 1. Research Objective

The objective of this experiment is to develop an adaptive RAG system for Indonesian multi-hop question answering.

The system adapts its retrieval strategy based on the reasoning complexity of each question:

- **2-hop questions** are processed using two-stage sequential retrieval.
- **4-hop questions** are processed using ranking-based retrieval with transfer learning.

The retrieved documents are then used for answer generation and evaluated using lexical and semantic metrics.

## 2. Datasets

The experiment uses two multi-hop question answering datasets translated into Indonesian:

| Dataset | Reasoning Type | Usage |
|---|---|---|
| HotpotQA | 2-hop | 2-hop retrieval and answer generation |
| 2WikiMultihopQA | 4-hop | 4-hop retrieval and answer generation |

The datasets were translated into Indonesian using NLLB-200 1.3B.

### Dataset Size

| Split | HotpotQA | 2WikiMultihopQA | Total |
|---|---:|---:|---:|
| Training | 16,000 | 16,000 | 32,000 |
| Evaluation | 4,000 | 4,000 | 8,000 |
| Total | 20,000 | 20,000 | 40,000 |

## 3. System Pipeline

```text
Question
→ Query Complexity Classification
→ Adaptive Routing
→ Retrieval
→ Answer Generation
→ Evaluation
```

The classifier predicts whether the input question is a 2-hop or 4-hop question.

```text
2-hop prediction → Two-stage sequential retrieval
4-hop prediction → Ranking-based retrieval
```

## 4. Query Complexity Classification

| Item | Description |
|---|---|
| Base model | bert-base-multilingual-cased |
| Input | Question only |
| Labels | 0 = 2-hop, 1 = 4-hop |
| Training data | 16,000 HotpotQA + 16,000 2WikiMultihopQA |
| Evaluation data | 4,000 HotpotQA + 4,000 2WikiMultihopQA |

### Classification Result

| Metric | Result |
|---|---:|
| Accuracy | 99.98% |
| Correct predictions | 7,998 / 8,000 |
| 2-hop correct | 3,999 / 4,000 |
| 4-hop correct | 3,999 / 4,000 |

The classifier achieved very high accuracy and was used as the control signal for adaptive routing.

## 5. Retrieval Methods

### 5.1 2-Hop Retrieval

The 2-hop retrieval path uses a two-stage sequential strategy:

1. **Stage 1** retrieves the first supporting document.
2. **Stage 2** retrieves the second supporting document using the question and the first retrieved document as context.

| Component | Model | Strategy |
|---|---|---|
| Stage 1 | XLM-RoBERTa-base | Question-document scoring |
| Stage 2 | XLM-RoBERTa-base | Contextual second-hop retrieval |

### 5.2 4-Hop Retrieval

The 4-hop retrieval path uses ranking-based retrieval.

| Component | Model | Strategy |
|---|---|---|
| 4-hop retrieval | XLM-RoBERTa-base | Ranking-based retrieval |
| Training strategy | Transfer learning | Initialized from 2-hop retrieval model |

The 4-hop retrieval model was fine-tuned from the 2-hop model using a lower learning rate and fewer epochs.

## 6. Full-Pipeline Retrieval Results

| Retrieval Path | Correct / Total | Accuracy |
|---|---:|---:|
| 2-hop path | 3,584 / 4,000 | 89.60% |
| 4-hop path | 3,869 / 4,000 | 96.73% |
| Overall | 7,453 / 8,000 | 93.16% |

The overall retrieval accuracy of the adaptive system is **93.16%**.

## 7. Baseline Comparison

| Method | Overall Accuracy |
|---|---:|
| TF-IDF K=3 | 64.14% |
| BM25 K=3 | 66.46% |
| DPR K=3 | 82.90% |
| DPR K=4 | 87.31% |
| Proposed adaptive method | 93.16% |

The proposed method outperformed the best baseline, DPR K=4, by **5.85 percentage points**.

## 8. Ablation Study

### 8.1 Fixed-K Retrieval vs Adaptive Routing

| Setting | Overall Retrieval EM | 2-Hop Retrieval EM | 4-Hop Retrieval EM |
|---|---:|---:|---:|
| Fixed K=2 | 31.19% | 62.65% | 0.00% |
| Fixed K=3 | 38.67% | 77.68% | 0.00% |
| Fixed K=4 | 90.97% | 85.12% | 96.78% |
| Proposed adaptive method | 93.16% | 89.60% | 96.73% |

Adaptive routing improves performance because it allows the system to retrieve a different number of supporting documents depending on question complexity.

### 8.2 2-Hop Single-Stage Retrieval Ablation

| Model | At Least 1 Gold | Both Gold |
|---|---:|---:|
| Binary | 99.04% | 76.19% |
| Ranking | 98.94% | 75.66% |

This result shows that single-stage retrieval can often retrieve at least one supporting document, but retrieving both supporting documents simultaneously is more difficult. This supports the use of a two-stage sequential retrieval strategy for 2-hop questions.

### 8.3 Stemming Ablation

| Setting | First-Hop Recall@K |
|---|---:|
| Without stemming | 58.30% |
| With stemming | 95.83% |

Stemming improved first-hop retrieval performance by **37.53 percentage points**.

### 8.4 Transfer Learning Ablation for 4-Hop Retrieval

| Setting | 4-Hop Retrieval Result |
|---|---:|
| Zero-shot transfer from 2-hop | 26.62% |
| Fine-tuned 4-hop model | 100.00% |

Transfer learning followed by fine-tuning improved 4-hop component retrieval performance by **73.38 percentage points**.

The 100.00% result refers to component-level evaluation in a closed candidate reranking setting, not open-domain retrieval.

## 9. Answer Generation Results

| Metric | Overall |
|---|---:|
| Exact Match | 63.11% |
| F1-score | 75.67% |

### Answer Generation by Retrieval Correctness

| Metric | Result |
|---|---:|
| EM given retrieval correct | 64.86% |
| F1 given retrieval correct | 77.55% |

The difference between retrieval performance and answer generation performance shows that correct retrieval does not always guarantee exact answer generation.

## 10. Semantic Evaluation

Lexical metrics such as Exact Match can be too strict for Indonesian QA because semantically correct answers may use different wording or translation variants.

### 10.1 BERTScore Evaluation

| Metric | Result |
|---|---:|
| BERTScore accuracy | 69.03% |
| Mean BERTScore F1 | 0.9157 |
| Median BERTScore F1 | 1.00 |
| Threshold | 0.85 |

Confidence distribution:

| Confidence | Count | Percentage |
|---|---:|---:|
| High | 5,034 | 62.93% |
| Medium | 1,466 | 18.33% |
| Low | 1,500 | 18.75% |

### 10.2 LLM-as-Judge Evaluation

| Metric | Result |
|---|---:|
| LLM-as-Judge accuracy | 88.61% |
| Correct answers | 7,089 / 8,000 |
| High confidence | 7,967 |
| Medium confidence | 33 |
| Low confidence | 0 |

The LLM-as-Judge result is higher than Exact Match because it evaluates semantic correctness rather than strict lexical equality.

Examples:

| Reference Answer | Predicted Answer | EM | Semantic Judgment |
|---|---|---:|---|
| publik | keduanya publik | 0 | Correct |
| Cid (Sidney) Corman | Cid Corman | 0 | Correct |
| Industri PPG | PPG Industries | 0 | Correct |
| Italia | opera Italia | 0 | Correct |

## 11. Main Results Summary

| Component | Result |
|---|---:|
| Query complexity classification | 99.98% |
| Full-pipeline retrieval | 93.16% |
| 2-hop retrieval path | 89.60% |
| 4-hop retrieval path | 96.73% |
| Answer generation EM | 63.11% |
| Answer generation F1 | 75.67% |
| BERTScore accuracy | 69.03% |
| LLM-as-Judge accuracy | 88.61% |

## 12. Key Findings

1. Adaptive routing improves retrieval performance compared with fixed-K retrieval.
2. Query complexity classification can effectively route questions into 2-hop and 4-hop retrieval paths.
3. Stemming is important for Indonesian retrieval because it reduces morphological variation.
4. Transfer learning helps adapt the 2-hop retrieval model to 4-hop retrieval.
5. Semantic evaluation is necessary because Exact Match underestimates semantically correct answers.

## 13. Limitations

1. The datasets are translated from English rather than originally written in Indonesian.
2. The retrieval setting uses candidate documents provided in the dataset, not open-domain retrieval from a large external corpus.
3. The answer generation component uses an external LLM API.
4. The 100.00% 4-hop retrieval result is limited to component-level closed candidate reranking.
5. Some automatic metrics may not fully capture factual correctness or answer completeness.

## 14. Future Work

Future work can explore:

1. Building a native Indonesian multi-hop QA dataset.
2. Extending the system to open-domain retrieval.
3. Comparing additional retrieval models such as IndoBERT, DeBERTa, and hybrid retrieval.
4. Evaluating answer generation with open-source self-hosted LLMs.
5. Adding human evaluation for factual correctness, answer completeness, and semantic relevance.
