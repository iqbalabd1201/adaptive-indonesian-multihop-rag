# Adaptive Indonesian Multi-Hop RAG

This repository contains the implementation artifacts for an adaptive Retrieval-Augmented Generation (RAG) system for Indonesian multi-hop question answering.

The system consists of three main components:
1. Query complexity classification for 2-hop and 4-hop questions.
2. Adaptive retrieval using two-stage sequential retrieval for 2-hop questions and ranking-based retrieval with transfer learning for 4-hop questions.
3. Answer generation and evaluation using Exact Match, F1-score, BERTScore, and LLM-as-Judge.

## Repository Structure

- `src/`: Python scripts for translation, training, retrieval, full pipeline evaluation, and semantic evaluation.
- `configs/`: Example configuration files.
- `examples/`: Sample prompts and output examples.
- `docs/`: Summary of experiments and reported results.

## Main Results

| Component | Metric | Result |
|---|---:|---:|
| Query complexity classification | Accuracy | 99.98% |
| Retrieval full pipeline | Overall accuracy | 93.16% |
| Retrieval 2-hop | Accuracy | 89.60% |
| Retrieval 4-hop | Accuracy | 96.73% |
| Answer generation | Exact Match | 63.11% |
| Answer generation | F1-score | 75.67% |
| Semantic evaluation | BERTScore | 69.03% |
| Semantic evaluation | LLM-as-Judge | 88.61% |

## Notes

The original experiments were conducted in Google Colab. For repository documentation, the notebook artifacts are reorganized into Python scripts to improve readability and reproducibility.
