# Experiment Artifacts

This document describes the external experiment artifacts for the project:

**Adaptive Retrieval-Augmented Generation for Indonesian Multi-Hop Question Answering**

The artifact folder is stored externally to keep this GitHub repository lightweight. It contains datasets, trained model checkpoints, and complete evaluation outputs used in the experiment.

## External Artifact Folder

Datasets, model checkpoints, and complete evaluation outputs are available at:

```text
https://drive.google.com/drive/folders/1fdnMLv5pD2qaIK5wJ603VQ9Po2yITkNT?usp=sharing
```

## Artifact Contents

The external folder contains:

```text
Dataset/
├── Raw/
├── Translated/
└── Stemmed/

Model/
├── classifier_model.pt
├── hotpot_first_hop.pt
├── hotpot_second_hop.pt
└── 2wiki_ranking_4Hop.pt

Output/
├── pipeline_results.json
├── bertscore_results.json
└── llm_judged_results.json
```

## Dataset Files

The `Dataset/` directory contains three stages of data preparation.

| Folder        | Description                                                             |
| ------------- | ----------------------------------------------------------------------- |
| `Raw/`        | Original English dataset samples used in the experiment.                |
| `Translated/` | Indonesian-translated datasets generated using NLLB-200.                |
| `Stemmed/`    | Stemmed Indonesian datasets used for retrieval training and evaluation. |

## Model Files

The `Model/` directory contains trained model checkpoints used in the experiment.

| File                    | Description                                                        |
| ----------------------- | ------------------------------------------------------------------ |
| `classifier_model.pt`   | Query complexity classifier checkpoint.                            |
| `hotpot_first_hop.pt`   | First-hop retrieval model for 2-hop HotpotQA questions.            |
| `hotpot_second_hop.pt`  | Second-hop retrieval model for 2-hop HotpotQA questions.           |
| `2wiki_ranking_4Hop.pt` | Ranking-based retrieval model for 4-hop 2WikiMultihopQA questions. |

## Output Files

The `Output/` directory contains complete evaluation outputs.

| File                      | Description                                                                                                                                  |
| ------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| `pipeline_results.json`   | Full adaptive RAG pipeline output, including predicted complexity, retrieved document indices, generated answers, Exact Match, and F1-score. |
| `bertscore_results.json`  | Full BERTScore evaluation output for generated answers.                                                                                      |
| `llm_judged_results.json` | Full LLM-as-Judge evaluation output for generated answers.                                                                                   |

