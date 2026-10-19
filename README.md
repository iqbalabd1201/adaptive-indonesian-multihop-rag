# Adaptive candidate reranking for Indonesian multi-hop QA — v2 experiments

Code for the revised experiments of *Benchmark-Defined Adaptive Candidate Reranking for Indonesian
Multi-Hop Question Answering* (JCTA submission 18163, revision round 2).

## v2 protocol

* **No Sastrawi stemming.** Text fed to the scorers is lowercased and whitespace-normalised only.
* **Dedicated validation split.** 10% of each 16,000-question training set (seed 42) is held out:
  14,400 + 14,400 training questions and 1,600 + 1,600 validation questions.
  The 8,000-question evaluation set (4,000 HotpotQA + 4,000 2WikiMultihopQA bridge-comparison) is used once,
  after every checkpoint and configuration has been fixed.
* **The four-document reranker configuration is chosen on validation** from a grid of
  {transfer, scratch} × {5e-6, 2e-5} with ranking loss, plus transfer + BCE.

## Run order (Google Colab, one script per cell)

| Step | Script | Hardware | Approx. time (A100) | Output (`FE2H/v2/`) |
|---|---|---|---|---|
| 0 | `v2_00_prepare.py` | CPU | 5 min | `prepared/{train,val,test}.json` |
| 1 | `v2_01_router.py` | GPU | 10–20 min | `models/router/`, `results/router_*.json` |
| 2 | `v2_02_stage1.py` | GPU | 30–45 min | `models/stage1/` |
| 3 | `v2_03_stage2.py` | GPU | 40–60 min | `models/stage2/` |
| 4 | `v2_04_fourdoc.py` | GPU | 2–2.5 h (resumable) | `models/fourdoc/best/`, `results/fourdoc_*.json` |
| 5 | `v2_05_pipeline_eval.py` | GPU | 15–25 min (incl. latency) | `results/pipeline_v2*.json` |
| 6 | `v2_06_answers.py` | API (+CPU) | 1–3 h | `results/answers_v2.json` |
| 7 | `v2_07_probes_crosssource.py` | CPU (+GPU for part 2) | 5 min | `results/routing_diagnostics_v2.json` |

Steps 1 and 7 (part 1) are independent of steps 2–4. Step 6 needs an OpenAI API key, which goes in the
`OPENAI_API_KEY` placeholder. Never commit a real key.

## Data

* HotpotQA, distractor setting (https://hotpotqa.github.io), and 2WikiMultihopQA (https://github.com/Alab-NII/2wikimultihop).
  From 2WikiMultihopQA, only the bridge-comparison instances with exactly four gold supporting-document titles are used.
  Each question keeps the original ten-document distractor pool of its benchmark.
* Both datasets are translated EN→ID with `facebook/nllb-200-1.3B`. Questions, answers, titles and sentences are
  translated, and each question keeps its original ten-document candidate pool and gold labels.
  The translation pipeline is described in the companion benchmark paper [26].
* Expected input files on Drive: `processed_data/{train_16k,val_4k}_nllb13b_id.json` and
  `2wiki_processed/{training_backup_4hop_16k,test_4hop_4k}_nllb13b_id*.json`.

## Hyper-parameters

See `v2_common.py` and the `CFG` dictionary at the top of each script. The values are reproduced in the
configuration table of the paper.
