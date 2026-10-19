# -*- coding: utf-8 -*-
"""
v2 STEP 5b - Latency of the selection stage only (GPU, ~5 min). Run after step 5.
Online setting: one question at a time, all candidates of a stage scored in one batch, fp16,
CUDA-synchronised wall-clock time, 20 warm-up questions. Covers routing + reranking (not the LLM call).
Adds 'latency_selection_stage' to FE2H/v2/results/pipeline_v2.json.
"""
from google.colab import drive
drive.mount('/content/drive')
import sys; sys.path.insert(0, '/content/drive/MyDrive/FE2H/v2_code')
from v2_common import *
import time
from transformers import AutoTokenizer

test = load_json(f'{PREP}/test.json')
true = [d['branch'] for d in test]
fsel = load_json(f'{RESULTS}/fourdoc_selection.json')['selected']
R2 = [i for i in range(len(test)) if true[i] == '2']; R4 = [i for i in range(len(test)) if true[i] == '4']
N_LAT, WARMUP = 500, 20
lat_idx = sorted(random.Random(SEED).sample(R2, N_LAT // 2) + random.Random(SEED + 1).sample(R4, N_LAT // 2))

r_tok = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')
r_mod = Router().to(device); r_mod.load_state_dict(torch.load(f'{MODELS}/router/router.pt', map_location=device)); r_mod.eval()
s1m, s1t = load_scorer(f'{MODELS}/stage1'); s2m, s2t = load_scorer(f'{MODELS}/stage2'); s4m, s4t = load_scorer(fsel['path'])

def sync():
    if device.type == 'cuda': torch.cuda.synchronize()

@torch.no_grad()
def route_one(d):
    e = r_tok([d['question_raw']], max_length=128, padding='max_length', truncation=True, return_tensors='pt').to(device)
    # router runs in fp32, as during training and evaluation (its -1e9 attention mask overflows fp16)
    return '2' if int(r_mod(e['input_ids'], e['attention_mask']).argmax(-1)) == 0 else '4'

def run_adaptive(d):
    if route_one(d) == '2':
        two_stage(s1m, s1t, s2m, s2t, d['q'], d['docs']); return 2
    score_docs(s4m, s4t, d['q'], d['docs']); return 1
def run_single_k4(d):
    score_docs(s4m, s4t, d['q'], d['docs']); return 1
def run_branch_k4(d):
    m, t = (s1m, s1t) if route_one(d) == '2' else (s4m, s4t)
    score_docs(m, t, d['q'], d['docs']); return 1

latency = {'gpu': torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu', 'n': N_LAT, 'warmup': WARMUP, 'methods': {}}
for name, fn in (('adaptive', run_adaptive), ('branch_matched_K4', run_branch_k4), ('single_model_K4', run_single_k4)):
    for i in lat_idx[:WARMUP]: fn(test[i])
    ms, stages = [], []
    for i in lat_idx:
        sync(); t0 = time.perf_counter(); stages.append(fn(test[i])); sync()
        ms.append(1000 * (time.perf_counter() - t0))
    ms = np.array(ms); b2 = np.array([true[i] == '2' for i in lat_idx])
    pairs = [len(test[i]['docs']) + (len(test[i]['docs']) - 1 if s == 2 else 0) for i, s in zip(lat_idx, stages)]
    v = {'mean_ms': float(ms.mean()), 'median_ms': float(np.median(ms)), 'p95_ms': float(np.percentile(ms, 95)),
         'mean_ms_2doc': float(ms[b2].mean()), 'mean_ms_4doc': float(ms[~b2].mean()),
         'cross_encoder_pairs_per_q': float(np.mean(pairs))}
    latency['methods'][name] = v
    print(f"{name:18s} mean {v['mean_ms']:6.1f} ms | median {v['median_ms']:6.1f} | p95 {v['p95_ms']:6.1f} | "
          f"2-doc {v['mean_ms_2doc']:6.1f} | 4-doc {v['mean_ms_4doc']:6.1f} | pairs/q {v['cross_encoder_pairs_per_q']:.1f}")
print('GPU:', latency['gpu'])
out = load_json(f'{RESULTS}/pipeline_v2.json'); out['latency_selection_stage'] = latency
save_json(out, f'{RESULTS}/pipeline_v2.json', indent=2)
print('Saved latency into', f'{RESULTS}/pipeline_v2.json')
