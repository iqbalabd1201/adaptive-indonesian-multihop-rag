# -*- coding: utf-8 -*-
"""
v2 STEP 5 - Final evaluation of the v2 pipeline on the 8,000-question evaluation set (GPU, ~10-20 min).
Everything here uses models selected on validation (steps 1-4); the evaluation set is touched only now.

Reports (overall / 2-doc / 4-doc):
  * adaptive pipeline (predicted routing) and oracle routing
  * branch-matched fixed-K (same scorers + same routing, top-K, K = 2,3,4)
  * single-model fixed-K (the four-document reranker for every question, K = 2,3,4)
  * set metrics: SDRR (containment), exact-set match, document precision, recall, docs/question
  * ranking metrics (Reviewer B #9): MRR, MAP, nDCG@10, Recall@2, Recall@4
  * McNemar exact test + paired bootstrap 95% CI (10,000 resamples)
  * selection-strategy ablation (Table 7) and v1-vs-v2 comparison
  * latency of routing + reranking per question (500 questions, batch = one question, fp16)
Writes FE2H/v2/results/pipeline_v2.json and pipeline_v2_selections.json (input for step 6).
"""
from google.colab import drive
drive.mount('/content/drive')
import sys; sys.path.insert(0, '/content/drive/MyDrive/FE2H/v2_code')
from v2_common import *
from tqdm.auto import tqdm

test = load_json(f'{PREP}/test.json')
route = [r['pred'] for r in load_json(f'{RESULTS}/router_test_preds.json')]
true = [d['branch'] for d in test]
G = [gold_set(d['labels']) for d in test]
fsel = load_json(f'{RESULTS}/fourdoc_selection.json')['selected']
print('four-document reranker selected on validation:', fsel)

# ---------------- scores (cached, resumable) ----------------
SC_F = f'{RESULTS}/test_scores_v2.json'
sc = load_json(SC_F) if os.path.exists(SC_F) else {}
if len(sc) < len(test):
    s1m, s1t = load_scorer(f'{MODELS}/stage1')
    s2m, s2t = load_scorer(f'{MODELS}/stage2')
    s4m, s4t = load_scorer(fsel['path'])
    for i, d in enumerate(tqdm(test, desc='scoring')):
        if str(i) in sc: continue
        s1 = score_docs(s1m, s1t, d['q'], d['docs'])
        a = int(np.argmax(s1)); cand = [j for j in range(len(d['docs'])) if j != a]
        s2 = [None] * len(d['docs'])
        if cand:
            v = score_pairs(s2m, s2t, [d['q']] * len(cand), [d['docs'][a] + ' ' + d['docs'][j] for j in cand])
            for j, x in zip(cand, v): s2[j] = float(x)
        s4 = score_docs(s4m, s4t, d['q'], d['docs'])
        sc[str(i)] = {'s1': s1.tolist(), 'a': a, 's2': s2, 's4': s4.tolist()}
        if i % 500 == 0: save_json(sc, SC_F)
    save_json(sc, SC_F)

def top(s, k): return [int(j) for j in np.argsort(-np.asarray(s))[:k]]
def two_stage_sel(x):
    if all(v is None for v in x['s2']): return [x['a']]
    j = max((v, j) for j, v in enumerate(x['s2']) if v is not None)[1]
    return [x['a'], j]
def two_stage_order(x):
    rest = sorted([j for j, v in enumerate(x['s2']) if v is not None], key=lambda j: -x['s2'][j])
    return [x['a']] + rest

N = len(test); S = [sc[str(i)] for i in range(N)]
methods = {}
methods['adaptive'] = [two_stage_sel(S[i]) if route[i] == '2' else top(S[i]['s4'], 4) for i in range(N)]
methods['adaptive_oracle'] = [two_stage_sel(S[i]) if true[i] == '2' else top(S[i]['s4'], 4) for i in range(N)]
for K in (2, 3, 4):
    methods[f'branch_matched_K{K}'] = [top(S[i]['s1'], K) if route[i] == '2' else top(S[i]['s4'], K) for i in range(N)]
    methods[f'single_model_K{K}'] = [top(S[i]['s4'], K) for i in range(N)]

R2 = [i for i in range(N) if true[i] == '2']; R4 = [i for i in range(N) if true[i] == '4']; RA = list(range(N))
per = {m: [set_metrics(sel[i], G[i]) for i in range(N)] for m, sel in methods.items()}

def summ(m, idx):
    p = per[m]
    return {k: pct([p[i][k] for i in idx]) if k != 'n' else float(np.mean([p[i]['n'] for i in idx]))
            for k in ('contain', 'exact', 'precision', 'recall', 'n')}

out = {'set_metrics': {}, 'ranking': {}, 'tests': {}, 'selected_fourdoc': fsel}
print(f"\n{'method':22s} {'scope':6s} {'SDRR':>7s} {'exact':>7s} {'prec':>7s} {'recall':>7s} {'docs':>5s}")
for m in methods:
    out['set_metrics'][m] = {}
    for sname, idx in (('all', RA), ('2-doc', R2), ('4-doc', R4)):
        r = summ(m, idx); out['set_metrics'][m][sname] = r
        print(f"{m:22s} {sname:6s} {r['contain']:7.2f} {r['exact']:7.2f} {r['precision']:7.2f} {r['recall']:7.2f} {r['n']:5.2f}")

# ---------------- ranking metrics ----------------
def rank_summary(order_fn, idx):
    rows = [ranking_metrics(order_fn(i), G[i]) for i in idx]
    o = {k: pct([r[k] for r in rows]) for k in ('mrr', 'map', 'ndcg')}
    for k in (2, 4):
        o[f'recall@{k}'] = pct([len(set(order_fn(i)[:k]) & G[i]) / max(len(G[i]), 1) for i in idx])
    return o
rk = {
    '2-doc | first-stage scorer (single-stage)': rank_summary(lambda i: top(S[i]['s1'], 10), R2),
    '2-doc | two-stage (anchor + conditioned)': rank_summary(lambda i: two_stage_order(S[i]), R2),
    '2-doc | four-doc reranker (single model)': rank_summary(lambda i: top(S[i]['s4'], 10), R2),
    '4-doc | four-doc reranker': rank_summary(lambda i: top(S[i]['s4'], 10), R4),
}
print(f"\n{'ranking':45s} {'MRR':>6s} {'MAP':>6s} {'nDCG@10':>8s} {'R@2':>6s} {'R@4':>6s}")
for k, v in rk.items():
    print(f"{k:45s} {v['mrr']:6.2f} {v['map']:6.2f} {v['ndcg']:8.2f} {v['recall@2']:6.2f} {v['recall@4']:6.2f}")
out['ranking'] = rk

# ---------------- significance ----------------
def compare(m1, m2, key, idx):
    a = [per[m1][i][key] for i in idx]; b = [per[m2][i][key] for i in idx]
    d, lo, hi = bootstrap_ci(a, b)
    res = {'diff_pts': d, 'ci95': [lo, hi]}
    if key in ('contain', 'exact'):
        x, y, p = mcnemar(a, b); res.update(only_first=x, only_second=y, p=p)
    return res
print('\nPaired comparisons (first - second):')
for m2 in ('single_model_K4', 'branch_matched_K4'):
    for key in ('contain', 'precision'):
        for sname, idx in (('all', RA), ('2-doc', R2), ('4-doc', R4)):
            r = compare('adaptive', m2, key, idx); out['tests'][f'adaptive_vs_{m2}|{key}|{sname}'] = r
            print(f"  adaptive vs {m2:18s} {key:9s} {sname:6s} diff {r['diff_pts']:+6.2f} CI [{r['ci95'][0]:+.2f}, {r['ci95'][1]:+.2f}]"
                  + (f"  McNemar {r['only_first']}/{r['only_second']} p={r['p']:.3g}" if 'p' in r else ''))
r = compare('adaptive_oracle', 'adaptive', 'contain', RA); out['tests']['oracle_vs_predicted_routing'] = r
print(f"  oracle vs predicted routing: {r}")

# ---------------- selection-strategy ablation (2-doc, true branch) ----------------
strat = {}
for K in (2, 3, 4):
    f = [set_metrics(top(S[i]['s1'], K), G[i]) for i in R2]
    strat[f'single-stage top-{K}'] = {'docs': float(np.mean([x['n'] for x in f])), 'sdrr': pct([x['contain'] for x in f]),
                                      'precision': pct([x['precision'] for x in f]), 'flags': [x['contain'] for x in f]}
f = [set_metrics(two_stage_sel(S[i]), G[i]) for i in R2]
strat['two-stage sequential'] = {'docs': float(np.mean([x['n'] for x in f])), 'sdrr': pct([x['contain'] for x in f]),
                                 'precision': pct([x['precision'] for x in f]), 'flags': [x['contain'] for x in f]}
x, y, p = mcnemar(strat['two-stage sequential']['flags'], strat['single-stage top-2']['flags'])
d, lo, hi = bootstrap_ci(strat['two-stage sequential']['flags'], strat['single-stage top-2']['flags'])
print('\nSelection strategy (2-doc, 4,000):')
for k, v in strat.items(): print(f"  {k:24s} docs {v['docs']:.2f} | SDRR {v['sdrr']:.2f} | precision {v['precision']:.2f}")
print(f'  two-stage vs top-2: {d:+.2f} pts CI [{lo:+.2f}, {hi:+.2f}], McNemar {x}/{y} p={p:.3g}')
out['selection_strategy'] = {k: {kk: vv for kk, vv in v.items() if kk != 'flags'} for k, v in strat.items()}
out['selection_strategy_test'] = {'diff': d, 'ci95': [lo, hi], 'only_two_stage': x, 'only_top2': y, 'p': p}

full2 = [i for i in R2 if len(test[i]['docs']) == 10 and len(G[i]) == 2]
comp = {'n_full_pools': len(full2),
        'top1_gold': pct([int(np.argmax(S[i]['s1'])) in G[i] for i in full2]),
        'at_least_one_in_top2': pct([len(set(top(S[i]['s1'], 2)) & G[i]) >= 1 for i in full2]),
        'both_in_top2': pct([set(top(S[i]['s1'], 2)) == G[i] for i in full2])}
print('component level (full pools):', comp); out['component_level'] = comp

# ---------------- v1 vs v2 ----------------
if os.path.exists(RAW['v1_pipeline']):
    v1 = load_json(RAW['v1_pipeline'])['results']
    a1 = [int(bool(v1[i]['all_correct'])) for i in range(N)]; a2 = [per['adaptive'][i]['contain'] for i in range(N)]
    for sname, idx in (('all', RA), ('2-doc', R2), ('4-doc', R4)):
        x, y, p = mcnemar([a2[i] for i in idx], [a1[i] for i in idx]); d, lo, hi = bootstrap_ci([a2[i] for i in idx], [a1[i] for i in idx])
        out['tests'][f'v2_vs_v1|{sname}'] = {'v1': pct([a1[i] for i in idx]), 'v2': pct([a2[i] for i in idx]),
                                             'diff': d, 'ci95': [lo, hi], 'only_v2': x, 'only_v1': y, 'p': p}
        print(f'v2 vs v1 [{sname}]: v1 {pct([a1[i] for i in idx]):.2f} -> v2 {pct([a2[i] for i in idx]):.2f} '
              f'({d:+.2f}, CI [{lo:+.2f}, {hi:+.2f}]) McNemar {x}/{y} p={p:.3g}')

# ---------------- latency of the selection stage (Reviewer A #7, Reviewer B #9) ----------------
# Online setting: one question at a time, all ten candidates scored in one batch per stage, fp16,
# CUDA-synchronised wall-clock time. Covers routing + reranking only (not the LLM call).
import time
from transformers import AutoTokenizer
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

latency = {'gpu': torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu', 'n': N_LAT, 'methods': {}}
for name, fn in (('adaptive', run_adaptive), ('branch_matched_K4', run_branch_k4), ('single_model_K4', run_single_k4)):
    for i in lat_idx[:WARMUP]: fn(test[i])
    ms, stages = [], []
    for i in lat_idx:
        sync(); t0 = time.perf_counter(); stages.append(fn(test[i])); sync()
        ms.append(1000 * (time.perf_counter() - t0))
    ms = np.array(ms); b2 = np.array([true[i] == '2' for i in lat_idx])
    pairs = [len(test[i]['docs']) + (len(test[i]['docs']) - 1 if (name == 'adaptive' and s == 2) else 0)
             for i, s in zip(lat_idx, stages)]
    latency['methods'][name] = {'mean_ms': float(ms.mean()), 'median_ms': float(np.median(ms)), 'p95_ms': float(np.percentile(ms, 95)),
                                'mean_ms_2doc': float(ms[b2].mean()), 'mean_ms_4doc': float(ms[~b2].mean()),
                                'cross_encoder_pairs_per_q': float(np.mean(pairs))}
    v = latency['methods'][name]
    print(f"latency {name:18s} mean {v['mean_ms']:6.1f} ms | median {v['median_ms']:6.1f} | p95 {v['p95_ms']:6.1f} | "
          f"2-doc {v['mean_ms_2doc']:6.1f} | 4-doc {v['mean_ms_4doc']:6.1f} | pairs/q {v['cross_encoder_pairs_per_q']:.1f}")
out['latency_selection_stage'] = latency

save_json(out, f'{RESULTS}/pipeline_v2.json', indent=2)
save_json([{'idx': i, 'route': route[i], 'true': true[i], 'adaptive': methods['adaptive'][i],
            'fixedk4': methods['branch_matched_K4'][i], 'adaptive_correct': per['adaptive'][i]['contain']}
           for i in range(N)], f'{RESULTS}/pipeline_v2_selections.json')
print('\nSaved', f'{RESULTS}/pipeline_v2.json')
