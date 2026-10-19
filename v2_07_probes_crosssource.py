# -*- coding: utf-8 -*-
"""
v2 STEP 7 - Routing diagnostics with class-wise metrics (Reviewer A #9, Reviewer B #1, #10).
  Part 1 (CPU, ~3-5 min): four shallow-feature probes + the v2 router on the 8,000-question evaluation set
          -> accuracy, per-class precision/recall/F1, macro-F1, confusion matrix.
          Protocol: trained on the v2 training split (14,400 + 14,400 questions), evaluated on the
          balanced evaluation set (4,000 + 4,000); hyper-parameters fixed a priori (no tuning on test).
  Part 2 (GPU, ~1 min): cross-source test re-run with the v2 router on the SAME translated questions
          saved by the v1 experiment (exp2_per_question.csv, exp2b_control.csv); no new translation.
"""
from google.colab import drive
drive.mount('/content/drive')
import sys; sys.path.insert(0, '/content/drive/MyDrive/FE2H/v2_code')
from v2_common import *
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, accuracy_score

train = load_json(f'{PREP}/train.json'); test = load_json(f'{PREP}/test.json')
Xtr = [d['question_raw'] for d in train]; ytr = np.array([int(d['branch'] == '4') for d in train])
Xte = [d['question_raw'] for d in test]; yte = np.array([int(d['branch'] == '4') for d in test])
print('train class counts (2-doc, 4-doc):', np.bincount(ytr), '| test:', np.bincount(yte))

def mask_entities(q):
    q = re.sub(r'"[^"]*"|“[^”]*”|\'[^\']*\'', ' ENT ', q)            # quoted titles
    toks = q.split()
    out = [t if (k == 0 or not t[:1].isupper()) else 'ENT' for k, t in enumerate(toks)]  # capitalised words (not sentence-initial)
    return re.sub(r'\d+', 'NUM', ' '.join(out))

def style_feats(qs):
    f = []
    for q in qs:
        w = q.split()
        f.append([len(q), len(w), np.mean([len(x) for x in w]) if w else 0, q.count(','), q.count('"') + q.count("'"),
                  sum(c.isdigit() for c in q), sum(x[:1].isupper() for x in w) / max(len(w), 1), q.count('?'),
                  q.count('('), q.count('-'), q.count(':')])
    return np.array(f, float)

probes = {
    'Word-level TF-IDF + LR': (make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
                                             LogisticRegression(max_iter=2000, C=1.0)), Xtr, Xte),
    'Character n-gram TF-IDF + LR': (make_pipeline(TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5), min_df=3, sublinear_tf=True),
                                                   LogisticRegression(max_iter=2000, C=1.0)), Xtr, Xte),
    'Entity-masked TF-IDF + LR': (make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
                                                LogisticRegression(max_iter=2000, C=1.0)),
                                  [mask_entities(q) for q in Xtr], [mask_entities(q) for q in Xte]),
    'Length/style features + LR': (make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)), style_feats(Xtr), style_feats(Xte)),
}

def report(name, y, p):
    P, R, F, S = precision_recall_fscore_support(y, p, labels=[0, 1], zero_division=0)
    cm = confusion_matrix(y, p, labels=[0, 1])
    r = {'accuracy': 100 * accuracy_score(y, p), 'macro_f1': 100 * F.mean(),
         '2-doc': {'P': 100 * P[0], 'R': 100 * R[0], 'F1': 100 * F[0]}, '4-doc': {'P': 100 * P[1], 'R': 100 * R[1], 'F1': 100 * F[1]},
         'cm[true][pred]': cm.tolist()}
    print(f"{name:32s} acc {r['accuracy']:6.2f} | 2-doc P/R/F1 {r['2-doc']['P']:.2f}/{r['2-doc']['R']:.2f}/{r['2-doc']['F1']:.2f}"
          f" | 4-doc P/R/F1 {r['4-doc']['P']:.2f}/{r['4-doc']['R']:.2f}/{r['4-doc']['F1']:.2f} | CM {cm.tolist()}")
    return r

out = {'class_counts': {'train': np.bincount(ytr).tolist(), 'test': np.bincount(yte).tolist()}, 'probes': {}}
for name, (clf, a, b) in probes.items():
    clf.fit(a, ytr); out['probes'][name] = report(name, yte, clf.predict(b))
# the most informative words of the word-level probe (source of predictability)
vec, lr = probes['Word-level TF-IDF + LR'][0].named_steps.values()
coef = lr.coef_[0]; vocab = np.array(vec.get_feature_names_out())
out['top_4doc_features'] = vocab[np.argsort(-coef)[:20]].tolist(); out['top_2doc_features'] = vocab[np.argsort(coef)[:20]].tolist()
print('most 4-doc-indicative n-grams:', out['top_4doc_features'])
print('most 2-doc-indicative n-grams:', out['top_2doc_features'])
rp = load_json(f'{RESULTS}/router_test_preds.json')
out['probes']['Proposed router (mBERT, v2)'] = report('Proposed router (mBERT, v2)', yte, np.array([int(r['pred'] == '4') for r in rp]))
film = np.array(['film' in q.lower() for q in Xtr])
out['film_share_train'] = {'4-doc': 100 * film[ytr == 1].mean(), '2-doc': 100 * film[ytr == 0].mean()}
print('share of training questions containing "film":', out['film_share_train'])

# ---------------- Part 2: cross-source test with the v2 router ----------------
from transformers import AutoTokenizer
E2 = f'{BASE}/exp2_router_outputs'
if os.path.exists(f'{E2}/exp2_per_question.csv') and torch.cuda.is_available():
    df = pd.read_csv(f'{E2}/exp2_per_question.csv')
    if os.path.exists(f'{E2}/exp2b_control.csv'):
        ctrl = pd.read_csv(f'{E2}/exp2b_control.csv'); ctrl['type'] = 'bridge_comparison (control)'
        df = pd.concat([df[df['type'] != 'bridge_comparison'], ctrl], ignore_index=True)
    tok = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')
    rt = Router().to(device); rt.load_state_dict(torch.load(f'{MODELS}/router/router.pt', map_location=device)); rt.eval()
    pr = []
    with torch.no_grad():
        for i in range(0, len(df), 64):
            e = tok(df['question_id'].iloc[i:i + 64].tolist(), max_length=128, padding='max_length', truncation=True, return_tensors='pt').to(device)
            pr += rt(e['input_ids'], e['attention_mask']).argmax(-1).cpu().tolist()
    df['pred_v2'] = ['4-hop' if p == 1 else '2-hop' for p in pr]
    df['has_film'] = df['question_id'].str.lower().str.contains('film')
    cs = {}
    for t, g in df.groupby('type'):
        cs[t] = {'n': len(g), 'to_2doc': 100 * (g['pred_v2'] == '2-hop').mean(), 'to_4doc': 100 * (g['pred_v2'] == '4-hop').mean()}
    two = df[df['n_gold_docs'] < 4]
    cs['all two-document types'] = {'n': len(two), 'to_2doc': 100 * (two['pred_v2'] == '2-hop').mean(), 'to_4doc': 100 * (two['pred_v2'] == '4-hop').mean()}
    comp = df[df['type'] == 'comparison']
    cs['comparison with "film"'] = {'n': int(comp['has_film'].sum()), 'to_4doc': 100 * (comp[comp['has_film']]['pred_v2'] == '4-hop').mean()}
    cs['comparison without "film"'] = {'n': int((~comp['has_film']).sum()), 'to_4doc': 100 * (comp[~comp['has_film']]['pred_v2'] == '4-hop').mean()}
    for k, v in cs.items(): print(f'  {k:32s} {v}')
    out['cross_source_v2'] = cs
    df.to_csv(f'{RESULTS}/cross_source_v2.csv', index=False)
else:
    print('cross-source part skipped (needs GPU and exp2_router_outputs/exp2_per_question.csv)')
save_json(out, f'{RESULTS}/routing_diagnostics_v2.json', indent=2)
print('Saved', f'{RESULTS}/routing_diagnostics_v2.json')
