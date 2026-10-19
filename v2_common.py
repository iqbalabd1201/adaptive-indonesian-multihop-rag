# -*- coding: utf-8 -*-
"""
Shared utilities for the v2 (revision round 2) experiments.

v2 protocol (answers Reviewer A #8, #12 and Reviewer B #2, #3, #11):
  * no Sastrawi stemming: scorer input = lowercase + whitespace normalisation only
  * a dedicated validation split (10% of each training set, seed 42) is used for ALL
    checkpoint / configuration selection; the 8,000-question evaluation set is used once, at the end
  * the four-document reranker configuration is chosen on the validation split
Copy this folder to  /content/drive/MyDrive/FE2H/v2_code/  before running the numbered scripts.
"""
import os, re, json, random, math
import numpy as np
import torch, torch.nn as nn

BASE = '/content/drive/MyDrive/FE2H'
V2 = f'{BASE}/v2'
PREP = f'{V2}/prepared'
MODELS = f'{V2}/models'
RESULTS = f'{V2}/results'
for d in (V2, PREP, MODELS, RESULTS):
    os.makedirs(d, exist_ok=True)

RAW = dict(
    hotpot_train=f'{BASE}/processed_data/train_16k_nllb13b_id.json',
    hotpot_test=f'{BASE}/processed_data/val_4k_nllb13b_id.json',
    wiki_train_candidates=[
        f'{BASE}/2wiki_processed/training_backup_4hop_16k_nllb13b_id.json',
        f'{BASE}/2wiki_processed/train_4hop_16k_nllb13b_id_with_labels.json',
        f'{BASE}/2wiki_processed/train_4hop_16k_nllb13b_id.json',
    ],
    wiki_train_stem_labels=f'{BASE}/2wiki_preprocessed_stem_only/train_16k_stemmed_mapped.json',
    wiki_test=f'{BASE}/2wiki_processed/test_4hop_4k_nllb13b_id_with_labels.json',
    v1_pipeline=f'{BASE}/pipeline_results/pipeline_with_answers_20260105_230942.json',
)

SEED = 42
VAL_FRAC = 0.10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_seed(s=SEED):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def load_json(p):
    with open(p, encoding='utf-8') as f:
        return json.load(f)


def save_json(obj, p, indent=None):
    tmp = p + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent,
                  default=lambda o: o.item() if hasattr(o, 'item') else (o.tolist() if hasattr(o, 'tolist') else str(o)))
    os.replace(tmp, p)


# ------------------------------------------------------------------ text
def norm(t):
    """v2 scorer normalisation: lowercase + whitespace (NO stemming)."""
    t = re.sub(r'\s+', ' ', (t or '').lower()).strip()
    return t or 'empty'


def doc_string(title, sents):
    return norm(title) + ' ' + ' '.join(norm(s) for s in sents)


def gold_set(labels):
    return {i for i, l in enumerate(labels) if l == 1}


# ------------------------------------------------------------------ models
class Scorer(nn.Module):
    """XLM-R-base cross-encoder: [CLS] -> dropout -> linear -> scalar (same as v1 RetrievalModel)."""
    def __init__(self, name='xlm-roberta-base'):
        super().__init__()
        from transformers import AutoModel
        self.encoder = AutoModel.from_pretrained(name)
        self.dropout = nn.Dropout(0.1)
        self.scorer = nn.Linear(self.encoder.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
        return self.scorer(self.dropout(h)).squeeze(-1)


def save_scorer(model, tok, path):
    os.makedirs(path, exist_ok=True)
    torch.save(model.state_dict(), f'{path}/model.pt')
    tok.save_pretrained(path)


def load_scorer(path):
    from transformers import AutoTokenizer
    m = Scorer().to(device)
    m.load_state_dict(torch.load(f'{path}/model.pt', map_location=device))
    return m.eval(), AutoTokenizer.from_pretrained(path)


class Router(nn.Module):
    """Benchmark-defined routing classifier (identical to the v1 BERTHierarchicalModel)."""
    def __init__(self, bert_model='bert-base-multilingual-cased', freeze_layers=8):
        super().__init__()
        from transformers import AutoModel
        self.encoder = AutoModel.from_pretrained(bert_model)
        for p in self.encoder.embeddings.parameters():
            p.requires_grad = False
        for i in range(freeze_layers):
            for p in self.encoder.encoder.layer[i].parameters():
                p.requires_grad = False
        self.num_heads, self.head_dim = 12, 64
        self.query_proj = nn.Linear(768, 768)
        self.key_proj = nn.Linear(768, 768)
        self.value_proj = nn.Linear(768, 768)
        self.complexity_attention = nn.Sequential(nn.Linear(768, 256), nn.Tanh(), nn.Linear(256, 1))
        self.classifier = nn.Sequential(nn.Linear(768, 512), nn.ReLU(), nn.Dropout(0.3),
                                        nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, 2))

    def forward(self, input_ids, attention_mask):
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        b, s, d = h.shape
        Q = self.query_proj(h).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key_proj(h).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value_proj(h).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        sc = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        sc = sc.masked_fill(attention_mask.unsqueeze(1).unsqueeze(2) == 0, -1e9)
        a = torch.matmul(torch.softmax(sc, -1), V).transpose(1, 2).contiguous().view(b, s, d)
        cs = self.complexity_attention(a).squeeze(-1).masked_fill(attention_mask == 0, -1e9)
        pooled = (a * torch.softmax(cs, -1).unsqueeze(-1)).sum(1)
        return self.classifier(pooled)


@torch.no_grad()
def score_pairs(model, tok, firsts, seconds, max_len=512, bs=32):
    """Raw scores (logits) for a list of (text_a, text_b) pairs."""
    out = []
    for i in range(0, len(firsts), bs):
        e = tok(firsts[i:i + bs], seconds[i:i + bs], max_length=max_len, truncation=True,
                padding=True, return_tensors='pt', return_token_type_ids=False).to(device)
        with torch.autocast('cuda', dtype=torch.float16, enabled=device.type == 'cuda'):
            s = model(e['input_ids'], e['attention_mask'])
        out.extend(s.float().cpu().tolist())
    return np.array(out)


def score_docs(model, tok, q, docs, max_len=512):
    return score_pairs(model, tok, [q] * len(docs), docs, max_len=max_len)


def two_stage(s1_model, s1_tok, s2_model, s2_tok, q, docs):
    """Returns (anchor, second, s1_scores, s2_scores_full) ; s2_scores_full[anchor] = -inf."""
    s1 = score_docs(s1_model, s1_tok, q, docs)
    a = int(np.argmax(s1))
    cand = [i for i in range(len(docs)) if i != a]
    s2c = score_pairs(s2_model, s2_tok, [q] * len(cand), [docs[a] + ' ' + docs[i] for i in cand])
    s2 = np.full(len(docs), -np.inf); s2[cand] = s2c
    return a, int(cand[int(np.argmax(s2c))]), s1, s2


# ------------------------------------------------------------------ metrics
def set_metrics(selected, gold):
    sel, g = set(selected), set(gold)
    return dict(contain=int(g <= sel), exact=int(sel == g),
                precision=len(sel & g) / max(len(sel), 1), recall=len(sel & g) / max(len(g), 1),
                n=len(sel))


def ranking_metrics(order, gold, k=10):
    """MRR (first gold), MAP and nDCG@k with binary relevance for one ranked list."""
    g = set(gold)
    if not g:
        return dict(mrr=0.0, map=0.0, ndcg=0.0)
    rr, hits, ap, dcg = 0.0, 0, 0.0, 0.0
    for r, d in enumerate(order[:k], 1):
        if d in g:
            hits += 1
            if rr == 0: rr = 1.0 / r
            ap += hits / r
            dcg += 1.0 / math.log2(r + 1)
    idcg = sum(1.0 / math.log2(r + 1) for r in range(1, min(len(g), k) + 1))
    return dict(mrr=rr, map=ap / len(g), ndcg=dcg / idcg)


def mcnemar(a, b):
    """a, b: equal-length 0/1 arrays. Returns (only_a, only_b, exact two-sided p)."""
    from scipy.stats import binomtest
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    x, y = int((a & ~b).sum()), int((b & ~a).sum())
    return x, y, (binomtest(x, x + y, 0.5).pvalue if x + y else 1.0)


def bootstrap_ci(a, b=None, n=10000, seed=SEED):
    """95% percentile CI of mean(a) (or of mean(a)-mean(b), paired) in percentage points."""
    rng = np.random.default_rng(seed)
    a = np.asarray(a, float)
    d = a if b is None else a - np.asarray(b, float)
    idx = rng.integers(0, len(d), size=(n, len(d)))
    m = d[idx].mean(1) * 100
    return float(d.mean() * 100), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def holm(pvals):
    """Holm-Bonferroni adjusted p-values (same order as input)."""
    p = np.asarray(pvals, float); m = len(p); order = np.argsort(p)
    adj = np.empty(m); run = 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (m - r) * p[i])); adj[i] = run
    return adj.tolist()


def pct(x):
    return 100.0 * float(np.mean(x))
