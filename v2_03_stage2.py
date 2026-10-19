# -*- coding: utf-8 -*-
"""
v2 STEP 3 - Second-stage (anchor-conditioned) scorer for the two-document branch (GPU, ~40-60 min on A100).
Input pair: (question, anchor_doc + " " + candidate_doc), XLM-R-base, pointwise BCE.
Training pairs: each of the two gold documents is used as anchor; the other gold document is the
positive and N_NEG distractors (resampled every epoch, seed 42) are negatives.
lr 2e-5, 2 epochs, batch 16, max_len 512, warmup 10%, seed 42.
Checkpoint = epoch with the highest VALIDATION two-stage SDRR, where the anchor is the one actually
chosen by the v2 first-stage scorer (as at inference time).
"""
from google.colab import drive
drive.mount('/content/drive')
import sys; sys.path.insert(0, '/content/drive/MyDrive/FE2H/v2_code')
from v2_common import *
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

CFG = dict(model='xlm-roberta-base', lr=2e-5, epochs=2, batch_size=16, max_len=512, n_neg=4,
           warmup_ratio=0.1, weight_decay=0.01, seed=SEED, loss='BCE (pointwise)',
           anchors='both gold documents (teacher forcing)')
OUT = f'{MODELS}/stage2'

ok = lambda d: d['branch'] == '2' and len(d['docs']) == 10 and sum(d['labels']) == 2
train = [d for d in load_json(f'{PREP}/train.json') if ok(d)]
val = [d for d in load_json(f'{PREP}/val.json') if ok(d)]
print('train:', len(train), '| val:', len(val))

# anchors for validation = v2 first-stage predictions (computed once)
s1m, s1t = load_scorer(f'{MODELS}/stage1')
val_anchor = [int(np.argmax(score_docs(s1m, s1t, d['q'], d['docs']))) for d in tqdm(val, desc='val anchors')]
print('val anchor accuracy (top-1 gold):', pct([a in gold_set(d['labels']) for a, d in zip(val_anchor, val)]))
del s1m; torch.cuda.empty_cache()

def make_pairs(epoch):
    r = random.Random(SEED + epoch); pairs = []
    for d in train:
        g = sorted(gold_set(d['labels'])); neg = [i for i in range(10) if i not in g]
        for a, p in ((g[0], g[1]), (g[1], g[0])):
            pairs.append((d['q'], d['docs'][a] + ' ' + d['docs'][p], 1.0))
            for n in r.sample(neg, CFG['n_neg']):
                pairs.append((d['q'], d['docs'][a] + ' ' + d['docs'][n], 0.0))
    return pairs

class Pairs(Dataset):
    def __init__(self, pairs, tok): self.p, self.tok = pairs, tok
    def __len__(self): return len(self.p)
    def __getitem__(self, i):
        q, x, l = self.p[i]
        e = self.tok(q, x, max_length=CFG['max_len'], padding='max_length', truncation=True,
                     return_tensors='pt', return_token_type_ids=False)
        return e['input_ids'][0], e['attention_mask'][0], torch.tensor(l)

def evaluate(model, tok):
    model.eval(); ok2 = []
    for d, a in tqdm(list(zip(val, val_anchor)), desc='val two-stage', leave=False):
        cand = [i for i in range(10) if i != a]
        s = score_pairs(model, tok, [d['q']] * 9, [d['docs'][a] + ' ' + d['docs'][i] for i in cand])
        ok2.append({a, cand[int(np.argmax(s))]} == gold_set(d['labels']))
    model.train()
    return pct(ok2)

set_seed(CFG['seed'])
tok = AutoTokenizer.from_pretrained(CFG['model'])
model = Scorer(CFG['model']).to(device)
n_steps = len(make_pairs(0)) // CFG['batch_size'] * CFG['epochs']
opt = torch.optim.AdamW(model.parameters(), lr=CFG['lr'], weight_decay=CFG['weight_decay'])
sch = get_linear_schedule_with_warmup(opt, int(CFG['warmup_ratio'] * n_steps), n_steps)
scaler = torch.amp.GradScaler('cuda')
log, best = [], (-1, None)
for ep in range(1, CFG['epochs'] + 1):
    loader = DataLoader(Pairs(make_pairs(ep), tok), batch_size=CFG['batch_size'], shuffle=True,
                        num_workers=2, pin_memory=True, drop_last=True)
    model.train(); tot = 0
    for ids, am, y in tqdm(loader, desc=f'stage2 epoch {ep}'):
        with torch.autocast('cuda', dtype=torch.float16):
            loss = F.binary_cross_entropy_with_logits(model(ids.to(device), am.to(device)).float(), y.to(device))
        opt.zero_grad(); scaler.scale(loss).backward(); scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(opt); scaler.update(); sch.step()
        tot += loss.item()
    v = evaluate(model, tok)
    log.append({'epoch': ep, 'train_loss': tot / len(loader), 'val_two_stage_sdrr': v})
    print(f'epoch {ep}: loss {tot / len(loader):.4f} | val two-stage SDRR {v:.2f}%')
    if v > best[0]:
        best = (v, ep); save_scorer(model, tok, OUT)
print('selected epoch (validation):', best[1])
save_json({'config': CFG, 'log': log, 'selected_epoch': best[1]}, f'{RESULTS}/stage2_results.json', indent=2)
print('Saved', OUT)
