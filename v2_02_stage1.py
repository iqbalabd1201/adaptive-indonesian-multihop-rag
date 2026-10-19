# -*- coding: utf-8 -*-
"""
v2 STEP 2 - First-stage scorer for the two-document branch, NO stemming (GPU, ~30-45 min on A100).
Pointwise BCE, XLM-R-base, lr 2e-5, 3 epochs, batch 16, max_len 384 (train) / 512 (eval), seed 42
(= the unstemmed arm of the stemming ablation, Section 4.7.5).
Checkpoint = epoch with the highest VALIDATION top-1 accuracy (tie -> higher top-2 both-gold).
Also used as: anchor selector in the two-stage branch, scorer of the branch-matched fixed-K baseline,
and transfer source for the four-document reranker.
"""
from google.colab import drive
drive.mount('/content/drive')
import sys; sys.path.insert(0, '/content/drive/MyDrive/FE2H/v2_code')
from v2_common import *
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

CFG = dict(model='xlm-roberta-base', lr=2e-5, epochs=3, batch_size=16, max_len_train=384, max_len_eval=512,
           warmup_ratio=0.1, weight_decay=0.01, seed=SEED, loss='BCE (pointwise)')
OUT = f'{MODELS}/stage1'

ok = lambda d: d['branch'] == '2' and len(d['docs']) == 10 and sum(d['labels']) == 2
train = [d for d in load_json(f'{PREP}/train.json') if ok(d)]
val = [d for d in load_json(f'{PREP}/val.json') if ok(d)]
print('train 2-doc questions:', len(train), '| val:', len(val))

class Pairs(Dataset):
    def __init__(self, data, tok):
        self.p = [(d['q'], x, float(l)) for d in data for x, l in zip(d['docs'], d['labels'])]; self.tok = tok
    def __len__(self): return len(self.p)
    def __getitem__(self, i):
        q, x, l = self.p[i]
        e = self.tok(q, x, max_length=CFG['max_len_train'], padding='max_length', truncation=True,
                     return_tensors='pt', return_token_type_ids=False)
        return e['input_ids'][0], e['attention_mask'][0], torch.tensor(l)

def evaluate(model, tok, data):
    model.eval(); t1, t2 = [], []
    for d in tqdm(data, desc='val', leave=False):
        s = score_docs(model, tok, d['q'], d['docs'], CFG['max_len_eval'])
        o = np.argsort(-s); g = gold_set(d['labels'])
        t1.append(int(o[0]) in g); t2.append(set(o[:2].tolist()) == g)
    model.train()
    return pct(t1), pct(t2)

set_seed(CFG['seed'])
tok = AutoTokenizer.from_pretrained(CFG['model'])
model = Scorer(CFG['model']).to(device)
loader = DataLoader(Pairs(train, tok), batch_size=CFG['batch_size'], shuffle=True, num_workers=2, pin_memory=True)
opt = torch.optim.AdamW(model.parameters(), lr=CFG['lr'], weight_decay=CFG['weight_decay'])
steps = len(loader) * CFG['epochs']
sch = get_linear_schedule_with_warmup(opt, int(CFG['warmup_ratio'] * steps), steps)
scaler = torch.amp.GradScaler('cuda')
log, best = [], (-1, -1, None)
for ep in range(1, CFG['epochs'] + 1):
    model.train(); tot = 0
    for ids, am, y in tqdm(loader, desc=f'stage1 epoch {ep}'):
        with torch.autocast('cuda', dtype=torch.float16):
            loss = F.binary_cross_entropy_with_logits(model(ids.to(device), am.to(device)).float(), y.to(device))
        opt.zero_grad(); scaler.scale(loss).backward(); scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(opt); scaler.update(); sch.step()
        tot += loss.item()
    a1, a2 = evaluate(model, tok, val)
    log.append({'epoch': ep, 'train_loss': tot / len(loader), 'val_top1': a1, 'val_top2_both': a2})
    print(f'epoch {ep}: loss {tot / len(loader):.4f} | val top-1 {a1:.2f}% | val top-2 both gold {a2:.2f}%')
    if (a1, a2) > best[:2]:
        best = (a1, a2, ep); save_scorer(model, tok, OUT)
print('selected epoch (validation):', best[2])
save_json({'config': CFG, 'log': log, 'selected_epoch': best[2]}, f'{RESULTS}/stage1_results.json', indent=2)
print('Saved', OUT)
