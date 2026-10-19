# -*- coding: utf-8 -*-
"""
v2 STEP 4 - Four-document reranker: training-regime grid + validation-based selection
(GPU, ~25-30 min per configuration on A100 -> ~2-2.5 h for all five; resumable).

Grid (all NO stemming, XLM-R-base, 2 epochs, batch 4 x grad-accum 4, max_len 384 train / 512 eval,
warmup 5%, wd 0.01, margin 0.5, seed 42):
   init      x  learning rate  (ranking loss)   -> full 2x2: separates transfer from learning rate
   transfer     5e-6  /  2e-5
   scratch      5e-6  /  2e-5
 + transfer, 5e-6, pointwise BCE loss           -> loss effect at fixed init and lr
"transfer" = initialised from the v2 first-stage two-document scorer (step 2).

The configuration AND epoch used in the pipeline = highest VALIDATION SDRR@4.
Test SDRR is also recorded for every (config, epoch) but only for the ablation table, never for selection.
"""
from google.colab import drive
drive.mount('/content/drive')
import sys; sys.path.insert(0, '/content/drive/MyDrive/FE2H/v2_code')
from v2_common import *
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

GRID = [
    dict(name='transfer_rank_5e-6', init='transfer', loss='rank', lr=5e-6),
    dict(name='transfer_rank_2e-5', init='transfer', loss='rank', lr=2e-5),
    dict(name='scratch_rank_5e-6',  init='scratch',  loss='rank', lr=5e-6),
    dict(name='scratch_rank_2e-5',  init='scratch',  loss='rank', lr=2e-5),
    dict(name='transfer_bce_5e-6',  init='transfer', loss='bce',  lr=5e-6),
]
CFG = dict(model='xlm-roberta-base', epochs=2, batch_size=4, grad_accum=4, max_len_train=384, max_len_eval=512,
           warmup_ratio=0.05, weight_decay=0.01, margin=0.5, seed=SEED)
OUT = f'{MODELS}/fourdoc'; os.makedirs(OUT, exist_ok=True)
RES_F = f'{RESULTS}/fourdoc_grid.json'

ok = lambda d: d['branch'] == '4' and len(d['docs']) == 10 and sum(d['labels']) == 4
train = [d for d in load_json(f'{PREP}/train.json') if ok(d)]
val = [d for d in load_json(f'{PREP}/val.json') if d['branch'] == '4']
test = [d for d in load_json(f'{PREP}/test.json') if d['branch'] == '4']
print('train:', len(train), '| val:', len(val), '| test:', len(test))

class Q(Dataset):
    def __init__(self, data, tok): self.d, self.tok = data, tok
    def __len__(self): return len(self.d)
    def __getitem__(self, i):
        s = self.d[i]
        e = self.tok([s['q']] * 10, s['docs'], max_length=CFG['max_len_train'], padding='max_length',
                     truncation=True, return_tensors='pt', return_token_type_ids=False)
        return e['input_ids'], e['attention_mask'], torch.tensor(s['labels'], dtype=torch.float)

def rank_loss(sc, lab, m):
    b = sc.size(0)
    pos = sc.masked_select(lab == 1).view(b, -1); neg = sc.masked_select(lab == 0).view(b, -1)
    return F.relu(m - (pos.unsqueeze(2) - neg.unsqueeze(1))).mean()

def sdrr_at4(model, tok, data, desc):
    model.eval(); flags = []
    for d in tqdm(data, desc=desc, leave=False):
        s = score_docs(model, tok, d['q'], d['docs'], CFG['max_len_eval'])
        flags.append(gold_set(d['labels']) <= set(np.argsort(-s)[:4].tolist()))
    model.train()
    return flags

res = load_json(RES_F) if os.path.exists(RES_F) else {}
for g in GRID:
    if g['name'] in res and len(res[g['name']]['epochs']) == CFG['epochs']:
        print('done already:', g['name']); continue
    print('\n' + '=' * 70 + f"\n{g['name']}\n" + '=' * 70)
    set_seed(CFG['seed'])
    if g['init'] == 'transfer':
        model, tok = load_scorer(f'{MODELS}/stage1'); model.train()
    else:
        tok = AutoTokenizer.from_pretrained(CFG['model']); model = Scorer(CFG['model']).to(device)
    loader = DataLoader(Q(train, tok), batch_size=CFG['batch_size'], shuffle=True, num_workers=2, pin_memory=True)
    opt = torch.optim.AdamW(model.parameters(), lr=g['lr'], weight_decay=CFG['weight_decay'])
    steps = len(loader) * CFG['epochs'] // CFG['grad_accum']
    sch = get_linear_schedule_with_warmup(opt, int(CFG['warmup_ratio'] * steps), steps)
    scaler = torch.amp.GradScaler('cuda')
    run = {'config': g, 'epochs': {}}
    for ep in range(1, CFG['epochs'] + 1):
        tot = 0
        for step, (ids, am, lab) in enumerate(tqdm(loader, desc=f"{g['name']} epoch {ep}")):
            b, n, L = ids.shape; lab = lab.to(device)
            with torch.autocast('cuda', dtype=torch.float16):
                sc = model(ids.view(b * n, L).to(device), am.view(b * n, L).to(device)).view(b, n).float()
            loss = rank_loss(sc, lab, CFG['margin']) if g['loss'] == 'rank' else F.binary_cross_entropy_with_logits(sc, lab)
            scaler.scale(loss / CFG['grad_accum']).backward()
            if (step + 1) % CFG['grad_accum'] == 0:
                scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update(); sch.step(); opt.zero_grad()
            tot += loss.item()
        vf = sdrr_at4(model, tok, val, 'val'); tf = sdrr_at4(model, tok, test, 'test')
        run['epochs'][str(ep)] = {'train_loss': tot / len(loader), 'val_sdrr': pct(vf), 'test_sdrr': pct(tf),
                                  'test_flags': [int(x) for x in tf]}
        print(f"epoch {ep}: loss {tot / len(loader):.4f} | VAL SDRR@4 {pct(vf):.2f}% | test SDRR@4 {pct(tf):.2f}%")
        # keep only the best-on-validation checkpoint (saves Drive space; strict '>' keeps the earlier one on ties)
        bst = res.get('_best', {'val_sdrr': -1})
        if pct(vf) > bst['val_sdrr']:
            save_scorer(model, tok, f'{OUT}/best')
            res['_best'] = {'config': g['name'], 'epoch': ep, 'val_sdrr': pct(vf), 'path': f'{OUT}/best'}
            print('   -> new best on validation, saved to', f'{OUT}/best')
        res[g['name']] = run; save_json(res, RES_F)
    del model; torch.cuda.empty_cache()

# ---------- selection on validation only ----------
sel = res['_best']
print('\nSELECTED on validation:', sel)
res = {k: v for k, v in res.items() if k != '_best'}

print(f"\n{'configuration':22s} {'ep':>3s} {'loss':>8s} {'VAL':>8s} {'TEST':>8s}  McNemar(test) vs selected")
selflags = res[sel['config']]['epochs'][str(sel['epoch'])]['test_flags']
table = []
for k, r in res.items():
    for e, v in r['epochs'].items():
        x, y, p = mcnemar(v['test_flags'], selflags)
        table.append(dict(config=k, epoch=int(e), train_loss=v['train_loss'], val=v['val_sdrr'], test=v['test_sdrr'],
                          only_this=x, only_selected=y, p=p))
        print(f"{k:22s} {e:>3s} {v['train_loss']:8.4f} {v['val_sdrr']:7.2f}% {v['test_sdrr']:7.2f}%  {x}/{y} p={p:.3g}")

# 2x2 factorial summary at each epoch (init x lr, ranking loss), test SDRR
for e in ('1', '2'):
    g = {k: res[k]['epochs'][e]['test_sdrr'] for k in res if k.endswith(('rank_5e-6', 'rank_2e-5'))}
    if len(g) == 4:
        ti = ((g['transfer_rank_5e-6'] + g['transfer_rank_2e-5']) - (g['scratch_rank_5e-6'] + g['scratch_rank_2e-5'])) / 2
        lr = ((g['transfer_rank_2e-5'] + g['scratch_rank_2e-5']) - (g['transfer_rank_5e-6'] + g['scratch_rank_5e-6'])) / 2
        ia = (g['transfer_rank_2e-5'] - g['scratch_rank_2e-5']) - (g['transfer_rank_5e-6'] - g['scratch_rank_5e-6'])
        print(f'epoch {e}: main effect transfer {ti:+.2f} | main effect lr 2e-5 {lr:+.2f} | interaction {ia:+.2f} (points)')
save_json({'selected': sel, 'table': table, 'cfg': CFG}, f'{RESULTS}/fourdoc_selection.json', indent=2)
