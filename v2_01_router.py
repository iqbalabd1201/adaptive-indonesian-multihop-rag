# -*- coding: utf-8 -*-
"""
v2 STEP 1 - Routing classifier with a dedicated validation split (GPU, ~10-20 min on A100).
Same architecture/optimiser as v1 (mBERT-base-cased, first 8 layers + embeddings frozen,
AdamW 2e-5, wd 0.01, warmup 10%, clip 1.0, max_len 128, 5 epochs, seed 42).
Checkpoint = epoch with the highest VALIDATION accuracy. Test set is predicted once at the end.
Input = raw question (the router never used stemming).
"""
from google.colab import drive
drive.mount('/content/drive')
import sys; sys.path.insert(0, '/content/drive/MyDrive/FE2H/v2_code')
from v2_common import *
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

CFG = dict(lr=2e-5, weight_decay=0.01, warmup_ratio=0.1, epochs=5, batch_size=32, max_len=128, seed=SEED)
OUT = f'{MODELS}/router'; os.makedirs(OUT, exist_ok=True)

train, val, test = (load_json(f'{PREP}/{s}.json') for s in ('train', 'val', 'test'))
tok = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')

def batches(data, shuffle):
    xs = [(d['question_raw'], 0 if d['branch'] == '2' else 1) for d in data]
    return DataLoader(xs, batch_size=CFG['batch_size'], shuffle=shuffle,
                      collate_fn=lambda b: (list(zip(*b))[0], torch.tensor(list(zip(*b))[1])))

@torch.no_grad()
def predict(model, data):
    model.eval(); preds, probs = [], []
    for qs, _ in batches(data, False):
        e = tok(list(qs), max_length=CFG['max_len'], padding='max_length', truncation=True, return_tensors='pt').to(device)
        p = torch.softmax(model(e['input_ids'], e['attention_mask']).float(), -1)
        preds += p.argmax(-1).cpu().tolist(); probs += p[:, 1].cpu().tolist()
    return np.array(preds), np.array(probs)

set_seed(CFG['seed'])
model = Router().to(device)
loader = batches(train, True)
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=CFG['lr'], weight_decay=CFG['weight_decay'])
steps = len(loader) * CFG['epochs']
sch = get_linear_schedule_with_warmup(opt, int(CFG['warmup_ratio'] * steps), steps)
yv = np.array([0 if d['branch'] == '2' else 1 for d in val])
log, best = [], (-1, None)
for ep in range(1, CFG['epochs'] + 1):
    model.train(); tot = 0
    for qs, y in tqdm(loader, desc=f'router epoch {ep}'):
        e = tok(list(qs), max_length=CFG['max_len'], padding='max_length', truncation=True, return_tensors='pt').to(device)
        loss = F.cross_entropy(model(e['input_ids'], e['attention_mask']), y.to(device))
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sch.step()
        tot += loss.item()
    pv, _ = predict(model, val)
    acc = 100 * (pv == yv).mean()
    log.append({'epoch': ep, 'train_loss': tot / len(loader), 'val_acc': acc})
    print(f'epoch {ep}: loss {tot / len(loader):.4f} | val acc {acc:.2f}%')
    if acc > best[0]:
        best = (acc, ep); torch.save(model.state_dict(), f'{OUT}/router.pt')
print('selected epoch (validation):', best[1])

# ---------- final test evaluation (once) ----------
model.load_state_dict(torch.load(f'{OUT}/router.pt', map_location=device))
pt, prob = predict(model, test)
yt = np.array([0 if d['branch'] == '2' else 1 for d in test])
P, R, F1, S = precision_recall_fscore_support(yt, pt, labels=[0, 1], zero_division=0)
cm = confusion_matrix(yt, pt, labels=[0, 1])
res = {'config': CFG, 'log': log, 'selected_epoch': best[1], 'test_acc': 100 * (pt == yt).mean(),
       'per_class': {'2-doc': dict(P=P[0], R=R[0], F1=F1[0], n=int(S[0])), '4-doc': dict(P=P[1], R=R[1], F1=F1[1], n=int(S[1]))},
       'macro_f1': float(F1.mean()), 'confusion_matrix[true][pred]': cm.tolist(),
       'misrouted_test_idx': np.where(pt != yt)[0].tolist()}
print(json.dumps({k: v for k, v in res.items() if k != 'log'}, indent=2, default=float))
save_json([{'idx': i, 'pred': '2' if p == 0 else '4', 'prob4': float(pr)} for i, (p, pr) in enumerate(zip(pt, prob))],
          f'{RESULTS}/router_test_preds.json')
save_json(res, f'{RESULTS}/router_results.json', indent=2)
print('Saved router to', OUT)
