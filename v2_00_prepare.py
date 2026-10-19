# -*- coding: utf-8 -*-
"""
v2 STEP 0 - Prepare data (CPU, ~2-5 minutes). Run once.
  * HotpotQA train 16k and 2Wiki bridge-comparison train 16k -> 90% train / 10% validation (seed 42)
  * evaluation set = the same 8,000 questions as v1 (4,000 HotpotQA then 4,000 2Wiki, same order)
  * scorer text WITHOUT stemming (lowercase + whitespace normalisation)
Outputs: FE2H/v2/prepared/{train,val,test}.json  + split_ids.json
"""
from google.colab import drive
drive.mount('/content/drive')
import sys; sys.path.insert(0, '/content/drive/MyDrive/FE2H/v2_code')
from v2_common import *
from collections import Counter

# ---------- locate the UNSTEMMED 2Wiki training file ----------
wiki_train_path = next((p for p in RAW['wiki_train_candidates'] if os.path.exists(p)), None)
if wiki_train_path is None:
    print('Could not find the unstemmed 2Wiki training file. JSON files in 2wiki_processed:')
    for f in sorted(os.listdir(f'{BASE}/2wiki_processed')):
        print('   ', f)
    raise SystemExit('Put the right path at the top of RAW["wiki_train_candidates"] in v2_common.py')
print('2Wiki train file:', wiki_train_path)

hot_tr = load_json(RAW['hotpot_train'])
hot_te = load_json(RAW['hotpot_test'])[:4000]
wik_tr = load_json(wiki_train_path)
wik_te = load_json(RAW['wiki_test'])[:4000]
print(f'raw sizes: hotpot train {len(hot_tr)}, hotpot test {len(hot_te)}, 2wiki train {len(wik_tr)}, 2wiki test {len(wik_te)}')

# 2Wiki train may lack doc_labels -> take them from the stemmed, label-mapped file (same _id, same doc order)
if not all('doc_labels' in d for d in wik_tr[:50]):
    print('2Wiki train has no doc_labels -> mapping from', RAW['wiki_train_stem_labels'])
    lab = {d['_id']: d['doc_labels'] for d in load_json(RAW['wiki_train_stem_labels'])}
    miss = 0
    for d in wik_tr:
        if d.get('_id') in lab and len(lab[d['_id']]) == len(d['context']):
            d['doc_labels'] = lab[d['_id']]
        else:
            miss += 1
    print('  items without usable labels:', miss)

def looks_stemmed(data):
    """Unstemmed NLLB output keeps capitals and affixes; stemmed text is all lower-case."""
    txt = ' '.join(d['question'] for d in data[:200])
    return sum(c.isupper() for c in txt) < 20
for name, d in [('hotpot train', hot_tr), ('2wiki train', wik_tr), ('hotpot test', hot_te), ('2wiki test', wik_te)]:
    assert not looks_stemmed(d), f'{name} looks stemmed/lower-cased - wrong file'
print('example raw question:', wik_tr[0]['question'])


def item(d, branch, split, src):
    labels = d.get('doc_labels', [])
    return dict(uid=f'{split}-{branch}-{src}', _id=d.get('_id'), branch=branch, src=src,
                question_raw=d['question'], q=norm(d['question']),
                docs=[doc_string(t, s) for t, s in d['context']],
                labels=labels, answer=d.get('answer', ''))


def full_pool(it, k):
    return len(it['docs']) == 10 and len(it['labels']) == 10 and sum(it['labels']) == k


rng = random.Random(SEED)
splits = {'train': [], 'val': []}
split_ids = {}
for name, data, br, k in [('hotpot', hot_tr, '2', 2), ('wiki', wik_tr, '4', 4)]:
    idx = list(range(len(data))); rng.shuffle(idx)
    n_val = int(round(VAL_FRAC * len(idx)))
    split_ids[name] = {'val': sorted(idx[:n_val]), 'train': sorted(idx[n_val:])}
    for sp in ('train', 'val'):
        for i in split_ids[name][sp]:
            splits[sp].append(item(data[i], br, sp, i))

test = [item(d, '2', 'test', i) for i, d in enumerate(hot_te)] + \
       [item(d, '4', 'test', i) for i, d in enumerate(wik_te)]
assert len(test) == 8000

# sanity: same order as the v1 pipeline output
if os.path.exists(RAW['v1_pipeline']):
    v1 = load_json(RAW['v1_pipeline'])['results']
    assert all(v1[i]['question'] == test[i]['question_raw'] for i in range(0, 8000, 499)), 'order mismatch with v1'
    print('test order matches v1 pipeline output')

# no question overlap between train/val and test
test_q = {t['question_raw'] for t in test}
for sp in ('train', 'val'):
    ov = sum(x['question_raw'] in test_q for x in splits[sp])
    print(f'{sp}: {len(splits[sp])} items | overlap with test questions: {ov}')

for sp, data in [('train', splits['train']), ('val', splits['val']), ('test', test)]:
    c = Counter(x['branch'] for x in data)
    fp = Counter(x['branch'] for x in data if full_pool(x, 2 if x['branch'] == '2' else 4))
    print(f'{sp:5s}: 2-doc {c["2"]:6d} (full pools {fp["2"]}) | 4-doc {c["4"]:6d} (full pools {fp["4"]})')
    save_json(data, f'{PREP}/{sp}.json')
save_json({'wiki_train_file': wiki_train_path, 'seed': SEED, 'val_frac': VAL_FRAC, **split_ids},
          f'{PREP}/split_ids.json')
print('example normalised doc:', test[0]['docs'][0][:160])
print('Saved to', PREP)
