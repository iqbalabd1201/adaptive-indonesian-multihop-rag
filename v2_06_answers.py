# -*- coding: utf-8 -*-
"""
v2 STEP 6 - Answer generation and evidence-context ablation on the v2 pipeline (API; no GPU needed
except BERTScore, which also runs on CPU). Resumable: every answer is cached on Drive.

Part A  all 8,000 questions with v2 adaptive evidence -> EM, F1, BERTScore, LLM-as-Judge (Table 10, 11)
Part B  1,000-question context ablation (same stratified subset as v1, seed 42), four contexts:
        all ten candidates | branch-matched fixed-K=4 (v2) | adaptive (v2) | gold documents
        run N_RUNS times (default 2) to measure run-to-run variance of the generator (Reviewer B #10)
Part C  uncertainty: bootstrap 95% CIs, paired McNemar vs adaptive with Holm correction,
        context tokens (o200k_base)

API calls (default): A 8,000 gen + 8,000 judge ; B 1,000 x 4 x 2 gen (1,000 reused from A)
Approximate cost: USD 15-30 (GPT-5-mini + gpt-4o-mini judge). Time: 1-3 h with MAX_WORKERS=8.
"""
from google.colab import drive
drive.mount('/content/drive')
import subprocess, sys
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'openai', 'tiktoken', 'bert-score'], check=True)
sys.path.insert(0, '/content/drive/MyDrive/FE2H/v2_code')
from v2_common import *
import string, time, threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.auto import tqdm
from openai import OpenAI
import tiktoken

OPENAI_API_KEY = "PASTE_YOUR_OPENAI_API_KEY_HERE"   # do NOT commit a real key
GEN_MODEL, JUDGE_MODEL = 'gpt-5-mini', 'gpt-4o-mini'
MAX_WORKERS, N_RUNS, N_PER_BRANCH = 8, 2, 500
CONDITIONS = ['all10', 'fixedk4', 'adaptive', 'gold']
assert OPENAI_API_KEY.startswith('sk-'), 'Paste your OpenAI key first.'
OUT = f'{RESULTS}/answers'; os.makedirs(OUT, exist_ok=True)
GEN_CACHE, JUDGE_CACHE = f'{OUT}/gen_cache.json', f'{OUT}/judge_cache.json'

# ---------------- data ----------------
hot = load_json(RAW['hotpot_test'])[:4000]; wik = load_json(RAW['wiki_test'])[:4000]
raw = hot + wik                                        # original (unnormalised) text for the prompt
sel = load_json(f'{RESULTS}/pipeline_v2_selections.json')
assert len(raw) == len(sel) == 8000
gold_idx = [sorted(gold_set(d.get('doc_labels', []))) for d in raw]

rnd = random.Random(SEED)                             # identical to the v1 ablation subset
subset = sorted(rnd.sample(range(0, 4000), N_PER_BRANCH) + rnd.sample(range(4000, 8000), N_PER_BRANCH))

def ctx(i, cond):
    return {'all10': list(range(len(raw[i]['context']))), 'fixedk4': sel[i]['fixedk4'],
            'adaptive': sel[i]['adaptive'], 'gold': gold_idx[i]}[cond]

# ---------------- prompt (same as the v1 context ablation; flexible number of documents) ----------------
FEWSHOT_2 = [
    ('Aktor Amerika mana yang menjadi salah satu bintang dalam Marry Me?',
     ['Marry Me adalah sitkom Amerika yang awalnya ditayangkan di NBC dari 14 Oktober 2014 hingga 17 Februari 2015. Serial ini dibintangi Casey Wilson dan Ken Marino dan ditayangkan Selasa malam pukul 9 malam (ET / PT) / 8 malam (CT) sebagai bagian dari musim televisi 2014-15.',
      'Kenneth Joseph "Ken" Marino (lahir 19 Desember 1968) adalah seorang aktor, komedian, sutradara, dan penulis skenario Amerika. Dia adalah anggota pemeran di MTV "The State" dan telah membintangi acara seperti "Party Down", "Marry Me", "Burning Love", "Brooklyn Nine-Nine" dan "Children\'s Hospital".'], 'Ken Marino'),
    ('Aktor apa yang muncul di film Best Seller dan juga di First Blood?',
     ['Best Seller adalah film thriller kriminal tahun 1987 yang ditulis oleh Larry Cohen, disutradarai oleh John Flynn dan dibintangi James Woods dan Brian Dennehy. Film ini menceritakan kisah Cleve (Woods), seorang hitman karir, yang ingin mengubah kisah hidupnya menjadi buku yang ditulis oleh Dennis Meechum (Dennehy), seorang petugas polisi veteran dan penulis terlaris.',
      'Brian Manion Dennehy (lahir 9 Juli 1938) adalah seorang aktor film, panggung, dan televisi Amerika. Pemenang satu Golden Globe, dua Tony Awards dan penerima enam nominasi Primetime Emmy Award, ia mendapatkan pengakuan awal untuk perannya sebagai Sheriff Will Teasle antagonis dalam "First Blood" (1982). Dia memiliki banyak peran dalam film seperti "Gorky Park, Silverado, Cocoon, F/X, Romeo + Juliet", dan "Knight of Cups".'], 'Brian Dennehy'),
    ('Apakah Justine Henin dan Virginia Ruano Pascual mantan pemain tenis profesional?',
     ['Justine Henin (lahir 1 Juni 1982), dikenal antara tahun 2002 dan 2007 sebagai Justine Hénin-Hardenne, adalah mantan pemain tenis profesional Belgia yang dikenal karena gaya bermain all-court dan terutama menjadi salah satu dari sedikit pemain wanita yang menggunakan backhand dengan satu tangan. Dia menghabiskan total 117 minggu sebagai No. 1 dunia dan menjadi No. 1 akhir tahun pada tahun 2003, 2006 dan 2007.',
      'Virginia Ruano Pascual (lahir 21 September 1973) adalah mantan pemain tenis profesional Spanyol. Dia lahir di Madrid, Spanyol.'], 'ya'),
    ('Apakah John Collier dan Martin Amis adalah warga negara yang sama?',
     ['John Henry Noyes Collier (3 Mei 1901 - 6 April 1980) adalah seorang penulis dan penulis skenario kelahiran Inggris yang terkenal karena cerita pendeknya, banyak di antaranya muncul di "The New Yorker" dari tahun 1930-an hingga 1950-an. Sebagian besar dikumpulkan dalam "The John Collier Reader" (Knopf, 1972); koleksi sebelumnya termasuk volume 1951, "Fancies and Goodnights" yang terkenal, yang memenangkan Penghargaan Fantasi Internasional dan masih tercetak.',
      'Martin Louis Amis (lahir 25 Agustus 1949) adalah seorang novelis Inggris. Novel-novelnya yang paling terkenal adalah "Money" (1984) dan "London Fields" (1989). Dia telah menerima James Tait Black Memorial Prize untuk memoirnya "Experience" dan telah terdaftar untuk Booker Prize dua kali sampai saat ini.'], 'ya'),
]
FEWSHOT_4 = [
    ("Apakah sutradara kedua film Tanya's Island dan Madhurappathinezhu dari negara yang sama?",
     ["Tanya's Island adalah film fantasi Kanada-Amerika tahun 1980 yang disutradarai oleh Alfred Sole.",
      'Alfred Sole (lahir 2 Juli 1943) adalah seorang desainer produksi, sutradara, produser dan penulis Amerika.',
      'Madhurappathinezhu adalah film Malayalam India tahun 1975, disutradarai oleh Hariharan.',
      'Hariharan adalah sutradara film India yang telah menyutradarai lebih dari 50 film Malayalam.'], 'tidak'),
    ('Apakah kedua sutradara film Nutty Professor II: The Klumps dan Yvonne From Paris berasal dari negara yang sama?',
     ['Nutty Professor II: The Klumps adalah film komedi romantis fiksi ilmiah Amerika tahun 2000 yang disutradarai oleh Peter Segal.',
      'Peter Segal (lahir 1962) adalah seorang sutradara, produser, penulis skenario, dan aktor Amerika Serikat.',
      'Yvonne from Paris adalah film komedi bisu Amerika tahun 1919 yang disutradarai oleh Emmett J. Flynn.',
      'Emmett J. Flynn adalah seorang sutradara, penulis skenario, aktor, dan produser Amerika.'], 'ya'),
]
RULES = """PENTING:
1. Jawab HANYA dalam bahasa Indonesia
2. Jangan gunakan judul film dalam bahasa Inggris
3. Gunakan nama yang sama dengan dokumen
4. Untuk ya/tidak: gunakan "ya", "tidak", atau "bisa"

Contoh BENAR:
- "bulan berdarah" (bukan "Bloody Moon")
- "ya" atau "bisa" (bukan "yes")
- "dendam ursus" (bukan "The Vengeance of Ursus")

Contoh SALAH:
- "Bloody Moon" ❌
- "yes" ❌
- "The Vengeance of Ursus" ❌

ATURAN TAMBAHAN:
1. Jawaban harus SANGAT SINGKAT (1-5 kata)
2. Jangan tambahkan penjelasan
3. Salin format dari dokumen jika menyebutkan nama

Jawaban:"""
SYSTEM = 'Anda adalah asisten QA yang HANYA memberikan jawaban dalam bahasa Indonesia. JANGAN PERNAH gunakan bahasa Inggris dalam jawaban Anda.'

def doc_text(title, sents, max_len=1500):
    t = title + ' ' + ' '.join(sents)
    return t[:max_len] + '...' if len(t) > max_len else t

def build_prompt(question, docs, branch):
    p = 'Berdasarkan pertanyaan dan dokumen-dokumen yang diberikan, berikan jawaban yang singkat dan tepat.\n\nBerikut adalah contoh-contoh:\n\n'
    for k, (q, ds, a) in enumerate(FEWSHOT_2 if branch == '2' else FEWSHOT_4, 1):
        p += f'Contoh {k}:\nPertanyaan: {q}\n' + ''.join(f'Dokumen {j}: {d}\n' for j, d in enumerate(ds, 1)) + f'Jawaban: {a}\n\n'
    p += '=' * 70 + '\nSEKARANG JAWAB PERTANYAAN INI:\n' + '=' * 70 + '\n\n'
    p += f'Pertanyaan: {question}\n' + ''.join(f'Dokumen {j}: {d}\n' for j, d in enumerate(docs, 1))
    return p + '\n' + RULES

def prompt_for(i, cond):
    return build_prompt(raw[i]['question'], [doc_text(*raw[i]['context'][j]) for j in ctx(i, cond)], sel[i]['route'])

# ---------------- metrics ----------------
def normalize(s):
    s = ''.join(ch for ch in (s or '').lower() if ch not in set(string.punctuation))
    return ' '.join(re.sub(r'\b(a|an|the|yang|sebuah|satu)\b', ' ', s).split())
def em(p, g): return int(normalize(p) == normalize(g))
def f1(p, g):
    pt, gt = normalize(p).split(), normalize(g).split()
    if not pt or not gt: return float(pt == gt)
    c = sum((Counter(pt) & Counter(gt)).values())
    return 0.0 if c == 0 else 2 * (c / len(pt)) * (c / len(gt)) / (c / len(pt) + c / len(gt))
def clean(a):
    a = (a or '').replace('Jawaban:', '').replace('jawaban:', '').strip().strip('"').strip("'").strip()
    return a.rstrip('.,!?;:').strip()

MONTHS = 'januari februari maret april mei juni juli agustus september oktober november desember'.split()
def answer_type(g):
    n = normalize(g)
    if n in {'ya', 'tidak', 'yes', 'no', 'bisa', 'benar', 'salah'}: return 'Boolean'
    if re.search(r'\d', g) or any(m in n.split() for m in MONTHS): return 'Date/number'
    if any(w[:1].isupper() for w in g.split()): return 'Named entity/title'
    return 'Common noun/attribute'

# ---------------- API ----------------
client = OpenAI(api_key=OPENAI_API_KEY)
gen = load_json(GEN_CACHE) if os.path.exists(GEN_CACHE) else {}
jud = load_json(JUDGE_CACHE) if os.path.exists(JUDGE_CACHE) else {}
lock = threading.Lock()

STOP = threading.Event()          # set when the API reports exhausted credit / invalid key

def call(model, messages, **kw):
    """Returns the reply text, or None on failure (a failed call is NEVER written to the cache)."""
    for attempt in range(6):
        if STOP.is_set():
            return None
        try:
            r = client.chat.completions.create(model=model, messages=messages, **kw)
            return (r.choices[0].message.content or '').strip()
        except Exception as e:
            msg = str(e).lower()
            if any(t in msg for t in ('insufficient_quota', 'exceeded your current quota', 'billing',
                                      'invalid_api_key', 'incorrect api key')):
                if not STOP.is_set():
                    print(f'\n*** API STOPPED: {e}\n*** Progress is saved. Top up / fix the key, then re-run this cell.')
                STOP.set(); return None
            time.sleep(min(2 ** attempt, 30))
    return None

SAVE_EVERY = 100

def run_jobs(jobs, store, path, fn, desc):
    """Parallel calls; checkpoint to Drive every SAVE_EVERY answers and on any stop/interrupt."""
    done = failed = 0
    ex = ThreadPoolExecutor(MAX_WORKERS)
    try:
        futs = [ex.submit(fn, j) for j in jobs]
        for f in tqdm(as_completed(futs), total=len(futs), desc=desc):
            k, v = f.result()
            with lock:
                if v is None:
                    failed += 1; continue
                store[k] = v; done += 1
                if done % SAVE_EVERY == 0: save_json(store, path)
            # after a STOP the remaining jobs return immediately (None), so every finished answer is still collected
    finally:
        ex.shutdown(wait=True, cancel_futures=True)
        save_json(store, path)
        print(f'{desc}: saved {len(store)} cached answers ({done} new, {failed} failed and NOT cached)')
    if STOP.is_set():
        raise SystemExit('Stopped because of the API (credit or key). Re-run the cell after fixing it; nothing will be repeated.')
    if failed:
        print(f'WARNING: {failed} calls failed (network / rate limit). Re-run the cell to retry only those.')

def gen_job(j):
    key, prompt = j
    out = call(GEN_MODEL, [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': prompt}])
    return key, (None if out is None else clean(out))

JUDGE_PROMPT = """Anda adalah penilai jawaban tanya-jawab. Tentukan apakah JAWABAN MODEL bermakna sama dengan JAWABAN REFERENSI untuk pertanyaan berikut. Abaikan perbedaan ejaan, huruf besar/kecil, format tanggal/angka, gelar, nama lengkap vs nama singkat, dan terjemahan judul yang setara. Jawab hanya dengan satu kata: BENAR atau SALAH.

Pertanyaan: {q}
Jawaban referensi: {g}
Jawaban model: {p}"""
def judge_job(j):
    key, q, g, p = j
    out = call(JUDGE_MODEL, [{'role': 'user', 'content': JUDGE_PROMPT.format(q=q, g=g, p=p)}], temperature=0)
    return key, (None if out is None else int(out.strip().upper().startswith('BENAR')))

# ---------------- Part A: all 8,000 (run 1, adaptive) ----------------
keyA = lambda i: f'r1|adaptive|{i}'
jobs = [(keyA(i), prompt_for(i, 'adaptive')) for i in range(8000) if keyA(i) not in gen]
# ---------------- Part B: 1,000 x 4 contexts x N_RUNS ----------------
jobs += [(f'r{r}|{c}|{i}', prompt_for(i, c)) for r in range(1, N_RUNS + 1) for c in CONDITIONS for i in subset
         if f'r{r}|{c}|{i}' not in gen and not (r == 1 and c == 'adaptive')]
print(f'generation calls remaining: {len(jobs)}')
run_jobs(jobs, gen, GEN_CACHE, gen_job, 'generation')

jj = [(keyA(i), raw[i]['question'], raw[i]['answer'], gen[keyA(i)]) for i in range(8000) if keyA(i) in gen and keyA(i) not in jud]
print(f'judge calls remaining: {len(jj)}')
run_jobs(jj, jud, JUDGE_CACHE, judge_job, 'judge')

missing_gen = [j for j in [keyA(i) for i in range(8000)] + [f'r{r}|{c}|{i}' for r in range(1, N_RUNS + 1) for c in CONDITIONS for i in subset] if j not in gen]
missing_jud = [keyA(i) for i in range(8000) if keyA(i) not in jud]
if missing_gen or missing_jud:
    raise SystemExit(f'Not finished: {len(missing_gen)} answers and {len(missing_jud)} judgements still missing. Re-run the cell.')

# ---------------- Part A results ----------------
from bert_score import score as bscore
preds = [gen[keyA(i)] or '-' for i in range(8000)]; refs = [raw[i]['answer'] or '-' for i in range(8000)]
_, _, BF = bscore(preds, refs, lang='id', batch_size=64, verbose=False)   # bert-base-multilingual-cased, no rescaling
EM = np.array([em(p, g) for p, g in zip(preds, refs)]); F1 = np.array([f1(p, g) for p, g in zip(preds, refs)])
JU = np.array([jud[keyA(i)] for i in range(8000)]); BS = BF.numpy()
R2, R4 = list(range(4000)), list(range(4000, 8000))
partA = {}
print(f"\n{'scope':8s} {'EM':>16s} {'F1':>16s} {'BERTScore':>16s} {'LLM-judge':>16s}")
for name, idx in (('all', list(range(8000))), ('2-doc', R2), ('4-doc', R4)):
    row = {}
    for mname, arr in (('EM', EM), ('F1', F1), ('BERTScore', BS), ('Judge', JU)):
        m, lo, hi = bootstrap_ci(arr[idx]); row[mname] = {'mean': m, 'ci95': [lo, hi]}
    partA[name] = row
    print(f"{name:8s} " + ' '.join(f"{row[k]['mean']:6.2f} [{row[k]['ci95'][0]:.1f},{row[k]['ci95'][1]:.1f}]" for k in ('EM', 'F1', 'BERTScore', 'Judge')))
types = [answer_type(raw[i]['answer']) for i in range(8000)]
by_type = {}
print('\nby answer type (n, share, EM, F1, judge with 95% CI on EM):')
for t in ['Named entity/title', 'Boolean', 'Date/number', 'Common noun/attribute']:
    idx = [i for i in range(8000) if types[i] == t]
    m, lo, hi = bootstrap_ci(EM[idx])
    by_type[t] = {'n': len(idx), 'share': 100 * len(idx) / 8000, 'EM': m, 'EM_ci95': [lo, hi],
                  'F1': pct(F1[idx]), 'Judge': pct(JU[idx]), 'BERTScore': pct(BS[idx])}
    print(f"  {t:22s} n={len(idx):5d} ({100 * len(idx) / 8000:5.2f}%) EM {m:.2f} [{lo:.1f},{hi:.1f}] F1 {pct(F1[idx]):.2f} judge {pct(JU[idx]):.2f}")

# ---------------- Part B/C: context ablation with uncertainty ----------------
enc = tiktoken.get_encoding('o200k_base')
toks = {c: np.array([sum(len(enc.encode(doc_text(*raw[i]['context'][j]))) for j in ctx(i, c)) for i in subset]) for c in CONDITIONS}
S2 = [i for i in subset if i < 4000]; S4 = [i for i in subset if i >= 4000]
pos = {i: k for k, i in enumerate(subset)}
def ans(r, c, i): return gen[f'r{r}|{c}|{i}']
partB = {'n': len(subset), 'runs': N_RUNS, 'conditions': {}, 'tests': {}}
print(f"\n{'context':10s} {'docs':>5s} {'tokens':>7s} " + ' '.join(f'{"EM run"+str(r):>20s}' for r in range(1, N_RUNS + 1)) + f" {'F1 run1':>8s}")
for c in CONDITIONS:
    row = {'docs': float(np.mean([len(ctx(i, c)) for i in subset])), 'tokens': float(toks[c].mean()),
           'tokens_2doc': float(np.mean([toks[c][pos[i]] for i in S2])), 'tokens_4doc': float(np.mean([toks[c][pos[i]] for i in S4]))}
    for r in range(1, N_RUNS + 1):
        e = np.array([em(ans(r, c, i), raw[i]['answer']) for i in subset]); f = np.array([f1(ans(r, c, i), raw[i]['answer']) for i in subset])
        m, lo, hi = bootstrap_ci(e)
        row[f'run{r}'] = {'EM': m, 'EM_ci95': [lo, hi], 'F1': pct(f), 'EM_2doc': pct(e[[pos[i] for i in S2]]),
                          'EM_4doc': pct(e[[pos[i] for i in S4]]), 'em_flags': e.tolist()}
    row['EM_mean_over_runs'] = float(np.mean([row[f'run{r}']['EM'] for r in range(1, N_RUNS + 1)]))
    partB['conditions'][c] = row
    print(f"{c:10s} {row['docs']:5.2f} {row['tokens']:7.1f} " + ' '.join(
        f"{row[f'run{r}']['EM']:6.2f} [{row[f'run{r}']['EM_ci95'][0]:.1f},{row[f'run{r}']['EM_ci95'][1]:.1f}]" for r in range(1, N_RUNS + 1))
        + f" {row['run1']['F1']:8.2f}")
# run-to-run agreement (same context, same prompt, different call)
for c in CONDITIONS:
    if N_RUNS >= 2:
        a, b = partB['conditions'][c]['run1']['em_flags'], partB['conditions'][c]['run2']['em_flags']
        x, y, p = mcnemar(a, b)
        partB['conditions'][c]['run1_vs_run2'] = {'diff': pct(a) - pct(b), 'only_run1': x, 'only_run2': y, 'p': p,
                                                  'answer_agreement': pct([ans(1, c, i) == ans(2, c, i) for i in subset])}
        print(f"  {c:10s} run1 vs run2: EM diff {pct(a) - pct(b):+.2f}, McNemar p={p:.3g}, identical answers {partB['conditions'][c]['run1_vs_run2']['answer_agreement']:.1f}%")
# adaptive vs each other context, per run, Holm-corrected over the 3 comparisons x 3 scopes
for r in range(1, N_RUNS + 1):
    names, ps, rows = [], [], []
    for c in ('all10', 'fixedk4', 'gold'):
        for scope, idx in (('all', subset), ('2-doc', S2), ('4-doc', S4)):
            a = [partB['conditions']['adaptive'][f'run{r}']['em_flags'][pos[i]] for i in idx]
            b = [partB['conditions'][c][f'run{r}']['em_flags'][pos[i]] for i in idx]
            x, y, p = mcnemar(a, b); d, lo, hi = bootstrap_ci(a, b)
            names.append(f'run{r}|adaptive_vs_{c}|{scope}'); ps.append(p)
            rows.append({'diff': d, 'ci95': [lo, hi], 'only_adaptive': x, 'only_other': y, 'p': p})
    for n_, r_, ph in zip(names, rows, holm(ps)):
        r_['p_holm'] = ph; partB['tests'][n_] = r_
        print(f"  {n_:32s} diff {r_['diff']:+6.2f} CI [{r_['ci95'][0]:+.2f},{r_['ci95'][1]:+.2f}] p={r_['p']:.3g} p_holm={ph:.3g}")
tA = toks['adaptive'].mean()
partB['token_reduction'] = {c: 100 * (1 - tA / toks[c].mean()) for c in ('all10', 'fixedk4')}
print('token reduction of adaptive vs all10 / fixedk4 (%):', partB['token_reduction'])

save_json({'partA': partA, 'by_type': by_type, 'partB': partB, 'subset': subset,
           'judge_prompt': JUDGE_PROMPT, 'gen_model': GEN_MODEL, 'judge_model': JUDGE_MODEL},
          f'{RESULTS}/answers_v2.json', indent=1)
print('\nSaved', f'{RESULTS}/answers_v2.json')
