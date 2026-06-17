# Error Analysis Samples

Dokumen ini berisi contoh analisis error dari output full pipeline adaptive Indonesian multi-hop RAG.

Tujuan dokumen ini adalah menyediakan contoh ringkas untuk analisis kualitatif hasil retrieval dan answer generation.

Contoh diambil dari output pipeline dengan format:
- `question`
- `answer`
- `predicted_answer`
- `retrieved_indices`
- `gold_indices`
- `all_correct`
- `em`
- `f1`

## 1. Ringkasan Jenis Error

| Kategori | Kondisi | Makna |
|---|---|---|
| Lexical mismatch | Retrieval benar, EM = 0, F1 > 0 | Jawaban prediksi secara makna dekat/benar, tetapi tidak identik secara leksikal dengan jawaban referensi. |
| Retrieval incomplete but answer correct | Retrieval tidak mengambil semua dokumen gold, tetapi EM = 1 | Model masih dapat menjawab benar meskipun retrieval tidak sempurna. |
| Wrong answer despite correct retrieval | Retrieval benar, tetapi EM = 0 dan F1 = 0 | Dokumen pendukung sudah benar, tetapi answer generation memilih jawaban yang salah. |

## 2. Contoh 1: Lexical Mismatch

| Field | Value |
|---|---|
| Question | Apakah Virginia Commonwealth University dan University of California publik atau swasta? |
| Reference Answer | publik |
| Predicted Answer | keduanya publik |
| True Complexity | 2-hop |
| Predicted Complexity | 2-hop |
| Retrieval Correct | true |
| Exact Match | 0 |
| F1 | 0.6667 |

### Analisis

Pada contoh ini, sistem berhasil mengambil dokumen pendukung yang benar. Jawaban prediksi adalah **“keduanya publik”**, sedangkan jawaban referensi adalah **“publik”**.

Exact Match bernilai 0 karena kedua string tidak identik. Namun, secara semantik jawaban tersebut menyampaikan informasi yang sama, yaitu kedua universitas bersifat publik.

### Implikasi

Kasus ini menunjukkan bahwa Exact Match terlalu ketat untuk mengevaluasi jawaban singkat dalam bahasa Indonesia. Evaluasi semantik seperti BERTScore dan LLM-as-Judge dibutuhkan untuk menangkap kesamaan makna.

## 3. Contoh 2: Entity Name Variation

| Field | Value |
|---|---|
| Question | Siapa yang hidup lebih lama, Cid Corman atau Katherine Mansfield? |
| Reference Answer | Cid (Sidney) Corman |
| Predicted Answer | Cid Corman |
| True Complexity | 2-hop |
| Predicted Complexity | 2-hop |
| Retrieval Correct | true |
| Exact Match | 0 |
| F1 | 0.8000 |

### Analisis

Jawaban prediksi **“Cid Corman”** merujuk pada entitas yang sama dengan jawaban referensi **“Cid (Sidney) Corman”**. Perbedaannya hanya terletak pada tambahan nama tengah dalam jawaban referensi.

### Implikasi

Kasus ini memperlihatkan adanya variasi penyebutan nama entitas. Exact Match gagal mengenali bahwa kedua jawaban merujuk pada orang yang sama.

## 4. Contoh 3: Translation Variation

| Field | Value |
|---|---|
| Question | Siapa yang beroperasi di sebagian besar negara di seluruh dunia, Dollar Tree atau PPG Industries? |
| Reference Answer | Industri PPG |
| Predicted Answer | PPG Industries |
| True Complexity | 2-hop |
| Predicted Complexity | 2-hop |
| Retrieval Correct | true |
| Exact Match | 0 |
| F1 | 0.5000 |

### Analisis

Jawaban prediksi **“PPG Industries”** dan jawaban referensi **“Industri PPG”** merujuk pada entitas yang sama. Perbedaan terjadi karena variasi terjemahan nama perusahaan.

### Implikasi

Pada dataset hasil terjemahan, variasi nama entitas antara bahasa Inggris dan bahasa Indonesia dapat menyebabkan EM bernilai 0 meskipun jawaban sebenarnya benar secara semantik.

## 5. Contoh 4: Retrieval Incomplete but Answer Correct

| Field | Value |
|---|---|
| Question | Majalah mana yang dimulai pada tanggal sebelumnya, Sejarah Angkatan Laut atau The Open Road for Boys? |
| Reference Answer | Jalan Terbuka untuk Anak Laki-Laki |
| Predicted Answer | Jalan Terbuka untuk Anak Laki-Laki |
| Retrieved Indices | [6, 0] |
| Gold Indices | [4, 6] |
| Retrieval Correct | false |
| Exact Match | 1 |
| F1 | 1.0000 |

### Analisis

Pada contoh ini, sistem tidak mengambil seluruh dokumen gold karena `retrieved_indices` berbeda dari `gold_indices`. Namun, answer generation tetap menghasilkan jawaban yang benar.

### Implikasi

Kasus ini menunjukkan bahwa kesalahan retrieval tidak selalu menyebabkan jawaban akhir salah. Sebagian pertanyaan masih dapat dijawab jika salah satu dokumen yang relevan sudah cukup mengandung informasi kunci.

## 6. Contoh 5: Wrong Answer Despite Correct Retrieval

| Field | Value |
|---|---|
| Question | Siapa yang baru-baru ini memenangkan Hadiah Nobel Sastra, Jacinto Benavente y Martínez atau Rohinton Mistry? |
| Reference Answer | Rohinton Mistry |
| Predicted Answer | Jacinto Benavente y Martínez |
| Retrieval Correct | true |
| Exact Match | 0 |
| F1 | 0.0000 |

### Analisis

Pada contoh ini, retrieval berhasil mengambil dokumen pendukung yang benar, tetapi answer generation memilih entitas yang salah. Jawaban referensi adalah **“Rohinton Mistry”**, sedangkan jawaban prediksi adalah **“Jacinto Benavente y Martínez”**.

### Implikasi

Kesalahan ini menunjukkan bahwa answer generation dapat gagal melakukan perbandingan temporal atau memilih entitas yang sesuai meskipun informasi pendukung sudah tersedia.

## 7. Kesimpulan Error Analysis

Berdasarkan contoh-contoh di atas, terdapat tiga temuan utama:

1. **Exact Match terlalu ketat** untuk kasus variasi leksikal, nama entitas, dan terjemahan.
2. **Retrieval yang tidak sempurna tidak selalu menyebabkan jawaban salah**, terutama jika dokumen yang terambil sudah memuat informasi kunci.
3. **Answer generation masih menjadi sumber error**, terutama ketika model harus memilih entitas berdasarkan perbandingan atau penalaran dari dokumen yang sudah benar.

Error analysis ini mendukung penggunaan evaluasi multi-metrik, yaitu Exact Match, F1-score, BERTScore, dan LLM-as-Judge.
