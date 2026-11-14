# 🛍️ Audit Penjualan Ritel, Pajak & Profitabilitas (Multi-Agen AI · Groq)

Dataset: **Supermarket Sales** (Kaggle `faresashraf1001/supermarket-sales`). Dibaca otomatis dari repository; upload manual opsional.

| Agen | Tugas |
|---|---|
| 1. Data Sanitizer | Kelengkapan kolom & baris, standardisasi Date/Time, cek Invoice ID ganda |
| 2. Tax & Accounting Reconciliation | Unit price x Quantity = cogs; Tax = 5% cogs; Total = cogs + Tax; gross income & margin |
| 3. Fraud & Cashless Anomaly | Nilai ekstrem, tunai bernilai tinggi, di luar jam operasional, transaksi kembar, rating anjlok cabang-jam |
| 4. Profitability & RFM | Gross income per produk/cabang/tipe pelanggan, volume tinggi-laba rendah, RFM segmen |
| 5. Executive Reporting & Copilot | Kertas kerja (MD/TXT/CSV) + chat dengan data |

## Struktur
```
app.py  agents.py  sample_data.py  requirements.txt  README.md
.streamlit/config.toml   (tema cream & maroon)
.streamlit/secrets.toml  (template API key)
SuperMarket Analysis.csv (dataset Anda, di root repo atau folder data/)
```
Nama file dikenali otomatis: `SuperMarket Analysis.csv`, `supermarket_sales.csv`, atau file apa pun yang namanya mengandung "supermarket". Jika tidak ada, app memakai data sintetis (dengan peringatan).

## Jalankan lokal
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy
Push ke GitHub, lalu di https://share.streamlit.io pilih repo, branch `main`, file `app.py`. Di **Advanced settings → Secrets** isi:
```toml
GROQ_API_KEY = "gsk_xxxxxxxx"
```
⚠️ Jangan menaruh key asli di `secrets.toml` yang di-push.

## Catatan
- Dataset aslinya sintetis dan bersih: uji rekonsiliasi biasanya **0 temuan**; temuan datang dari uji statistik (nilai ekstrem, tunai tinggi, rating). Data sintetis bawaan sengaja memuat anomali untuk demo.
- Di dataset ini `gross income` = `Tax 5%` dan `cogs` = penjualan sebelum pajak, sehingga margin konstan 4,76%. Simulasi memakai asumsi HPP agar laba bermakna.
- Model `mixtral-8x7b-32768` mungkin sudah dihentikan Groq; kode otomatis beralih ke `llama-3.1-8b-instant`.
