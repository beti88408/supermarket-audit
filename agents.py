"""agents.py - 5 Agen AI (Groq API) untuk Audit Penjualan Ritel, Pajak & Profitabilitas (Supermarket Sales).
Hitungan/statistik dikerjakan pandas (akurat & hemat token); LLM dipakai untuk penalaran (pemetaan kolom,
opini rekonsiliasi, rekomendasi, narasi, ringkasan, chat) dengan konteks ringkas."""
import json
import re
import time
from collections import Counter
from datetime import datetime

import numpy as np
import pandas as pd

MODELS = ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3-32b", "meta-llama/llama-4-scout-17b-16e-instruct"]
FALLBACK = "openai/gpt-oss-20b"
_LAST_CALL = [0.0]

REQ = ["invoice_id", "branch", "city", "customer_type", "gender", "product_line", "unit_price", "quantity", "tax_5",
       "total", "date", "time", "payment", "cogs", "gross_margin_percentage", "gross_income", "rating"]
NUMC = ["unit_price", "quantity", "tax_5", "total", "cogs", "gross_margin_percentage", "gross_income", "rating"]
CRIT = ["invoice_id", "unit_price", "quantity", "total", "cogs"]
CATC = ["branch", "city", "customer_type", "gender", "product_line", "payment"]
ALIAS = {"tax": "tax_5", "tax5": "tax_5", "tax_5_percent": "tax_5", "invoice": "invoice_id", "gross_margin": "gross_margin_percentage",
         "customer": "customer_type", "productline": "product_line", "sales": "total", "total_sales": "total"}


# ------------------------------------------------------------------ LLM client
class LLM:
    """Pembungkus Groq: throttle antar-panggilan, retry+backoff saat rate-limit, fallback model."""

    def __init__(self, api_key=None, model=MODELS[0], min_interval=1.2, max_retries=4):
        self.model, self.min_interval, self.max_retries = model, min_interval, max_retries
        self.errors, self.calls, self.client = [], 0, None
        if api_key:
            try:
                from groq import Groq
                self.client = Groq(api_key=api_key)
            except Exception as e:
                self.errors.append(f"Inisialisasi Groq gagal: {e}")

    @property
    def active(self):
        return self.client is not None

    def chat(self, system, user, json_mode=False, max_tokens=800, temperature=0.1, history=None):
        if not self.active:
            return ""
        msgs = [{"role": "system", "content": system}] + (history or []) + [{"role": "user", "content": user}]
        kw = dict(model=self.model, messages=msgs, temperature=temperature, max_tokens=max_tokens)
        if json_mode:
            kw["response_format"] = {"type": "json_object"}
        for attempt in range(self.max_retries):
            wait = self.min_interval - (time.time() - _LAST_CALL[0])
            if wait > 0:
                time.sleep(wait)
            try:
                _LAST_CALL[0] = time.time()
                self.calls += 1
                r = self.client.chat.completions.create(**kw)
                return r.choices[0].message.content or ""
            except Exception as e:
                m = str(e).lower()
                if "401" in m or "invalid api key" in m:
                    self.errors.append("API key tidak valid.")
                    return ""
                if "429" in m or "rate" in m:
                    time.sleep(min(2 ** attempt * 3, 30))
                    continue
                if any(k in m for k in ("decommission", "not found", "does not exist")) and kw["model"] != FALLBACK:
                    self.errors.append(f"Model {kw['model']} tidak tersedia, beralih ke {FALLBACK}.")
                    kw["model"] = FALLBACK
                    continue
                self.errors.append(str(e)[:200])
                time.sleep(1.5)
        self.errors.append("Panggilan Groq gagal setelah beberapa percobaan.")
        return ""

    def json(self, system, user, **kw):
        t = self.chat(system, user, json_mode=True, **kw)
        if not t:
            return {}
        try:
            return json.loads(t)
        except Exception:
            m = re.search(r"\{.*\}", t, re.S)
            try:
                return json.loads(m.group(0)) if m else {}
            except Exception:
                return {}


# ------------------------------------------------------------------ util
def to_num(v):
    if pd.isna(v):
        return np.nan
    if isinstance(v, (int, float, np.number)):
        return float(v)
    s = re.sub(r"[^\d.,\-]", "", str(v))
    if s in ("", "-"):
        return np.nan
    if "." in s and "," in s:
        s = s.replace(".", "").replace(",", ".") if s.rfind(",") > s.rfind(".") else s.replace(",", "")
    elif "," in s:
        s = s.replace(",", "") if re.fullmatch(r"-?\d{1,3}(,\d{3})+", s) else s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return np.nan


def robust_z(x, g):
    x, g = pd.Series(np.asarray(x, float)), pd.Series(np.asarray(g))
    med = x.groupby(g).transform("median")
    mad = (x - med).abs().groupby(g).transform("median")
    mad = mad.where(mad > 0, med.abs() * 0.1 + 1e-9)
    return (0.6745 * (x - med) / mad).values, med.values


def md(df):
    try:
        return df.to_markdown(index=False)
    except Exception:
        return df.to_string(index=False)


def _best_parse(s, fmts, **kw):
    s = s.astype(str).str.strip()
    best, bn, bf = None, -1, None
    for f in fmts:
        p = pd.to_datetime(s, format=f, errors="coerce")
        if p.notna().sum() > bn:
            best, bn, bf = p, p.notna().sum(), f
    return best, bf


class Findings:
    """Penampung temuan per baris dari semua agen: skor, alasan, kategori, dampak."""

    def __init__(self, n):
        self.n, self.score, self.impact = n, np.zeros(n), np.zeros(n)
        self.reasons, self.cats, self.extra = [[] for _ in range(n)], [[] for _ in range(n)], {}

    def add(self, mask, pts, text, cat, imp=None):
        mask = np.asarray(mask, bool)
        imp = np.zeros(self.n) if imp is None else np.nan_to_num(np.abs(np.asarray(imp, float)))
        for i in np.flatnonzero(mask):
            self.score[i] += pts
            self.impact[i] = max(self.impact[i], imp[i])
            self.reasons[i].append(text if isinstance(text, str) else text[i])
            if cat not in self.cats[i]:
                self.cats[i].append(cat)


# ------------------------------------------------------------------ Agen 1
class SanitizerAgent:
    def __init__(self, llm):
        self.llm = llm

    def run(self, df):
        rep = {"baris_awal": len(df), "catatan": []}
        df = df.copy()
        df.columns = [re.sub(r"[^0-9a-z]+", "_", str(c).strip().lower()).strip("_") for c in df.columns]
        df = df.loc[:, ~df.columns.duplicated()]
        df = df.rename(columns={k: v for k, v in ALIAS.items() if k in df.columns and v not in df.columns})
        miss, mapping = [c for c in REQ if c not in df.columns], {}
        if miss and self.llm.active:
            out = self.llm.json(
                'Anda Data Sanitizer. Petakan kolom dataset ke nama kolom standar. Balas HANYA JSON: {"mapping":{"standar":"kolom_asli atau null"}}',
                f"Kolom standar yang hilang: {miss}\nKolom dataset: {list(df.columns)}\nContoh: {df.head(2).astype(str).to_dict('records')}",
                max_tokens=500)
            for k, v in (out.get("mapping") or {}).items():
                if k in miss and v in df.columns and v not in REQ:
                    df = df.rename(columns={v: k})
                    mapping[k] = v
        miss = [c for c in REQ if c not in df.columns]
        rep["kolom_hilang"], rep["pemetaan_kolom_ai"] = miss, mapping
        if miss:
            return df, rep
        rep["kelengkapan_%"] = round(float(100 * (1 - df[REQ].isna().sum().sum() / max(len(df) * len(REQ), 1))), 2)
        n0 = len(df)
        df = df.dropna(how="all")
        rep["baris_kosong_dihapus"] = n0 - len(df)
        for c in df.select_dtypes("object").columns:
            df[c] = df[c].astype(str).str.strip().replace({"nan": np.nan, "": np.nan, "None": np.nan, "NaT": np.nan})
        for c in NUMC:
            if not pd.api.types.is_numeric_dtype(df[c]):
                df[c] = df[c].map(to_num)
        # --- standardisasi tanggal & jam
        if pd.api.types.is_datetime64_any_dtype(df["date"]):
            rep["format_tanggal"] = "datetime native"
        else:
            df["date"], rep["format_tanggal"] = _best_parse(df["date"], ["%m/%d/%Y", "%d/%m/%Y", "%Y-%m-%d", "%m/%d/%y", "%d-%m-%Y"])
        t, rep["format_jam"] = _best_parse(df["time"], ["%H:%M", "%H:%M:%S", "%I:%M %p", "%I:%M:%S %p"])
        df["time"] = t.dt.strftime("%H:%M")
        df["jam"] = t.dt.hour.fillna(-1).astype(int)
        df["waktu"] = df["date"] + pd.to_timedelta(t.dt.hour.fillna(0) * 60 + t.dt.minute.fillna(0), unit="m")
        df["bulan"] = df["date"].dt.strftime("%Y-%m")
        df["hari"] = df["date"].dt.day_name()
        n1 = len(df)
        df = df.dropna(subset=CRIT + ["date"])
        rep["baris_data_kritis_kosong"] = n1 - len(df)
        for c in CATC:
            df[c] = df[c].fillna("N/A").astype(str)
        df["rating"] = df["rating"].fillna(df["rating"].median())
        df["tax_5"] = df["tax_5"].fillna(0)
        df["gross_income"] = df["gross_income"].fillna(df["total"] - df["cogs"])
        df["gross_margin_percentage"] = df["gross_margin_percentage"].fillna(df["gross_margin_percentage"].median())
        n2 = len(df)
        df = df.drop_duplicates().reset_index(drop=True)
        rep["baris_duplikat_penuh_dihapus"] = n2 - len(df)
        rep["baris_dengan_invoice_ganda"] = int(df.duplicated("invoice_id", keep=False).sum())
        rep["baris_akhir"] = len(df)
        if len(df):
            rep["rentang_tanggal"] = f"{df['date'].min():%Y-%m-%d} s/d {df['date'].max():%Y-%m-%d}"
            ok = df["jam"][df["jam"] >= 0]
            rep["rentang_jam"] = f"{ok.min()}:00 - {ok.max()}:59" if len(ok) else "-"
        if len(df) != 1000:
            rep["catatan"].append(f"Dataset referensi berisi 1.000 transaksi; file ini berisi {len(df)} baris.")
        return df, rep


# ------------------------------------------------------------------ Agen 2
class ReconAgent:
    """Tax & Accounting Reconciliation Agent."""

    def __init__(self, llm):
        self.llm = llm

    def run(self, d, F, params):
        tol, rate = params.get("tol", 0.01), params.get("tax_rate", 0.05)
        up, q, cg, tx, tt, gi, gm = (d[c].values for c in ("unit_price", "quantity", "cogs", "tax_5", "total", "gross_income", "gross_margin_percentage"))
        stats = {}

        def chk(name, mask, pts, texts, diff):
            F.add(mask, pts, texts, name, diff)
            stats[name] = {"jumlah_baris": int(np.sum(mask)), "total_selisih": round(float(np.abs(diff)[mask].sum()), 2)}

        e_cogs = up * q
        chk("Rekonsiliasi Harga x Qty", np.abs(cg - e_cogs) > tol, 40,
            [f"cogs {a:.2f} != Unit price x Quantity {b:.2f}" for a, b in zip(cg, e_cogs)], cg - e_cogs)
        e_tax = rate * cg
        chk("Rekonsiliasi Pajak 5%", np.abs(tx - e_tax) > tol, 40,
            [f"Tax {a:.4f} != {rate:.0%} x cogs ({b:.4f})" for a, b in zip(tx, e_tax)], tx - e_tax)
        e_tot = cg + tx
        chk("Rekonsiliasi Total", np.abs(tt - e_tot) > tol, 40,
            [f"Total {a:.4f} != cogs + Tax ({b:.4f})" for a, b in zip(tt, e_tot)], tt - e_tot)
        chk("Gross Income", np.abs(gi - (tt - cg)) > tol, 25,
            [f"Gross income {a:.4f} != Total - cogs ({b:.4f})" for a, b in zip(gi, tt - cg)], gi - (tt - cg))
        e_gm = np.where(tt != 0, gi / np.where(tt != 0, tt, 1) * 100, 0)
        chk("Margin %", np.abs(gm - e_gm) > 0.05, 15,
            [f"Gross margin % {a:.3f} != gross income/Total ({b:.3f})" for a, b in zip(gm, e_gm)], np.zeros(len(d)))
        chk("Data Tidak Wajar", (up <= 0) | (q <= 0) | (tt <= 0), 40, "Harga/kuantitas/total nol atau negatif", np.zeros(len(d)))
        dup = d.duplicated("invoice_id", keep=False).values
        chk("Invoice ID Ganda", dup, 40, "Invoice ID muncul lebih dari sekali (indikasi double posting/penomoran bermasalah)", tt)
        tax_tbl = d.groupby(["bulan", "branch"]).agg(penjualan=("cogs", "sum"), pajak_tercatat=("tax_5", "sum")).reset_index()
        tax_tbl["pajak_seharusnya"] = tax_tbl["penjualan"] * rate
        tax_tbl["selisih"] = tax_tbl["pajak_tercatat"] - tax_tbl["pajak_seharusnya"]
        tax_tbl["tarif_efektif_%"] = tax_tbl["pajak_tercatat"] / tax_tbl["penjualan"] * 100
        F.extra["tax_table"] = tax_tbl.round(2)
        recon = pd.DataFrame([{"pengujian": k, **v} for k, v in stats.items()])
        F.extra["recon"] = recon
        bad = recon[recon["jumlah_baris"] > 0]
        note = self.llm.chat("Anda Tax & Accounting Reconciliation Agent. Tulis opini rekonsiliasi singkat (maks 90 kata, Bahasa Indonesia, Markdown) "
                             "berdasarkan hasil uji; sebut pengujian bermasalah & implikasi pajak/pembukuan.",
                             json.dumps({"baris_diuji": len(d), "hasil": stats, "tarif_efektif_%": round(float(d.tax_5.sum() / d.cogs.sum() * 100), 4)}),
                             max_tokens=300, temperature=0.2)
        if not note:
            note = (f"Seluruh {len(d)} transaksi lolos rekonsiliasi harga x qty, pajak 5%, total, dan gross income (toleransi {tol})."
                    if bad.empty else "Ditemukan penyimpangan pada: " + "; ".join(f"**{r.pengujian}** ({r.jumlah_baris} baris, selisih {r.total_selisih:,.2f})" for r in bad.itertuples()) + ".")
        F.extra["recon_note"] = note


# ------------------------------------------------------------------ Agen 3
class FraudAgent:
    """Fraud & Cashless Anomaly Agent (+ rekomendasi tindak lanjut)."""

    def __init__(self, llm):
        self.llm = llm

    def run(self, d, F, params):
        oh, ch = params.get("open_hour", 10), params.get("close_hour", 21)
        tot, jam = d["total"].values, d["jam"].values
        z, med = robust_z(tot, d["product_line"])
        F.add(z > 3.5, 30, [f"Nilai belanja {t:.2f} outlier ekstrem (median lini {m:.2f})" for t, m in zip(tot, med)], "Nilai Ekstrem")
        p99, p95 = np.quantile(tot, .99), np.quantile(tot, .95)
        F.add(tot >= p99, 12, f"Nilai belanja termasuk 1% tertinggi (>= {p99:.2f})", "Nilai Ekstrem")
        cash = (d["payment"].str.lower() == "cash").values
        F.add(cash & (tot >= p95), 20, "Transaksi tunai bernilai tinggi (risiko skimming / minim jejak digital)", "Anomali Pembayaran")
        F.add((jam >= 0) & ((jam < oh) | (jam >= ch)), 30, f"Transaksi di luar jam operasional ({oh}:00-{ch}:00)", "Di Luar Jam Operasional")
        key = ["branch", "date", "time", "product_line", "unit_price", "quantity", "payment"]
        F.add(d.duplicated(key, keep=False).values, 35, "Transaksi kembar: isi identik dengan Invoice ID berbeda (indikasi double posting)", "Transaksi Kembar", tot)
        # rating: sel cabang x jam
        r = d["rating"].values
        m, s = r.mean(), (r.std() or 1.0)
        g = d.groupby(["branch", "jam"])["rating"].agg(["mean", "count"])
        g["z"] = (g["mean"] - m) / (s / np.sqrt(g["count"]))
        bad = g[(g["z"] < -3) & (g["count"] >= 10)]
        badset = set(bad.index)
        F.add(np.array([(b, j) in badset for b, j in zip(d["branch"], d["jam"])]), 15, "Rating rendah anomali pada kombinasi cabang-jam ini", "Anomali Rating")
        gf = [f"Cabang {b} jam {j}:00: rating rata-rata {row['mean']:.2f} (n={int(row['count'])}) jauh di bawah rata-rata global {m:.2f}" for (b, j), row in bad.iterrows()]
        mm = d.groupby(["bulan", "branch"])["rating"].mean().unstack("branch")
        for b in mm:
            for bl, v in mm[b].diff()[lambda x: x < -1].items():
                gf.append(f"Cabang {b}: rating rata-rata turun {abs(v):.2f} poin pada {bl} dibanding bulan sebelumnya")
        pm = pd.crosstab(d["branch"], d["payment"], normalize="index") * 100
        overall = d["payment"].value_counts(normalize=True) * 100
        for b in pm.index:
            for p in pm.columns:
                if abs(pm.loc[b, p] - overall[p]) > 10:
                    gf.append(f"Cabang {b}: porsi {p} {pm.loc[b, p]:.1f}% menyimpang >10 poin dari rata-rata {overall[p]:.1f}%")
        F.extra.update(group_findings=gf, payment_mix=pm.round(1), rating_cells=g.reset_index().round(2))

    def enrich(self, res):
        recs = {}
        top = res[res["risk_score"] > 0].nlargest(12, "risk_score")
        if self.llm.active and len(top):
            lines = "\n".join(f"{r['invoice_id']}|{int(r['risk_score'])}|{str(r['temuan'])[:200]}" for _, r in top.iterrows())
            out = self.llm.json('Anda auditor ritel senior. Untuk tiap temuan beri tindakan tindak lanjut singkat (maks 15 kata). '
                                'Balas HANYA JSON: {"items":[{"id":"...","tindakan":"..."}]}', "id|skor|temuan\n" + lines, max_tokens=1200)
            recs = {str(i.get("id")): i.get("tindakan") for i in out.get("items", []) if isinstance(i, dict)}
        default = "Verifikasi struk/jurnal penjualan dan konfirmasi ke manajer cabang"
        return [recs.get(str(i), default if s > 0 else "") for i, s in zip(res["invoice_id"], res["risk_score"])]


# ------------------------------------------------------------------ Agen 4
class ProfitAgent:
    """Profitability & RFM Customer Agent."""

    def __init__(self, llm):
        self.llm = llm

    @staticmethod
    def agg(res, by):
        g = res.groupby(by).agg(transaksi=("invoice_id", "count"), qty=("quantity", "sum"), penjualan=("cogs", "sum"),
                                gross_income=("gross_income", "sum"), total=("total", "sum"), rating=("rating", "mean"))
        g["margin_%"] = g["gross_income"] / g["total"] * 100
        g["pangsa_gi_%"] = g["gross_income"] / g["gross_income"].sum() * 100
        g["pangsa_qty_%"] = g["qty"] / g["qty"].sum() * 100
        g["efisiensi"] = g["pangsa_gi_%"] / g["pangsa_qty_%"]
        return g.sort_values("gross_income", ascending=False).round(2)

    def run(self, res):
        prod, br, ct = self.agg(res, "product_line"), self.agg(res, "branch"), self.agg(res, "customer_type")
        lowm = prod[(prod["pangsa_qty_%"] > prod["pangsa_qty_%"].median()) & (prod["efisiensi"] < 1)]
        heat = res.pivot_table(index="product_line", columns="branch", values="gross_income", aggfunc="sum").round(2)
        ref = res["date"].max() + pd.Timedelta(days=1)
        seg = (res.assign(segmen=res["customer_type"] + " | " + res["gender"] + " | Cabang " + res["branch"]).groupby("segmen")
               .agg(recency=("date", lambda s: (ref - s.max()).days), frekuensi=("invoice_id", "count"), monetary=("total", "sum")))
        seg["basket"] = seg["monetary"] / seg["frekuensi"]
        pr = lambda s, asc=True: s.rank(pct=True, ascending=asc)
        seg["rfm_score"] = ((pr(seg["recency"], False) + pr(seg["frekuensi"]) + pr(seg["monetary"])) / 3 * 100).round(0)
        seg["label"] = np.select([seg["rfm_score"] >= 70, seg["rfm_score"] >= 40], ["Champions", "Potensial"], "Perlu Reaktivasi")
        seg = seg.round(2).sort_values("rfm_score", ascending=False).reset_index()
        payload = json.dumps({"produk": prod[["qty", "gross_income", "margin_%", "pangsa_gi_%", "efisiensi", "rating"]].to_dict("index"),
                              "cabang": br[["gross_income", "pangsa_gi_%", "rating"]].to_dict("index"),
                              "tipe_pelanggan": ct[["transaksi", "gross_income", "pangsa_gi_%"]].to_dict("index"),
                              "volume_tinggi_laba_rendah": list(lowm.index), "rfm_atas": seg.head(3).to_dict("records"),
                              "rfm_bawah": seg.tail(3).to_dict("records")}, default=str, ensure_ascii=False)
        txt = self.llm.chat("Anda Profitability & RFM Customer Agent. Tulis analisis profitabilitas ringkas (maks 200 kata, Bahasa Indonesia, Markdown): "
                            "kontribusi gross income per lini produk/cabang/segmen Member vs Normal, produk volume tinggi-margin rendah, "
                            "dan 3 rekomendasi bernomor.", payload, max_tokens=700, temperature=0.3)
        if not txt:
            txt = (f"- Lini produk penyumbang gross income terbesar: **{prod.index[0]}** ({prod['pangsa_gi_%'].iloc[0]:.1f}%).\n"
                   f"- Cabang terkuat: **{br.index[0]}** ({br['pangsa_gi_%'].iloc[0]:.1f}%).\n"
                   f"- Kontribusi per tipe pelanggan: {', '.join(f'{k} {v:.1f}%' for k, v in ct['pangsa_gi_%'].items())}.\n"
                   f"- Volume tinggi namun kontribusi laba di bawah proporsi: {', '.join(lowm.index) or 'tidak ada'}.\n"
                   "- Rekomendasi: (1) tinjau harga/bundling lini volume-tinggi laba-rendah, (2) reaktivasi segmen RFM rendah, (3) replikasi praktik cabang terkuat.")
        return {"prod": prod, "branch": br, "ctype": ct, "lowm": list(lowm.index), "heat": heat, "rfm": seg, "narrative": txt}

    @staticmethod
    def simulate(df, price_pct, vol_pct, member_disc, tax_rate, hpp_pct, base_tax=5.0):
        f = np.where(df["customer_type"].str.lower() == "member", 1 - member_disc / 100, 1.0)
        s0 = df["unit_price"].values * df["quantity"].values
        s1 = s0 * (1 + price_pct / 100) * (1 + vol_pct / 100) * f
        by = pd.DataFrame({"lini": df["product_line"].values, "Baseline": s0, "Skenario": s1}).groupby("lini").sum().reset_index()
        return {"sales0": s0.sum(), "sales1": s1.sum(), "tax0": s0.sum() * base_tax / 100, "tax1": s1.sum() * tax_rate / 100,
                "total0": s0.sum() * (1 + base_tax / 100), "total1": s1.sum() * (1 + tax_rate / 100),
                "laba0": s0.sum() * (1 - hpp_pct / 100), "laba1": s1.sum() * (1 - hpp_pct / 100), "by_prod": by}


# ------------------------------------------------------------------ Agen 5
class ReportingAgent:
    @staticmethod
    def summary(llm, res, kpis, extra):
        cats = Counter(c for s in res["kategori"] for c in s.split(", ") if c)
        payload = json.dumps({"kpi": kpis, "kategori": dict(cats.most_common(6)), "rekonsiliasi": extra["recon_note"][:500],
                              "temuan_kelompok": extra["group_findings"][:5]}, ensure_ascii=False, default=str)
        txt = llm.chat("Anda Executive Reporting Agent. Tulis ringkasan eksekutif audit penjualan ritel (maks 140 kata, Bahasa Indonesia, Markdown, "
                       "tanpa heading): temuan utama, dampak, prioritas tindakan.", payload, max_tokens=450, temperature=0.3)
        return txt or (f"Dari **{kpis['total_baris']}** transaksi diuji ditemukan **{kpis['anomali']}** anomali ({kpis['pct']:.1f}%), "
                       f"**{kpis['tinggi']}** berisiko Tinggi, estimasi nilai terdampak **{kpis['dampak']:,.2f}**. "
                       f"Kategori dominan: {', '.join(k for k, _ in cats.most_common(3)) or '-'}. (Ringkasan otomatis tanpa AI.)")

    @staticmethod
    def context(res, kpis, extra, profit):
        top = res[res["risk_score"] > 0].nlargest(15, "risk_score")[["invoice_id", "branch", "total", "risk_score", "risk_level", "temuan"]].copy()
        top["temuan"] = top["temuan"].str.slice(0, 150)
        mon = res.groupby(["bulan", "branch"])["total"].sum().unstack("branch").round(0).reset_index()
        return (f"Dataset: transaksi supermarket. Kolom: {list(res.columns)}\nKPI: {json.dumps(kpis, default=str)}\n"
                f"Per lini produk:\n{md(profit['prod'].reset_index())}\nPer cabang:\n{md(profit['branch'].reset_index())}\n"
                f"Per tipe pelanggan:\n{md(profit['ctype'].reset_index())}\nPenjualan total per bulan x cabang:\n{md(mon)}\n"
                f"Bauran pembayaran per cabang (%):\n{md(extra['payment_mix'].reset_index())}\n"
                f"Temuan kelompok: {extra['group_findings'][:6]}\nRekonsiliasi: {extra['recon_note'][:300]}\nTemuan transaksi teratas:\n{md(top)}")

    @staticmethod
    def chat(llm, question, r, history):
        if not llm.active:
            return "⚠️ AI tidak aktif. Isi GROQ_API_KEY di sidebar untuk memakai Copilot."
        extra = ""
        ids = re.findall(r"\d{3}-\d{2}-\d{4}", question)
        if ids:
            rows = r["results"][r["results"]["invoice_id"].isin(ids)].astype(str).to_dict("records")[:5]
            extra = "\nRecord yang disebut:\n" + json.dumps(rows, ensure_ascii=False)[:2500]
        hist = [{"role": m["role"], "content": m["content"][:600]} for m in history[-6:]]
        out = llm.chat("Anda AI Audit Copilot untuk data penjualan supermarket. Jawab berdasarkan KONTEKS (agregat & sampel, bukan seluruh baris). "
                       "Bila informasi tidak ada, jelaskan keterbatasannya. Singkat, Bahasa Indonesia, Markdown.",
                       f"KONTEKS:\n{r['context']}{extra}\n\nPERTANYAAN: {question}", max_tokens=700, temperature=0.2, history=hist)
        return out or "⚠️ Gagal mendapat respons: " + "; ".join(llm.errors[-2:])

    @staticmethod
    def report_md(res, kpis, srep, extra, profit, summary):
        top = res[res["risk_score"] > 0].nlargest(15, "risk_score")[["invoice_id", "branch", "total", "risk_score", "risk_level", "temuan", "rekomendasi_ai"]].copy()
        top["temuan"] = top["temuan"].str.slice(0, 200)
        return f"""# Kertas Kerja Audit - Penjualan Ritel, Pajak & Profitabilitas
_Dibuat: {datetime.now():%d %B %Y %H:%M} | Sistem Multi-Agen AI (Groq) | Dataset: Supermarket Sales_

## 1. Ringkasan Eksekutif
{summary}

## 2. Kualitas & Kelengkapan Data (Agen 1)
- Baris awal: {srep['baris_awal']} | Baris diuji: {srep['baris_akhir']} | Kelengkapan sel: {srep['kelengkapan_%']}%
- Format tanggal terdeteksi: `{srep['format_tanggal']}` | Format jam: `{srep['format_jam']}` | Rentang: {srep.get('rentang_tanggal', '-')}
- Baris kosong dihapus: {srep['baris_kosong_dihapus']} | Data kritis kosong: {srep['baris_data_kritis_kosong']} | Duplikat penuh: {srep['baris_duplikat_penuh_dihapus']} | Baris ber-Invoice ID ganda: {srep['baris_dengan_invoice_ganda']}
- Catatan: {'; '.join(srep['catatan']) or '-'}

## 3. Rekonsiliasi Pajak & Pembukuan (Agen 2)
{extra['recon_note']}

{md(extra['recon'])}

Rekap pajak per bulan & cabang:

{md(extra['tax_table'])}

## 4. Fraud & Anomali (Agen 3)
- Total anomali: {kpis['anomali']} ({kpis['pct']:.1f}%) | Tinggi {kpis['tinggi']} / Sedang {kpis['sedang']} / Rendah {kpis['rendah']}
- Temuan kelompok:
{chr(10).join('  - ' + g for g in extra['group_findings']) or '  - Tidak ada'}

Temuan transaksi prioritas (Top 15):

{md(top)}

## 5. Profitabilitas & RFM (Agen 4)
{profit['narrative']}

{md(profit['prod'].reset_index())}

## 6. Keterbatasan
- Pada dataset Supermarket Sales, `gross income` identik dengan `Tax 5%` dan `cogs` setara penjualan sebelum pajak, sehingga "margin" konstan (4,76%) dan bukan laba kotor sebenarnya. Analisis profit memakai definisi dataset.
- Tidak ada ID pelanggan; RFM dihitung pada tingkat segmen (tipe pelanggan x gender x cabang).
- Temuan adalah indikasi awal dan wajib diverifikasi dengan dokumen sumber.
"""


# ------------------------------------------------------------------ Orkestrasi
def run_pipeline(df_raw, llm, params=None, progress=None):
    params, p = params or {}, progress or (lambda f, t: None)
    p(.05, "🧹 Agen 1 - Data Sanitizer & Ingestion...")
    clean, srep = SanitizerAgent(llm).run(df_raw)
    if srep["kolom_hilang"]:
        raise ValueError(f"Kolom wajib tidak ditemukan: {srep['kolom_hilang']}")
    if clean.empty:
        raise ValueError("Tidak ada baris valid setelah pembersihan data.")
    F = Findings(len(clean))
    p(.25, "🧾 Agen 2 - Tax & Accounting Reconciliation...")
    ReconAgent(llm).run(clean, F, params)
    p(.45, "🕵️ Agen 3 - Fraud & Cashless Anomaly...")
    fraud = FraudAgent(llm)
    fraud.run(clean, F, params)
    res = clean.copy()
    sc = np.minimum(F.score, 100)
    res["risk_score"] = sc.round(0)
    res["risk_level"] = np.select([sc >= 60, sc >= 30, sc > 0], ["Tinggi", "Sedang", "Rendah"], "Normal")
    res["temuan"] = ["; ".join(x) for x in F.reasons]
    res["kategori"] = [", ".join(x) for x in F.cats]
    res["estimasi_dampak"] = F.impact.round(2)
    res["rekomendasi_ai"] = fraud.enrich(res)
    n_flag = int((sc > 0).sum())
    kpis = {"total_baris": len(res), "anomali": n_flag, "pct": n_flag / len(res) * 100,
            "tinggi": int((res.risk_level == "Tinggi").sum()), "sedang": int((res.risk_level == "Sedang").sum()),
            "rendah": int((res.risk_level == "Rendah").sum()), "dampak": float(res["estimasi_dampak"].sum()),
            "total_penjualan": float(res["total"].sum()), "total_pajak": float(res["tax_5"].sum()),
            "gross_income": float(res["gross_income"].sum()), "rating": float(res["rating"].mean())}
    p(.7, "📈 Agen 4 - Profitability & RFM Customer...")
    profit = ProfitAgent(llm).run(res)
    p(.88, "📝 Agen 5 - Executive Reporting & Copilot...")
    extra = F.extra
    summary = ReportingAgent.summary(llm, res, kpis, extra)
    p(1.0, "Selesai")
    return {"results": res, "kpis": kpis, "sanitizer": srep, "recon": extra["recon"], "recon_note": extra["recon_note"],
            "tax_table": extra["tax_table"], "group_findings": extra["group_findings"], "payment_mix": extra["payment_mix"],
            "profit": profit, "summary": summary, "report_md": ReportingAgent.report_md(res, kpis, srep, extra, profit, summary),
            "context": ReportingAgent.context(res, kpis, extra, profit),
            "llm_errors": list(dict.fromkeys(llm.errors)), "llm_calls": llm.calls}
