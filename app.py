"""app.py - Audit Penjualan Ritel, Pajak & Analisis Profitabilitas Pelanggan (Supermarket Sales) - Multi-Agen AI (Groq)."""
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import agents as ag
from sample_data import generate_supermarket

st.set_page_config(page_title="Audit Penjualan Ritel & Profitabilitas", page_icon="🛍️", layout="wide")

# ============================================================ TEMA CREAM & MAROON
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Inter:wght@400;500;600&display=swap');
:root{--maroon:#6B1D2F;--maroon2:#800020;--cream:#FDFBF7;--beige:#F3ECE1;--rose:#D8B4BC;--ink:#2B1A1E;--gold:#C9A24B;}
html, body, [class*="css"], .stApp {font-family:'Inter',sans-serif; color:var(--ink);}
.stApp {background:var(--cream);}
.block-container {padding-top:1.6rem; max-width:1300px;}
h1,h2,h3,h4 {font-family:'Playfair Display',serif !important; color:var(--maroon) !important; letter-spacing:.2px;}
.hero {background:linear-gradient(120deg,#6B1D2F 0%,#800020 55%,#8C2F39 100%); padding:26px 32px; border-radius:16px;
       box-shadow:0 6px 18px rgba(107,29,47,.25); margin-bottom:18px; border-bottom:4px solid var(--gold);}
.hero h1 {color:#FDFBF7 !important; margin:0 0 4px 0; font-size:2rem;}
.hero p {color:#F3ECE1; margin:0; font-size:.98rem; opacity:.92;}
section[data-testid="stSidebar"] {background:var(--beige); border-right:1px solid var(--rose);}
section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {color:var(--maroon) !important;}
div[data-testid="stMetric"] {background:var(--beige); border:1px solid var(--rose); border-left:5px solid var(--maroon);
       border-radius:12px; padding:14px 16px; box-shadow:0 3px 10px rgba(107,29,47,.10);}
div[data-testid="stMetricLabel"] p {color:#6f4a52; font-weight:500; font-size:.85rem;}
div[data-testid="stMetricValue"] {color:var(--maroon); font-family:'Playfair Display',serif; font-weight:700;}
.stButton>button, .stDownloadButton>button {background:var(--maroon); color:#FDFBF7; border:1px solid var(--maroon); border-radius:10px;
       font-weight:600; padding:.5rem 1rem; transition:all .15s ease;}
.stButton>button:hover, .stDownloadButton>button:hover {background:var(--maroon2); border-color:var(--maroon2); color:#fff; box-shadow:0 4px 12px rgba(128,0,32,.3);}
.stButton>button:focus {box-shadow:0 0 0 3px rgba(107,29,47,.25) !important; color:#fff;}
.stTabs [data-baseweb="tab-list"] {gap:6px; border-bottom:1px solid var(--rose);}
.stTabs [data-baseweb="tab"] {background:var(--beige); border-radius:10px 10px 0 0; color:#6f4a52; padding:10px 18px; font-weight:500;}
.stTabs [aria-selected="true"] {background:var(--maroon) !important; color:#FDFBF7 !important;}
.stTabs [data-baseweb="tab-highlight"] {background-color:var(--gold) !important;}
div[data-testid="stExpander"] {background:var(--beige); border:1px solid var(--rose); border-radius:12px;}
div[data-testid="stChatMessage"] {background:var(--beige); border:1px solid var(--rose); border-radius:12px; box-shadow:0 2px 8px rgba(107,29,47,.08);}
div[data-testid="stDataFrame"] {border:1px solid var(--rose); border-radius:12px; overflow:hidden;}
div[data-testid="stAlert"] {border-radius:12px;}
hr {border-color:var(--rose) !important;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

MAROON = ["#6B1D2F", "#B5523B", "#C9A24B", "#A4505E", "#8C2F39", "#E0B98A", "#D8B4BC"]  # maroon, terracotta, gold, burgundy...
SCALE = ["#F3ECE1", "#D8B4BC", "#A4505E", "#6B1D2F"]
LEVEL_COLORS = {"Tinggi": "#6B1D2F", "Sedang": "#B5523B", "Rendah": "#C9A24B", "Normal": "#E4D5BD"}


def show(fig):
    fig.update_layout(colorway=MAROON, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#FDFBF7", font=dict(color="#3A1F26", family="Inter, sans-serif"),
                      title_font=dict(family="Playfair Display, serif", size=18, color="#6B1D2F"), margin=dict(t=55, l=10, r=10, b=10),
                      legend=dict(bgcolor="rgba(0,0,0,0)"))
    fig.update_xaxes(gridcolor="#EADFCF", linecolor="#D8B4BC")
    fig.update_yaxes(gridcolor="#EADFCF", linecolor="#D8B4BC")
    st.plotly_chart(fig, theme=None)


fmt = lambda v: f"{v:,.2f}"

# ============================================================ DATASET REPOSITORY (otomatis)
BASE_DIR = Path(__file__).parent
CANDIDATES = ["SuperMarket Analysis.csv", "supermarket_sales.csv", "supermarket_sales - Sheet1.csv", "Supermarket Sales.csv",
              "SuperMarket Analysis.xlsx", "supermarket_sales.xlsx"]


def _norm(n):
    return re.sub(r"[^a-z0-9]", "", n.lower())


def find_default_file():
    """Cari dataset di root repo / folder data/: nama persis dulu, lalu nama mengandung 'supermarket', lalu CSV berkolom 'Invoice ID'."""
    dirs = [d for d in (BASE_DIR, BASE_DIR / "data") if d.is_dir()]
    for d in dirs:
        for n in CANDIDATES:
            if (d / n).is_file():
                return d / n
    files = [p for d in dirs for p in sorted(d.iterdir()) if p.suffix.lower() in (".csv", ".xlsx", ".xls")]
    for p in files:
        if "supermarket" in _norm(p.name):
            return p
    for p in files:
        if p.suffix.lower() == ".csv":
            try:
                if "invoiceid" in _norm(p.open(encoding="utf-8", errors="ignore").readline()):
                    return p
            except Exception:
                pass
    return None


@st.cache_data(show_spinner="Membaca dataset dari repository...")
def read_table(path: str, mtime: float):
    """Dibaca sekali & di-cache; mtime membuat cache diperbarui bila file di repo berubah."""
    if path.lower().endswith(".csv"):
        try:
            return pd.read_csv(path, sep=None, engine="python")
        except UnicodeDecodeError:
            return pd.read_csv(path, sep=None, engine="python", encoding="latin-1")
    return pd.read_excel(path)


def default_df():
    p = find_default_file()
    if p is None:
        return None, None
    try:
        return read_table(str(p), p.stat().st_mtime), p.name
    except Exception as e:
        st.sidebar.error(f"Gagal membaca {p.name}: {e}")
        return None, None


def load_upload(f):
    if f.name.lower().endswith(".csv"):
        f.seek(0)
        return pd.read_csv(f, sep=None, engine="python")
    return pd.read_excel(f)


def secret_key():
    try:
        k = st.secrets["GROQ_API_KEY"]
        return "" if "ISI_" in k else k
    except Exception:
        return ""


# ============================================================ SIDEBAR
with st.sidebar:
    st.header("⚙️ Konfigurasi")
    key_in = st.text_input("GROQ_API_KEY", type="password", help="Kosongkan untuk memakai st.secrets['GROQ_API_KEY']")
    api_key = key_in or secret_key()
    st.caption("🔑 API key terdeteksi" if api_key else "⚠️ API key belum diisi: agen berjalan mode deterministik (tanpa narasi AI)")
    model = st.selectbox("Model Groq", ag.MODELS)
    c1, c2 = st.columns(2)
    open_h = c1.number_input("Jam buka", 0, 23, 10)
    close_h = c2.number_input("Jam tutup", 1, 24, 21)
    st.divider()
    up = st.file_uploader("Unggah file lain (opsional)", type=["csv", "xlsx"])
    reload_btn = st.button("🔄 Muat Ulang Dataset Repository")
    sample_btn = st.button("🧪 Data Sampel Sintetis")
    autorun_cb = st.checkbox("Jalankan analisis otomatis saat dibuka", value=False,
                             help="Aktif = agen (dan Groq bila ada API key) langsung berjalan saat data dimuat.")
    run_btn = st.button("🚀 Jalankan Analisis Multi-Agen", type="primary")

# ============================================================ STATE & DATA
ss = st.session_state
if ss.get("df_raw") is None:  # 1) dataset repository dimuat otomatis
    d, n = default_df()
    if d is not None:
        ss.df_raw, ss.source, ss.auto_done = d, f"{n} (dataset repository)", False
    else:  # fallback agar aplikasi tetap bisa didemokan
        ss.df_raw, ss.source, ss.auto_done = generate_supermarket(), "Data sintetis (CSV repository tidak ditemukan)", False
        ss.fallback = True
if reload_btn:
    d, n = default_df()
    if d is None:
        st.sidebar.warning("File dataset tidak ditemukan di repository.")
    else:
        ss.df_raw, ss.source, ss.result, ss.chat, ss.auto_done, ss.fallback = d, f"{n} (dataset repository)", None, [], False, False
if up is not None and ss.get("upload_sig") != (up.name, up.size):  # 2) upload manual menimpa
    try:
        ss.df_raw, ss.source, ss.upload_sig = load_upload(up), up.name, (up.name, up.size)
        ss.result, ss.chat, ss.auto_done, ss.fallback = None, [], False, False
    except Exception as e:
        st.sidebar.error(f"Gagal membaca file: {e}")
if sample_btn:
    ss.df_raw, ss.source, ss.result, ss.chat, ss.auto_done, ss.fallback = generate_supermarket(), "Data sampel sintetis", None, [], False, True
st.sidebar.caption(f"📂 Sumber data aktif: **{ss.source}**")

auto = autorun_cb and ss.get("result") is None and not ss.get("auto_done")
if run_btn or auto:
    ss.auto_done = True
    llm = ag.LLM(api_key, model)
    bar = st.progress(0.0, text="Memulai...")
    try:
        ss.result = ag.run_pipeline(ss.df_raw, llm, {"open_hour": int(open_h), "close_hour": int(close_h)}, lambda f, t: bar.progress(f, text=t))
        ss.chat = []
    except Exception as e:
        st.error(f"Analisis gagal: {e}")
    finally:
        bar.empty()

# ============================================================ HEADER
st.markdown('<div class="hero"><h1>🛍️ Audit Penjualan Ritel, Pajak & Profitabilitas</h1>'
            '<p>Sistem Multi-Agen AI untuk rekonsiliasi pajak, deteksi fraud, analisis profitabilitas &amp; RFM · Supermarket Sales · Groq API</p></div>',
            unsafe_allow_html=True)
r = ss.get("result")
if ss.get("fallback"):
    st.warning("File CSV supermarket tidak ditemukan di repository, jadi dipakai **data sintetis** (berisi anomali yang sengaja disuntik). "
               "Letakkan `SuperMarket Analysis.csv` atau `supermarket_sales.csv` di root repo untuk memakai data asli.")
if r is None:
    st.info("Data sudah dimuat otomatis. Klik **🚀 Jalankan Analisis Multi-Agen** di sidebar untuk memulai (upload manual bersifat opsional).")
    st.subheader(f"Pratinjau data: {ss.source} · {len(ss.df_raw):,} baris")
    st.dataframe(ss.df_raw.head(50))
    st.stop()

res, k, pf = r["results"], r["kpis"], r["profit"]
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Ringkasan & KPI", "🔍 Hasil Audit & Anomali", "💎 Profitabilitas & RFM", "🎛️ Simulasi What-If", "🤖 Copilot & Export"])

# ------------------------------------------------------------ TAB 1
with tab1:
    c = st.columns(6)
    c[0].metric("Transaksi diuji", f"{k['total_baris']:,}")
    c[1].metric("Anomali", k["anomali"], f"{k['pct']:.1f}%", delta_color="off")
    c[2].metric("Risiko Tinggi", k["tinggi"])
    c[3].metric("Total penjualan", fmt(k["total_penjualan"]))
    c[4].metric("Total pajak 5%", fmt(k["total_pajak"]))
    c[5].metric("Rating rata-rata", f"{k['rating']:.2f}")
    st.caption(f"Estimasi nilai terdampak (selisih rekonsiliasi + potensi double posting): **{fmt(k['dampak'])}** · satuan mata uang sesuai dataset.")
    a, b = st.columns(2)
    lv = res["risk_level"].value_counts().reindex(list(LEVEL_COLORS)).dropna().reset_index()
    lv.columns = ["Tingkat", "Jumlah"]
    with a:
        show(px.pie(lv, names="Tingkat", values="Jumlah", hole=.5, color="Tingkat", color_discrete_map=LEVEL_COLORS, title="Distribusi Tingkat Risiko"))
    cats = Counter(x for s in res["kategori"] for x in s.split(", ") if x)
    with b:
        if cats:
            cd = pd.DataFrame(cats.most_common(), columns=["Kategori", "Jumlah"]).sort_values("Jumlah")
            show(px.bar(cd, x="Jumlah", y="Kategori", orientation="h", title="Temuan per Kategori", color="Jumlah", color_continuous_scale=SCALE))
        else:
            st.success("Tidak ada anomali terdeteksi.")
    a, b = st.columns(2)
    mon = res.groupby(["bulan", "branch"])["total"].sum().reset_index()
    with a:
        show(px.bar(mon, x="bulan", y="total", color="branch", barmode="group", title="Penjualan per Bulan & Cabang"))
    with b:
        daily = res.groupby("date")["total"].sum().reset_index()
        show(px.area(daily, x="date", y="total", title="Tren Penjualan Harian"))
    st.subheader("Ringkasan Eksekutif")
    st.markdown(r["summary"])
    with st.expander("🧾 Rekonsiliasi Pajak & Pembukuan (Agen 2)", expanded=True):
        st.markdown(r["recon_note"])
        st.dataframe(r["recon"])
        st.caption("Rekap pajak per bulan & cabang")
        st.dataframe(r["tax_table"])
    with st.expander("🕵️ Temuan Kelompok: Rating, Pembayaran (Agen 3)"):
        for g in r["group_findings"] or ["Tidak ada temuan kelompok."]:
            st.markdown(f"- {g}")
        st.caption("Bauran metode pembayaran per cabang (%)")
        st.dataframe(r["payment_mix"])
    with st.expander("🧹 Laporan Data Sanitizer (Agen 1)"):
        st.json(r["sanitizer"])
    if r["llm_errors"]:
        st.warning("Catatan LLM: " + " | ".join(r["llm_errors"]))
    elif r["llm_calls"] == 0:
        st.caption("ℹ️ Berjalan tanpa Groq (API key kosong): narasi & rekomendasi memakai templat otomatis.")

# ------------------------------------------------------------ TAB 2
with tab2:
    f1, f2, f3 = st.columns([2, 2, 3])
    levels = f1.multiselect("Tingkat risiko", list(LEVEL_COLORS), default=["Tinggi", "Sedang", "Rendah"])
    all_cats = sorted(cats)
    sel_cats = f2.multiselect("Kategori temuan", all_cats)
    q = f3.text_input("Cari (Invoice ID / cabang / alasan)")
    df = res[res["risk_level"].isin(levels)]
    if sel_cats:
        df = df[df["kategori"].apply(lambda s: any(c in s.split(", ") for c in sel_cats))]
    if q:
        df = df[df.astype(str).apply(lambda s: s.str.contains(q, case=False, na=False)).any(axis=1)]
    front = [c for c in ["invoice_id", "branch", "city", "customer_type", "product_line", "payment", "total", "risk_score", "risk_level",
                         "kategori", "temuan", "rekomendasi_ai", "estimasi_dampak"] if c in df]
    df = df.sort_values("risk_score", ascending=False)
    df = df[front + [c for c in df.columns if c not in front]]
    st.caption(f"{len(df):,} baris ditampilkan")
    st.dataframe(df, height=520, column_config={"risk_score": st.column_config.ProgressColumn("Skor Risiko", min_value=0, max_value=100, format="%d")})
    st.download_button("⬇️ Unduh tampilan terfilter (CSV)", df.to_csv(index=False).encode("utf-8-sig"), "hasil_audit_filter.csv", "text/csv")

# ------------------------------------------------------------ TAB 3
with tab3:
    st.subheader("Kontribusi Gross Income & Segmen Pelanggan")
    prod = pf["prod"].reset_index()
    a, b = st.columns(2)
    with a:
        show(px.bar(prod.sort_values("gross_income"), x="gross_income", y="product_line", orientation="h", color="efisiensi",
                    color_continuous_scale=SCALE, title="Gross Income per Lini Produk (warna = efisiensi laba/volume)"))
    with b:
        show(px.imshow(pf["heat"], text_auto=".0f", aspect="auto", color_continuous_scale=SCALE, title="Gross Income: Lini Produk x Cabang"))
    a, b = st.columns(2)
    ct = pf["ctype"].reset_index()
    with a:
        show(px.bar(ct, x="customer_type", y=["gross_income", "penjualan"], barmode="group", title="Member vs Normal"))
    with b:
        br = pf["branch"].reset_index()
        show(px.bar(br, x="branch", y="gross_income", color="rating", color_continuous_scale=SCALE, title="Gross Income per Cabang (warna = rating)"))
    if pf["lowm"]:
        st.warning("**Volume tinggi, kontribusi laba di bawah proporsi:** " + ", ".join(pf["lowm"]))
    else:
        st.success("Tidak ada lini produk dengan volume tinggi namun kontribusi laba di bawah proporsi.")
    st.subheader("Segmentasi RFM (tipe pelanggan x gender x cabang)")
    rfm = pf["rfm"]
    show(px.scatter(rfm, x="frekuensi", y="monetary", size="basket", color="rfm_score", hover_name="segmen", color_continuous_scale=SCALE,
                    title="Frekuensi vs Monetary (ukuran = rata-rata keranjang)"))
    st.dataframe(rfm)
    with st.expander("📈 Analisis Profitabilitas (Agen 4)", expanded=True):
        st.markdown(pf["narrative"])
        st.dataframe(prod)

# ------------------------------------------------------------ TAB 4
with tab4:
    st.subheader("Simulasi Skenario (What-If)")
    s1, s2, s3 = st.columns(3)
    price_pct = s1.slider("Perubahan harga jual (%)", -20, 20, 0)
    vol_pct = s2.slider("Perubahan volume penjualan (%)", -30, 30, 0)
    disc = s3.slider("Diskon khusus Member (%)", 0, 20, 0)
    s4, s5 = st.columns(2)
    tax_new = s4.slider("Tarif pajak penjualan (%)", 0.0, 20.0, 5.0, step=0.5)
    hpp = s5.slider("Asumsi HPP (% dari penjualan)", 50, 95, 75)
    s = ag.ProfitAgent.simulate(res, price_pct, vol_pct, disc, tax_new, hpp)
    m = st.columns(4)
    m[0].metric("Penjualan sebelum pajak", fmt(s["sales1"]), fmt(s["sales1"] - s["sales0"]))
    m[1].metric("Pajak terutang", fmt(s["tax1"]), fmt(s["tax1"] - s["tax0"]), delta_color="off")
    m[2].metric("Total penerimaan", fmt(s["total1"]), fmt(s["total1"] - s["total0"]))
    m[3].metric("Laba kotor (asumsi HPP)", fmt(s["laba1"]), fmt(s["laba1"] - s["laba0"]))
    bp = s["by_prod"].melt(id_vars="lini", var_name="Skenario", value_name="Penjualan")
    show(px.bar(bp, x="lini", y="Penjualan", color="Skenario", barmode="group", title="Penjualan per Lini Produk: Baseline vs Skenario"))
    st.caption("Baseline = Unit price x Quantity dengan pajak 5%. Laba kotor memakai asumsi HPP Anda karena kolom gross income dataset identik dengan pajak 5%.")

# ------------------------------------------------------------ TAB 5
with tab5:
    st.subheader("💬 Chat with Sales Data")
    ss.setdefault("chat", [])
    for msg in ss.chat:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    if prompt := st.chat_input("Mis. 'Cabang mana yang rating-nya paling mencurigakan?' atau sebut Invoice ID"):
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Copilot menganalisis..."):
                ans = ag.ReportingAgent.chat(ag.LLM(api_key, model), prompt, r, ss.chat)
            st.markdown(ans)
        ss.chat += [{"role": "user", "content": prompt}, {"role": "assistant", "content": ans}]
    st.divider()
    st.subheader("📥 Export Kertas Kerja Audit")
    e = st.columns(3)
    e[0].download_button("⬇️ Kertas Kerja (Markdown)", r["report_md"], "kertas_kerja_audit.md", "text/markdown")
    e[1].download_button("⬇️ Kertas Kerja (TXT)", r["report_md"], "kertas_kerja_audit.txt", "text/plain")
    e[2].download_button("⬇️ Seluruh hasil (CSV)", res.to_csv(index=False).encode("utf-8-sig"), "hasil_audit_lengkap.csv", "text/csv")
    with st.expander("Pratinjau laporan"):
        st.markdown(r["report_md"])
