"""sample_data.py - Generator data sintetis berformat Supermarket Sales (fallback bila CSV repo tidak ada)."""
import numpy as np
import pandas as pd

LINES = ["Health and beauty", "Electronic accessories", "Home and lifestyle", "Sports and travel",
         "Food and beverages", "Fashion accessories"]


def generate_supermarket(n: int = 1000, seed: int = 11, anomalies: bool = True) -> pd.DataFrame:
    """Kolom & format identik dengan file Kaggle (Date m/d/yyyy, Time H:MM). anomalies=True menyuntik temuan uji."""
    n = max(n, 100)
    rng = np.random.default_rng(seed)
    br = rng.choice(["A", "B", "C"], n)
    city = {"A": "Yangon", "B": "Mandalay", "C": "Naypyitaw"}
    price = np.round(rng.uniform(10, 100, n), 2)
    qty = rng.integers(1, 11, n).astype(float)
    pay = rng.choice(["Ewallet", "Cash", "Credit card"], n, p=[.34, .34, .32])
    jam, mnt = rng.integers(10, 21, n), rng.integers(0, 60, n)
    day = pd.Timestamp("2019-01-01") + pd.to_timedelta(rng.integers(0, 89, n), unit="D")
    rating = np.round(rng.uniform(4, 10, n), 1)
    p = rng.permutation(n)
    if anomalies:
        price[p[:4]], qty[p[:4]], pay[p[:4]] = rng.uniform(380, 450, 4).round(2), 10, "Cash"   # nilai ekstrem
        jam[p[4:10]] = rng.choice([6, 7, 22, 23], 6)                                          # luar jam operasional
        m = (br == "B") & (jam == 14)
        rating[m] = np.round(rng.uniform(4, 5.5, m.sum()), 1)                                 # rating anjlok cabang B jam 14
    cogs = np.round(price * qty, 2)
    tax = np.round(cogs * .05, 4)
    gi = tax.copy()
    if anomalies:
        tax[p[10:16]] = np.round(tax[p[10:16]] * .6, 4)                                       # pajak dipotong
    total = np.round(cogs + tax, 4)
    if anomalies:
        total[p[16:21]] = np.round(total[p[16:21]] + rng.uniform(5, 30, 5), 4)                # total digelembungkan
        cogs[p[21:25]] = np.round(cogs[p[21:25]] * 1.1, 2)                                    # cogs tidak cocok
    ids = np.array([f"{a:03d}-{b:02d}-{c:04d}" for a, b, c in zip(rng.integers(100, 900, n), rng.integers(10, 99, n), rng.integers(1000, 9999, n))])
    if anomalies:
        ids[p[25:28]] = ids[p[28:31]]                                                         # invoice ganda
    df = pd.DataFrame({
        "Invoice ID": ids, "Branch": br, "City": [city[b] for b in br],
        "Customer type": rng.choice(["Member", "Normal"], n), "Gender": rng.choice(["Female", "Male"], n),
        "Product line": rng.choice(LINES, n), "Unit price": price, "Quantity": qty.astype(int), "Tax 5%": tax,
        "Total": total, "Date": [f"{d.month}/{d.day}/{d.year}" for d in day],
        "Time": [f"{h}:{m:02d}" for h, m in zip(jam, mnt)], "Payment": pay, "cogs": cogs,
        "gross margin percentage": 4.761904762, "gross income": gi, "Rating": rating})
    if anomalies:                                                                             # transaksi kembar
        dup = df.iloc[p[31:34]].copy()
        dup["Invoice ID"] = ["999-99-0001", "999-99-0002", "999-99-0003"]
        df = pd.concat([df, dup], ignore_index=True)
    return df
