# ============================================================
# SISTEM ANALISIS DATA TRANSAKSI TOKO
# Integrasi Data Mining & Business Intelligence — UAS
# ============================================================
# Fridolin Alisa Kosat · 51230032 · Teknologi Informasi
# Fakultas Pertanian, Sains dan Kesehatan — Universitas Timor
# ============================================================

import io
import base64
import warnings
from datetime import datetime
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from mlxtend.frequent_patterns import apriori, association_rules

warnings.filterwarnings("ignore")

# ============================================================
# KONFIGURASI & TEMA WARNA (SOFT ELEGANT)
# ============================================================
st.set_page_config(
    page_title="Sistem Analisis Data Transaksi Toko",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

NAVY      = "#1e293b"
TEAL      = "#0d9488"
TEAL_DK   = "#0f766e"
TEAL_LT   = "#ccfbf1"
MINT      = "#d1fae5"
SKY       = "#e0f2fe"
SKY_DK    = "#0284c7"
LAVENDER  = "#f3e8ff"
LAV_DK    = "#8b5cf6"
PEACH     = "#fef3c7"
PEACH_DK  = "#f59e0b"
ROSE      = "#ffe4e6"
ROSE_DK   = "#ef4444"
WHITE     = "#ffffff"
GRAY      = "#475569"
GRAY_LT   = "#94a3b8"
BG_APP    = "#f8fafc"
BORDER    = "#e2e8f0"

ASSETS = Path("assets")
ASSETS.mkdir(exist_ok=True)

PALETTE = ["#0d9488", "#0284c7", "#8b5cf6", "#f59e0b",
           "#ef4444", "#10b981", "#f97316", "#6366f1"]

# ============================================================
# CSS GLOBAL
# ============================================================
st.markdown(f"""
<style>
    .stApp {{
        background: linear-gradient(135deg, #f8fafc 0%, #f0fdfa 50%, #eff6ff 100%);
    }}
    .block-container {{
        padding-top: 1.2rem; padding-bottom: 2rem;
        max-width: 1350px; background-color: {WHITE};
        border-radius: 18px; padding-left: 2rem; padding-right: 2rem;
        box-shadow: 0 8px 32px rgba(13,148,136,.08), 0 2px 8px rgba(0,0,0,.04);
        margin-top: 1rem;
    }}
    .hero {{
        position: relative; border-radius: 18px; overflow: hidden;
        margin-bottom: 1.4rem;
        box-shadow: 0 12px 32px rgba(13,148,136,.18);
        border: 1px solid {BORDER};
    }}
    .hero img {{ width:100%; height:270px; object-fit:cover; display:block; }}
    .hero .ov {{
        position:absolute; inset:0;
        background: linear-gradient(120deg, rgba(15,23,42,.88) 0%, rgba(13,148,136,.72) 60%, rgba(139,92,246,.55) 100%);
        display:flex; flex-direction:column; justify-content:center;
        padding: 2rem 2.6rem; color:#fff;
    }}
    .hero h1 {{ font-size:1.85rem; margin:0 0 .4rem 0; font-weight:800; letter-spacing:-.5px;
                text-shadow: 0 2px 8px rgba(0,0,0,.2); }}
    .hero p  {{ font-size:.98rem; margin:0; opacity:.95; }}

    .kpi {{
        background: linear-gradient(135deg, {WHITE} 0%, {TEAL_LT} 100%);
        border: 1px solid #99f6e4; border-left: 5px solid {TEAL};
        color: {NAVY}; border-radius: 14px; padding: 1rem 1.2rem;
        box-shadow: 0 6px 18px rgba(13,148,136,.12);
        transition: transform .2s, box-shadow .2s;
    }}
    .kpi:hover {{ transform: translateY(-2px); box-shadow: 0 10px 24px rgba(13,148,136,.2); }}
    .kpi h3 {{ margin:0; font-size:.72rem; font-weight:700; color:{GRAY};
               text-transform:uppercase; letter-spacing:.6px; }}
    .kpi h2 {{ margin:.3rem 0 0 0; font-size:1.4rem; font-weight:800; color:{NAVY}; }}

    .kpi.sky {{ background: linear-gradient(135deg, {WHITE} 0%, {SKY} 100%);
                border-color:#bae6fd; border-left-color:{SKY_DK};
                box-shadow: 0 6px 18px rgba(2,132,199,.12); }}
    .kpi.lav {{ background: linear-gradient(135deg, {WHITE} 0%, {LAVENDER} 100%);
                border-color:#e9d5ff; border-left-color:{LAV_DK};
                box-shadow: 0 6px 18px rgba(139,92,246,.12); }}
    .kpi.peach {{ background: linear-gradient(135deg, {WHITE} 0%, {PEACH} 100%);
                  border-color:#fde68a; border-left-color:{PEACH_DK};
                  box-shadow: 0 6px 18px rgba(245,158,11,.12); }}
    .kpi.rose {{ background: linear-gradient(135deg, {WHITE} 0%, {ROSE} 100%);
                 border-color:#fecdd3; border-left-color:{ROSE_DK};
                 box-shadow: 0 6px 18px rgba(239,68,68,.12); }}

    .section-title {{
        font-size:1.15rem; font-weight:800; color:{NAVY};
        margin:1.3rem 0 .8rem 0; border-left:5px solid {TEAL};
        padding:.45rem .85rem;
        background: linear-gradient(90deg, {TEAL_LT} 0%, {MINT} 60%, transparent 100%);
        border-radius: 0 10px 10px 0;
    }}

    .soft-card {{
        background: {WHITE}; border:1px solid {BORDER}; border-radius:14px;
        padding:1.1rem 1.3rem; box-shadow:0 3px 12px rgba(0,0,0,.04);
        margin-bottom:.9rem;
    }}
    .soft-card.teal  {{ background: linear-gradient(135deg, {WHITE} 0%, {TEAL_LT} 100%);
                        border-color:#99f6e4; border-left:4px solid {TEAL}; }}
    .soft-card.blue  {{ background: linear-gradient(135deg, {WHITE} 0%, {SKY} 100%);
                        border-color:#bae6fd; border-left:4px solid {SKY_DK}; }}
    .soft-card.amber {{ background: linear-gradient(135deg, {WHITE} 0%, {PEACH} 100%);
                        border-color:#fde68a; border-left:4px solid {PEACH_DK}; }}
    .soft-card.purple{{ background: linear-gradient(135deg, {WHITE} 0%, {LAVENDER} 100%);
                        border-color:#e9d5ff; border-left:4px solid {LAV_DK}; }}
    .soft-card.rose  {{ background: linear-gradient(135deg, {WHITE} 0%, {ROSE} 100%);
                        border-color:#fecdd3; border-left:4px solid {ROSE_DK}; }}

    .profile-card {{
        background: linear-gradient(180deg, {WHITE} 0%, {SKY} 60%, {LAVENDER} 100%);
        border:1px solid #bae6fd; border-radius:18px;
        padding:1.6rem 1.4rem;
        box-shadow: 0 8px 24px rgba(2,132,199,.12);
        text-align:center;
    }}
    .profile-card .logo-wrap {{ display:flex; justify-content:center; margin-bottom:1rem; }}
    .profile-card .photo-wrap {{ display:flex; justify-content:center; margin-bottom:1rem; }}
    .profile-card .name {{ font-size:1.08rem; font-weight:800; color:{NAVY}; margin:0; }}
    .profile-card .npm {{ font-size:.85rem; color:{GRAY}; margin-top:.25rem; }}
    .profile-card .univ {{ font-size:.92rem; color:{TEAL}; font-weight:800; margin-top:.4rem;
                           letter-spacing:.3px; }}

    .stButton>button {{
        background: linear-gradient(135deg, {TEAL} 0%, {SKY_DK} 100%);
        color:#fff; border:0; border-radius:10px;
        font-weight:700; padding: .55rem 1.3rem; transition: all .2s;
        box-shadow: 0 6px 14px rgba(13,148,136,.25);
    }}
    .stButton>button:hover {{
        background: linear-gradient(135deg, {TEAL_DK} 0%, #0369a1 100%);
        color:#fff; transform: translateY(-2px);
        box-shadow: 0 10px 20px rgba(13,148,136,.35);
    }}

    section[data-testid="stSidebar"] {{
        background: linear-gradient(180deg, {NAVY} 0%, #0f172a 50%, #134e4a 100%);
    }}
    section[data-testid="stSidebar"] * {{ color: #f1f5f9 !important; }}
    section[data-testid="stSidebar"] .stRadio label:hover {{ color: {TEAL_LT} !important; }}
    section[data-testid="stSidebar"] .menu-header {{
        font-size:.7rem; font-weight:800; letter-spacing:1.2px;
        text-transform:uppercase; color:{TEAL_LT} !important;
        margin: 1rem 0 .35rem 0; padding-left:.3rem;
        border-left: 3px solid {TEAL};
    }}

    .stTabs [data-baseweb="tab-list"] {{ gap: .4rem; border-bottom: 2px solid {BORDER}; }}
    .stTabs [data-baseweb="tab"] {{
        background:{BG_APP}; border-radius:10px 10px 0 0; padding:.55rem 1.1rem;
        font-size:.9rem; font-weight: 600; color: {GRAY};
    }}
    .stTabs [aria-selected="true"] {{
        background: linear-gradient(135deg, {TEAL_LT} 0%, {MINT} 100%) !important;
        color:{TEAL_DK} !important; border-bottom: 3px solid {TEAL};
    }}

    .stTextInput > div > div > input {{
        border-radius: 10px !important; border: 1.5px solid #cbd5e1 !important;
        padding: .65rem .9rem !important; font-size: .92rem !important;
    }}
    .stTextInput > div > div > input:focus {{
        border-color: {TEAL} !important; box-shadow: 0 0 0 3px {TEAL_LT} !important;
    }}
    .stForm [data-testid="stFormSubmitButton"] > button {{
        background: linear-gradient(135deg, {NAVY} 0%, {TEAL} 100%) !important;
        color:#fff !important; border:0 !important; border-radius:10px !important;
        font-weight:700 !important; padding:.7rem 1rem !important;
        box-shadow: 0 6px 16px rgba(13,148,136,.3) !important;
    }}

    .step {{
        display:flex; gap:.85rem; align-items:center;
        background: linear-gradient(90deg, {WHITE} 0%, {TEAL_LT} 100%);
        border:1px solid #99f6e4; border-radius:12px;
        padding:.8rem 1.1rem; margin-bottom:.65rem;
        box-shadow: 0 3px 10px rgba(13,148,136,.06);
        transition: transform .2s;
    }}
    .step:hover {{ transform: translateX(4px); }}
    .step .num {{
        background: linear-gradient(135deg, {TEAL} 0%, {SKY_DK} 100%);
        color:#fff; width:30px; height:30px; min-width:30px;
        border-radius:50%; display:flex; align-items:center; justify-content:center;
        font-weight:800; font-size:.85rem;
        box-shadow: 0 4px 10px rgba(13,148,136,.3);
    }}
    .step .txt {{ font-size:.9rem; color:{NAVY}; }}
</style>
""", unsafe_allow_html=True)

# ============================================================
# HELPER GAMBAR & UI
# ============================================================
def img_b64(path: Path):
    if path.exists():
        with open(path, "rb") as f:
            data = base64.b64encode(f.read()).decode()
        ext = "png" if path.suffix.lower() == ".png" else "jpeg"
        return f"data:image/{ext};base64,{data}"
    return None

def logo_html(size=64):
    b64 = img_b64(ASSETS / "logo_unimor.png")
    if b64:
        return (f'<img src="{b64}" width="{size}" '
                f'style="border-radius:14px;object-fit:contain;'
                f'box-shadow:0 6px 16px rgba(13,148,136,.25);"/>')
    return (f'<div style="width:{size}px;height:{size}px;border-radius:14px;'
            f'background:linear-gradient(135deg,{TEAL},{SKY_DK},{LAV_DK});'
            f'display:flex;align-items:center;justify-content:center;'
            f'color:#fff;font-weight:800;font-size:{size//4.5:.0f}px;'
            f'letter-spacing:.5px;box-shadow:0 6px 16px rgba(13,148,136,.3);">UNIMOR</div>')

def profile_photo(size=120):
    b64 = img_b64(ASSETS / "foto_profil.jpg")
    if b64:
        return (f'<img src="{b64}" width="{size}" height="{size}" '
                f'style="border-radius:50%;object-fit:cover;'
                f'border:4px solid {WHITE};'
                f'box-shadow:0 8px 22px rgba(13,148,136,.35);"/>')
    return (f'<div style="width:{size}px;height:{size}px;border-radius:50%;'
            f'background:linear-gradient(135deg,{TEAL},{SKY_DK},{LAV_DK});'
            f'display:flex;align-items:center;justify-content:center;'
            f'color:#fff;font-weight:800;font-size:{size//3.5:.0f}px;'
            f'border:4px solid {WHITE};'
            f'box-shadow:0 8px 22px rgba(13,148,136,.35);">FK</div>')

def profile_card_html():
    return f"""
    <div class="profile-card">
        <div class="logo-wrap">{logo_html(90)}</div>
        <div class="photo-wrap">{profile_photo(120)}</div>
        <p class="name">Fridolin Alisa Kosat</p>
        <div class="npm"><b>NPM</b> 51230032 · Teknologi Informasi</div>
        <div class="npm">Fakultas Pertanian, Sains dan Kesehatan</div>
        <div class="univ">Universitas Timor</div>
    </div>
    """

def hero_image():
    for name in ["hero_banner.jpg", "hero_banner.png", "hero.jpg", "hero.png"]:
        p = ASSETS / name
        if p.exists():
            return img_b64(p)
    return ("https://images.pexels.com/photos/264636/pexels-photo-264636.jpeg"
            "?auto=compress&cs=tinysrgb&w=1600")

def hero_banner(title, subtitle):
    src = hero_image()
    st.markdown(f"""
    <div class="hero">
        <img src="{src}" onerror="this.style.display='none'"/>
        <div class="ov">
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

def kpi(label, value, tone=""):
    cls = f"kpi {tone}".strip()
    return f'<div class="{cls}"><h3>{label}</h3><h2>{value}</h2></div>'

def section(title):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)

def soft_card(content, tone=""):
    cls = f"soft-card {tone}".strip()
    st.markdown(f'<div class="{cls}">{content}</div>', unsafe_allow_html=True)

# ============================================================
# UTIL EXCEL
# ============================================================
def to_excel_bytes(sheets):
    buf = io.BytesIO()
    try:
        with pd.ExcelWriter(buf, engine="openpyxl") as w:
            for name, df in sheets.items():
                if df is None or len(df) == 0:
                    pd.DataFrame({"Info": ["Data kosong"]}).to_excel(w, sheet_name=name[:31], index=False)
                else:
                    df.to_excel(w, sheet_name=name[:31], index=False)
        return buf.getvalue()
    except Exception:
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, 'w') as z:
            for name, df in sheets.items():
                csv_data = df.to_csv(index=False) if df is not None else "Kosong"
                z.writestr(f"{name[:31]}.csv", csv_data)
        return zip_buf.getvalue()

# ============================================================
# LOGIN PAGE (SAMA SEPERTI KODE AWAL)
# ============================================================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = None

if not st.session_state.logged_in:
    st.markdown("<div style='height:4vh'></div>", unsafe_allow_html=True)
    _, mid, _ = st.columns([1, 1.15, 1])

    with mid:
        st.markdown(f"""
        <div style="
            background: linear-gradient(180deg, {WHITE} 0%, {SKY} 70%, {LAVENDER} 100%);
            border:1px solid #bae6fd; border-radius:20px;
            padding:2.2rem 2rem 1.6rem 2rem;
            box-shadow:0 16px 40px rgba(2,132,199,.12);
            text-align:center; margin-bottom:1rem;
        ">
            <div style="display:flex; justify-content:center; margin-bottom:1.1rem;">
                {logo_html(100)}
            </div>
            <h1 style="margin:0 0 .35rem 0; font-size:1.4rem; font-weight:800;
                color:{NAVY}; line-height:1.25;">
                Sistem Analisis Data<br/>Transaksi Toko
            </h1>
            <p style="margin:0 0 1.2rem 0; color:{TEAL}; font-size:.9rem;
                font-weight:700; letter-spacing:.4px;">
                Integrasi Data Mining &amp; Business Intelligence
            </p>
            <div style="display:flex; justify-content:center; margin:1.1rem 0 1rem 0;">
                {profile_photo(125)}
            </div>
            <div style="color:{NAVY}; font-weight:800; font-size:1rem; margin-bottom:.2rem;">
                Fridolin Alisa Kosat
            </div>
            <div style="color:{GRAY}; font-size:.82rem; margin-bottom:.35rem;">
                NPM 51230032 · Teknologi Informasi
            </div>
            <div style="color:{TEAL}; font-weight:800; font-size:.9rem;">
                Universitas Timor
            </div>
        </div>
        """, unsafe_allow_html=True)

        with st.form("login_form", clear_on_submit=False):
            username = st.text_input("👤 Username", placeholder="Masukkan username")
            password = st.text_input("🔒 Password", type="password", placeholder="Masukkan password")
            submit = st.form_submit_button("🔐 Masuk", use_container_width=True)

        if submit:
            if username == "admin" and password == "admin123":
                st.session_state.logged_in = True
                st.session_state.username = username
                st.rerun()
            else:
                st.error("❌ Username atau password salah.")

        st.markdown(f"""
        <div style="margin-top:1rem; padding:.9rem 1.1rem; border-radius:12px;
            background: linear-gradient(135deg, {PEACH} 0%, {ROSE} 100%);
            border-left:4px solid {PEACH_DK}; font-size:.85rem;
            color:{NAVY}; line-height:1.55;">
            <b style="color:#b45309">🔑 Login Demo</b><br/>
            <span style="color:{GRAY}">Username:</span>
            <code style="background:#fff;padding:.15rem .5rem;border-radius:6px;color:{NAVY};font-weight:700">admin</code>
            &nbsp;·&nbsp;
            <span style="color:{GRAY}">Password:</span>
            <code style="background:#fff;padding:.15rem .5rem;border-radius:6px;color:{NAVY};font-weight:700">admin123</code>
        </div>
        """, unsafe_allow_html=True)

    st.stop()

# ============================================================
# SESSION STATE
# ============================================================
for key, default in {
    "datasets": {}, "data_raw": None, "data_clean": None, "log_clean": None,
    "apriori_results": None, "kmeans_results": None, "history": [],
    "rfm": None, "basket": None, "snapshot": None, "kmeans_eval": None,
    "scaling_info": None, "basket_summary": None, "rfm_summary": None,
    "kmeans_k": 4, "active_menu": "🏠 Beranda",
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ============================================================
# SIDEBAR — MENU SESUAI TAHAPAN UAS (FLAT RADIO, ANTI-EROR)
# ============================================================
MENU_ITEMS = [
    "🏠 Beranda",
    "1️⃣ Upload Dataset",
    "1️⃣ Business Understanding & EDA",
    "2️⃣ Preprocessing Data",
    "3️⃣ Feature Engineering (Basket, RFM, Scaling)",
    "4️⃣ Apriori / Market Basket Analysis",
    "5️⃣ K-Means / Segmentasi Pelanggan",
    "6️⃣ Dashboard Business Intelligence",
    "7️⃣ Rekomendasi Keputusan",
    "8️⃣ Unduh Hasil",
    "8️⃣ Riwayat Analisis",
]

# Pastikan active_menu valid
if st.session_state.active_menu not in MENU_ITEMS:
    st.session_state.active_menu = MENU_ITEMS[0]

with st.sidebar:
    st.markdown(
        f'<div style="display:flex;gap:.7rem;align-items:center;margin-bottom:.3rem">'
        f'{logo_html(44)}'
        f'<div><b style="font-size:.95rem">Sistem Analisis</b><br>'
        f'<small style="color:{TEAL_LT}">Data Transaksi Toko</small></div></div>',
        unsafe_allow_html=True
    )
    st.markdown("---")

    # Status data
    _status = []
    if st.session_state.get("data_raw") is not None: _status.append("✅ Data_Raw")
    if st.session_state.get("data_clean") is not None: _status.append("✅ Data_Clean")
    if st.session_state.get("apriori_results") is not None: _status.append("✅ Apriori")
    if st.session_state.get("kmeans_results") is not None: _status.append("✅ K-Means")
    st.caption("**Status:** " + (" · ".join(_status) if _status else "belum ada data"))

    # Info tahapan UAS
    st.markdown(
        f'<div style="background:rgba(13,148,136,.15); border-left:3px solid {TEAL};'
        f'padding:.5rem .7rem; border-radius:8px; font-size:.72rem;'
        f'color:{TEAL_LT} !important; margin:.6rem 0; line-height:1.5;">'
        f'<b>TAHAPAN UAS</b><br>'
        f'1. EDA (10%)<br>'
        f'2. Preprocessing (25%)<br>'
        f'3. Feature Engineering<br>'
        f'4. Apriori (20%)<br>'
        f'5. K-Means (20%)<br>'
        f'6. Dashboard BI (15%)<br>'
        f'7. Rekomendasi (10%)'
        f'</div>',
        unsafe_allow_html=True
    )

    st.markdown('<div class="menu-header">📋 Menu Navigasi</div>', unsafe_allow_html=True)

    menu = st.radio(
        "Navigasi",
        MENU_ITEMS,
        index=MENU_ITEMS.index(st.session_state.active_menu),
        label_visibility="collapsed",
        key="menu_radio",
    )
    st.session_state.active_menu = menu

    st.markdown("---")
    st.caption(f"👤 Login: **{st.session_state.username}**")
    if st.button("🚪 Logout", use_container_width=True):
        st.session_state.logged_in = False
        st.session_state.username = None
        st.rerun()

# ============================================================
# MAPPING KOLOM & PREPROCESSING
# ============================================================
COLUMN_MAP = {
    "Transaction_ID":     ["transaction_id", "id_transaksi", "invoice", "no_transaksi"],
    "Tanggal":            ["transaction_date", "tanggal", "date", "tgl"],
    "Pelanggan":          ["customer_id", "pelanggan", "customer", "id_pelanggan"],
    "Wilayah":            ["region", "wilayah", "area"],
    "Tipe_Pelanggan":     ["customer_type", "tipe_pelanggan", "tipe"],
    "Produk_ID":          ["product_id", "id_produk"],
    "Produk":             ["product_name", "produk", "product", "nama_produk"],
    "Kategori":           ["category", "kategori"],
    "Qty":                ["quantity", "qty", "jumlah"],
    "Harga":              ["unit_price", "harga", "price"],
    "Diskon":             ["discount_rate", "diskon", "discount"],
    "Line_Total":         ["line_total", "total", "total_baris"],
    "Metode_Pembayaran":  ["payment_method", "metode_pembayaran", "payment"],
}

def auto_map(df):
    cols = {c.lower().strip(): c for c in df.columns}
    m = {}
    for canon, alts in COLUMN_MAP.items():
        for a in alts:
            if a in cols:
                m[canon] = cols[a]
                break
    return m

def to_canonical(df, mapping):
    inv = {v: k for k, v in mapping.items()}
    return df.rename(columns=inv)

def preprocess(df_in):
    log = []
    df = df_in.copy()
    log.append(("Baris awal", "-", len(df), "-", "-", len(df)))

    if "Tanggal" in df.columns:
        df["Tanggal"] = pd.to_datetime(df["Tanggal"], errors="coerce")
        b = len(df)
        df = df.dropna(subset=["Tanggal"])
        log.append(("Tanggal tidak valid", "Tanggal", b - len(df), "Dihapus", "Wajib untuk RFM", len(df)))

    for c in ["Pelanggan", "Produk", "Kategori", "Metode_Pembayaran", "Wilayah", "Tipe_Pelanggan"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip().str.title()
            df.loc[df[c].isin(["Nan", "None", ""]), c] = np.nan

    for c in ["Qty", "Harga", "Diskon"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    key = [c for c in ["Transaction_ID", "Pelanggan", "Produk", "Qty", "Harga"] if c in df.columns]
    if key:
        b = len(df)
        df = df.dropna(subset=key)
        log.append(("Missing kolom kunci", ", ".join(key), b - len(df), "Dihapus", "Tidak lengkap", len(df)))

    if "Qty" in df.columns:
        b = len(df)
        df = df[df["Qty"] > 0]
        log.append(("Qty ≤ 0", "Qty", b - len(df), "Dihapus", "Tidak valid", len(df)))

    if "Harga" in df.columns:
        b = len(df)
        df = df[df["Harga"] > 0]
        log.append(("Harga ≤ 0", "Harga", b - len(df), "Dihapus", "Tidak valid", len(df)))

    if "Diskon" in df.columns:
        df["Diskon"] = df["Diskon"].fillna(0).clip(0, 1)

    b = len(df)
    df = df.drop_duplicates()
    log.append(("Duplikat", "Semua", b - len(df), "Dihapus", "Baris ganda", len(df)))

    if "Qty" in df.columns and len(df) > 10:
        q1, q3 = df["Qty"].quantile([.25, .75])
        up = q3 + 3 * (q3 - q1)
        b = len(df)
        df = df[df["Qty"] <= up]
        log.append((f"Outlier Qty > {up:.0f}", "Qty", b - len(df), "Dihapus", "Ekstrem >3 IQR", len(df)))

    if {"Qty", "Harga"}.issubset(df.columns):
        disk = df["Diskon"] if "Diskon" in df.columns else 0
        df["Penjualan_Bersih"] = df["Qty"] * df["Harga"] * (1 - disk)

    df = df.reset_index(drop=True)
    log.append(("Baris akhir", "-", len(df), "-", "-", len(df)))
    log_df = pd.DataFrame(log, columns=["Jenis Masalah", "Kolom", "Jumlah", "Tindakan", "Alasan", "Total Baris"])
    return df, log_df

# ============================================================
# FEATURE ENGINEERING HELPERS
# ============================================================
def build_basket(df):
    if not {"Transaction_ID", "Produk"}.issubset(df.columns):
        return None
    b = df.groupby(["Transaction_ID", "Produk"])["Qty"].sum().unstack().fillna(0)
    return (b > 0).astype(int)

def build_basket_summary(df):
    if not {"Transaction_ID", "Produk"}.issubset(df.columns):
        return None
    sizes = df.groupby("Transaction_ID")["Produk"].nunique()
    item_freq = df.groupby("Produk")["Transaction_ID"].nunique().sort_values(ascending=False)
    return {
        "n_trx": len(sizes), "n_prod": df["Produk"].nunique(),
        "avg_size": sizes.mean(), "max_size": sizes.max(), "median_size": sizes.median(),
        "sizes": sizes, "item_freq": item_freq
    }

def build_rfm(df):
    if not {"Pelanggan", "Tanggal", "Transaction_ID"}.issubset(df.columns):
        return None, None
    snap = df["Tanggal"].max() + pd.Timedelta(days=1)
    money = "Penjualan_Bersih" if "Penjualan_Bersih" in df.columns else "Harga"
    rfm = (df.groupby("Pelanggan")
             .agg(Recency=("Tanggal", lambda s: (snap - s.max()).days),
                  Frequency=("Transaction_ID", "nunique"),
                  Monetary=(money, "sum"))
             .reset_index())
    return rfm, snap

def build_rfm_summary(rfm):
    if rfm is None: return None
    return {
        "R_min": rfm["Recency"].min(), "R_max": rfm["Recency"].max(),
        "R_mean": rfm["Recency"].mean(), "R_median": rfm["Recency"].median(),
        "F_min": rfm["Frequency"].min(), "F_max": rfm["Frequency"].max(),
        "F_mean": rfm["Frequency"].mean(), "F_median": rfm["Frequency"].median(),
        "M_min": rfm["Monetary"].min(), "M_max": rfm["Monetary"].max(),
        "M_mean": rfm["Monetary"].mean(), "M_median": rfm["Monetary"].median(),
        "M_total": rfm["Monetary"].sum(),
    }

def prepare_matrix(rfm, log_transform=True):
    X = rfm[["Recency", "Frequency", "Monetary"]].copy()
    if log_transform:
        X["Log_Frequency"] = np.log1p(X["Frequency"])
        X["Log_Monetary"]  = np.log1p(X["Monetary"])
        feats = ["Recency", "Log_Frequency", "Log_Monetary"]
    else:
        feats = ["Recency", "Frequency", "Monetary"]
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X[feats])
    return Xs, feats, scaler, X

def run_apriori(basket, min_sup, min_conf):
    if basket is None or basket.shape[1] < 2:
        return pd.DataFrame(), pd.DataFrame()
    b = basket.astype(bool)
    try:
        freq = apriori(b, min_support=min_sup, use_colnames=True, max_len=4)
    except Exception:
        freq = apriori(b, min_support=min_sup, use_colnames=True, max_len=3)
    if freq.empty:
        return freq, pd.DataFrame()
    try:
        rules = association_rules(freq, metric="confidence", min_threshold=min_conf)
    except Exception:
        return freq, pd.DataFrame()
    if rules.empty:
        return freq.sort_values("support", ascending=False), rules
    rules = rules[(rules["lift"] > 1) & (rules["support"] > 0) & (rules["confidence"] > 0)]
    rules = rules.sort_values(["lift", "confidence", "support"], ascending=False).reset_index(drop=True)
    return freq.sort_values("support", ascending=False), rules

def evaluate_k(rfm, ks=range(2, 9)):
    Xs, _, _, _ = prepare_matrix(rfm)
    rows = []
    for k in ks:
        if k >= len(rfm): break
        km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(Xs)
        try:
            sil = silhouette_score(Xs, km.labels_)
        except Exception:
            sil = np.nan
        rows.append({"k": k, "Inertia": km.inertia_, "Silhouette": sil})
    return pd.DataFrame(rows)

def run_kmeans(rfm, k, log_transform=True):
    Xs, feats, scaler, X = prepare_matrix(rfm, log_transform)
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(Xs)
    rfm_c = rfm.copy()
    rfm_c["Cluster"] = km.labels_
    sil = silhouette_score(Xs, km.labels_) if k > 1 else np.nan
    return rfm_c, km, sil, scaler, feats, Xs, X

def label_clusters(rfm_c):
    prof = (rfm_c.groupby("Cluster")
                 .agg(R=("Recency", "median"), F=("Frequency", "median"), M=("Monetary", "median"))
                 .reset_index())
    prof["Score"] = prof["M"].rank() + prof["F"].rank() + (1 / (prof["R"] + 1)).rank()
    prof = prof.sort_values("Score", ascending=False).reset_index(drop=True)
    base_labels = [
        "Loyal / Bernilai Tinggi", "Potensial", "Baru / Aktif",
        "Berisiko Tidak Kembali", "Perlu Perhatian",
        "Segmen 6", "Segmen 7", "Segmen 8",
    ]
    mp = {int(row["Cluster"]): base_labels[i] if i < len(base_labels) else f"Segmen {i+1}"
          for i, row in prof.iterrows()}
    rfm_c["Label"] = rfm_c["Cluster"].map(mp)
    return rfm_c, prof, mp

# ============================================================
# HALAMAN: BERANDA
# ============================================================
if menu == "🏠 Beranda":
    hero_banner("Sistem Analisis Data Transaksi Toko", "Integrasi Data Mining dan Business Intelligence")

    c1, c2 = st.columns([2, 1], gap="large")
    with c1:
        section("Tujuan Sistem")
        soft_card("""
        Sistem ini membantu manajemen toko memahami pola pembelian pelanggan
        melalui pendekatan <b>Market Basket Analysis</b> dan <b>segmentasi pelanggan
        (RFM + K-Means)</b>. Hasilnya disajikan dalam dashboard Business
        Intelligence yang dilengkapi rekomendasi keputusan bisnis.
        """, tone="teal")

        section("Alur Analisis Sesuai Petunjuk UAS")
        steps = [
            ("TAHAP 1 — EDA (10%)", "Upload dataset + Business Understanding + EDA."),
            ("TAHAP 2 — Preprocessing (25%)", "Audit kualitas & bersihkan Data_Clean."),
            ("TAHAP 3 — Feature Engineering", "Basket Matrix, RFM, dan Scaling."),
            ("TAHAP 4 — Apriori (20%)", "Temukan aturan asosiasi antarproduk."),
            ("TAHAP 5 — K-Means (20%)", "Segmentasi pelanggan berdasarkan RFM."),
            ("TAHAP 6 — Dashboard BI (15%)", "Rangkuman KPI dan tren penjualan."),
            ("TAHAP 7 — Rekomendasi (10%)", "Keputusan berbasis bukti + ekspor hasil."),
        ]
        for i, (t, d) in enumerate(steps, 1):
            st.markdown(f"""
            <div class="step">
                <div class="num">{i}</div>
                <div class="txt"><b>{t}</b> — {d}</div>
            </div>
            """, unsafe_allow_html=True)

    with c2:
        section("Profil Pembuat")
        st.markdown(profile_card_html(), unsafe_allow_html=True)

# ============================================================
# HALAMAN: UPLOAD DATASET (TUGAS 1-2)
# ============================================================
elif menu == "1️⃣ Upload Dataset":
    hero_banner("Upload Dataset", "Unggah Data_Raw — format XLSX, XLS, atau CSV")
    soft_card("""
    <b>TAHAP 1 — Business Understanding & EDA (10%)</b><br>
    Unggah file Excel dengan sheet <code>Data_Raw</code> (18.513 baris) beserta
    <code>Data_Dictionary</code> dan <code>Data_Produk</code>. Sistem otomatis
    memetakan kolom.
    """, tone="blue")

    up = st.file_uploader("Pilih file dataset", type=["xlsx", "xls", "csv"],
        help="Sistem otomatis mendeteksi sheet Data_Raw, Data_Dictionary, Data_Produk.")

    if up is not None:
        st.success(f"✅ File diterima: **{up.name}**")
        try:
            if up.name.lower().endswith((".xlsx", ".xls")):
                xls = pd.ExcelFile(up)
                sheets = {n: pd.read_excel(xls, sheet_name=n) for n in xls.sheet_names}
            else:
                sheets = {"Data_Raw": pd.read_csv(up)}

            st.session_state.datasets = sheets
            raw_key = next((k for k in sheets if "raw" in k.lower()), list(sheets)[0])
            data_raw = sheets[raw_key].copy()
            mapping = auto_map(data_raw)
            data_raw = to_canonical(data_raw, mapping)
            st.session_state.data_raw = data_raw

            section("📋 Sheet Terdeteksi")
            st.dataframe(pd.DataFrame([{"Sheet": k, "Baris": v.shape[0], "Kolom": v.shape[1]} for k, v in sheets.items()]), use_container_width=True, hide_index=True)

            section("🔗 Pemetaan Kolom Otomatis")
            map_df = pd.DataFrame([{"Kolom Canonical": k, "Kolom Dataset": mapping.get(k, "❌ Tidak ditemukan")} for k in COLUMN_MAP])
            st.dataframe(map_df, use_container_width=True, hide_index=True)

            section("👀 Preview Data_Raw (15 baris)")
            st.dataframe(data_raw.head(15), use_container_width=True)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Baris", f"{len(data_raw):,}")
            c2.metric("Kolom", data_raw.shape[1])
            if "Tanggal" in data_raw.columns:
                tgl = pd.to_datetime(data_raw["Tanggal"], errors="coerce").dropna()
                c3.metric("Min", str(tgl.min().date()) if len(tgl) else "-")
                c4.metric("Max", str(tgl.max().date()) if len(tgl) else "-")

            st.session_state.history.append({
                "Waktu": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Dataset": up.name, "Baris": len(data_raw), "Status": "Uploaded"})

        except Exception as e:
            st.error(f"❌ Gagal membaca file: {e}")
    else:
        soft_card("""
        <b>Format:</b> <code>.xlsx</code> / <code>.xls</code> (multi-sheet) atau <code>.csv</code><br><br>
        <b>Kolom minimal:</b> <code>Transaction_ID</code>, <code>Transaction_Date</code>,
        <code>Customer_ID</code>, <code>Product_Name</code>, <code>Quantity</code>,
        <code>Unit_Price</code><br>
        <b>Opsional:</b> <code>Discount_Rate</code>, <code>Category</code>,
        <code>Region</code>, <code>Customer_Type</code>, <code>Payment_Method</code>
        """, tone="blue")

# ============================================================
# HALAMAN: EDA (TUGAS 3)
# ============================================================
elif menu == "1️⃣ Business Understanding & EDA":
    hero_banner("Business Understanding & EDA", "Pahami permasalahan bisnis dan kondisi awal dataset")
    df = st.session_state.data_raw
    if df is None:
        st.warning("⚠️ Belum ada dataset. Buka menu **1️⃣ Upload Dataset**.")
        st.stop()

    section("Tugas 1 — Pertanyaan Bisnis")
    soft_card("""
    <b>1.</b> Produk apa yang sering dibeli bersama? → dasar bundling & tata letak rak.<br>
    <b>2.</b> Segmen pelanggan mana yang paling bernilai? → prioritas retensi.<br>
    <b>3.</b> Kapan penjualan meningkat/menurun? → perencanaan promosi musiman.
    """, tone="teal")

    section("Tugas 2 — Ringkasan Dataset")
    d = df.copy()
    if "Penjualan_Bersih" not in d and {"Qty", "Harga"}.issubset(d.columns):
        disk = d["Diskon"].fillna(0) if "Diskon" in d.columns else 0
        d["Penjualan_Bersih"] = d["Qty"] * d["Harga"] * (1 - disk)

    n_trx = d["Transaction_ID"].nunique() if "Transaction_ID" in d.columns else 0
    n_cust = d["Pelanggan"].nunique() if "Pelanggan" in d.columns else 0
    n_prod = d["Produk"].nunique() if "Produk" in d.columns else 0
    total = d["Penjualan_Bersih"].sum() if "Penjualan_Bersih" in d.columns else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Baris", f"{len(d):,}")
    c2.metric("Transaksi Unik", f"{n_trx:,}")
    c3.metric("Pelanggan", f"{n_cust:,}")
    c4.metric("Produk", f"{n_prod:,}")

    if "Tanggal" in d.columns:
        tgl = pd.to_datetime(d["Tanggal"], errors="coerce").dropna()
        if len(tgl):
            st.caption(f"📅 Rentang: **{tgl.min().date()}** → **{tgl.max().date()}**")
    st.metric("Total Penjualan Sementara", f"Rp {total:,.0f}")

    section("Tugas 3 — Audit Kualitas Data")
    audit = {
        "Missing values": int(d.isna().sum().sum()),
        "Duplikat": int(d.duplicated().sum()),
        "Tanggal tidak valid": int(pd.to_datetime(d["Tanggal"], errors="coerce").isna().sum()) if "Tanggal" in d.columns else 0,
        "Qty ≤ 0": int((d["Qty"] <= 0).sum()) if "Qty" in d.columns else 0,
        "Harga ≤ 0": int((d["Harga"] <= 0).sum()) if "Harga" in d.columns else 0,
    }
    st.dataframe(pd.DataFrame(audit.items(), columns=["Masalah", "Jumlah"]), use_container_width=True, hide_index=True)

    section("Tugas 3 — Visualisasi EDA (minimal 2)")
    tab1, tab2, tab3 = st.tabs(["📈 Tren Bulanan", "🏆 Top 10 Produk", "💰 Distribusi Transaksi"])

    with tab1:
        if "Tanggal" in d.columns and "Penjualan_Bersih" in d.columns:
            tmp = d.dropna(subset=["Tanggal"]).copy()
            tmp["Bulan"] = pd.to_datetime(tmp["Tanggal"]).dt.to_period("M").astype(str)
            tren = tmp.groupby("Bulan")["Penjualan_Bersih"].sum().reset_index()
            fig = px.line(tren, x="Bulan", y="Penjualan_Bersih", markers=True, title="Tren Penjualan Bulanan", color_discrete_sequence=[TEAL])
            fig.update_layout(plot_bgcolor="white", title_font_size=16, height=400)
            fig.update_traces(line=dict(width=3))
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        if "Produk" in d.columns and "Penjualan_Bersih" in d.columns:
            top = (d.groupby("Produk")["Penjualan_Bersih"].sum().nlargest(10).reset_index().sort_values("Penjualan_Bersih"))
            fig = px.bar(top, x="Penjualan_Bersih", y="Produk", orientation="h", title="Top 10 Produk Berdasarkan Penjualan", color="Penjualan_Bersih", color_continuous_scale=["#ccfbf1", TEAL, NAVY])
            fig.update_layout(plot_bgcolor="white", height=420, title_font_size=16, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        if "Transaction_ID" in d.columns and "Penjualan_Bersih" in d.columns:
            nilai = d.groupby("Transaction_ID")["Penjualan_Bersih"].sum()
            fig, ax = plt.subplots(figsize=(10, 3.8))
            sns.histplot(nilai, bins=40, ax=ax, color=TEAL, kde=True)
            ax.set_title("Distribusi Nilai Transaksi", fontsize=13, fontweight="bold", color=NAVY)
            ax.set_xlabel("Rupiah"); ax.set_ylabel("Frekuensi")
            sns.despine(ax=ax)
            st.pyplot(fig)

    section("Tugas 3 — Insight Awal (minimal 3)")
    soft_card("""
    - Dataset memuat baris ganda dan nilai Qty/Harga tidak wajar → audit & preprocessing wajib sebelum analisis.<br>
    - Penjualan tidak merata antarproduk (pola <i>long-tail</i>) — segelintir produk mendominasi revenue.<br>
    - Pola musiman dapat terlihat dari tren bulanan; promosi disesuaikan dengan bulan penjualan rendah.
    """, tone="amber")

# ============================================================
# HALAMAN: PREPROCESSING (TUGAS 4-5)
# ============================================================
elif menu == "2️⃣ Preprocessing Data":
    hero_banner("Preprocessing Data", "Audit dan bersihkan data secara transparan")
    df = st.session_state.data_raw
    if df is None:
        st.warning("⚠️ Belum ada dataset. Buka menu **1️⃣ Upload Dataset**.")
        st.stop()

    section("Tugas 4 — Jalankan Preprocessing")
    if st.button("🧼 Jalankan Preprocessing", use_container_width=False):
        clean, log = preprocess(df)
        st.session_state.data_clean = clean
        st.session_state.log_clean = log
        st.session_state.basket = build_basket(clean)
        st.session_state.basket_summary = build_basket_summary(clean)
        rfm, snap = build_rfm(clean)
        st.session_state.rfm = rfm
        st.session_state.rfm_summary = build_rfm_summary(rfm)
        st.session_state.snapshot = snap
        st.session_state.history.append({
            "Waktu": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Dataset": "Data_Raw → Data_Clean",
            "Baris": len(clean), "Status": "Preprocessed"})
        st.success("✅ Preprocessing selesai. Lanjutkan ke Feature Engineering.")

    clean = st.session_state.data_clean
    if clean is not None:
        c1, c2, c3 = st.columns(3)
        c1.metric("Baris Data_Raw", f"{len(df):,}")
        c2.metric("Baris Data_Clean", f"{len(clean):,}")
        c3.metric("Efisiensi", f"{len(clean)/max(len(df),1)*100:.1f}%")

        section("Tugas 5 — Log Preprocessing (transparan)")
        st.dataframe(st.session_state.log_clean, use_container_width=True, hide_index=True)

        section("Preview Data_Clean (10 baris)")
        st.dataframe(clean.head(10), use_container_width=True)

        st.download_button(
            "⬇️ Unduh Data_Clean (Excel/CSV)",
            to_excel_bytes({"Data_Clean": clean, "Log_Preprocessing": st.session_state.log_clean}),
            file_name="Data_Clean.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    else:
        st.info("Klik tombol **Jalankan Preprocessing** untuk memproses data.")

# ============================================================
# HALAMAN: FEATURE ENGINEERING (TUGAS 6-8)
# ============================================================
elif menu == "3️⃣ Feature Engineering (Basket, RFM, Scaling)":
    hero_banner("Feature Engineering", "Basket Matrix, RFM, dan Scaling Data")

    clean = st.session_state.data_clean
    if clean is None:
        st.warning("⚠️ Jalankan preprocessing dulu di menu **2️⃣ Preprocessing Data**.")
        st.stop()

    tab_basket, tab_rfm, tab_scaling = st.tabs([
        "🧺 Tugas 6 — Market Basket Matrix",
        "📊 Tugas 7 — Analisis RFM",
        "⚖️ Tugas 8 — Scaling Data"
    ])

    # --- TAB 1: BASKET ---
    with tab_basket:
        b = st.session_state.basket_summary
        if b is None:
            b = build_basket_summary(clean)
            st.session_state.basket_summary = b

        section("Ringkasan Basket")
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(kpi("Transaksi", f"{b['n_trx']:,}", "teal"), unsafe_allow_html=True)
        c2.markdown(kpi("Produk Unik", f"{b['n_prod']:,}", "sky"), unsafe_allow_html=True)
        c3.markdown(kpi("Rata-rata Basket", f"{b['avg_size']:.2f}", "lav"), unsafe_allow_html=True)
        c4.markdown(kpi("Basket Maks", f"{b['max_size']}", "peach"), unsafe_allow_html=True)

        section("📐 Matriks Basket (5 transaksi pertama)")
        st.caption("Baris = transaksi, Kolom = produk, Nilai 1 = produk dibeli.")
        basket_df = st.session_state.basket
        if basket_df is not None:
            st.dataframe(basket_df.head(5).iloc[:, :12], use_container_width=True)

        section("Distribusi Ukuran Basket")
        c1, c2 = st.columns([3, 2])
        with c1:
            fig, ax = plt.subplots(figsize=(9, 3.6))
            sns.histplot(b["sizes"], bins=range(1, int(b["max_size"]) + 2), ax=ax, color=TEAL, discrete=True)
            ax.set_title("Jumlah Produk Unik per Transaksi", fontsize=13, fontweight="bold", color=NAVY)
            ax.set_xlabel("Basket Size"); ax.set_ylabel("Frekuensi")
            sns.despine(ax=ax)
            st.pyplot(fig)
        with c2:
            soft_card(f"""
            <b>Interpretasi</b><br>
            Sebagian besar transaksi berisi <b>{int(b['median_size'])}</b> produk unik.
            Peluang cross-selling masih terbuka pada transaksi dengan basket size kecil.
            """, tone="amber")

        section("Top 10 Produk Paling Sering Muncul")
        top_item = b["item_freq"].head(10).reset_index()
        top_item.columns = ["Produk", "Frekuensi Transaksi"]
        fig = px.bar(top_item.sort_values("Frekuensi Transaksi"), x="Frekuensi Transaksi", y="Produk", orientation="h", color="Frekuensi Transaksi", color_continuous_scale=["#ccfbf1", TEAL, NAVY], title="Top 10 Produk Berdasarkan Frekuensi Transaksi")
        fig.update_layout(plot_bgcolor="white", height=400, title_font_size=15, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

        soft_card("""
        <b>Langkah berikutnya:</b> matriks basket ini digunakan sebagai input pada menu
        <b>4️⃣ Apriori / Market Basket Analysis</b> untuk menemukan kombinasi produk.
        """, tone="teal")

    # --- TAB 2: RFM ---
    with tab_rfm:
        rfm = st.session_state.rfm
        if rfm is None:
            st.warning("⚠️ RFM belum dihitung. Jalankan preprocessing.")
        else:
            r = st.session_state.rfm_summary
            section("Tugas 7 — Ringkasan RFM")
            c1, c2, c3 = st.columns(3)
            with c1:
                soft_card(f"""
                <b>Recency (R)</b><br>
                Min: <b>{r['R_min']:.0f}</b> hari · Maks: <b>{r['R_max']:.0f}</b> hari<br>
                Rata-rata: <b>{r['R_mean']:.1f}</b> hari<br>
                Median: <b>{r['R_median']:.0f}</b> hari
                """, tone="teal")
            with c2:
                soft_card(f"""
                <b>Frequency (F)</b><br>
                Min: <b>{r['F_min']:.0f}</b>x · Maks: <b>{r['F_max']:.0f}</b>x<br>
                Rata-rata: <b>{r['F_mean']:.1f}</b>x<br>
                Median: <b>{r['F_median']:.0f}</b>x
                """, tone="blue")
            with c3:
                soft_card(f"""
                <b>Monetary (M)</b><br>
                Min: <b>Rp {r['M_min']:,.0f}</b><br>
                Maks: <b>Rp {r['M_max']:,.0f}</b><br>
                Median: <b>Rp {r['M_median']:,.0f}</b>
                """, tone="amber")

            st.caption(f"Snapshot date: **{st.session_state.snapshot.date()}** · Jumlah pelanggan: **{len(rfm):,}**")

            section("📋 Tabel RFM")
            st.dataframe(rfm.head(20), use_container_width=True, hide_index=True)

            section("Distribusi Fitur RFM")
            t1, t2, t3 = st.tabs(["Recency", "Frequency", "Monetary"])
            with t1:
                fig, ax = plt.subplots(figsize=(9, 3.4))
                sns.histplot(rfm["Recency"], bins=30, ax=ax, color=TEAL, kde=True)
                ax.set_title("Distribusi Recency (hari)", fontsize=13, fontweight="bold", color=NAVY)
                sns.despine(ax=ax); st.pyplot(fig)
            with t2:
                fig, ax = plt.subplots(figsize=(9, 3.4))
                sns.histplot(rfm["Frequency"], bins=30, ax=ax, color="#0ea5e9", kde=True)
                ax.set_title("Distribusi Frequency", fontsize=13, fontweight="bold", color=NAVY)
                sns.despine(ax=ax); st.pyplot(fig)
            with t3:
                fig, ax = plt.subplots(figsize=(9, 3.4))
                sns.histplot(rfm["Monetary"], bins=30, ax=ax, color="#f59e0b", kde=True)
                ax.set_title("Distribusi Monetary (Rupiah)", fontsize=13, fontweight="bold", color=NAVY)
                sns.despine(ax=ax); st.pyplot(fig)

            section("🔍 Deteksi Outlier")
            soft_card("""
            Distribusi Monetary sering <i>right-skewed</i> (ada pelanggan bernilai sangat tinggi).
            Distribusi Frequency juga cenderung miring. Karena itu tahap <b>Scaling Data</b>
            menerapkan <b>log-transform + StandardScaler</b> sebelum K-Means.
            """, tone="rose")

    # --- TAB 3: SCALING ---
    with tab_scaling:
        rfm = st.session_state.rfm
        if rfm is None:
            st.warning("⚠️ RFM belum dihitung. Jalankan preprocessing.")
        else:
            section("Tugas 8 — Konfigurasi Scaling")
            c1, c2 = st.columns(2)
            log_tr = c1.checkbox("Terapkan log1p pada Frequency & Monetary", value=True)
            std_sc = c2.checkbox("Terapkan StandardScaler (Z-score)", value=True)

            if st.button("⚖️ Hitung Scaling", use_container_width=False):
                X = rfm[["Recency", "Frequency", "Monetary"]].copy()
                if log_tr:
                    X["Frequency_t"] = np.log1p(X["Frequency"])
                    X["Monetary_t"]  = np.log1p(X["Monetary"])
                    feats = ["Recency", "Frequency_t", "Monetary_t"]
                else:
                    feats = ["Recency", "Frequency", "Monetary"]

                before = X[feats].agg(["mean", "std", "min", "max"]).T
                if std_sc:
                    scaler = StandardScaler()
                    Xs = pd.DataFrame(scaler.fit_transform(X[feats]), columns=feats)
                    after = Xs.agg(["mean", "std", "min", "max"]).T
                    info = {"feats": feats, "log": log_tr, "std": std_sc, "scaler": scaler, "Xs": Xs, "before": before, "after": after}
                else:
                    info = {"feats": feats, "log": log_tr, "std": False, "scaler": None, "Xs": X[feats], "before": before, "after": before}

                st.session_state.scaling_info = info
                st.session_state.history.append({
                    "Waktu": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Dataset": "Scaling", "Baris": len(X), "Status": f"log={log_tr}, std={std_sc}"})
                st.success("✅ Scaling selesai.")

            info = st.session_state.scaling_info
            if info:
                section("Sebelum Scaling (statistik asli)")
                st.dataframe(info["before"].round(2), use_container_width=True)

                if info["std"]:
                    section("Sesudah Scaling (Z-score: mean≈0, std≈1)")
                    st.dataframe(info["after"].round(3), use_container_width=True)

                section("Visualisasi Efek Scaling")
                c1, c2 = st.columns(2)
                with c1:
                    fig, ax = plt.subplots(figsize=(8, 3.4))
                    sns.boxplot(data=info["Xs"], ax=ax, palette=["#0d9488", "#0ea5e9", "#f59e0b"])
                    ax.set_title("Distribusi Setelah Scaling", fontsize=13, fontweight="bold", color=NAVY)
                    ax.tick_params(axis="x", rotation=15)
                    sns.despine(ax=ax); st.pyplot(fig)
                with c2:
                    soft_card(f"""
                    <b>Fitur yang digunakan:</b> {', '.join(info['feats'])}<br>
                    <b>Log-transform:</b> {'✅ Ya' if info['log'] else '❌ Tidak'}<br>
                    <b>StandardScaler:</b> {'✅ Ya' if info['std'] else '❌ Tidak'}<br><br>
                    Matriks hasil scaling menjadi input untuk <b>K-Means Clustering</b>.
                    """, tone="teal")

                section("Preview Matriks Hasil Scaling (10 baris)")
                st.dataframe(info["Xs"].head(10).round(3), use_container_width=True, hide_index=True)

            soft_card("""
            <b>Mengapa scaling penting?</b><br>
            Recency (hari), Frequency (kali), dan Monetary (Rupiah) memiliki skala sangat berbeda.
            Tanpa scaling, jarak Euclidean didominasi Monetary. Log-transform meredam outlier,
            StandardScaler menyetarakan skala.
            """, tone="amber")

# ============================================================
# HALAMAN: APRIORI (TUGAS 9-12)
# ============================================================
elif menu == "4️⃣ Apriori / Market Basket Analysis":
    hero_banner("Apriori / Market Basket Analysis", "Temukan produk yang sering dibeli bersama")
    basket = st.session_state.basket
    if basket is None:
        st.warning("⚠️ Jalankan preprocessing dulu di menu **2️⃣ Preprocessing Data**.")
        st.stop()

    soft_card("""
    <b>TAHAP 4 — Apriori (20%)</b><br>
    Input: matriks transaksi × produk dari <b>Tugas 6</b>. Tentukan minimum support
    dan confidence, lalu temukan aturan asosiasi terbaik dengan kriteria pemeringkatan
    (lift → confidence → support).
    """, tone="blue")

    section("Tugas 10 — Parameter Apriori")
    c1, c2 = st.columns(2)
    min_sup = c1.slider("Minimum Support (%)", 0.5, 10.0, 1.0, 0.5) / 100
    min_conf = c2.slider("Minimum Confidence (%)", 10, 80, 20, 5) / 100
    st.caption(f"Matriks: **{basket.shape[0]:,}** transaksi × **{basket.shape[1]:,}** produk.")

    if st.button("🔍 Jalankan Apriori", use_container_width=False):
        freq, rules = run_apriori(basket, min_sup, min_conf)
        st.session_state.apriori_results = {"freq": freq, "rules": rules, "params": (min_sup, min_conf)}
        st.session_state.history.append({
            "Waktu": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Dataset": "Apriori", "Baris": len(rules), "Status": f"sup={min_sup}, conf={min_conf}"})

    res = st.session_state.apriori_results
    if res is not None:
        freq, rules = res["freq"], res["rules"]

        section("Frequent Itemsets")
        if freq.empty:
            st.warning("Tidak ada itemset memenuhi support. Turunkan support.")
        else:
            st.dataframe(freq.head(20), use_container_width=True, hide_index=True)

        section("Tugas 11 — Association Rules (Top 10)")
        if rules.empty:
            st.warning("Tidak ada aturan memenuhi confidence. Turunkan confidence.")
        else:
            show = rules.head(10).copy()
            show["antecedents"] = show["antecedents"].apply(lambda s: ", ".join(sorted(s)))
            show["consequents"] = show["consequents"].apply(lambda s: ", ".join(sorted(s)))
            st.dataframe(show[["antecedents", "consequents", "support", "confidence", "lift"]], use_container_width=True, hide_index=True)

            section("Tugas 12 — Visualisasi Aturan (Support × Confidence × Lift)")
            fig = px.scatter(show, x="support", y="confidence", size="lift", color="lift", hover_data=["antecedents", "consequents"], color_continuous_scale=["#ccfbf1", TEAL, NAVY], title="Sebaran Aturan Asosiasi")
            fig.update_layout(plot_bgcolor="white", height=440, title_font_size=16)
            st.plotly_chart(fig, use_container_width=True)

            section("🕸️ Network Graph Hubungan Produk")
            G = nx.Graph()
            for _, r in rules.head(20).iterrows():
                for a in r["antecedents"]:
                    for c in r["consequents"]:
                        G.add_edge(a, c, weight=r["lift"])
            if len(G.nodes) > 0:
                pos = nx.spring_layout(G, seed=42, k=0.9)
                ex, ey = [], []
                for e in G.edges():
                    x0, y0 = pos[e[0]]; x1, y1 = pos[e[1]]
                    ex += [x0, x1, None]; ey += [y0, y1, None]
                nx_ = [pos[n][0] for n in G.nodes()]
                ny_ = [pos[n][1] for n in G.nodes()]
                fn = go.Figure()
                fn.add_trace(go.Scatter(x=ex, y=ey, mode="lines", line=dict(width=1.2, color="#94a3b8"), hoverinfo="none"))
                fn.add_trace(go.Scatter(x=nx_, y=ny_, mode="markers+text", text=list(G.nodes()), textposition="top center", marker=dict(size=18, color=TEAL, line=dict(width=2, color=NAVY)), hoverinfo="text"))
                fn.update_layout(showlegend=False, height=480, title="Hubungan Produk (Top 20 Rules)", plot_bgcolor="white")
                st.plotly_chart(fn, use_container_width=True)

            section("💡 Tugas 12 — 3 Rekomendasi Cross-Selling / Bundling")
            for i, (_, r) in enumerate(rules.head(3).iterrows(), 1):
                a = ", ".join(sorted(r["antecedents"]))
                c = ", ".join(sorted(r["consequents"]))
                soft_card(f"""
                <b>{i}. {a} → {c}</b><br>
                Support <b>{r['support']*100:.2f}%</b> · Confidence <b>{r['confidence']*100:.1f}%</b> · Lift <b>{r['lift']:.2f}</b><br>
                <span style="color:{GRAY}">Aksi: buat paket bundling, letakkan berdekatan di rak, atau tampilkan sebagai "sering dibeli bersama".</span>
                """, tone="teal")

# ============================================================
# HALAMAN: K-MEANS (TUGAS 13-16)
# ============================================================
elif menu == "5️⃣ K-Means / Segmentasi Pelanggan":
    hero_banner("K-Means / Segmentasi Pelanggan", "Segmentasi pelanggan berdasarkan RFM")
    rfm = st.session_state.rfm
    if rfm is None:
        st.warning("⚠️ Jalankan preprocessing dulu di menu **2️⃣ Preprocessing Data**.")
        st.stop()

    soft_card("""
    <b>TAHAP 5 — K-Means (20%)</b><br>
    Input: fitur RFM (Tugas 7) yang telah di-scaling (Tugas 8). Uji beberapa kandidat k
    (2–8), evaluasi dengan Elbow + Silhouette, profil klaster, dan strategi pemasaran per segmen.
    """, tone="blue")

    st.caption(f"Pelanggan: **{len(rfm):,}** · Snapshot: **{st.session_state.snapshot.date()}**")

    section("Tugas 14 — Evaluasi K (2–8)")
    if st.button("🎯 Evaluasi K", use_container_width=False):
        st.session_state.kmeans_eval = evaluate_k(rfm, range(2, 9))

    eval_df = st.session_state.kmeans_eval
    best_k = st.session_state.kmeans_k
    if eval_df is not None and len(eval_df):
        st.dataframe(eval_df, use_container_width=True, hide_index=True)
        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots(figsize=(6, 3.4))
            ax.plot(eval_df["k"], eval_df["Inertia"], "o-", color=TEAL, linewidth=2.5, markersize=8)
            ax.set_title("Elbow Method", fontweight="bold", color=NAVY)
            ax.set_xlabel("k"); ax.set_ylabel("Inertia")
            ax.grid(alpha=.3); sns.despine(ax=ax); st.pyplot(fig)
        with c2:
            fig, ax = plt.subplots(figsize=(6, 3.4))
            ax.plot(eval_df["k"], eval_df["Silhouette"], "o-", color=NAVY, linewidth=2.5, markersize=8)
            ax.set_title("Silhouette Score", fontweight="bold", color=NAVY)
            ax.set_xlabel("k"); ax.set_ylabel("Score")
            ax.grid(alpha=.3); sns.despine(ax=ax); st.pyplot(fig)

        valid = eval_df.dropna(subset=["Silhouette"])
        best_k = int(valid.loc[valid["Silhouette"].idxmax(), "k"]) if len(valid) else 4
        st.info(f"💡 Rekomendasi K = **{best_k}** (Silhouette tertinggi).")

    section("Tugas 14 — Pilih K (Manual)")
    k_manual = st.slider("Jumlah klaster (K)", 2, 8, int(best_k))
    if st.button("▶️ Jalankan K-Means", use_container_width=False):
        try:
            rfm_c, km, sil, scaler, feats, Xs, X = run_kmeans(rfm, k_manual)
            rfm_c, prof, mp = label_clusters(rfm_c)
            st.session_state.kmeans_results = {"rfm_c": rfm_c, "prof": prof, "sil": sil, "k": k_manual}
            st.session_state.kmeans_k = k_manual
            st.session_state.history.append({
                "Waktu": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Dataset": "K-Means", "Baris": len(rfm_c), "Status": f"K={k_manual}, Sil={sil:.3f}"})
            st.success(f"✅ K-Means selesai. K = {k_manual}, Silhouette = {sil:.3f}")
        except Exception as e:
            st.error(f"❌ Gagal: {e}")

    res = st.session_state.kmeans_results
    if res is not None:
        rfm_c, prof, sil, k = res["rfm_c"], res["prof"], res["sil"], res["k"]
        st.metric("Silhouette Score", f"{sil:.3f}")

        section(f"Tugas 15 — Profil Klaster (K = {k})")
        prof_show = prof.copy()
        prof_show["Jumlah"] = rfm_c.groupby("Cluster").size().values
        prof_show["% Pelanggan"] = (prof_show["Jumlah"] / len(rfm_c) * 100).round(1)
        prof_show["Revenue (Rp)"] = (rfm_c.groupby("Cluster")["Monetary"].sum().round(0).values)
        prof_show["% Revenue"] = (rfm_c.groupby("Cluster")["Monetary"].sum() / rfm_c["Monetary"].sum() * 100).round(1).values
        prof_show = prof_show.rename(columns={"R": "Recency (median)", "F": "Frequency (median)", "M": "Monetary (median)", "Score": "Skor"})
        st.dataframe(prof_show, use_container_width=True, hide_index=True)

        section("Tugas 15 — Scatter Plot Klaster (sampel 1.000)")
        samp = rfm_c.sample(min(1000, len(rfm_c)), random_state=42)
        c1, c2 = st.columns(2)
        with c1:
            fig = px.scatter(samp, x="Frequency", y="Monetary", color="Label", hover_data=["Pelanggan", "Recency"], title="Frequency vs Monetary", color_discrete_sequence=PALETTE)
            fig.update_layout(plot_bgcolor="white", height=420, title_font_size=15)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.scatter(samp, x="Recency", y="Monetary", color="Label", hover_data=["Pelanggan", "Frequency"], title="Recency vs Monetary", color_discrete_sequence=PALETTE)
            fig.update_layout(plot_bgcolor="white", height=420, title_font_size=15)
            st.plotly_chart(fig, use_container_width=True)

        section("Tugas 16 — Strategi Pemasaran per Segmen")
        for lbl in rfm_c["Label"].unique():
            sub = rfm_c[rfm_c["Label"] == lbl]
            soft_card(f"""
            <b>{lbl}</b> — {len(sub)} pelanggan · Median Recency <b>{sub['Recency'].median():.0f} hari</b> · Frequency <b>{sub['Frequency'].median():.0f}x</b> · Monetary <b>Rp {sub['Monetary'].median():,.0f}</b><br>
            <span style="color:{GRAY}; font-size:.88rem;">Strategi: sesuaikan penawaran dengan karakter segmen.</span>
            """, tone="blue")

# ============================================================
# HALAMAN: DASHBOARD BI (TUGAS 17-19)
# ============================================================
elif menu == "6️⃣ Dashboard Business Intelligence":
    hero_banner("Business Intelligence Dashboard", "Rangkuman KPI dan tren penjualan")
    clean = st.session_state.data_clean
    if clean is None:
        st.warning("⚠️ Jalankan preprocessing dulu di menu **2️⃣ Preprocessing Data**.")
        st.stop()

    soft_card("""
    <b>TAHAP 6 — Dashboard BI (15%)</b><br>
    KPI total penjualan, transaksi, pelanggan, rata-rata basket, dan produk.
    Filter tanggal & kategori untuk eksplorasi. Hasil Apriori dan profil K-Means
    ditampilkan di panel bawah.
    """, tone="blue")

    with st.sidebar:
        st.markdown("### 🔎 Filter Dashboard")
        df_f = clean.copy()
        if "Tanggal" in df_f.columns:
            tgl = pd.to_datetime(df_f["Tanggal"], errors="coerce")
            dmin, dmax = tgl.min().date(), tgl.max().date()
            rng = st.date_input("Rentang Tanggal", [dmin, dmax], min_value=dmin, max_value=dmax)
            if isinstance(rng, (tuple, list)) and len(rng) == 2:
                mask = (tgl.dt.date >= rng[0]) & (tgl.dt.date <= rng[1])
                df_f = df_f[mask]
        if "Kategori" in df_f.columns:
            kats = sorted(df_f["Kategori"].dropna().unique())
            sel = st.multiselect("Kategori", kats, default=kats)
            if sel:
                df_f = df_f[df_f["Kategori"].isin(sel)]

    tp = df_f["Penjualan_Bersih"].sum() if "Penjualan_Bersih" in df_f else 0
    nt = df_f["Transaction_ID"].nunique() if "Transaction_ID" in df_f else 0
    nc = df_f["Pelanggan"].nunique() if "Pelanggan" in df_f else 0
    bs = (df_f.groupby("Transaction_ID")["Produk"].nunique().mean() if {"Transaction_ID", "Produk"}.issubset(df_f.columns) else 0)
    np_ = df_f["Produk"].nunique() if "Produk" in df_f else 0

    tones = ["teal", "sky", "lav", "peach", "rose"]
    labels = ["Total Penjualan", "Transaksi", "Pelanggan", "Rata-rata Basket", "Produk"]
    vals = [f"Rp {tp:,.0f}", f"{nt:,}", f"{nc:,}", f"{bs:.2f}", f"{np_:,}"]
    c1, c2, c3, c4, c5 = st.columns(5)
    for col, lbl, val, tn in zip([c1, c2, c3, c4, c5], labels, vals, tones):
        col.markdown(kpi(lbl, val, tn), unsafe_allow_html=True)

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        if "Tanggal" in df_f.columns:
            tmp = df_f.dropna(subset=["Tanggal"]).copy()
            tmp["Bulan"] = pd.to_datetime(tmp["Tanggal"]).dt.to_period("M").astype(str)
            tren = tmp.groupby("Bulan")["Penjualan_Bersih"].sum().reset_index()
            fig = px.line(tren, x="Bulan", y="Penjualan_Bersih", markers=True, title="Tren Penjualan Bulanan", color_discrete_sequence=[TEAL])
            fig.update_traces(line=dict(width=3))
            fig.update_layout(plot_bgcolor="white", height=400, title_font_size=16)
            st.plotly_chart(fig, use_container_width=True)
    with c2:
        if "Produk" in df_f.columns:
            top = (df_f.groupby("Produk")["Penjualan_Bersih"].sum().nlargest(10).reset_index().sort_values("Penjualan_Bersih"))
            fig = px.bar(top, x="Penjualan_Bersih", y="Produk", orientation="h", title="Top 10 Produk", color="Penjualan_Bersih", color_continuous_scale=["#ccfbf1", TEAL, NAVY])
            fig.update_layout(plot_bgcolor="white", height=400, title_font_size=16, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        section("Ringkasan Apriori")
        res = st.session_state.apriori_results
        if res and len(res["rules"]):
            r = res["rules"].head(5).copy()
            r["Rule"] = r.apply(lambda x: f"{', '.join(sorted(x['antecedents']))} → {', '.join(sorted(x['consequents']))}", axis=1)
            st.dataframe(r[["Rule", "support", "confidence", "lift"]], use_container_width=True, hide_index=True)
        else:
            st.info("Belum dijalankan. Buka menu **4️⃣ Apriori**.")
    with c2:
        section("Profil Klaster")
        res_km = st.session_state.kmeans_results
        if res_km:
            st.dataframe(res_km["prof"], use_container_width=True, hide_index=True)
        else:
            st.info("Belum dijalankan. Buka menu **5️⃣ K-Means**.")

    st.caption("⚠️ Filter tanggal/kategori hanya memengaruhi visualisasi dashboard, bukan model.")

# ============================================================
# HALAMAN: REKOMENDASI (TUGAS 20-22)
# ============================================================
elif menu == "7️⃣ Rekomendasi Keputusan":
    hero_banner("Rekomendasi Keputusan Bisnis", "Keputusan berbasis bukti data")

    soft_card("""
    <b>TAHAP 7 — Rekomendasi (10%)</b><br>
    Minimal 3 rekomendasi prioritas: penempatan/bundling produk, retensi pelanggan,
    dan tindakan pemasaran. Setiap rekomendasi memuat bukti data, tindakan, KPI,
    estimasi, asumsi, dan cara evaluasi.
    """, tone="blue")

    section("Tugas 20 — Tabel Rekomendasi Prioritas")
    reco = pd.DataFrame([
        {"Rekomendasi": "Bundling produk sering dibeli bersama", "Bukti": "Aturan asosiasi lift > 1", "Tindakan": "Buat paket bundling, tata rak berdekatan", "Prioritas": "Tinggi", "KPI": "Uplift basket_size"},
        {"Rekomendasi": "Program loyalitas segmen bernilai tinggi", "Bukti": "Profil cluster RFM (Monetary tinggi)", "Tindakan": "Reward non-diskon, akses awal produk baru", "Prioritas": "Tinggi", "KPI": "Retention rate 90 hari"},
        {"Rekomendasi": "Kampanye win-back segmen berisiko", "Bukti": "Recency tinggi + Monetary rendah", "Tindakan": "Voucher terbatas waktu", "Prioritas": "Menengah", "KPI": "Reactivation rate"},
    ])
    st.dataframe(reco, use_container_width=True, hide_index=True)

    section("Tugas 21 & 22 — Catatan Metodologi")
    soft_card("""
    <b>Catatan Metodologi:</b><br>
    - <b>Temuan faktual</b> = support, confidence, lift, profil RFM.<br>
    - <b>Estimasi</b> = dampak uplift/perkiraan manfaat (berbasis asumsi).<br>
    - <b>Dampak aktual</b> = hanya dapat diklaim setelah implementasi & pengukuran.<br><br>
    <b>Cara evaluasi dampak:</b> bandingkan KPI sebelum vs sesudah implementasi
    (A/B test, pre-post analysis, atau cohort analysis).
    """, tone="amber")

# ============================================================
# HALAMAN: UNDUH HASIL
# ============================================================
elif menu == "8️⃣ Unduh Hasil":
    hero_banner("Unduh Hasil Analisis", "Semua hasil dalam format Excel/CSV")
    clean = st.session_state.data_clean
    res_a = st.session_state.apriori_results
    res_k = st.session_state.kmeans_results

    section("1. Data")
    c1, c2 = st.columns(2)
    with c1:
        if clean is not None:
            st.download_button("⬇️ Data_Clean", to_excel_bytes({"Data_Clean": clean, "Log_Preprocessing": st.session_state.log_clean}), file_name="Data_Clean.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        else:
            st.info("Data_Clean belum tersedia.")
    with c2:
        if st.session_state.rfm is not None:
            st.download_button("⬇️ Data_RFM.csv", st.session_state.rfm.to_csv(index=False).encode(), file_name="Data_RFM.csv", mime="text/csv")

    section("2. Apriori")
    if res_a and len(res_a["rules"]):
        rules = res_a["rules"].copy()
        rules["antecedents"] = rules["antecedents"].apply(lambda s: ", ".join(sorted(s)))
        rules["consequents"] = rules["consequents"].apply(lambda s: ", ".join(sorted(s)))
        c1, c2 = st.columns(2)
        with c1:
            st.download_button("⬇️ Hasil_Apriori", to_excel_bytes({"Frequent_Itemsets": res_a["freq"], "Association_Rules": rules}), file_name="Hasil_Apriori.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        with c2:
            st.download_button("⬇️ Association_Rules.csv", rules.to_csv(index=False).encode(), file_name="Association_Rules.csv", mime="text/csv")
    else:
        st.info("Hasil Apriori belum tersedia.")

    section("3. K-Means")
    if res_k:
        rfm_c = res_k["rfm_c"].copy()
        c1, c2 = st.columns(2)
        with c1:
            st.download_button("⬇️ Hasil_Clustering", to_excel_bytes({"Pelanggan_Cluster": rfm_c, "Profil_Cluster": res_k["prof"], "Evaluasi_K": st.session_state.get("kmeans_eval", pd.DataFrame())}), file_name="Hasil_Clustering.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        with c2:
            st.download_button("⬇️ Data_Pelanggan_Cluster.csv", rfm_c.to_csv(index=False).encode(), file_name="Data_Pelanggan_Cluster.csv", mime="text/csv")
    else:
        st.info("Hasil K-Means belum tersedia.")

# ============================================================
# HALAMAN: RIWAYAT
# ============================================================
elif menu == "8️⃣ Riwayat Analisis":
    hero_banner("Riwayat Analisis", "Catatan aktivitas analisis pada sesi ini")
    if not st.session_state.history:
        st.info("Belum ada riwayat.")
    else:
        st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True, hide_index=True)