"""
Kişisel Finans Öneri Asistanı — Flask Web Arayüzü
"""
import os, sys, json
from pathlib import Path
import numpy as np
import pandas as pd
import skfuzzy as fuzz
from flask import Flask, render_template, jsonify, request
from werkzeug.utils import secure_filename
from statement_pdf_pipeline import process_pdf
from transaction_classifier import classify_dataframe, parse_amount

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OZET_PATH  = os.path.join(BASE_DIR, "ozet_hesaplar.csv")
BANKA_PATH = os.path.join(BASE_DIR, "banka_hareketleri.csv")
EXTRACTED_PATH = os.path.join(BASE_DIR, "extracted_transactions.csv")
PROFILE_PATH = os.path.join(BASE_DIR, "user_profile.json")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

CATEGORY_OPTIONS = [
    "Gelir",
    "Banka Ücreti",
    "Gıda",
    "Konut/Fatura",
    "Ulaştırma",
    "Eğitim",
    "Sağlık",
    "Borç/Kredi/Kart",
    "Alışveriş",
    "İsteğe Bağlı",
    "Diğer",
]
EXPENSE_TYPE_OPTIONS = ["Gelir", "Zorunlu", "Kısılabilir", "İsteğe Bağlı", "Belirsiz"]
CLASSIFICATION_COLUMNS = {
    "Kategori",
    "Gider Tipi",
    "Sınıflandırma Güveni",
    "Sınıflandırma Yöntemi",
    "Sınıflandırma Kuralı",
}

# ── Üyelik fonksiyon yardımcısı ──────────────────────────────
x = np.linspace(0, 1, 1000)
y = np.linspace(0, 1, 1000)

def interp(arr, mf, val):
    return float(fuzz.interp_membership(arr, mf, val))

def mf_to_points(x_arr, mf_arr, steps=200):
    """Grafik için (x,y) nokta listesi döndür."""
    idx = np.round(np.linspace(0, len(x_arr)-1, steps)).astype(int)
    return [{"x": round(float(x_arr[i]),4), "y": round(float(mf_arr[i]),4)} for i in idx]

def clamp01(value):
    return max(0.0, min(1.0, float(value)))

def load_bank_data():
    if os.path.exists(EXTRACTED_PATH):
        return pd.read_csv(EXTRACTED_PATH), "PDF OCR"
    return pd.read_csv(BANKA_PATH), "Demo CSV"

def load_user_profile():
    if not os.path.exists(PROFILE_PATH):
        return {}
    try:
        with open(PROFILE_PATH, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}

def save_user_profile(profile):
    with open(PROFILE_PATH, "w", encoding="utf-8") as handle:
        json.dump(profile, handle, ensure_ascii=False, indent=2)

def calculate_risk_score(payload):
    risk_keys = ("loss_reaction", "volatility_comfort", "growth_preference")
    scores = [float(payload.get(key, 0.5)) for key in risk_keys]
    return clamp01(sum(scores) / len(scores))

def calculate_investment_horizon(payload):
    return clamp01(float(payload.get("investment_horizon", 0.5)))

def _profile_defaults(profile):
    answers = profile.get("answers", {})
    risk = float(profile.get("risk_tolerance", 0.5))
    horizon = float(profile.get("investment_horizon", answers.get("investment_horizon", risk)))
    return {
        "risk_tolerance": clamp01(risk),
        "investment_horizon": clamp01(horizon),
        "answers": {
            "loss_reaction": clamp01(answers.get("loss_reaction", risk)),
            "investment_horizon": clamp01(answers.get("investment_horizon", horizon)),
            "volatility_comfort": clamp01(answers.get("volatility_comfort", risk)),
            "growth_preference": clamp01(answers.get("growth_preference", risk)),
        },
    }

def fallback_risk_score(fallback_ozet):
    if "Risk Tolerans (0-1)" in fallback_ozet:
        return clamp01(float(fallback_ozet["Risk Tolerans (0-1)"].iloc[0]))
    return 0.5

def calculate_financial_metrics(banka):
    gelirler = banka[banka["Tutar"] > 0]
    giderler = banka[banka["Tutar"] < 0].copy()
    giderler["Abs"] = giderler["Tutar"].abs()

    toplam_gelir = float(gelirler["Tutar"].sum())
    toplam_gider = float(giderler["Abs"].sum())
    zorunlu_gider = float(giderler[giderler["Gider Tipi"] == "Zorunlu"]["Abs"].sum())
    borc_gider = float(giderler[giderler["Kategori"] == "Borç/Kredi/Kart"]["Abs"].sum())
    esnek_gider_tipleri = ["Kısılabilir", "İsteğe Bağlı"]
    esnek_gider = float(giderler[giderler["Gider Tipi"].isin(esnek_gider_tipleri)]["Abs"].sum())

    esneklik = clamp01((toplam_gelir - zorunlu_gider) / toplam_gelir) if toplam_gelir > 0 else 0.0
    borc_yuku = clamp01(borc_gider / toplam_gelir) if toplam_gelir > 0 else 0.0
    tasarruf_orani = clamp01((toplam_gelir - toplam_gider) / toplam_gelir) if toplam_gelir > 0 else 0.0
    kisilabilir_oran = clamp01(esnek_gider / toplam_gider) if toplam_gider > 0 else 0.0

    aylik_gelir = gelirler.groupby("Ay")["Tutar"].sum()
    aylik_gider = giderler.groupby("Ay")["Abs"].sum()
    aylik_zorunlu = giderler[giderler["Gider Tipi"] == "Zorunlu"].groupby("Ay")["Abs"].sum()
    aylik_tasarruf = aylik_gelir.subtract(aylik_gider, fill_value=0)

    if len(aylik_gelir) >= 2 and float(aylik_gelir.mean()) > 0:
        duzenlilik = 1 - min(1.0, float(aylik_gelir.std()) / float(aylik_gelir.mean()))
    elif len(aylik_gelir) == 1:
        duzenlilik = 0.65
    else:
        duzenlilik = 0.0

    son_bakiye = None
    if "Bakiye" in banka.columns:
        parsed_balance = banka["Bakiye"].apply(lambda value: np.nan if pd.isna(value) else parse_amount(value))
        dated_balance = pd.DataFrame({"Tarih": banka["Tarih"], "Bakiye": parsed_balance}).dropna()
        if not dated_balance.empty:
            son_bakiye = float(dated_balance.sort_values("Tarih")["Bakiye"].iloc[-1])

    aylik_zorunlu_ort = float(aylik_zorunlu.mean()) if not aylik_zorunlu.empty else 0.0
    if son_bakiye is not None and aylik_zorunlu_ort > 0:
        tampon_ay = max(0.0, son_bakiye / aylik_zorunlu_ort)
        acil_tampon = clamp01(tampon_ay / 3.0)
    else:
        tampon_ay = None
        acil_tampon = 0.5

    return {
        "gelirler": gelirler,
        "giderler": giderler,
        "toplam_gelir": toplam_gelir,
        "toplam_gider": toplam_gider,
        "zorunlu_gider": zorunlu_gider,
        "borc_gider": borc_gider,
        "esnek_gider": esnek_gider,
        "esneklik": esneklik,
        "borc_yuku": borc_yuku,
        "tasarruf_orani": tasarruf_orani,
        "kisilabilir_oran": kisilabilir_oran,
        "duzenlilik": clamp01(duzenlilik),
        "acil_tampon": acil_tampon,
        "tampon_ay": tampon_ay,
        "aylik_gelir": aylik_gelir,
        "aylik_gider": aylik_gider,
        "aylik_tasarruf": aylik_tasarruf,
    }

def compute_fuzzy_inputs(banka, fallback_ozet):
    metrics = calculate_financial_metrics(banka)
    raw_profile = load_user_profile()
    profile = _profile_defaults(raw_profile)
    if raw_profile:
        risk = profile["risk_tolerance"]
        vade = profile["investment_horizon"]
    else:
        risk = fallback_risk_score(fallback_ozet)
        vade = 0.5
        profile = _profile_defaults({
            "risk_tolerance": risk,
            "investment_horizon": vade,
            "answers": {
                "loss_reaction": risk,
                "investment_horizon": vade,
                "volatility_comfort": risk,
                "growth_preference": risk,
            },
        })
    return {
        "esneklik": clamp01(metrics["esneklik"]),
        "duzenlilik": clamp01(metrics["duzenlilik"]),
        "risk": clamp01(risk),
        "vade": clamp01(vade),
        "tampon": clamp01(metrics["acil_tampon"]),
        "borc": clamp01(metrics["borc_yuku"]),
    }, metrics, profile

def prepare_bank_data(banka):
    banka = banka.copy()
    banka.columns = banka.columns.str.strip()
    if not CLASSIFICATION_COLUMNS.issubset(set(banka.columns)):
        banka = classify_dataframe(banka)
    banka["Tutar"] = banka["Tutar"].apply(parse_amount)
    banka["Tarih"] = pd.to_datetime(banka["Tarih"], errors="coerce", dayfirst=True)
    banka["Ay"] = banka["Tarih"].dt.to_period("M")
    return banka

def _raw_transaction_rows():
    if not os.path.exists(EXTRACTED_PATH):
        return []
    df = pd.read_csv(EXTRACTED_PATH)
    df.columns = df.columns.str.strip()
    rows = []
    for idx, row in df.iterrows():
        rows.append({
            "id": int(idx),
            "tarih": str(row.get("Tarih", "")),
            "aciklama": str(row.get("Açıklama", "")),
            "tutar": parse_amount(row.get("Tutar", 0)),
            "bakiye": None if pd.isna(row.get("Bakiye", None)) else parse_amount(row.get("Bakiye", 0)),
            "kategori": str(row.get("Kategori", "Diğer")),
            "gider_tipi": str(row.get("Gider Tipi", "Belirsiz")),
            "guven": float(row.get("Sınıflandırma Güveni", 0) or 0),
            "yontem": str(row.get("Sınıflandırma Yöntemi", "")),
            "kural": str(row.get("Sınıflandırma Kuralı", "")),
            "ham_ocr": str(row.get("Ham OCR Satırı", "")),
        })
    return rows

def run_fuzzy():
    # ── CSV Oku ──────────────────────────────────────────────
    ozet  = pd.read_csv(OZET_PATH)
    ozet.columns  = ozet.columns.str.strip()
    banka, banka_kaynak = load_bank_data()

    # ── Banka Analizi ───────────────────────────────────────
    banka = prepare_bank_data(banka)
    inputs, metrics, profile = compute_fuzzy_inputs(banka, ozet)
    esn = inputs["esneklik"]
    duz = inputs["duzenlilik"]
    risk = inputs["risk"]
    vade = inputs["vade"]
    tampon = inputs["tampon"]
    borc = inputs["borc"]

    giderler = metrics["giderler"]
    aylik_gelir = metrics["aylik_gelir"]
    aylik_gider = metrics["aylik_gider"]
    aylik_tasarruf = metrics["aylik_tasarruf"]

    kat_gider = giderler.groupby("Kategori")["Abs"].sum().sort_values(ascending=False)
    dusuk_guven = banka[banka["Sınıflandırma Güveni"] < 0.50].copy()

    # Aylık trend için ay bazında gelir/gider
    aylar = sorted(set(list(aylik_gelir.index) + list(aylik_gider.index)))
    monthly_labels = [str(a) for a in aylar]
    monthly_gelir  = [round(float(aylik_gelir.get(a, 0)), 2) for a in aylar]
    monthly_gider  = [round(float(aylik_gider.get(a, 0)), 2) for a in aylar]
    monthly_tas    = [round(float(aylik_tasarruf.get(a, 0)), 2) for a in aylar]

    # ── Üyelik Fonksiyonları ─────────────────────────────────
    mf = {
        "duz": {
            "Düzensiz": fuzz.trapmf(x,[0.0,0.0,0.30,0.50]),
            "Orta"    : fuzz.trapmf(x,[0.30,0.45,0.55,0.70]),
            "Düzenli" : fuzz.trapmf(x,[0.50,0.70,1.0,1.0]),
        },
        "esn": {
            "Düşük"  : fuzz.trapmf(x,[0.0,0.0,0.25,0.45]),
            "Orta"   : fuzz.trapmf(x,[0.30,0.45,0.55,0.70]),
            "Yüksek" : fuzz.trapmf(x,[0.55,0.75,1.0,1.0]),
        },
        "risk": {
            "Düşük"  : fuzz.trapmf(x,[0.0,0.0,0.25,0.45]),
            "Orta"   : fuzz.trapmf(x,[0.30,0.45,0.55,0.70]),
            "Yüksek" : fuzz.trapmf(x,[0.55,0.75,1.0,1.0]),
        },
        "vade": {
            "Kısa" : fuzz.trapmf(x,[0.0,0.0,0.25,0.45]),
            "Orta" : fuzz.trapmf(x,[0.30,0.45,0.55,0.70]),
            "Uzun" : fuzz.trapmf(x,[0.55,0.75,1.0,1.0]),
        },
        "tampon": {
            "Zayıf" : fuzz.trapmf(x,[0.0,0.0,0.25,0.45]),
            "Orta"  : fuzz.trapmf(x,[0.30,0.45,0.60,0.78]),
            "Güçlü" : fuzz.trapmf(x,[0.62,0.80,1.0,1.0]),
        },
        "borc": {
            "Düşük"  : fuzz.trapmf(x,[0.0,0.0,0.15,0.30]),
            "Orta"   : fuzz.trapmf(x,[0.20,0.35,0.48,0.65]),
            "Yüksek" : fuzz.trapmf(x,[0.50,0.70,1.0,1.0]),
        },
    }
    mf_out = {
        "Güvenilir": fuzz.trapmf(y,[0.0,0.0,0.20,0.33]),
        "Dengeli"  : fuzz.trapmf(y,[0.25,0.38,0.52,0.65]),
        "Agresif"  : fuzz.trapmf(y,[0.55,0.67,1.0,1.0]),
    }

    # ── Bulanıklaştırma ──────────────────────────────────────
    deg_duz  = {k: interp(x, v, duz)  for k,v in mf["duz"].items()}
    deg_esn  = {k: interp(x, v, esn)  for k,v in mf["esn"].items()}
    deg_risk = {k: interp(x, v, risk) for k,v in mf["risk"].items()}
    deg_vade = {k: interp(x, v, vade) for k,v in mf["vade"].items()}
    deg_tampon = {k: interp(x, v, tampon) for k,v in mf["tampon"].items()}
    deg_borc = {k: interp(x, v, borc) for k,v in mf["borc"].items()}

    # ── Kurallar (Mamdani, min) ──────────────────────────────
    kurallar = [
        (deg_risk["Düşük"], "Güvenilir",
         "K1", "Risk = Düşük", "Güvenilir"),
        (min(deg_tampon["Zayıf"], deg_borc["Yüksek"]), "Güvenilir",
         "K2", "Tampon=Zayıf & Borç=Yüksek", "Güvenilir"),
        (min(deg_tampon["Zayıf"], deg_vade["Kısa"]), "Güvenilir",
         "K3", "Tampon=Zayıf & Vade=Kısa", "Güvenilir"),
        (min(deg_borc["Yüksek"], deg_esn["Düşük"]), "Güvenilir",
         "K4", "Borç=Yüksek & Esn=Düşük", "Güvenilir"),
        (min(deg_risk["Orta"], deg_esn["Düşük"], deg_duz["Düzensiz"]), "Güvenilir",
         "K5", "Risk=Orta & Esn=Düşük & Gel=Düzensiz", "Güvenilir"),
        (min(deg_risk["Yüksek"], deg_vade["Kısa"]), "Dengeli",
         "K6", "Risk=Yüksek & Vade=Kısa", "Dengeli"),
        (min(deg_risk["Orta"], deg_esn["Orta"], deg_tampon["Orta"]), "Dengeli",
         "K7", "Risk=Orta & Esn=Orta & Tampon=Orta", "Dengeli"),
        (min(deg_risk["Orta"], deg_vade["Uzun"], deg_borc["Düşük"]), "Dengeli",
         "K8", "Risk=Orta & Vade=Uzun & Borç=Düşük", "Dengeli"),
        (min(deg_esn["Yüksek"], deg_duz["Düzenli"], deg_tampon["Güçlü"]), "Dengeli",
         "K9", "Esn=Yüksek & Gel=Düzenli & Tampon=Güçlü", "Dengeli"),
        (min(deg_risk["Yüksek"], deg_esn["Orta"], deg_vade["Orta"], deg_borc["Düşük"]), "Dengeli",
         "K10", "Risk=Yüksek & Esn=Orta & Vade=Orta & Borç=Düşük", "Dengeli"),
        (min(deg_risk["Orta"], deg_esn["Yüksek"], deg_vade["Uzun"], deg_tampon["Güçlü"]), "Dengeli",
         "K11", "Risk=Orta & Esn=Yüksek & Vade=Uzun & Tampon=Güçlü", "Dengeli"),
        (min(deg_risk["Yüksek"], deg_vade["Uzun"], deg_esn["Yüksek"], deg_tampon["Güçlü"], deg_borc["Düşük"]), "Agresif",
         "K12", "Risk=Yüksek & Vade=Uzun & Esn=Yüksek & Tampon=Güçlü & Borç=Düşük", "Agresif"),
        (min(deg_risk["Yüksek"], deg_vade["Uzun"], deg_esn["Yüksek"], deg_duz["Düzenli"]), "Agresif",
         "K13", "Risk=Yüksek & Vade=Uzun & Esn=Yüksek & Gel=Düzenli", "Agresif"),
        (min(deg_risk["Yüksek"], deg_vade["Orta"], deg_esn["Yüksek"], deg_tampon["Güçlü"], deg_borc["Düşük"]), "Agresif",
         "K14", "Risk=Yüksek & Vade=Orta & Esn=Yüksek & Tampon=Güçlü & Borç=Düşük", "Agresif"),
        (min(deg_risk["Yüksek"], deg_vade["Uzun"], deg_borc["Orta"], deg_tampon["Güçlü"]), "Dengeli",
         "K15", "Risk=Yüksek & Vade=Uzun & Borç=Orta & Tampon=Güçlü", "Dengeli"),
    ]

    rules_out = []
    for akt, label, kid, cond, sonuc in kurallar:
        rules_out.append({
            "id": kid, "condition": cond,
            "conclusion": sonuc, "activation": round(akt, 4),
            "active": akt > 0.001
        })

    # ── Agregasyon & Defuzz ───────────────────────────────────
    max_g = max(a for a,l,*_ in kurallar if l=="Güvenilir")
    max_d = max(a for a,l,*_ in kurallar if l=="Dengeli")
    max_a = max(a for a,l,*_ in kurallar if l=="Agresif")

    agr_g = np.fmin(max_g, mf_out["Güvenilir"])
    agr_d = np.fmin(max_d, mf_out["Dengeli"])
    agr_a = np.fmin(max_a, mf_out["Agresif"])
    agregasyon = np.fmax(agr_g, np.fmax(agr_d, agr_a))

    defuzz_val = fuzz.defuzz(y, agregasyon, "centroid") if np.sum(agregasyon)>0 else 0.165

    if defuzz_val <= 0.33:   profil = "Güvenilir"
    elif defuzz_val <= 0.66: profil = "Dengeli"
    else:                    profil = "Agresif"

    portfolyolar = {
        "Güvenilir": {
            "desc": "Sermaye koruma odaklı, düşük riskli portföy",
            "color": "#22d3ee",
            "items": {"Devlet Tahvili/Bono":45,"Mevduat/TL Mevduat":25,
                      "Altın":15,"Hisse Senedi (BIST-30)":10,"Döviz (USD/EUR)":5}
        },
        "Dengeli": {
            "desc": "Risk-getiri dengesi gözeten, orta vadeli portföy",
            "color": "#a78bfa",
            "items": {"Hisse Senedi (BIST-100)":30,"Devlet Tahvili/Bono":25,
                      "Altın":20,"Döviz (USD/EUR)":15,"Kripto (BTC/ETH)":10}
        },
        "Agresif": {
            "desc": "Yüksek getiri hedefli, yüksek risk toleranslı portföy",
            "color": "#fb923c",
            "items": {"Hisse Senedi (BIST/Yurt dışı)":45,"Kripto (BTC/ETH)":25,
                      "Emtia (Petrol/Altın)":15,"Döviz":10,"Girişim/Fon":5}
        },
    }

    # MF grafik verileri
    mf_charts = {
        "gelir_duzenliligi": {
            "value": round(duz,4),
            "curves": {k: mf_to_points(x, v) for k,v in mf["duz"].items()},
            "degrees": {k: round(v,4) for k,v in deg_duz.items()}
        },
        "esneklik_orani": {
            "value": round(esn,4),
            "curves": {k: mf_to_points(x, v) for k,v in mf["esn"].items()},
            "degrees": {k: round(v,4) for k,v in deg_esn.items()}
        },
        "risk_toleransi": {
            "value": round(risk,4),
            "curves": {k: mf_to_points(x, v) for k,v in mf["risk"].items()},
            "degrees": {k: round(v,4) for k,v in deg_risk.items()}
        },
        "yatirim_vadesi": {
            "value": round(vade,4),
            "curves": {k: mf_to_points(x, v) for k,v in mf["vade"].items()},
            "degrees": {k: round(v,4) for k,v in deg_vade.items()}
        },
        "acil_durum_tamponu": {
            "value": round(tampon,4),
            "curves": {k: mf_to_points(x, v) for k,v in mf["tampon"].items()},
            "degrees": {k: round(v,4) for k,v in deg_tampon.items()}
        },
        "borc_yuku": {
            "value": round(borc,4),
            "curves": {k: mf_to_points(x, v) for k,v in mf["borc"].items()},
            "degrees": {k: round(v,4) for k,v in deg_borc.items()}
        },
    }

    output_chart = {
        "curves": {k: mf_to_points(y, v) for k,v in mf_out.items()},
        "aggregation": mf_to_points(y, agregasyon),
        "defuzz": round(float(defuzz_val),4),
        "activations": {"Güvenilir": round(max_g,4), "Dengeli": round(max_d,4), "Agresif": round(max_a,4)}
    }

    return {
        "inputs": {
            "esneklik": round(esn,4),
            "duzenlilik": round(duz,4),
            "risk": round(risk,4),
            "vade": round(vade,4),
            "tampon": round(tampon,4),
            "borc": round(borc,4),
        },
        "profile": profile,
        "options": {"categories": CATEGORY_OPTIONS, "expense_types": EXPENSE_TYPE_OPTIONS},
        "mf_charts": mf_charts,
        "output_chart": output_chart,
        "rules": rules_out,
        "defuzz": round(float(defuzz_val),4),
        "profil": profil,
        "portfolyo": portfolyolar[profil],
        "banka": {
            "kaynak": banka_kaynak,
            "toplam_gelir": round(float(metrics["toplam_gelir"]),2),
            "toplam_gider": round(float(metrics["toplam_gider"]),2),
            "aylik_gelir_ort": round(float(aylik_gelir.mean()),2),
            "aylik_gider_ort": round(float(aylik_gider.mean()),2),
            "aylik_tasarruf_ort": round(float(aylik_tasarruf.mean()),2),
            "istege_bagli": round(float(metrics["esnek_gider"]),2),
            "borc_yuku_orani": round(float(metrics["borc_yuku"]),4),
            "acil_durum_tamponu": round(float(metrics["acil_tampon"]),4),
            "tampon_ay": None if metrics["tampon_ay"] is None else round(float(metrics["tampon_ay"]),2),
            "tasarruf_orani": round(float(metrics["tasarruf_orani"]),4),
            "kisilabilir_oran": round(float(metrics["kisilabilir_oran"]),4),
            "kategoriler": {k: round(float(v),2) for k,v in kat_gider.items()},
            "siniflandirma": {
                "toplam": int(len(banka)),
                "dusuk_guven_adedi": int(len(dusuk_guven)),
                "yontemler": {k: int(v) for k, v in banka["Sınıflandırma Yöntemi"].value_counts().items()},
                "dusuk_guven_ornekleri": [
                    {
                        "tarih": "" if pd.isna(row["Tarih"]) else str(row["Tarih"].date()),
                        "aciklama": str(row.get("Açıklama", "")),
                        "tutar": round(float(row.get("Tutar", 0)), 2),
                        "kategori": str(row.get("Kategori", "")),
                        "gider_tipi": str(row.get("Gider Tipi", "")),
                        "guven": round(float(row.get("Sınıflandırma Güveni", 0)), 2),
                    }
                    for _, row in dusuk_guven.head(10).iterrows()
                ],
            },
            "monthly": {
                "labels": monthly_labels,
                "gelir": monthly_gelir,
                "gider": monthly_gider,
                "tasarruf": monthly_tas
            },
            "transactions": _raw_transaction_rows()[:100]
        }
    }

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/data")
def api_data():
    data = run_fuzzy()
    return jsonify(data)

@app.route("/api/transactions", methods=["GET", "POST"])
def transactions():
    if request.method == "GET":
        return jsonify({
            "transactions": _raw_transaction_rows(),
            "categories": CATEGORY_OPTIONS,
            "expense_types": EXPENSE_TYPE_OPTIONS,
        })

    if not os.path.exists(EXTRACTED_PATH):
        return jsonify({"error": "Düzenlenecek OCR işlemi bulunamadı."}), 404

    payload = request.get_json(silent=True) or {}
    updates = payload.get("transactions", [])
    update_by_id = {int(item["id"]): item for item in updates if "id" in item}
    df = pd.read_csv(EXTRACTED_PATH)
    df.columns = df.columns.str.strip()
    if "Tutar" in df.columns:
        df["Tutar"] = df["Tutar"].apply(parse_amount)
    if "Bakiye" in df.columns:
        df["Bakiye"] = df["Bakiye"].apply(lambda value: None if pd.isna(value) else parse_amount(value))

    for idx, update in update_by_id.items():
        if idx < 0 or idx >= len(df):
            continue
        if "tarih" in update:
            df.at[idx, "Tarih"] = str(update["tarih"])
        if "aciklama" in update:
            df.at[idx, "Açıklama"] = str(update["aciklama"])
        if "tutar" in update:
            df.at[idx, "Tutar"] = parse_amount(update["tutar"])
        if "kategori" in update and update["kategori"] in CATEGORY_OPTIONS:
            df.at[idx, "Kategori"] = update["kategori"]
        if "gider_tipi" in update and update["gider_tipi"] in EXPENSE_TYPE_OPTIONS:
            df.at[idx, "Gider Tipi"] = update["gider_tipi"]
        df.at[idx, "Sınıflandırma Yöntemi"] = "manual"
        df.at[idx, "Sınıflandırma Kuralı"] = "user_correction"
        df.at[idx, "Sınıflandırma Güveni"] = 1.0

    df.to_csv(EXTRACTED_PATH, index=False, encoding="utf-8-sig")
    return jsonify({"ok": True, "transactions": len(df)})

@app.route("/api/upload-statement", methods=["POST"])
def upload_statement():
    uploaded = request.files.get("statement")
    if uploaded is None or not uploaded.filename:
        return jsonify({"error": "PDF dosyası seçilmedi."}), 400
    if not uploaded.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Sadece PDF dosyası yüklenebilir."}), 400

    filename = secure_filename(uploaded.filename) or "statement.pdf"
    pdf_path = os.path.join(UPLOAD_DIR, filename)
    uploaded.save(pdf_path)

    rows = process_pdf(Path(pdf_path), Path(EXTRACTED_PATH))
    return jsonify({
        "ok": True,
        "transactions": len(rows),
        "output": os.path.basename(EXTRACTED_PATH),
    })

@app.route("/api/risk-profile", methods=["GET", "POST"])
def risk_profile():
    if request.method == "GET":
        profile = load_user_profile()
        profile = _profile_defaults(profile)
        return jsonify({
            "risk_tolerance": round(float(profile.get("risk_tolerance", 0.5)), 4),
            "investment_horizon": round(float(profile.get("investment_horizon", 0.5)), 4),
            "answers": profile.get("answers", {}),
        })

    payload = request.get_json(silent=True) or {}
    score = calculate_risk_score(payload)
    horizon = calculate_investment_horizon(payload)
    profile = {
        "risk_tolerance": round(score, 4),
        "investment_horizon": round(horizon, 4),
        "answers": {k: clamp01(v) for k, v in payload.items()},
    }
    save_user_profile(profile)
    return jsonify(profile)

if __name__ == "__main__":
    app.run(debug=True, port=5050)
