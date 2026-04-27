"""
=============================================================
 KİŞİSEL FİNANS ÖNERİ ASİSTANI - BULANIK MANTIK SİSTEMİ
 Yöntem : Mamdani Çıkarımı + Centroid Durulaştırma
 Kütüphane: scikit-fuzzy (skfuzzy)
=============================================================
Kurulum:
    pip install scikit-fuzzy numpy pandas matplotlib
=============================================================
"""

import os
import sys
import numpy as np
import pandas as pd

# skfuzzy kontrolü
try:
    import skfuzzy as fuzz
except ImportError:
    print("HATA: scikit-fuzzy kurulu değil.")
    print("Lütfen şu komutu çalıştırın: pip install scikit-fuzzy")
    sys.exit(1)

# ─────────────────────────────────────────
# 1. CSV DOSYALARINI OKU
# ─────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OZET_PATH = os.path.join(BASE_DIR, "ozet_hesaplar.csv")
BANKA_PATH = os.path.join(BASE_DIR, "banka_hareketleri.csv")

print("\n" + "="*60)
print("  KİŞİSEL FİNANS ÖNERİ ASİSTANI — BULANIK MANTIK")
print("="*60)

# Dosya varlık kontrolü
for path, name in [(OZET_PATH, "ozet_hesaplar.csv"), (BANKA_PATH, "banka_hareketleri.csv")]:
    if not os.path.exists(path):
        print(f"\n⚠  UYARI: '{name}' bulunamadı → {path}")
        print("   Lütfen dosyayı aynı klasöre yerleştirin ve tekrar çalıştırın.")
        sys.exit(1)

ozet = pd.read_csv(OZET_PATH)
banka = pd.read_csv(BANKA_PATH)

# Sütun adlarını temizle (baştaki/sondaki boşluklar)
ozet.columns = ozet.columns.str.strip()
banka.columns = banka.columns.str.strip()

# Girdi değerlerini al
esneklik_orani   = float(ozet["Esneklik Oranı (0-1)"].iloc[0])
duzenlilik_skoru = float(ozet["Gelir Düzenlilik Skoru (0-1)"].iloc[0])
risk_toleransi   = float(ozet["Risk Tolerans (0-1)"].iloc[0])

print(f"\n📂 Özet veriler okundu:")
print(f"   Esneklik Oranı      : {esneklik_orani:.4f}")
print(f"   Gelir Düzenliliği   : {duzenlilik_skoru:.4f}")
print(f"   Risk Toleransı      : {risk_toleransi:.4f}")

# ─────────────────────────────────────────
# 2. BANKA HAREKETLERİ ANALİZİ
# ─────────────────────────────────────────
print("\n" + "-"*60)
print("  📊 BANKA HAREKETLERİ ANALİZİ")
print("-"*60)

banka["Tutar"] = pd.to_numeric(banka["Tutar"], errors="coerce")
banka["Tarih"] = pd.to_datetime(banka["Tarih"], errors="coerce")
banka["Ay"]    = banka["Tarih"].dt.to_period("M")

gelirler  = banka[banka["Tutar"] > 0]
giderler  = banka[banka["Tutar"] < 0].copy()
giderler["Tutar_Abs"] = giderler["Tutar"].abs()

# Aylık ortalamalar
aylik_gelir  = gelirler.groupby("Ay")["Tutar"].sum()
aylik_gider  = giderler.groupby("Ay")["Tutar_Abs"].sum()
aylik_tasarruf = aylik_gelir.subtract(aylik_gider, fill_value=0)

print(f"   Aylık Ort. Gelir      : {aylik_gelir.mean():>10.2f} TL")
print(f"   Aylık Ort. Gider      : {aylik_gider.mean():>10.2f} TL")
print(f"   Aylık Ort. Tasarruf   : {aylik_tasarruf.mean():>10.2f} TL")

# Kategoriye göre gider dağılımı
kat_gider = (giderler.groupby("Kategori")["Tutar_Abs"]
             .sum()
             .sort_values(ascending=False))
print(f"\n   Kategoriye Göre Toplam Giderler:")
for kat, tutar in kat_gider.items():
    print(f"     {kat:<28}: {tutar:>10.2f} TL")

# İsteğe bağlı gider → tasarruf potansiyeli
istege_bagli = kat_gider.get("İsteğe Bağlı", 0)
print(f"\n   💡 Tasarruf Potansiyeli (İsteğe Bağlı Giderler): {istege_bagli:.2f} TL/yıl")
print(f"      (~{istege_bagli/12:.2f} TL/ay)")

# ─────────────────────────────────────────
# 3. ÜYELİK FONKSİYONLARI TANIMLA
# ─────────────────────────────────────────
x = np.linspace(0, 1, 1000)   # Evrensel küme (0-1)

# ── Gelir Düzenliliği ──────────────────────
# Düzensiz: [0, 0, 0.3, 0.5]  → trapezoid
# Orta    : [0.3, 0.5, 0.5, 0.7]  → trapezoid
# Düzenli : [0.5, 0.7, 1, 1]  → trapezoid
mf_duz_duzensiz = fuzz.trapmf(x, [0.0, 0.0, 0.30, 0.50])
mf_duz_orta     = fuzz.trapmf(x, [0.30, 0.45, 0.55, 0.70])
mf_duz_duzenli  = fuzz.trapmf(x, [0.50, 0.70, 1.0, 1.0])

# ── Esneklik Oranı ─────────────────────────
mf_esn_dusuk  = fuzz.trapmf(x, [0.0, 0.0, 0.25, 0.45])
mf_esn_orta   = fuzz.trapmf(x, [0.30, 0.45, 0.55, 0.70])
mf_esn_yuksek = fuzz.trapmf(x, [0.55, 0.75, 1.0, 1.0])

# ── Risk Toleransı ─────────────────────────
mf_risk_dusuk  = fuzz.trapmf(x, [0.0, 0.0, 0.25, 0.45])
mf_risk_orta   = fuzz.trapmf(x, [0.30, 0.45, 0.55, 0.70])
mf_risk_yuksek = fuzz.trapmf(x, [0.55, 0.75, 1.0, 1.0])

# ── Çıktı: Yatırım Profili ─────────────────
y = np.linspace(0, 1, 1000)
mf_guvenilir = fuzz.trapmf(y, [0.0,  0.0,  0.20, 0.33])
mf_dengeli   = fuzz.trapmf(y, [0.25, 0.38, 0.52, 0.65])
mf_agresif   = fuzz.trapmf(y, [0.55, 0.67, 1.0,  1.0])

# ─────────────────────────────────────────
# 4. BULANIKLAŞTIRMA
# ─────────────────────────────────────────
def interpol(x_arr, mf, val):
    """Verilen değer için üyelik derecesini döndür."""
    return float(fuzz.interp_membership(x_arr, mf, val))

# Gelir Düzenliliği
deg_duz = {
    "Düzensiz": interpol(x, mf_duz_duzensiz, duzenlilik_skoru),
    "Orta"    : interpol(x, mf_duz_orta,     duzenlilik_skoru),
    "Düzenli" : interpol(x, mf_duz_duzenli,  duzenlilik_skoru),
}
# Esneklik Oranı
deg_esn = {
    "Düşük"  : interpol(x, mf_esn_dusuk,  esneklik_orani),
    "Orta"   : interpol(x, mf_esn_orta,   esneklik_orani),
    "Yüksek" : interpol(x, mf_esn_yuksek, esneklik_orani),
}
# Risk Toleransı
deg_risk = {
    "Düşük"  : interpol(x, mf_risk_dusuk,  risk_toleransi),
    "Orta"   : interpol(x, mf_risk_orta,   risk_toleransi),
    "Yüksek" : interpol(x, mf_risk_yuksek, risk_toleransi),
}

print("\n" + "-"*60)
print("  🔍 BULANIKLAŞTIRMA SONUÇLARI")
print("-"*60)
print(f"  Gelir Düzenliliği ({duzenlilik_skoru:.4f}):")
for k, v in deg_duz.items():
    print(f"    μ({k:<10}) = {v:.4f}")
print(f"  Esneklik Oranı ({esneklik_orani:.4f}):")
for k, v in deg_esn.items():
    print(f"    μ({k:<10}) = {v:.4f}")
print(f"  Risk Toleransı ({risk_toleransi:.4f}):")
for k, v in deg_risk.items():
    print(f"    μ({k:<10}) = {v:.4f}")

# ─────────────────────────────────────────
# 5. MAMDANI KURALLARI (9 Kural, min operatörü)
# ─────────────────────────────────────────
# Kural Tablosu:
# K1 : Risk=Düşük                                          → Güvenilir
# K2 : Risk=Orta  & Esneklik=Düşük  & Gelir=Düzensiz      → Güvenilir
# K3 : Risk=Orta  & Esneklik=Düşük  & Gelir=Orta          → Güvenilir
# K4 : Risk=Orta  & Esneklik=Orta   & Gelir=Düzensiz      → Güvenilir
# K5 : Risk=Orta  & Esneklik=Orta   & Gelir=Orta          → Dengeli
# K6 : Risk=Orta  & Esneklik=Orta   & Gelir=Düzenli       → Dengeli
# K7 : Risk=Orta  & Esneklik=Yüksek & Gelir=Orta          → Dengeli
# K8 : Risk=Orta  & Esneklik=Yüksek & Gelir=Düzenli       → Agresif
# K9 : Risk=Yüksek                                         → Agresif

kurallar = [
    # (aktivasyon_kuvveti,  çıktı_kümesi_label)
    (deg_risk["Düşük"],
     "guvenilir", "K1: Risk=Düşük → Güvenilir"),

    (min(deg_risk["Orta"], deg_esn["Düşük"],  deg_duz["Düzensiz"]),
     "guvenilir", "K2: Risk=Orta & Esn=Düşük & Gel=Düzensiz → Güvenilir"),

    (min(deg_risk["Orta"], deg_esn["Düşük"],  deg_duz["Orta"]),
     "guvenilir", "K3: Risk=Orta & Esn=Düşük & Gel=Orta     → Güvenilir"),

    (min(deg_risk["Orta"], deg_esn["Orta"],   deg_duz["Düzensiz"]),
     "guvenilir", "K4: Risk=Orta & Esn=Orta  & Gel=Düzensiz → Güvenilir"),

    (min(deg_risk["Orta"], deg_esn["Orta"],   deg_duz["Orta"]),
     "dengeli",   "K5: Risk=Orta & Esn=Orta  & Gel=Orta     → Dengeli"),

    (min(deg_risk["Orta"], deg_esn["Orta"],   deg_duz["Düzenli"]),
     "dengeli",   "K6: Risk=Orta & Esn=Orta  & Gel=Düzenli  → Dengeli"),

    (min(deg_risk["Orta"], deg_esn["Yüksek"], deg_duz["Orta"]),
     "dengeli",   "K7: Risk=Orta & Esn=Yüksek& Gel=Orta     → Dengeli"),

    (min(deg_risk["Orta"], deg_esn["Yüksek"], deg_duz["Düzenli"]),
     "agresif",   "K8: Risk=Orta & Esn=Yüksek& Gel=Düzenli  → Agresif"),

    (deg_risk["Yüksek"],
     "agresif",   "K9: Risk=Yüksek → Agresif"),
]

print("\n" + "-"*60)
print("  ⚙  TETİKLENEN KURALLAR (aktivasyon kuvveti > 0)")
print("-"*60)
aktif_kural = False
for aktivasyon, _, aciklama in kurallar:
    if aktivasyon > 0:
        print(f"  [{aktivasyon:.4f}]  {aciklama}")
        aktif_kural = True
if not aktif_kural:
    print("  ⚠  Hiçbir kural tetiklenmedi! Girdi değerlerini kontrol edin.")

# ─────────────────────────────────────────
# 6. ÇIKTI KÜMELERİNİ BİRLEŞTİR (max-min agregasyon)
# ─────────────────────────────────────────
# Her çıktı etiketi için maksimum aktivasyonu bul
max_guvenilir = max(a for a, lbl, _ in kurallar if lbl == "guvenilir")
max_dengeli   = max(a for a, lbl, _ in kurallar if lbl == "dengeli")
max_agresif   = max(a for a, lbl, _ in kurallar if lbl == "agresif")

# Kırpılmış üyelik fonksiyonları (min operatörü ile kırp)
agr_guvenilir = np.fmin(max_guvenilir, mf_guvenilir)
agr_dengeli   = np.fmin(max_dengeli,   mf_dengeli)
agr_agresif   = np.fmin(max_agresif,   mf_agresif)

# Tüm çıktıları birleştir (max)
agregasyon = np.fmax(agr_guvenilir, np.fmax(agr_dengeli, agr_agresif))

# ─────────────────────────────────────────
# 7. DURULAŞTIRMA — CENTROİD
# ─────────────────────────────────────────
if np.sum(agregasyon) == 0:
    print("\n⚠  UYARI: Agregasyon boş! Varsayılan 'Güvenilir' atanıyor.")
    defuzz_val = 0.165
else:
    defuzz_val = fuzz.defuzz(y, agregasyon, "centroid")

print("\n" + "-"*60)
print("  📐 DURULAŞTIRMA")
print("-"*60)
print(f"  Centroid Çıktısı (0-1) : {defuzz_val:.4f}")
print(f"  Aktivasyon Kuvvetleri  :")
print(f"    Güvenilir = {max_guvenilir:.4f}")
print(f"    Dengeli   = {max_dengeli:.4f}")
print(f"    Agresif   = {max_agresif:.4f}")

# ─────────────────────────────────────────
# 8. PROFİL BELİRLE VE PORTFÖY DAĞILIMI
# ─────────────────────────────────────────
portfolyolar = {
    "Güvenilir": {
        "Açıklama": "Sermaye koruma odaklı, düşük riskli portföy",
        "Dağılım": {
            "Devlet Tahvili / Bono"  : 45,
            "Mevduat / TL Mevduat"   : 25,
            "Altın"                  : 15,
            "Hisse Senedi (BIST-30)" : 10,
            "Döviz (USD/EUR)"        :  5,
        }
    },
    "Dengeli": {
        "Açıklama": "Risk-getiri dengesi gözeten, orta vadeli portföy",
        "Dağılım": {
            "Hisse Senedi (BIST-100)": 30,
            "Devlet Tahvili / Bono"  : 25,
            "Altın"                  : 20,
            "Döviz (USD/EUR)"        : 15,
            "Kripto (BTC/ETH)"       : 10,
        }
    },
    "Agresif": {
        "Açıklama": "Yüksek getiri hedefli, risk toleransı yüksek portföy",
        "Dağılım": {
            "Hisse Senedi (BIST/Yurt dışı)": 45,
            "Kripto (BTC/ETH/Altcoin)"     : 25,
            "Emtia (Petrol/Altın)"          : 15,
            "Döviz (USD/EUR/Egzotik)"       : 10,
            "Girişim Sermayesi / Fon"        :  5,
        }
    },
}

if defuzz_val <= 0.33:
    profil = "Güvenilir"
elif defuzz_val <= 0.66:
    profil = "Dengeli"
else:
    profil = "Agresif"

profil_bilgi = portfolyolar[profil]

print("\n" + "="*60)
print("  🏆  ÖNERİLEN YATIRIM PROFİLİ")
print("="*60)
print(f"\n  ➤  {profil.upper()}")
print(f"     {profil_bilgi['Açıklama']}")
print(f"     Bulanık çıktı değeri: {defuzz_val:.4f}")
print(f"\n  📈 Portföy Dağılım Önerisi:")
print(f"  {'Varlık Sınıfı':<35} {'Ağırlık':>8}")
print(f"  {'-'*44}")
for varlik, oran in profil_bilgi["Dağılım"].items():
    bar = "█" * (oran // 3)
    print(f"  {varlik:<35} %{oran:>3}  {bar}")

# ─────────────────────────────────────────
# 9. ÖZET RAPOR
# ─────────────────────────────────────────
toplam_gelir = banka[banka["Tutar"] > 0]["Tutar"].sum()
toplam_gider = banka[banka["Tutar"] < 0]["Tutar"].abs().sum()
net_tasarruf = toplam_gelir - toplam_gider

print("\n" + "="*60)
print("  📋  ÖZET FİNANSAL RAPOR")
print("="*60)
print(f"  Toplam Gelir (12 ay)    : {toplam_gelir:>12.2f} TL")
print(f"  Toplam Gider (12 ay)    : {toplam_gider:>12.2f} TL")
print(f"  Net Tasarruf            : {net_tasarruf:>12.2f} TL")
print(f"  Esneklik Oranı          : %{esneklik_orani*100:.1f}")
print(f"  Gelir Düzenliliği       : %{duzenlilik_skoru*100:.1f}")
print(f"  Risk Toleransı          : %{risk_toleransi*100:.1f}")
print(f"\n  → Önerilen Profil       : {profil}")
print("="*60)
print("\n✅ Analiz tamamlandı.\n")
