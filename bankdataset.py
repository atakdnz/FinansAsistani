import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# -------------------------------
# 1. PARAMETRELER (12 AY, GERÇEKÇİ)
# -------------------------------
np.random.seed(42)          # Aynı sonuçları almak için (isterseniz 42'yi değiştirin)
random.seed(42)

ay_sayisi = 12              # ⬅️ 12 AYLIK VERİ
baslangic_tarihi = datetime(2024, 1, 1)

# Gelir bilgileri
aylik_maas = 45000
maas_std = 800              # Gelir düzensizliği için standart sapma (oynama miktarı)

# Ek gelirler (bazı aylarda)
ek_gelir_aylari = [3, 6, 9] # Mart, Haziran, Eylül aylarında ek gelir
ek_gelir_tutari = 5000

# -------------------------------
# 2. HARCAMA KATEGORİLERİ (Detaylı)
# -------------------------------
harcama_kategorileri = {
    "Konut": {
        "aciklama": ["Ev Kredisi", "Kira", "Elektrik Faturası", "Su Faturası", "Doğalgaz", "Site Aidatı", "İnternet"],
        "ortalama": [-12000, -1500, -900, -350, -1200, -500, -300],
        "olasilik": [0.20, 0.10, 0.20, 0.15, 0.15, 0.10, 0.10],
        "mevsimsel": {"Kis": ["Doğalgaz"], "Yaz": []}  # Kış aylarında doğalgaz artar
    },
    "Gıda": {
        "aciklama": ["Migros", "Şok Market", "Bim", "A101", "Manav", "Kasap", "Restoran"],
        "ortalama": [-2500, -1500, -1200, -1300, -400, -800, -1200],
        "olasilik": [0.20, 0.15, 0.15, 0.15, 0.10, 0.10, 0.15]
    },
    "Ulaştırma": {
        "aciklama": ["Shell Akaryakıt", "OPET", "Uber", "İETT Bilet", "Otoyol HGS", "Araç Bakım", "Kredi"],
        "ortalama": [-1200, -1000, -500, -200, -400, -1500, -2000],
        "olasilik": [0.25, 0.20, 0.15, 0.10, 0.10, 0.10, 0.10]
    },
    "İsteğe Bağlı": {
        "aciklama": ["Trendyol", "Netflix", "Spotify", "Sinema", "Dış Yemek", "Steam", "Tatil", "Hobi"],
        "ortalama": [-1000, -300, -150, -200, -800, -400, -3000, -500],
        "olasilik": [0.25, 0.10, 0.10, 0.05, 0.15, 0.10, 0.10, 0.15]
    }
}

# Mevsimsel eklemeler (Kış ayları: 12,1,2)
kis_aylari = [12, 1, 2]
yaz_aylari = [6, 7, 8]

# -------------------------------
# 3. VERİ OLUŞTURMA DÖNGÜSÜ
# -------------------------------
hareketler = []

for ay in range(ay_sayisi):
    ay_tarih = baslangic_tarihi + timedelta(days=30*ay)
    ay_no = ay_tarih.month   # 1-12
    ay_ismi = ay_tarih.strftime("%B")
    
    # ----- GELİR (Maaş) -----
    # Gelir düzenliliğini simüle et: her ay oynar
    bu_ay_maas = max(30000, aylik_maas + np.random.normal(0, maas_std))
    hareketler.append({
        "Tarih": ay_tarih.replace(day=1),
        "Açıklama": "Maaş Ödemesi",
        "Kategori": "Gelir",
        "Tutar": round(bu_ay_maas, 2)
    })
    
    # ----- EK GELİR (varsa) -----
    if (ay+1) in ek_gelir_aylari:
        hareketler.append({
            "Tarih": ay_tarih.replace(day=15),
            "Açıklama": "Ek Gelir (Freelance/Temettü)",
            "Kategori": "Gelir",
            "Tutar": round(ek_gelir_tutari + np.random.normal(0, 500), 2)
        })
    
    # ----- GİDERLER (ayda rastgele 15-25 hareket) -----
    harcama_sayisi = np.random.randint(15, 25)
    for _ in range(harcama_sayisi):
        # Kategori seçimi (ağırlıklandırma yapılabilir)
        kat = np.random.choice(list(harcama_kategorileri.keys()))
        kat_verisi = harcama_kategorileri[kat]
        
        # Açıklama seçimi (olasılıklara göre)
        aciklama = np.random.choice(kat_verisi["aciklama"], p=kat_verisi["olasilik"])
        ortalama_tutar = kat_verisi["ortalama"][kat_verisi["aciklama"].index(aciklama)]
        
        # Mevsimsel ayarlama
        if kat == "Konut" and aciklama == "Doğalgaz" and ay_no in kis_aylari:
            ortalama_tutar *= 1.8   # Kışın doğalgaz faturası 1.8 katı
        if kat == "İsteğe Bağlı" and aciklama == "Tatil" and ay_no in yaz_aylari:
            ortalama_tutar *= 1.5   # Yaz tatili daha pahalı
        
        # Tutarı rastgele oynat (±%20)
        tutar = round(np.random.normal(ortalama_tutar, abs(ortalama_tutar)*0.2), 2)
        if tutar > 0:
            tutar = -tutar  # Giderler negatif
        
        # Rastgele gün (1-28 arası, ama maaş günü (1) dışında olabilir)
        gun = np.random.randint(2, 29)
        hareketler.append({
            "Tarih": ay_tarih.replace(day=gun),
            "Açıklama": aciklama,
            "Kategori": kat,
            "Tutar": tutar
        })

# -------------------------------
# 4. VERİYİ DÜZENLE VE KAYDET (CSV)
# -------------------------------
df_hareket = pd.DataFrame(hareketler)
df_hareket = df_hareket.sort_values("Tarih").reset_index(drop=True)
df_hareket.to_csv("banka_hareketleri.csv", index=False, encoding="utf-8-sig")
print("✓ banka_hareketleri.csv oluşturuldu (12 aylık, detaylı).")

# -------------------------------
# 5. BULANIK MANTIK GİRDİLERİNİ HESAPLA
# -------------------------------
# Toplam Gelir
toplam_gelir = df_hareket[df_hareket["Tutar"] > 0]["Tutar"].sum()

# Zorunlu Giderler (Konut + bazı Ulaştırma? Bu örnekte sadece Konut)
zorunlu_kategoriler = ["Konut"]
zorunlu_gider = df_hareket[(df_hareket["Tutar"] < 0) & (df_hareket["Kategori"].isin(zorunlu_kategoriler))]["Tutar"].sum()
zorunlu_gider_mutlak = abs(zorunlu_gider)

# Esneklik Oranı = (Gelir - Zorunlu) / Gelir
esneklik_orani = (toplam_gelir - zorunlu_gider_mutlak) / toplam_gelir

# Gelir Düzenliliği: aylık gelirlerin standart sapması / ortalama
aylik_gelirler = df_hareket[df_hareket["Kategori"] == "Gelir"].groupby(df_hareket["Tarih"].dt.to_period("M"))["Tutar"].sum()
gelir_std = aylik_gelirler.std()
gelir_ortalama = aylik_gelirler.mean()
duzenlilik_skoru = 1 - min(1, gelir_std / gelir_ortalama)  # 0=çok düzensiz, 1=çok düzenli

if duzenlilik_skoru < 0.33:
    gelir_duzenliligi = "Düzensiz"
elif duzenlilik_skoru < 0.66:
    gelir_duzenliligi = "Orta"
else:
    gelir_duzenliligi = "Düzenli"

# Risk Toleransı (simule edelim - kullanıcı anketi yerine rastgele)
risk_toleransi = np.random.uniform(0, 1)   # 0-1 arası
if risk_toleransi < 0.33:
    risk_seviyesi = "Düşük"
elif risk_toleransi < 0.66:
    risk_seviyesi = "Orta"
else:
    risk_seviyesi = "Yüksek"

# Yatırım Profili Önerisi (basit kural - bulanık mantık sonrası kullanılacak)
if risk_toleransi > 0.66:
    onerilen_profil = "Agresif"
elif risk_toleransi > 0.33:
    onerilen_profil = "Dengeli"
else:
    onerilen_profil = "Güvenilir"

# Özet verileri bir DataFrame'te topla
ozet_veriler = {
    "Toplam Gelir (TL)": toplam_gelir,
    "Toplam Zorunlu Gider (TL)": zorunlu_gider_mutlak,
    "Esneklik Oranı (0-1)": esneklik_orani,
    "Gelir Düzenlilik Skoru (0-1)": duzenlilik_skoru,
    "Gelir Düzenlilik (Kategori)": gelir_duzenliligi,
    "Risk Tolerans (0-1)": risk_toleransi,
    "Risk Seviyesi": risk_seviyesi,
    "Yatırım Profili Önerisi (Basit Kural)": onerilen_profil
}
df_ozet = pd.DataFrame([ozet_veriler])
df_ozet.to_csv("ozet_hesaplar.csv", index=False, encoding="utf-8-sig")
print("✓ ozet_hesaplar.csv oluşturuldu (12 aylık metrikler).")

# -------------------------------
# 6. EKRANA ÖZET YAZDIR
# -------------------------------
print("\n========== PROJE GİRDİLERİ (Bulanık Mantık İçin) ==========")
print(f"Gelir Düzenliliği: {gelir_duzenliligi} (skor: {duzenlilik_skoru:.2f})")
print(f"Esneklik Oranı: {esneklik_orani:.2f}")
print(f"Risk Toleransı: {risk_toleransi:.2f} ({risk_seviyesi})")
print(f"Önerilen Yatırım Profili (Basit Kural): {onerilen_profil}")
print("\nNot: Bu verileri bulanık mantık modelinize giriş olarak kullanabilirsiniz.")