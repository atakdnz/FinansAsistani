# Finans Asistani

Kisisel finans onerisi icin banka hesap dokumunu okuyup harcama davranisini analiz eden Flask uygulamasi.

## Akis

1. Kullanici arayuzden PDF hesap dokumu yukler.
2. Backend PDF sayfalarini gorsellere render eder.
3. PaddleOCR dokumdeki metin kutularini okur.
4. OCR satirlari tarih, aciklama, tutar ve bakiye alanlarina ayrilir.
5. Deterministik siniflandirici aciklama ve tutara gore kategori/gider tipi atar.
6. Dusuk guvenli kayitlar icin embedding tabanli yardimci sinyal kullanilir.
7. Gelir, gider, esneklik orani ve gelir duzenliligi hesaplanir.
8. Kullanici risk toleransi sliderlarini doldurur.
9. Mamdani bulanik mantik ve centroid durulastirma ile yatirim profili uretilir.

## Ana Bilesenler

- `app.py`: Flask endpointleri, dashboard verisi ve bulanik mantik hesaplama.
- `statement_pdf_pipeline.py`: PDF render, PaddleOCR, satir ayrisma ve CSV uretimi.
- `transaction_classifier.py`: Kural tabanli deterministik islem siniflandirma.
- `embedding_classifier.py`: Dusuk guvenli kayitlar icin opsiyonel embedding onerisi.
- `templates/index.html`: PDF yukleme, risk profili ve analiz dashboard'u.
- `PROJECT_OVERVIEW.md`: Ekip icin kod rehberi, mevcut durum ve gelistirme onerileri.

## OCR Normalizasyonu

OCR ciktilari her zaman temiz gelmez. Ornegin Turkce karakterler veya banka
metinleri `ÖRENIM`, `ALIVERI`, `ALI§VERI`, `Komisyonu` gibi farkli
sekillerde okunabilir. Siniflandirma oncesinde metinler buyuk harfe cevrilir,
Turkce karakterler sade forma indirilir, noktalama ve fazla bosluklar
temizlenir. Bu sayede `YEMEKPAY`, `GETIR`, `BSMV`, `KOMISYON` gibi sinyaller
daha kararli yakalanir.

## Calistirma

PaddleOCR cache'i proje icinde tutulur:

```bash
PADDLE_PDX_CACHE_HOME=.paddlex_cache .venv_paddle/bin/python app.py
```

Uygulama varsayilan olarak:

```text
http://127.0.0.1:5050
```

## Git Disi Yerel Dosyalar

Asagidaki dosyalar yerel calisma urunudur ve commitlenmez:

- `uploads/`
- `extracted_transactions.csv`
- `user_profile.json`
- `.venv_*`
- `.paddlex_cache/`
- gercek hesap dokumu PDF dosyalari
