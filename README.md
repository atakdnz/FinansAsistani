# Finans Asistani

Kisisel finans onerisi icin banka hesap dokumunu okuyup harcama davranisini analiz eden Flask uygulamasi.

## Akis

1. Kullanici arayuzden PDF hesap dokumu yukler.
2. Backend PDF sayfalarini gorsellere render eder.
3. PaddleOCR dokumdeki metin kutularini okur.
4. OCR satirlari tarih, aciklama, tutar ve bakiye alanlarina ayrilir.
5. Deterministik siniflandirici aciklama ve tutara gore kategori/gider tipi atar.
6. Dusuk guvenli kayitlar icin embedding tabanli yardimci sinyal kullanilir.
7. 10 TL ve altindaki mikro tahsilatlar analiz ve arayuz listesinden ayrilir.
8. Gelir duzenliligi, esneklik, acil durum tamponu ve borc yuku hesaplanir.
9. Kullanici risk toleransi ve yatirim vadesi sliderlarini doldurur.
10. 6 girdili Mamdani bulanik mantik ve centroid durulastirma ile yatirim profili uretilir.
11. Kategori bazli kisa finansal oneriler uretilir.

## Ana Bilesenler

- `app.py`: Flask endpointleri, dashboard verisi ve bulanik mantik hesaplama.
- `statement_pdf_pipeline.py`: PDF render, PaddleOCR, satir ayrisma ve CSV uretimi.
- `transaction_classifier.py`: Kural tabanli deterministik islem siniflandirma.
- `embedding_classifier.py`: Dusuk guvenli kayitlar icin opsiyonel embedding onerisi.
- `templates/index.html`: PDF yukleme, OCR islem duzeltme tablosu, risk profili ve analiz dashboard'u.
- `PROJECT_OVERVIEW.md`: Ekip icin kod rehberi, mevcut durum ve gelistirme onerileri.

## Bulanik Mantik Girdileri

Ana fuzzy sistem 6 girdi kullanir:

- Gelir duzenliligi
- Esneklik orani
- Risk toleransi
- Yatirim vadesi
- Acil durum tamponu
- Borc yuku orani

Risk toleransi kullanicinin kayip ve dalgalanmaya psikolojik dayanimi olarak
hesaplanir. Yatirim vadesi ayri girdi olarak tutulur; uzun vade tek basina
agresif profil uretmez, sadece risk toleransi ve finansal tampon uygunsa daha
dalgali portfoylere alan acar.

## Mikro Islemler ve Oneriler

10 TL ve altindaki negatif islemler mikro tahsilat kabul edilir. Bu satirlar
ham CSV'de kalir, ancak dashboard islem tablosunda gosterilmez ve toplam gider,
kategori dagilimi, fuzzy girdiler ve aylik trend hesaplarini etkilemez.

Dashboard ayrica kategori bazli oneriler uretir. Borc yuku, gida/alisveris
agirligi, banka ucretleri, acil durum tamponu ve tasarruf orani gibi sinyaller
kisa oneri kartlarina donusturulur.

## OCR Normalizasyonu

OCR ciktilari her zaman temiz gelmez. Ornegin Turkce karakterler veya banka
metinleri `ÖRENIM`, `ALIVERI`, `ALI§VERI`, `Komisyonu` gibi farkli
sekillerde okunabilir. Siniflandirma oncesinde metinler buyuk harfe cevrilir,
Turkce karakterler sade forma indirilir, noktalama ve fazla bosluklar
temizlenir. Bu sayede `YEMEKPAY`, `GETIR`, `BSMV`, `KOMISYON` gibi sinyaller
daha kararli yakalanir.

Turkce metinler icin PaddleOCR tarafinda `latin_PP-OCRv5_mobile_rec`
recognition modeli kullanilir. Bu model Latin alfabeli diller arasinda
Turkce'yi de destekler ve `ÖĞRENİM`, `ALIŞVERİŞ`, `İŞYERİ` gibi kelimelerde
Ingilizce modele gore daha kararli sonuc verir.

PDF islendikten sonra OCR'dan cikan islemler arayuzde tablo olarak gosterilir.
Kullanici gerekirse tarih, aciklama, tutar, kategori veya gider tipini
duzeltebilir. Kaydedilen duzeltmeler `manual` yontemiyle saklanir ve sonraki
bulanik mantik hesabinda otomatik siniflandirma tarafindan ezilmez.

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
