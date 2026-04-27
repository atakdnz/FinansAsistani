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
9. Kullanici risk toleransi ve yatirim ufku uzunlugu sliderlarini doldurur.
10. 6 girdili Mamdani bulanik mantik ve centroid durulastirma ile yatirim profili uretilir.
11. Gider kategorilerinin payi hesaplanir ve kategoriye ozel aksiyon onerileri uretilir.

## Ana Bilesenler

- `app.py`: Flask endpointleri, dashboard verisi ve bulanik mantik hesaplama.
- `statement_pdf_pipeline.py`: PDF render, PaddleOCR, satir ayrisma ve CSV uretimi.
- `transaction_classifier.py`: Kural tabanli deterministik islem siniflandirma.
- `embedding_classifier.py`: Dusuk guvenli kayitlar icin opsiyonel embedding onerisi.
- `templates/index.html`: PDF yukleme, sade Oneri Paneli, Teknik Analiz modu, OCR islem duzeltme tablosu, risk profili, tasarruf simulatoru ve analiz dashboard'u.
- `PROJECT_OVERVIEW.md`: Ekip icin kod rehberi, mevcut durum ve gelistirme onerileri.

## Arayuz Modlari

Uygulama iki modla acilir:

- **Oneri Paneli:** Varsayilan kullanici ekrani. Yalnizca profil, portfoy,
  temel finans ozeti ve kisisel tasarruf onerilerini gosterir. PDF yukleme
  alani ile risk/vade sorulari sayfanin ustunde yan yana tutulur.
- **Teknik Analiz:** Sunum ve hoca sorulari icin detay ekranidir. OCR kontrolu,
  siniflandirma ozeti, uyelik fonksiyonlari, Mamdani kural aktivasyonlari,
  agregasyon ve centroid grafikleri burada kalir.

## Bulanik Mantik Girdileri

Ana fuzzy sistem 6 girdi kullanir:

- Gelir duzenliligi
- Esneklik orani
- Risk toleransi
- Yatirim vadesi
- Acil durum tamponu
- Borc yuku orani

Risk toleransi kullanicidan tek slider olarak alinir. Yatirim vadesi ayri girdi
olarak tutulur; uzun vade tek basina agresif profil uretmez, sadece risk
toleransi ve finansal tampon uygunsa daha dalgali portfoylere alan acar.

Kural tabani 25 Mamdani kuralindan olusur. Arayuzde yalnizca aktif kurallar
varsayilan olarak gosterilir; 0 aktivasyonlu pasif kurallar acilip kapanabilen
ayri bir bolumde tutulur. Mamdani min operatorunde bir kosulun uyeligi 0 ise
ilgili kuralin 0 gorunmesi normaldir.

## Mikro Islemler ve Oneriler

10 TL ve altindaki negatif islemler mikro tahsilat kabul edilir. Bu satirlar
ham CSV'de kalir, ancak dashboard islem tablosunda gosterilmez ve toplam gider,
kategori dagilimi, fuzzy girdiler ve aylik trend hesaplarini etkilemez.

Dashboard ayrica kategori bazli gider analizi uretir. Her kategori icin toplam
gider icindeki pay, tasarruf potansiyeli ve kategoriye ozel aksiyon metni
gosterilir. Ornegin gida icin liste/market disiplini, ulastirma icin toplu
tasima veya paylasim, konut/fatura icin uzun vadeli tarife ve abonelik kontrolu
onerilir.

## OCR Normalizasyonu

OCR ciktilari her zaman temiz gelmez. Ornegin Turkce karakterler veya banka
metinleri `ÖRENIM`, `ALIVERI`, `ALI§VERI`, `Komisyonu` gibi farkli
sekillerde okunabilir. Siniflandirma oncesinde metinler buyuk harfe cevrilir,
Turkce karakterler sade forma indirilir, noktalama ve fazla bosluklar
temizlenir. Bu sayede `YEMEKPAY`, `GETIR`, `BSMV`, `KOMISYON` gibi sinyaller
daha kararli yakalanir.

PDF satirlarinda tarih `26.04.2026` veya `26/04/2026` formatinda gelebilir.
Parser iki formati da kabul eder, fis numarasi ve `TL` birimi gibi tablo
kolonlarini aciklamadan temizler. Kredi karti dokumlerinde harcama tutarlari
eksi isareti olmadan gelebilecegi icin, aciklama gida/alisveris/ulasim gibi
gider sinyali tasiyorsa bu satir analizde gider olarak ele alinir.

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
- `last_pdf_name.txt`
- `.venv_*`
- `.paddlex_cache/`
- gercek hesap dokumu PDF dosyalari
