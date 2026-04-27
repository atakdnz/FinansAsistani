# Proje Teknik Ozeti

Bu dosya ekip uyelerinin kodu hizli anlamasi ve sunuma hazirlanmasi icin yazildi.

## Su Anki Durum

Proje artik sadece hazir CSV gosteren bir demo degil. Ana akis su sekilde calisiyor:

1. Kullanici arayuzden PDF hesap dokumu yukler.
2. `statement_pdf_pipeline.py` PDF sayfalarini gorsele cevirir.
3. PaddleOCR metin kutularini okur.
4. OCR satirlari tarih, aciklama, tutar ve bakiye alanlarina ayrilir.
5. `transaction_classifier.py` aciklama ve tutara gore kategori/gider tipi uretir.
6. Dusuk guvenli satirlar varsa `embedding_classifier.py` embedding benzerligi ile yardimci kategori onerir.
7. `app.py` gelir/gider metriklerini, esneklik oranini, gelir duzenliligini ve risk toleransini fuzzy sisteme verir.
8. Mamdani cikarimi ve centroid durulastirma ile yatirim profili uretilir.
9. `templates/index.html` sonucu dashboard olarak gosterir.

## Eski Halden Farki

Onceki kodda islem siniflandirmasi yoktu. `banka_hareketleri.csv` icinde hazir gelen `Kategori` kolonu dogru kabul ediliyordu. Bu yuzden sistem gercek banka dokumunu anlayamiyor, sadece ornek CSV toplamlarini gorsellestiriyordu.

Yeni durumda kategori karari kod tarafindan veriliyor:

```text
Referansli Havale BSMV Tahsilati -> Banka Ucreti / Zorunlu
YEMEKPAY YEMEK SEPET -> Gida / Kisilabilir
KK OTOMATIK ODEME -> Borc/Kredi/Kart / Zorunlu
ATM YATAN KART -> Gelir / Gelir
```

## OCR Normalizasyonu Nedir?

OCR ciktisi banka dokumlerinde her zaman duzgun gelmez. Ornekler:

```text
OGRENIM -> ORENIM
ALISVERIS -> ALIVERI veya ALI§VERI
Komisyonu Tahsilati -> Komisyonu Tahsilati
80.852,01 -> 80.852.01
```

Normalizasyon, siniflandirmadan once metni ortak ve daha kararli bir forma cekmektir. Kodda bu islem `transaction_classifier.py` icindeki `normalize_text` fonksiyonuyla yapilir:

- metni buyuk harfe cevirir,
- Turkce karakterleri sade hale getirir,
- aksan/bozuk karakter etkisini azaltir,
- noktalama ve fazla bosluklari temizler.

Bu sayede kural tabani bozuk OCR ciktilarinda bile `GETIR`, `YEMEKPAY`, `BSMV`, `KOMISYON`, `KK OTOMATIK ODEME` gibi sinyalleri yakalayabilir.

## Dosya Dosya Kod Rehberi

### `app.py`

Flask uygulamasinin merkezidir.

- `/` ana dashboard sayfasini dondurur.
- `/api/data` mevcut banka verisini fuzzy analizden gecirip JSON uretir.
- `/api/upload-statement` PDF yukler, PaddleOCR pipeline'ini calistirir ve `extracted_transactions.csv` uretir.
- `/api/risk-profile` risk toleransi cevaplarini kaydeder.
- Fuzzy uyelik fonksiyonlari, Mamdani kurallari, agregasyon ve centroid durulastirma burada calisir.

### `statement_pdf_pipeline.py`

PDF isleme hattidir.

- PDF sayfalarini `pypdfium2` ile gorsele cevirir.
- PaddleOCR ile metin kutularini alir.
- OCR kutularini satirlara gruplar.
- Satirlardan tarih, aciklama, tutar ve bakiye cikarir.
- Islemleri deterministik siniflandiriciya yollar.
- Dusuk guvenli kayitlar icin embedding yardimini cagirir.
- Sonucu `extracted_transactions.csv` dosyasina yazar.

### `transaction_classifier.py`

Ana islem siniflandiricisidir. LLM yerine deterministik kurallar ana karar mekanizmasidir.

Kural ornekleri:

- `BSMV`, `KOMISYON`, `MASRAF` -> Banka Ucreti
- `GETIR`, `YEMEKPAY`, `MIGROS`, `RESTORAN` -> Gida
- `KK OTOMATIK ODEME`, `KREDI KARTI`, `BORC ODEME` -> Borc/Kredi/Kart
- `ATM YATAN`, `MAAS`, `KYK` -> Gelir
- `KIRA`, `ELEKTRIK`, `DOGALGAZ`, `INTERNET` -> Konut/Fatura

Guven skoru model olasiligi degildir; kuralin gucunu ifade eder. Ornegin `BSMV` cok net banka ucreti sinyali oldugu icin yuksek guven alir.

### `embedding_classifier.py`

Opsiyonel yapay zeka destek katmanidir.

Embedding modeli kategori aciklamalari ile islem aciklamasini vektor uzayinda karsilastirir. Ana siniflandirici dusuk guven verirse bu modul yardimci sinyal uretir. Bu sayede projede yapay zeka modeli kullanimi teknik olarak vardir, ama ana karar mekanizmasi guvenilir deterministik kurallardir.

### `templates/index.html`

Kullanici arayuzudur.

- PDF yukleme alani vardir.
- Risk toleransi slider'lari vardir.
- Onerilen yatirim profili gosterilir.
- Gelir/gider ozetleri gosterilir.
- Siniflandirma yontemi ve dusuk guvenli kayit sayisi gosterilir.
- Fuzzy uyelik fonksiyonlari, kurallar, agregasyon ve centroid grafikleri gosterilir.

### `bankdataset.py`

Sentetik demo verisi uretir. Gercek kullanimda ana veri kaynagi degildir. PDF yuklenmemisse demo CSV ile dashboard calisabilsin diye tutuluyor.

### `fuzzy_finans.py`

Komut satiri/demo amacli eski fuzzy analiz dosyasidir. Ana web uygulamasi artik `app.py` uzerinden calisir. Ileride sadelestirme yapilacaksa fuzzy hesaplama ortak bir servis modulune tasinabilir.

## Onceki Analize Gore Ne Yapildi?

Yapildi:

- PDF yukleme eklendi.
- PaddleOCR ile PDF'ten veri cikarma eklendi.
- Gercek hesap dokumu uzerinde satir ayristirma calistirildi.
- Kullanici risk toleransi arayuzden alinir hale geldi.
- Risk toleransi fuzzy sisteme baglandi.
- Sentetik CSV'ye bagimlilik azaltildi.
- Deterministik siniflandirma sistemi eklendi.
- Embedding destekli yardimci siniflandirma eklendi.
- `requirements.txt` eklendi.
- Gereksiz LLM modeli ve eski sanal ortamlar temizlendi.
- Dashboard'a siniflandirma ozeti eklendi.

Kismen yapildi:

- Gider analizi kategori toplamlarini ve tasarruf potansiyelini gosteriyor, ama kategori bazli detayli oneri metinleri henuz sinirli.
- Embedding fallback var, fakat mevcut gercek PDF orneginde deterministik kurallar tum satirlari yuksek guvenle yakaladigi icin embedding devreye girmedi.
- Gelir duzenliligi gercek PDF tek aylik veri oldugunda notr varsayiliyor (`0.65`). Daha fazla aylik PDF yuklenirse daha anlamli hesaplanabilir.

Henuz yapilmadi:

- Saglik gibi daha fazla kategori eklenmedi.
- Kural tabani 27 fuzzy kombinasyona genisletilmedi.
- PDF parser farkli banka formatlarina karsi kapsamli test edilmedi.
- OCR sonucunu kullaniciya satir satir duzeltme ekrani eklenmedi.
- Fuzzy mantik kodu ayri bir servis modulune tasinmadi.

## Sunumda Nasil Anlatilir?

Kisa teknik cumle:

> Sistem, kullanicinin yukledigi banka PDF'ini PaddleOCR ile okuyarak islem satirlarina ayirir. Islemler once kural tabanli deterministik siniflandirici ile kategorize edilir. Belirsiz islemler icin embedding tabanli benzerlik modeli yardimci karar mekanizmasi olarak kullanilir. Sonrasinda gelir duzenliligi, esneklik orani ve kullanicidan alinan risk toleransi Mamdani bulanik cikarim sistemine verilerek yatirim profili olusturulur.

## Mantikli Sonraki Gelistirmeler

1. **Kural sayisini artirmak**

   Daha fazla merchant ve banka terimi eklenebilir. Ozellikle saglik, egitim, abonelik, market, e-ticaret, ulasim ve fatura kaliplari genisletilebilir.

2. **Gider oneri motoru eklemek**

   Kategori toplamlarina gore kisa oneriler uretilebilir:

   - Gida harcamasi yuksekse market/yemek disi ayrimi yapilabilir.
   - Ulasim yuksekse yakit/toplu tasima ayrimi yapilabilir.
   - Istege bagli gider yuksekse tasarruf potansiyeli yuzdesi verilebilir.

3. **OCR duzeltme ekrani eklemek**

   Kullanici OCR'dan cikan islem listesini gorup kategori veya tutari elle duzeltebilir. Bu demo icin cok guclu gorunur.

4. **Fuzzy kural tabanini genisletmek**

   Su an 9 temel kural var. 3 girdi x 3 seviye icin 27 kombinasyon daha sistematik yazilabilir. Ancak mevcut sistem calisir ve anlasilir durumda.

5. **Portfoy dagilimini daha aciklanabilir yapmak**

   Portfoy oranlari sunuma birebir uymak zorunda degil, ama her profil icin neden o dagilim secildigi kisa aciklanabilir.

6. **Birden fazla PDF veya ay destegi**

   Kullanici birkac aylik dokum yuklerse gelir duzenliligi daha gercekci hesaplanir.

## Mevcut Degerlendirme

Proje artik sunumdaki temel vaadi karsiliyor:

- PDF yukleme var.
- OCR var.
- Siniflandirma var.
- Yapay zeka destekli embedding katmani var.
- Kullanici risk girdisi var.
- Bulanik mantik var.
- Dashboard var.

Kusursuz urun degil, ama ders projesi ve canli demo icin savunulabilir ve teknik olarak tutarli bir seviyede.
