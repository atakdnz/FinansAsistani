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
7. Kullanici OCR'dan cikan islem satirlarini arayuzden kontrol edip duzeltebilir.
8. `app.py` 10 TL ve altindaki mikro tahsilatlari analiz ve arayuz listesinden ayirir.
9. `app.py` gelir duzenliligi, esneklik, risk toleransi, yatirim vadesi, acil durum tamponu ve borc yukunu fuzzy sisteme verir.
10. Mamdani cikarimi ve centroid durulastirma ile yatirim profili uretilir.
11. Kategori bazli finansal oneriler dashboard'da gosterilir.

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

PaddleOCR'da recognition modeli olarak `latin_PP-OCRv5_mobile_rec` kullaniliyor. Bu model Latin alfabeli diller icin egitildigi ve Turkce'yi destekledigi icin `en_PP-OCRv5_mobile_rec` modeline gore daha dogru sonuc verdi. Test PDF'inde `ÖĞRENİM`, `KREDİ`, `ALIŞVERİŞ`, `İŞYERİ` gibi kelimeler Latin modelle daha duzgun okundu.

Tablo formatli PDF'lerde parser `26.04.2026` ve `26/04/2026` tarihlerini
destekler. Fis numarasi ve `TL` birimi aciklamadan temizlenir. Kredi karti
dokumlerinde harcama tutarlari eksi isareti olmadan gelebilirse aciklama gider
sinyali tasidigi durumda tutar analizde gider olarak isaretlenir.

## Dosya Dosya Kod Rehberi

### `app.py`

Flask uygulamasinin merkezidir.

- `/` ana dashboard sayfasini dondurur.
- `/api/data` mevcut banka verisini fuzzy analizden gecirip JSON uretir.
- `/api/upload-statement` PDF yukler, PaddleOCR pipeline'ini calistirir ve `extracted_transactions.csv` uretir.
- `/api/risk-profile` risk toleransi cevaplarini kaydeder.
- `/api/transactions` OCR'dan cikan islem satirlarini listeler ve kullanici duzeltmelerini kaydeder.
- Fuzzy uyelik fonksiyonlari, Mamdani kurallari, agregasyon ve centroid durulastirma burada calisir.
- 10 TL ve altindaki mikro tahsilatlari ham veride tutar ama analiz ve tablo gosteriminden ayirir.
- Kategori paylarini ve kategoriye ozel aksiyonlari `build_category_analysis` ile uretir.
- `/api/data` icin `NaN`/sonsuz degerleri JSON'a cikmadan temizler.

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
- Varsayilan ekran sade `Oneri Paneli`dir; kullaniciya profil, portfoy, temel finans ozeti ve oneriler gosterilir.
- `Teknik Analiz` sekmesi mevcut detayli demo ekranini korur; OCR kontrolu, siniflandirma ozeti, fuzzy grafikler ve kural aktivasyonlari burada gosterilir.
- Risk toleransi ve yatirim ufku uzunlugu icin iki ayri slider/card vardir.
- Yatirim vadesi risk toleransindan ayri fuzzy girdi olarak kullanilir.
- Onerilen yatirim profili gosterilir.
- Gelir/gider ozetleri gosterilir.
- Siniflandirma yontemi ve dusuk guvenli kayit sayisi gosterilir.
- Tasarruf simulatoru ile gider kismaya gore tahmini birikim etkisi gosterilir.
- Fuzzy uyelik fonksiyonlari, kurallar, agregasyon ve centroid grafikleri gosterilir.
- Kategori bazli finansal oneriler gosterilir.
- Mikro islemler tabloya girmeden ozet olarak gosterilir.

### `bankdataset.py`

Sentetik demo verisi uretir. Gercek kullanimda ana veri kaynagi degildir. PDF yuklenmemisse demo CSV ile dashboard calisabilsin diye tutuluyor.

### `fuzzy_finans.py`

Komut satiri/demo amacli eski fuzzy analiz dosyasidir. Ana web uygulamasi artik `app.py` uzerinden calisir. Ileride sadelestirme yapilacaksa fuzzy hesaplama ortak bir servis modulune tasinabilir.

## Onceki Analize Gore Ne Yapildi?

Yapildi:

- PDF yukleme eklendi.
- PaddleOCR ile PDF'ten veri cikarma eklendi.
- Gercek hesap dokumu uzerinde satir ayristirma calistirildi.
- OCR islem kontrol ve manuel duzeltme ekrani eklendi.
- Kullanici risk toleransi arayuzden alinir hale geldi.
- Risk toleransi fuzzy sisteme baglandi.
- Yatirim vadesi risk toleransi ortalamasindan ayrilip bagimsiz fuzzy girdi yapildi.
- Acil durum tamponu ve borc yuku hesap dokumunden hesaplanip fuzzy sisteme eklendi.
- Fuzzy kural tabani 6 girdi kullanan 25 hedefli Mamdani kuralina genisletildi.
- 10 TL ve altindaki mikro islemler analiz ve arayuz listesinden ayrildi.
- Kategori bazli oneri kartlari eklendi.
- Pasif fuzzy kurallari arayuzde katlanabilir bolume tasindi; aktif kurallar varsayilan olarak ustte gosteriliyor.
- Sentetik CSV'ye bagimlilik azaltildi.
- Deterministik siniflandirma sistemi eklendi.
- Embedding destekli yardimci siniflandirma eklendi.
- Saglik kategorisi deterministik ve embedding siniflandirmaya eklendi.
- `requirements.txt` eklendi.
- Gereksiz LLM modeli ve eski sanal ortamlar temizlendi.
- Dashboard'a siniflandirma ozeti eklendi.

Kismen yapildi:

- Gider analizi kategori toplamlarini, tasarruf potansiyelini ve kategoriye ozel aksiyon metinlerini gosteriyor. Metin/esik kalitesi daha fazla gercek dokumle ince ayar gerektirebilir.
- Embedding fallback var, fakat mevcut gercek PDF orneginde deterministik kurallar tum satirlari yuksek guvenle yakaladigi icin embedding devreye girmedi.
- Gelir duzenliligi gercek PDF tek aylik veri oldugunda notr varsayiliyor (`0.65`). Daha fazla aylik PDF yuklenirse daha anlamli hesaplanabilir.

Henuz yapilmadi:

- PDF parser farkli banka formatlarina karsi kapsamli test edilmedi.
- Fuzzy mantik kodu ayri bir servis modulune tasinmadi.

## Sunumda Nasil Anlatilir?

Kisa teknik cumle:

> Sistem, kullanicinin yukledigi banka PDF'ini PaddleOCR ile okuyarak islem satirlarina ayirir. Islemler once kural tabanli deterministik siniflandirici ile kategorize edilir. 10 TL ve altindaki mikro tahsilatlar analizden ayrilir. Belirsiz islemler icin embedding tabanli benzerlik modeli yardimci karar mekanizmasi olarak kullanilir. Sonrasinda gelir duzenliligi, esneklik orani, risk toleransi, yatirim vadesi, acil durum tamponu ve borc yuku Mamdani bulanik cikarim sistemine verilerek yatirim profili ve kategori bazli oneriler olusturulur.

## Mantikli Sonraki Gelistirmeler

1. **Siniflandirma kural listesini genisletmek**

   Daha fazla merchant ve banka terimi eklenebilir. Ozellikle saglik, egitim, abonelik, market, e-ticaret, ulasim ve fatura kaliplari genisletilebilir.

2. **Gider oneri motorunu iyilestirmek**

   Kategori toplamlarina gore oneriler uretiliyor. Ileride su ayrimlar daha detayli hale getirilebilir:

   - Gida harcamasi yuksekse market/yemek disi ayrimi yapilabilir.
   - Ulasim yuksekse yakit/toplu tasima ayrimi yapilabilir.
   - Istege bagli gider yuksekse tasarruf potansiyeli yuzdesi verilebilir.

3. **OCR duzeltme ekranini demo icin iyilestirmek**

   Mevcut ekran islem satirlarini duzenleyebiliyor. Bir sonraki adimda dusuk guvenli satirlari daha belirgin gostermek, ham OCR satirini tooltip veya detay satiri olarak acmak ve kayit sonrasi daha net geri bildirim vermek iyi olur.

4. **Fuzzy kural agirliklarini ince ayar yapmak**

   Su an 6 girdiyi kullanan 25 hedefli kural var. Kurallar sunum icin yeterli seviyede; daha fazla ornek senaryoyla aktivasyon esikleri ve portfoy sonuc davranisi ince ayar yapilabilir.

5. **Portfoy dagilimini daha aciklanabilir yapmak**

   Portfoy oranlari sunuma birebir uymak zorunda degil, ama her profil icin neden o dagilim secildigi kisa aciklanabilir.

6. **Birden fazla PDF veya ay destegi**

   Kullanici birkac aylik dokum yuklerse gelir duzenliligi daha gercekci hesaplanir.

7. **Ek veri kaynaklarini tartismak**

   Ekip karari sonrasi su girdiler eklenebilir:

   - Harcama volatilitesi: aylik gider oynakligi. Giderler cok degisiyorsa finansal planlama riski artar.
   - Piyasa risk seviyesi: opsiyonel internet verisi. Demo bagimliligi yaratmamasi icin fallback ile tasarlanmalidir.

## Mevcut Degerlendirme

Proje artik sunumdaki temel vaadi karsiliyor:

- PDF yukleme var.
- OCR var.
- Siniflandirma var.
- Yapay zeka destekli embedding katmani var.
- Kullanici risk girdisi var.
- Bulanik mantik var.
- Kullanici icin sade Oneri Paneli ve sunum icin Teknik Analiz modu var.

Kusursuz urun degil, ama ders projesi ve canli demo icin savunulabilir ve teknik olarak tutarli bir seviyede.
