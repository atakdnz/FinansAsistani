# FinansAsistani Agent Memory

Bu dosya projeye baska bir coding agent devam ederse hizli ve dogru context alabilmesi icin yazildi.

## Proje Hedefi

Bu repo, Bulanik Mantik dersi icin gelistirilen kisisel finans asistanidir. Sunumda vaat edilen ana akis:

1. Kullanici banka hesap dokumu PDF'i yukler.
2. Sistem PDF'ten islemleri otomatik cikarir.
3. Islemler kategori ve gider tipine ayrilir.
4. Kullanici risk toleransi sorularini yanitlar.
5. Gelir duzenliligi, esneklik, risk toleransi, yatirim vadesi, acil durum tamponu ve borc yuku Mamdani bulanik mantik sistemine girer.
6. Sistem yatirim profili ve portfoy onerisi uretir.

Sunuma yuzde yuz birebir uyum zorunlu degil. Onemli olan calisan, savunulabilir ve teknik olarak tutarli bir ders projesidir.

## Kullanici Tercihleri

- PaddleOCR kalsin; PDF'ten metin cikarmak icin kullanilacak.
- VisionOCR kullanilmasin; baska cihazlarda test edilebilmesi gerekiyor.
- LLM tabanli siniflandirma ana yol olmasin. Qwen 0.8B denendi ama tutarsiz bulundu.
- Deterministik siniflandirma ana karar mekanizmasi olsun.
- Embedding modeli yardimci AI katmani olarak kalsin.
- Gereksiz buyuk modeller, eski sanal ortamlar ve LLM artefaktlari temiz kalmali.
- GitHub commitleri detayli olmali ve her committen sonra push yapilmali.

## GitHub Kurallari

Repo: `https://github.com/atakdnz/FinansAsistani`

Kullanici commit kurallari:

```bash
git config --global user.name "Atakan Akdeniz"
git config --global user.email "at.akdnz@gmail.com"
```

- Commit basligi present tense olsun.
- Ilk satir kisa olsun.
- Commit body'sinde maddelerle ne degistigi yazilsin.
- AI/Claude referansi kullanma.
- Emoji kullanma.
- Her committen hemen sonra mutlaka:

```bash
git push origin main
```

Bu ortamdaki sandbox nedeniyle `.git` yazan git komutlari cogunlukla escalated calistirilmalidir.

## Mevcut Commit Gecmisi

Son onemli commitler:

- `dea4881 Clarify fuzzy rule activation display`
- `bec1bc4 Refine fuzzy rule base`
- `62fc953 Merge teammate dashboard updates`
- `3fa119d Simplify risk profile inputs`
- `fb9976d Stabilize OCR parsing and fuzzy inputs`
- `4e40cbe Refine category expense guidance`
- `f2779e6 Add micro transaction filtering and recommendations`
- `d15d1c9 Expand fuzzy profile inputs`
- `c08bc51 Document optional fuzzy inputs`
- `2324b00 Add OCR correction workflow docs`
- `2b23bc7 Use Latin PaddleOCR recognition for Turkish statements`
- `6f508ce Document OCR finance assistant flow`

Yeni 6 girdili fuzzy degisiklikleri:

- Yatirim vadesi risk toleransi ortalamasindan ayrildi.
- Acil durum tamponu hesap dokumundeki bakiye ve zorunlu giderden hesaplandi.
- Borc yuku orani kredi/kart/borc odemeleri uzerinden hesaplandi.
- Kural tabani 25 hedefli Mamdani kuralina cikti.
- 10 TL ve altindaki mikro tahsilatlar analiz ve arayuz listesinden ayrildi.
- Kategori bazli oneri kartlari eklendi.
- Pasif fuzzy kurallari arayuzde katlanabilir bolume tasindi.
- Arayuz iki moda ayrildi: varsayilan `Oneri Paneli` kullanici odakli sade ekran, `Teknik Analiz` mevcut detayli fuzzy/OCR ekranidir.
- PDF yukleme alani ve risk/vade sorulari ustte yan yana tutulur; kullanici girdileri tek bolgede toplanir.

## Calisma Ortami

Ana klasor:

```text
/Users/atakan/Desktop/Projeler/FinansAsistani
```

PaddleOCR ile uygulama calistirma:

```bash
PADDLE_PDX_CACHE_HOME=.paddlex_cache .venv_paddle/bin/python app.py
```

Uygulama:

```text
http://127.0.0.1:5050
```

Test komutlari:

```bash
.venv_paddle/bin/python -m unittest discover -s tests
.venv_paddle/bin/python -m py_compile app.py transaction_classifier.py embedding_classifier.py statement_pdf_pipeline.py tests/test_transaction_classifier.py
```

Yerel/ignored dosyalar:

- `uploads/`
- `extracted_transactions.csv`
- `user_profile.json`
- `.venv_*`
- `.paddlex_cache/`
- gercek hesap dokumu PDF'leri

Onceden temizlenen gereksiz seyler:

- Qwen GGUF/LLM model dosyalari
- `.venv_ocr`
- `.venv`
- `.venv_app`
- buyuk ve kullanilmayan deneme artefaktlari

## Neden LLM Ana Siniflandirici Degil?

LM Studio ve llama.cpp ile Qwen 0.8B denenmistir. Model hizli olsa da banka islemlerinde kararsiz cevaplar verdi:

- Format disina cikti.
- BSMV/komisyon gibi net banka ucretlerini bazen yanlis siniflandirdi.
- Kredi karti ve ulasim/gida gibi alanlarda tutarsiz davrandi.
- Guven skoru vermesi prompttan kaynaklandi; bu skor gercek olasilik degildi.

Bu nedenle ana siniflandirma deterministik kurallara alindi. Embedding ise "AI modeli kullandik" denebilecek yardimci bir katman olarak tutuldu.

## OCR Karari

PaddleOCR kullaniliyor. Baslangicta Ingilizce recognition modeli Turkce karakterlerde zayifti. `latin_PP-OCRv5_mobile_rec` modeline gecildi.

`statement_pdf_pipeline.py` icinde:

```python
PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    text_detection_model_name="PP-OCRv5_mobile_det",
    text_recognition_model_name="latin_PP-OCRv5_mobile_rec",
    text_det_limit_side_len=1600,
    text_det_limit_type="max",
)
```

Latin model test PDF'inde `ÖĞRENİM`, `KREDİ`, `ALIŞVERİŞ`, `İŞYERİ` gibi kelimeleri Ingilizce modele gore daha iyi okudu.

OCR normalizasyonu, OCR'dan gelen metni siniflandirmaya uygun daha kararli forma cekmektir:

- Buyuk harfe cevirme.
- Turkce karakterleri sade forma indirme.
- Aksan/bozuk karakter etkisini azaltma.
- Noktalama ve fazla bosluklari temizleme.

Bu is `transaction_classifier.py` icindeki `normalize_text()` fonksiyonunda yapilir.

## Test Edilen PDF

Kullanici kucuk bir hesap dokumu PDF'i yukledi. Dosya adi yerelde `Hesap_Hareketleri_26042026-1.pdf` olarak goruldu. Pipeline bu PDF'ten 11 islem cikardi.

Ornek beklenen/gercek siniflandirmalar:

- `Referansli Havale BSMV Tahsilati -0.20` -> `Banka Ücreti / Zorunlu`
- `Referansli Havale Komisyonu Tahsilat -3.99` -> `Banka Ücreti / Zorunlu`
- `cig kofte / kahve -500.00` -> `Gıda / Kısılabilir`
- `ATM YATAN KART 8000.00` -> `Gelir / Gelir`
- `KK OTOMATIK ODEME -4092.71` -> `Borç/Kredi/Kart / Zorunlu`
- `KYK ÖĞRENİM KREDİSİ 4000.00` -> `Gelir / Gelir`
- `S/GETIR 1` -> `Gıda / Kısılabilir`
- `YEMEKPAY/YEMEK SEPET` -> `Gıda / Kısılabilir`

## Ana Dosyalar

### `app.py`

Flask uygulamasi. Endpointler:

- `/`: dashboard.
- `/api/data`: banka verisini, fuzzy sonucu ve grafik verilerini JSON dondurur.
- `/api/upload-statement`: PDF yukler, OCR pipeline'i calistirir, `extracted_transactions.csv` uretir.
- `/api/risk-profile`: kullanici risk cevaplarini kaydeder/okur.
- `/api/transactions`: OCR islemlerini listeler ve manuel duzeltmeleri kaydeder.
- `calculate_financial_metrics`: mikro islemleri ayirir, 6 fuzzy girdiye kaynak olan metrikleri hesaplar.
- `build_category_analysis`: kategori paylarini, tasarruf potansiyelini, kategoriye ozel aksiyon metinlerini ve toplam tasarruf ozetini uretir.

Fuzzy sistem su anda 6 girdi kullanir:

- `esneklik`: gelirden zorunlu giderler ciktiktan sonra kalan oran.
- `duzenlilik`: aylik gelirlerin duzenliligi.
- `risk`: kullanicinin kayip ve dalgalanmaya psikolojik toleransi.
- `vade`: kullanicinin yatirim ufku.
- `tampon`: son bakiye / aylik ortalama zorunlu gider uzerinden acil durum tamponu.
- `borc`: borc/kredi/kart odemeleri / toplam gelir.

Kural tabani su anda 25 hedefli Mamdani kuralidir. Uzun vade tek basina agresif profil uretmez; risk, esneklik, tampon ve borc yuku ile birlikte degerlendirilir.

Arayuzde kural listesi aktif kurallari varsayilan olarak gosterir. 0 aktivasyonlu pasif kurallar `pasif kural` adli katlanabilir bolumdedir. Pasif kural sayisinin yuksek olmasi normaldir; Mamdani min operatorunde bir kosulun uyeligi 0 ise kural aktivasyonu da 0 olur.

10 TL ve altindaki negatif islemler mikro tahsilat kabul edilir. Bu satirlar ham CSV'de kalir, ama dashboard islem tablosunda gosterilmez ve toplam gider, kategori dagilimi, aylik trend, fuzzy girdi hesaplari ve kategori analizini etkilemez.

PDF/OCR parser `dd.mm.yyyy` ve `dd/mm/yyyy` tarihlerini kabul eder. Tablo
formatli dokumlerde fis numarasi ve `TL` birimi aciklamadan temizlenir, `Donem
Basi Devir Bakiyesi` ve limit satirlari islem olarak alinmaz. Kredi karti
dokumlerinde bazi harcamalar eksi isareti olmadan gelebilir; aciklama gider
sinyali tasiyorsa runtime'da tutar negatife cevrilir ve `positive_card_amount`
kural ekiyle isaretlenir.

`/api/data` yaniti `sanitize_json` ile temizlenir. Bos veya basarisiz OCR
sonucunda aylik ortalamalar `NaN` uretemez; bos ortalamalar 0 veya gerekli
yerlerde `null` olarak dondurulur.

### `statement_pdf_pipeline.py`

PDF -> image -> PaddleOCR -> islem satiri -> siniflandirma -> CSV hattidir.

Dusuk guvenli deterministik kararlar icin `.venv_embed` varsa `embedding_classifier.py` yardimci onerisi cagirir.

### `transaction_classifier.py`

Ana siniflandirici. Deterministik kurallar:

- BSMV, komisyon, masraf -> Banka Ucreti / Zorunlu
- Getir, Yemeksepeti, market, restoran, kahve -> Gida / Kisilabilir
- Kredi karti, kart odeme, borc, taksit -> Borc/Kredi/Kart / Zorunlu
- ATM yatan, maas, KYK, iade -> Gelir
- Kira, elektrik, su, dogalgaz, internet -> Konut/Fatura / Zorunlu
- Ulasim/yakit terimleri -> Ulastirma / Zorunlu
- Okul/kurs/kirtasiye -> Egitim / Zorunlu
- Hastane, eczane, doktor, klinik, medikal -> Saglik / Zorunlu
- Netflix, Spotify, Steam, sinema, tatil -> Istegi Bagli
- Trendyol, Hepsiburada, Amazon, sanal pos alisveris -> Alisveris / Kisilabilir

Guven skoru model olasiligi degil, kural gucudur.

### `embedding_classifier.py`

Opsiyonel embedding fallback. Model:

```text
google/embeddinggemma-300m
```

Kategori prototipleri ile islem aciklamasi embedding uzayinda karsilastirilir. Ana karar deterministik oldugu icin bu modul yardimci sinyal olarak dusunulmeli.

### `templates/index.html`

Tek sayfalik dashboard:

- PDF yukleme.
- Varsayilan Oneri Paneli: profil skoru, portfoy, temel finans ozeti, iki risk/vade sorusu ve kisisel tasarruf onerileri.
- Teknik Analiz sekmesi: OCR kontrolu, siniflandirma, fuzzy uyelik grafikleri, kural aktivasyonlari, agregasyon ve centroid.
- OCR duzeltme tablosu Teknik Analiz'de acik gelir, ancak tablo kendi icinde scroll eder. Pagination kullanilmadi; bu sayede kaydetme mantigi tum satirlari DOM'da tutmaya devam eder.
- Risk toleransi ve yatirim ufku uzunlugu icin iki ayri slider/card.
- OCR islem kontrol ve manuel duzeltme tablosu.
- Gelir/gider ozetleri.
- Siniflandirma ozeti.
- Tasarruf simulatoru.
- Fuzzy uyelik grafikleri.
- Kural aktivasyonlari.
- Portfoy onerisi.
- Kategori bazli oneri kartlari.
- Mikro islem ozeti.

### `README.md`

Kisa kurulum/akis dokumani.

### `PROJECT_OVERVIEW.md`

Ekip icin daha detayli teknik anlatim ve sunum notlari.

## Kavramlar

### Esneklik Orani

`(toplam gelir - zorunlu gider) / toplam gelir`

Kisaca: Kullanici zorunlu harcamalardan sonra gelirinin ne kadarini esnek kullanabiliyor? Bu daha cok butce kapasitesidir.

### Kisilabilir Gider Orani

`kisilabilir giderler / toplam gider`

Kisaca: Harcamalarin ne kadari azaltmaya uygun? Bu daha cok tasarruf potansiyelidir. Esneklikten farklidir; gelirden bagimsiz olarak gider yapisini anlatir.

### Tasarruf Orani

`(toplam gelir - toplam gider) / toplam gelir`

Kisaca: Kullanici gelirinin ne kadarini ay sonunda elde tutuyor? Bu mevcut finansal davranisi gosterir.

Bu iki metrik su an dashboard/gelecek oneri motoru icin anlamli adaylardir; ana fuzzy sisteme henuz girdi olarak eklenmedi:

- `kisilabilir_gider_orani`: harcama azaltma potansiyeli.
- `tasarruf_orani`: finansal tampon/likidite davranisi.

## Mevcut Eksikler

Zorunlu olmayan ama projeyi guclendirecek alanlar:

1. Kategori bazli onerilerin metinlerini ve esiklerini daha fazla gercek dokumle iyilestirmek.
2. OCR duzeltme ekraninda dusuk guvenli satirlari vurgulamak.
3. Fuzzy kural agirliklarini daha fazla senaryoyla ince ayar yapmak.
4. Fuzzy hesaplamayi `app.py` icinden ayri bir servis modulune tasimak.

Farkli banka PDF'i bulmak zor olabilir; bu ders projesi icin sart degil. Mevcut akisin tek banka dokumunde stabil calismasi daha onemli.

## Fuzzy Girdi Kararlari

Acil durum tamponu, borc yuku ve yatirim vadesi artik uygulandi. Harcama volatilitesi ve piyasa risk seviyesi henuz uygulanmadi.

### Acil Durum Tamponu

Kaynak: Hesap dokumundeki son bakiye ve zorunlu giderler.

Formul:

```text
mevcut bakiye / aylik ortalama zorunlu gider
```

Etkisi:

- Tampon zayifsa agresif portfoy onerisi baskilanir.
- Tampon gucluyse kullanicinin risk toleransi daha rahat dikkate alinabilir.

Uyelik fonksiyonu son durumda daha yumusak ucgensel kumelerle ayarlidir.
Bu degisiklik, tek bir aralikta uzun sure 1.000 uyelik gorunmesini azaltir;
ornek olarak 0.129 tampon girdisi `Zayif` icin yaklasik 0.69 uyelik verir.

### Borc Yuku Orani

Kaynak: Hesap dokumundeki kredi, kredi karti, taksit ve borc odemeleri.

Formul:

```text
borc/kredi/kart odemeleri / toplam gelir
```

Etkisi:

- Borc yuku yuksekse sistem daha korumaci veya dengeli profile kayar.
- Borc yuku dusukse esneklik ve risk toleransi daha etkili olabilir.

### Yatirim Vadesi

Kaynak: Kullanicidan soru.

Ornek soru:

```text
Bu parayi ne kadar sure yatirimda tutabilirsiniz?
```

Ornek seviyeler:

- Kisa vade: 0-6 ay
- Orta vade: 6-24 ay
- Uzun vade: 24 ay ve uzeri

Etkisi:

- Kisa vadede dalgalanma riski daha tehlikelidir; sistem guvenilir/dengeli tarafa kayar.
- Uzun vadede kullanici dalgalanmalari bekleyebilir; risk toleransi da yuksekse dengeli/agresif portfoy daha savunulabilir olur.
- Uzun vade tek basina agresif profil demek degildir; acil durum tamponu, borc yuku ve kullanici risk toleransi ile birlikte yorumlanmalidir.

### Harcama Volatilitesi

Kaynak: Birden fazla aylik dokum varsa hesap dokumu.

Formul:

```text
aylik gider standart sapmasi / aylik ortalama gider
```

Etkisi:

- Giderler cok oynaksa finansal planlama riski artar, sistem daha temkinli olur.
- Stabil giderler daha ongorulebilir bir yatirim davranisini destekler.

### Piyasa Risk Seviyesi

Kaynak: Opsiyonel internet/API verisi.

Ornek sinyaller:

- BIST veya kur oynakligi
- faiz/enflasyon seviyesi
- altin veya doviz hareketliligi

Etkisi:

- Piyasa riski yuksekse agresif portfoy agirligi azaltilabilir.
- Demo sirasinda internet/API bagimliligi riskli oldugu icin opsiyonel ve fallback'li tasarlanmalidir.

## Onerilen Sonraki Implementasyon

En mantikli sonraki teknik adim:

1. Frontend'deki ders anlatimi gibi duran bolumleri sade bir demo akisi haline getir.
2. Kategori onerilerini daha net, daha az kalabalik ve demo odakli hale getir.
3. Fuzzy hesaplamayi ayri servis modulune tasiyarak `app.py` dosyasini sadeleştir.

## Dikkat Edilecek Noktalar

- `prepare_bank_data()` manuel duzeltmeleri ezmemek icin siniflandirma kolonlari zaten varsa tekrar classify etmez.
- `/api/transactions` POST duzeltmeleri `manual`, `user_correction`, `1.0` guvenle yazar.
- Gercek hesap dokumu ve uretilen CSV'ler commitlenmemeli.
- PaddleOCR cache `.paddlex_cache` icinde kalmali ve commitlenmemeli.
- Son tarayici kontrolunde kural aktivasyon ozeti, kapali pasif kural bolumu ve konsol hatasiz dashboard dogrulandi.
