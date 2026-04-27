# FinansAsistani Agent Memory

Bu dosya projeye baska bir coding agent devam ederse hizli ve dogru context alabilmesi icin yazildi.

## Proje Hedefi

Bu repo, Bulanik Mantik dersi icin gelistirilen kisisel finans asistanidir. Sunumda vaat edilen ana akis:

1. Kullanici banka hesap dokumu PDF'i yukler.
2. Sistem PDF'ten islemleri otomatik cikarir.
3. Islemler kategori ve gider tipine ayrilir.
4. Kullanici risk toleransi sorularini yanitlar.
5. Gelir duzenliligi, esneklik ve risk toleransi Mamdani bulanik mantik sistemine girer.
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

- `b8f22dc Initialize finance assistant project`
- `bd369a5 Add rule-based transaction classification`
- `96ced67 Add PDF upload and risk profile flow`
- `6f508ce Document OCR finance assistant flow`
- `2b23bc7 Use Latin PaddleOCR recognition for Turkish statements`

Bu MEMORY dosyasi yazilirken uncommitted degisiklikler:

- OCR islem kontrol ve manuel duzeltme endpointi.
- Arayuzde OCR satirlarini duzenleme tablosu.
- Saglik kategorisi.
- README ve proje ozeti guncellemeleri.

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

Fuzzy sistem su anda 3 girdi kullanir:

- `esneklik`: gelirden zorunlu giderler ciktiktan sonra kalan oran.
- `duzenlilik`: aylik gelirlerin duzenliligi.
- `risk`: kullanicinin risk toleransi.

Kural tabani su anda 9 Mamdani kuralidir. Bu yeterli calisir ama gelistirme icin genisletilebilir.

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
- Risk sliderlari.
- OCR islem kontrol ve manuel duzeltme tablosu.
- Gelir/gider ozetleri.
- Siniflandirma ozeti.
- Fuzzy uyelik grafikleri.
- Kural aktivasyonlari.
- Portfoy onerisi.

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

Bu iki yeni metrik fuzzy sisteme eklenebilir:

- `kisilabilir_gider_orani`: harcama azaltma potansiyeli.
- `tasarruf_orani`: finansal tampon/likidite davranisi.

## Mevcut Eksikler

Zorunlu olmayan ama projeyi guclendirecek alanlar:

1. Fuzzy girdi sayisini 3'ten 4 veya 5'e cikarmak.
2. Kural tabanini 9 kuraldan hedefli 14-18 kurala genisletmek.
3. Kategori bazli tasarruf onerileri uretmek.
4. OCR duzeltme ekraninda dusuk guvenli satirlari vurgulamak.
5. Farkli banka PDF formatlariyla test yapmak.
6. Fuzzy hesaplamayi `app.py` icinden ayri bir servis modulune tasimak.

Farkli banka PDF'i bulmak zor olabilir; bu ders projesi icin sart degil. Mevcut akisin tek banka dokumunde stabil calismasi daha onemli.

## Tartisilacak Ek Fuzzy Girdiler

Bu girdiler hemen uygulanmadi. Ekip karari sonrasi eklenebilir.

### Acil Durum Tamponu

Kaynak: Hesap dokumundeki son bakiye ve zorunlu giderler.

Formul:

```text
mevcut bakiye / aylik ortalama zorunlu gider
```

Etkisi:

- Tampon zayifsa agresif portfoy onerisi baskilanir.
- Tampon gucluyse kullanicinin risk toleransi daha rahat dikkate alinabilir.

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

1. `app.py` icinde `tasarruf_orani` ve `kisilabilir_gider_orani` hesapla.
2. Bunlari dashboard'da metrik olarak goster.
3. Fuzzy sisteme en az birini dorduncu girdi olarak ekle.
4. 9 kural yerine hedefli 14-18 kural yaz.
5. Kategori bazli basit oneri motoru ekle:
   - Gida yuksekse market/restoran ayrimi ve azaltma onerisi.
   - Istegi bagli yuksekse abonelik/eglence onerisi.
   - Banka ucreti yuksekse EFT/FAST/hesap paketi onerisi.
   - Borc/Kredi/Kart yuksekse borc yukunu azaltma onerisi.

Bu, sunumda "bulanik mantik kismi gelismis" gorunmesini saglar ve gercekten anlamli girdiler kullanir.

## Dikkat Edilecek Noktalar

- `prepare_bank_data()` manuel duzeltmeleri ezmemek icin siniflandirma kolonlari zaten varsa tekrar classify etmez.
- `/api/transactions` POST duzeltmeleri `manual`, `user_correction`, `1.0` guvenle yazar.
- Gercek hesap dokumu ve uretilen CSV'ler commitlenmemeli.
- PaddleOCR cache `.paddlex_cache` icinde kalmali ve commitlenmemeli.
- Browser UI son hali henuz tam gorsel testten gecmemis olabilir. Calisan server varsa `http://127.0.0.1:5050/` uzerinden test etmek iyi olur.
