import unittest
import os
import tempfile

import pandas as pd

import app as finance_app
from statement_pdf_pipeline import OcrToken, parse_transactions
from transaction_classifier import classify_dataframe, classify_transaction, parse_amount


class TransactionClassifierTests(unittest.TestCase):
    def test_parse_turkish_amounts(self):
        self.assertEqual(parse_amount("-4.092,71"), -4092.71)
        self.assertEqual(parse_amount("8.000,00"), 8000.00)
        self.assertEqual(parse_amount("8.000.00"), 8000.00)
        self.assertEqual(parse_amount("-0.20"), -0.20)

    def test_real_statement_like_rows(self):
        cases = [
            ("Referansli Havale BSMV Tahsilati", "-0.20", "Banka Ücreti", "Zorunlu"),
            ("Referansli Havale Komisyonu Tahsilat", "-3.99", "Banka Ücreti", "Zorunlu"),
            ("ci kofte, kalsiyum, kahve, codex ERAY OZ Ziraat Mobil Havale", "-500,00", "Gıda", "Kısılabilir"),
            ("ATM YATAN KART", "8.000,00", "Gelir", "Gelir"),
            ("KK OTOMATIK ODEME", "-4.092,71", "Borç/Kredi/Kart", "Zorunlu"),
            ("KREDI YURTLAR KURUMU OGRENIM KREDISI", "4.000,00", "Gelir", "Gelir"),
            ("S/GETIR 1 MUTABAKAT SANAL POS ALISVERIS", "-125,00", "Gıda", "Kısılabilir"),
            ("Starbucks - Sabah Kahvesi", "125,00", "Gıda", "Kısılabilir"),
            ("YEMEKPAY/YEMEK SEPET", "-176,00", "Gıda", "Kısılabilir"),
            ("ECZANE SAGLIK HARCAMASI", "-240,00", "Sağlık", "Zorunlu"),
            ("İş Bankası Kredi Kartı Ödemesi", "-2500,00", "Borç/Kredi/Kart", "Zorunlu"),
        ]

        for description, amount, category, expense_type in cases:
            with self.subTest(description=description):
                result = classify_transaction(description, amount)
                self.assertEqual(result.category, category)
                self.assertEqual(result.expense_type, expense_type)
                self.assertGreaterEqual(result.confidence, 0.80)

    def test_dataframe_classification_columns(self):
        df = pd.DataFrame(
            [
                {"Açıklama": "S/GETIR 1 MUTABAKAT", "Tutar": "-125,00"},
                {"Açıklama": "Bilinmeyen Harcama", "Tutar": "-50,00"},
            ]
        )

        result = classify_dataframe(df)

        self.assertEqual(result.loc[0, "Kategori"], "Gıda")
        self.assertEqual(result.loc[0, "Gider Tipi"], "Kısılabilir")
        self.assertEqual(result.loc[1, "Kategori"], "Diğer")
        self.assertEqual(result.loc[1, "Gider Tipi"], "Belirsiz")
        self.assertIn("Sınıflandırma Kuralı", result.columns)

    def test_transaction_correction_endpoint_accepts_string_amounts(self):
        fd, path = tempfile.mkstemp(suffix=".csv")
        os.close(fd)
        old_path = finance_app.EXTRACTED_PATH
        try:
            pd.DataFrame(
                [
                    {
                        "Tarih": "26.04.2026",
                        "Açıklama": "ECZANE",
                        "Tutar": "-240,00",
                        "Bakiye": "1.000,00",
                        "Kategori": "Diğer",
                        "Gider Tipi": "Belirsiz",
                        "Sınıflandırma Güveni": 0.35,
                        "Sınıflandırma Yöntemi": "fallback",
                        "Sınıflandırma Kuralı": "negative_unknown",
                    }
                ]
            ).to_csv(path, index=False)
            finance_app.EXTRACTED_PATH = path

            response = finance_app.app.test_client().post(
                "/api/transactions",
                json={
                    "transactions": [
                        {
                            "id": 0,
                            "kategori": "Sağlık",
                            "gider_tipi": "Zorunlu",
                            "tutar": "-240,00",
                        }
                    ]
                },
            )

            self.assertEqual(response.status_code, 200)
            rows = finance_app.app.test_client().get("/api/transactions").get_json()["transactions"]
            self.assertEqual(rows[0]["kategori"], "Sağlık")
            self.assertEqual(rows[0]["gider_tipi"], "Zorunlu")
            self.assertEqual(rows[0]["yontem"], "manual")
        finally:
            finance_app.EXTRACTED_PATH = old_path
            if os.path.exists(path):
                os.remove(path)

    def test_risk_score_keeps_investment_horizon_separate(self):
        payload = {
            "risk_tolerance": 0.2,
            "investment_horizon": 1.0,
        }

        self.assertAlmostEqual(finance_app.calculate_risk_score(payload), 0.2)
        self.assertEqual(finance_app.calculate_investment_horizon(payload), 1.0)

    def test_profile_defaults_migrates_legacy_risk_answer(self):
        profile = finance_app._profile_defaults({
            "answers": {
                "loss_reaction": 0.35,
                "investment_horizon": 0.8,
                "volatility_comfort": 1.0,
                "growth_preference": 0.35,
            }
        })

        self.assertEqual(profile["risk_tolerance"], 0.35)
        self.assertEqual(profile["investment_horizon"], 0.8)
        self.assertEqual(set(profile["answers"].keys()), {"risk_tolerance", "investment_horizon"})

    def test_prepare_bank_data_flips_positive_card_expenses(self):
        df = pd.DataFrame(
            [
                {
                    "Tarih": "26.04.2026",
                    "Açıklama": "Starbucks - Sabah Kahvesi",
                    "Tutar": "125,00",
                    "Bakiye": "28.345,50",
                    "Kategori": "Gelir",
                    "Gider Tipi": "Gelir",
                    "Sınıflandırma Güveni": 0.70,
                    "Sınıflandırma Yöntemi": "amount",
                    "Sınıflandırma Kuralı": "positive_amount",
                },
                {
                    "Tarih": "24.04.2026",
                    "Açıklama": "Aylık Faiz Getirisi",
                    "Tutar": "2.140,00",
                    "Bakiye": "152.140,00",
                    "Kategori": "Gelir",
                    "Gider Tipi": "Gelir",
                    "Sınıflandırma Güveni": 0.95,
                    "Sınıflandırma Yöntemi": "rule",
                    "Sınıflandırma Kuralı": "income_signal",
                },
            ]
        )

        prepared = finance_app.prepare_bank_data(df)

        self.assertEqual(prepared.loc[0, "Kategori"], "Gıda")
        self.assertEqual(prepared.loc[0, "Tutar"], -125.0)
        self.assertEqual(prepared.loc[1, "Kategori"], "Gelir")
        self.assertEqual(prepared.loc[1, "Tutar"], 2140.0)

    def test_financial_metrics_include_debt_and_emergency_buffer(self):
        df = pd.DataFrame(
            [
                {
                    "Tarih": pd.Timestamp("2026-04-01"),
                    "Tutar": 10000.0,
                    "Kategori": "Gelir",
                    "Gider Tipi": "Gelir",
                    "Bakiye": 15000.0,
                    "Ay": pd.Period("2026-04"),
                },
                {
                    "Tarih": pd.Timestamp("2026-04-03"),
                    "Tutar": -2000.0,
                    "Kategori": "Borç/Kredi/Kart",
                    "Gider Tipi": "Zorunlu",
                    "Bakiye": 13000.0,
                    "Ay": pd.Period("2026-04"),
                },
                {
                    "Tarih": pd.Timestamp("2026-04-04"),
                    "Tutar": -1000.0,
                    "Kategori": "Gıda",
                    "Gider Tipi": "Kısılabilir",
                    "Bakiye": 12000.0,
                    "Ay": pd.Period("2026-04"),
                },
                {
                    "Tarih": pd.Timestamp("2026-04-05"),
                    "Tutar": -4.0,
                    "Kategori": "Banka Ücreti",
                    "Gider Tipi": "Zorunlu",
                    "Bakiye": 11996.0,
                    "Ay": pd.Period("2026-04"),
                },
            ]
        )

        metrics = finance_app.calculate_financial_metrics(df)

        self.assertAlmostEqual(metrics["borc_yuku"], 0.2)
        self.assertAlmostEqual(metrics["esneklik"], 0.8)
        self.assertAlmostEqual(metrics["toplam_gider"], 3000.0)
        self.assertEqual(metrics["mikro_adedi"], 1)
        self.assertEqual(metrics["mikro_toplam"], 4.0)
        self.assertGreater(metrics["acil_tampon"], 0.99)

    def test_raw_transaction_rows_hide_micro_transactions(self):
        fd, path = tempfile.mkstemp(suffix=".csv")
        os.close(fd)
        old_path = finance_app.EXTRACTED_PATH
        try:
            pd.DataFrame(
                [
                    {
                        "Tarih": "26.04.2026",
                        "Açıklama": "BSMV",
                        "Tutar": "-0,20",
                        "Bakiye": "1.000,00",
                        "Kategori": "Banka Ücreti",
                        "Gider Tipi": "Zorunlu",
                        "Sınıflandırma Güveni": 0.95,
                        "Sınıflandırma Yöntemi": "rule",
                        "Sınıflandırma Kuralı": "bank_fee",
                    },
                    {
                        "Tarih": "26.04.2026",
                        "Açıklama": "MARKET",
                        "Tutar": "-240,00",
                        "Bakiye": "760,00",
                        "Kategori": "Gıda",
                        "Gider Tipi": "Kısılabilir",
                        "Sınıflandırma Güveni": 0.9,
                        "Sınıflandırma Yöntemi": "rule",
                        "Sınıflandırma Kuralı": "food_merchant",
                    },
                ]
            ).to_csv(path, index=False)
            finance_app.EXTRACTED_PATH = path

            visible_rows = finance_app._raw_transaction_rows()
            all_rows = finance_app._raw_transaction_rows(include_micro=True)

            self.assertEqual(len(visible_rows), 1)
            self.assertEqual(visible_rows[0]["aciklama"], "MARKET")
            self.assertEqual(len(all_rows), 2)
        finally:
            finance_app.EXTRACTED_PATH = old_path
            if os.path.exists(path):
                os.remove(path)

    def test_category_analysis_builds_specific_advice(self):
        df = pd.DataFrame(
            [
                {
                    "Tarih": pd.Timestamp("2026-04-01"),
                    "Tutar": 10000.0,
                    "Kategori": "Gelir",
                    "Gider Tipi": "Gelir",
                    "Bakiye": 15000.0,
                    "Ay": pd.Period("2026-04"),
                },
                {
                    "Tarih": pd.Timestamp("2026-04-02"),
                    "Tutar": -1800.0,
                    "Kategori": "Gıda",
                    "Gider Tipi": "Kısılabilir",
                    "Bakiye": 13200.0,
                    "Ay": pd.Period("2026-04"),
                },
                {
                    "Tarih": pd.Timestamp("2026-04-03"),
                    "Tutar": -900.0,
                    "Kategori": "Ulaştırma",
                    "Gider Tipi": "Kısılabilir",
                    "Bakiye": 12300.0,
                    "Ay": pd.Period("2026-04"),
                },
                {
                    "Tarih": pd.Timestamp("2026-04-04"),
                    "Tutar": -4.0,
                    "Kategori": "Banka Ücreti",
                    "Gider Tipi": "Zorunlu",
                    "Bakiye": 12296.0,
                    "Ay": pd.Period("2026-04"),
                },
            ]
        )

        metrics = finance_app.calculate_financial_metrics(df)
        kat_gider = metrics["giderler"].groupby("Kategori")["Abs"].sum().sort_values(ascending=False)
        analysis = finance_app.build_category_analysis(metrics, kat_gider)
        food_item = next(item for item in analysis["items"] if item["category"] == "Gıda")

        self.assertEqual(food_item["title"], "Gıda Bütçesi")
        self.assertGreater(food_item["percent"], 60)
        self.assertGreater(food_item["target_saving"], 0)
        self.assertIn("yatırım bütçesi", analysis["summary"])
        self.assertIn("mikro tahsilatlar", analysis["summary"])

    def test_pdf_parser_accepts_slash_dates_and_table_columns(self):
        tokens = [
            OcrToken("26/04/2026", 0.99, (10, 10, 90, 30), 1),
            OcrToken("88412033", 0.99, (120, 10, 190, 30), 1),
            OcrToken("Starbucks - Sabah Kahvesi", 0.99, (220, 10, 430, 30), 1),
            OcrToken("TL", 0.99, (460, 10, 490, 30), 1),
            OcrToken("-125,00", 0.99, (520, 10, 590, 30), 1),
            OcrToken("28.345,50", 0.99, (620, 10, 700, 30), 1),
            OcrToken("01/04/2026", 0.99, (10, 45, 90, 65), 1),
            OcrToken("88411290", 0.99, (120, 45, 190, 65), 1),
            OcrToken("Dönem Başı Devir Bakiyesi", 0.99, (220, 45, 430, 65), 1),
            OcrToken("TL", 0.99, (460, 45, 490, 65), 1),
            OcrToken("15.824,48", 0.99, (520, 45, 590, 65), 1),
            OcrToken("15.824,48", 0.99, (620, 45, 700, 65), 1),
        ]

        transactions = parse_transactions(tokens)

        self.assertEqual(len(transactions), 1)
        self.assertEqual(transactions[0].tarih, "26.04.2026")
        self.assertEqual(transactions[0].aciklama, "Starbucks - Sabah Kahvesi")
        self.assertEqual(transactions[0].tutar, -125.0)
        self.assertEqual(transactions[0].bakiye, 28345.5)

    def test_api_data_does_not_emit_nan_for_empty_extracted_file(self):
        fd, path = tempfile.mkstemp(suffix=".csv")
        os.close(fd)
        old_path = finance_app.EXTRACTED_PATH
        try:
            pd.DataFrame(
                columns=[
                    "Tarih",
                    "Açıklama",
                    "Kategori",
                    "Gider Tipi",
                    "Tutar",
                    "Bakiye",
                    "Sınıflandırma Güveni",
                    "Sınıflandırma Yöntemi",
                    "Sınıflandırma Kuralı",
                ]
            ).to_csv(path, index=False)
            finance_app.EXTRACTED_PATH = path

            response = finance_app.app.test_client().get("/api/data")

            self.assertEqual(response.status_code, 200)
            self.assertNotIn(b"NaN", response.data)
            self.assertEqual(response.get_json()["banka"]["aylik_gelir_ort"], 0.0)
        finally:
            finance_app.EXTRACTED_PATH = old_path
            if os.path.exists(path):
                os.remove(path)


if __name__ == "__main__":
    unittest.main()
