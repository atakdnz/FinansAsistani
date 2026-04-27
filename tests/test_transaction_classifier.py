import unittest
import os
import tempfile

import pandas as pd

import app as finance_app
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
            ("YEMEKPAY/YEMEK SEPET", "-176,00", "Gıda", "Kısılabilir"),
            ("ECZANE SAGLIK HARCAMASI", "-240,00", "Sağlık", "Zorunlu"),
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
            "loss_reaction": 0.2,
            "investment_horizon": 1.0,
            "volatility_comfort": 0.4,
            "growth_preference": 0.6,
        }

        self.assertAlmostEqual(finance_app.calculate_risk_score(payload), 0.4)
        self.assertEqual(finance_app.calculate_investment_horizon(payload), 1.0)

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
            ]
        )

        metrics = finance_app.calculate_financial_metrics(df)

        self.assertAlmostEqual(metrics["borc_yuku"], 0.2)
        self.assertAlmostEqual(metrics["esneklik"], 0.8)
        self.assertEqual(metrics["acil_tampon"], 1.0)


if __name__ == "__main__":
    unittest.main()
