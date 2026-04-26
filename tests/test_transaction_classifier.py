import unittest

import pandas as pd

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


if __name__ == "__main__":
    unittest.main()
