"""
Deterministic transaction classification for bank statement rows.

The classifier is intentionally rule-first. Small local LLMs were not stable
enough for primary classification, so this module extracts strong merchant and
banking signals before any optional AI layer is considered.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
import unicodedata
from typing import Iterable

import pandas as pd


TR_TRANSLATION = str.maketrans(
    {
        "ç": "c",
        "Ç": "C",
        "ğ": "g",
        "Ğ": "G",
        "ı": "i",
        "I": "I",
        "İ": "I",
        "ö": "o",
        "Ö": "O",
        "ş": "s",
        "Ş": "S",
        "ü": "u",
        "Ü": "U",
    }
)


@dataclass(frozen=True)
class ClassificationResult:
    category: str
    expense_type: str
    confidence: float
    method: str
    rule: str
    normalized_description: str


def normalize_text(value: object) -> str:
    text = "" if value is None else str(value)
    text = text.translate(TR_TRANSLATION)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.upper()
    text = re.sub(r"[^A-Z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def parse_amount(value: object) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if isinstance(value, float) and math.isnan(value):
            return 0.0
        return float(value)

    raw = str(value).strip()
    if not raw:
        return 0.0

    negative = raw.startswith("-") or raw.startswith("(")
    cleaned = re.sub(r"[^0-9,.\-]", "", raw).replace("-", "")
    if not cleaned:
        return 0.0

    last_comma = cleaned.rfind(",")
    last_dot = cleaned.rfind(".")

    if last_comma >= 0 and last_dot >= 0:
        decimal_sep = "," if last_comma > last_dot else "."
    elif last_comma >= 0:
        decimal_sep = "," if len(cleaned) - last_comma - 1 <= 2 else ""
    elif last_dot >= 0:
        parts = cleaned.split(".")
        decimal_sep = "." if len(parts[-1]) <= 2 else ""
    else:
        decimal_sep = ""

    if decimal_sep:
        integer_part, decimal_part = cleaned.rsplit(decimal_sep, 1)
        integer_part = re.sub(r"[^0-9]", "", integer_part)
        decimal_part = re.sub(r"[^0-9]", "", decimal_part)
        normalized = f"{integer_part}.{decimal_part}"
    else:
        normalized = re.sub(r"[^0-9]", "", cleaned)

    if not normalized or normalized == ".":
        return 0.0

    amount = float(normalized)
    return -amount if negative else amount


def _contains_any(text: str, terms: Iterable[str]) -> bool:
    return any(term in text for term in terms)


def classify_transaction(description: object, amount: object = 0.0) -> ClassificationResult:
    normalized = normalize_text(description)
    parsed_amount = parse_amount(amount)

    if parsed_amount > 0:
        if _contains_any(normalized, ("KREDI YURTLAR KURUMU", "KYK", "OGRENIM KREDISI")):
            return _result("Gelir", "Gelir", 0.90, "rule", "student_loan_income", normalized)
        if _contains_any(normalized, ("ATM YATAN", "MAAS", "UCRET", "FREELANCE", "TEMETTU", "IADE")):
            return _result("Gelir", "Gelir", 0.95, "rule", "income_signal", normalized)
        return _result("Gelir", "Gelir", 0.70, "amount", "positive_amount", normalized)

    if _contains_any(normalized, ("BSMV", "KOMISYON", "MASRAF", "HAVALE UCRETI", "EFT UCRETI", "FAST UCRETI")):
        return _result("Banka Ücreti", "Zorunlu", 0.95, "rule", "bank_fee", normalized)

    if _contains_any(normalized, ("KK OTOMATIK ODEME", "KREDI KARTI", "KART ODEME", "KREDI ODEMESI", "BORC ODEME", "TAKSIT")) or normalized == "KREDI":
        return _result("Borç/Kredi/Kart", "Zorunlu", 0.90, "rule", "debt_or_card_payment", normalized)

    if _contains_any(
        normalized,
        (
            "GETIR",
            "YEMEKPAY",
            "YEMEK SEPET",
            "MIGROS",
            "BIM",
            "A101",
            "SOK MARKET",
            "CARREFOUR",
            "MARKET",
            "RESTORAN",
            "DIS YEMEK",
            "KASAP",
            "MANAV",
            "CIG KOFTE",
            "KAHVE",
            "STARBUCKS",
        ),
    ):
        return _result("Gıda", "Kısılabilir", 0.90, "rule", "food_merchant", normalized)

    if _contains_any(
        normalized,
        (
            "KIRA",
            "EV KREDISI",
            "ELEKTRIK",
            "SU FATUR",
            "DOGALGAZ",
            "INTERNET",
            "AIDAT",
            "TELEKOM",
            "TURKCELL",
            "VODAFONE",
            "MILLENICOM",
        ),
    ):
        return _result("Konut/Fatura", "Zorunlu", 0.90, "rule", "housing_or_bill", normalized)

    if _contains_any(
        normalized,
        ("SHELL", "OPET", "BP", "PETROL", "AKARYAKIT", "UBER", "TAKSI", "IETT", "HGS", "OTOYOL", "METRO", "ARAC BAKIM"),
    ):
        return _result("Ulaştırma", "Zorunlu", 0.85, "rule", "transport", normalized)

    if _contains_any(normalized, ("OKUL", "UNIVERSITE", "DERSHANE", "KURS", "KIRTASIYE", "UDEMY", "COURSERA")):
        return _result("Eğitim", "Zorunlu", 0.82, "rule", "education", normalized)

    if _contains_any(normalized, ("NETFLIX", "SPOTIFY", "STEAM", "SINEMA", "TATIL", "HOBI")):
        return _result("İsteğe Bağlı", "İsteğe Bağlı", 0.88, "rule", "leisure_subscription", normalized)

    if _contains_any(normalized, ("TRENDYOL", "HEPSIBURADA", "AMAZON", "N11", "LCWAIKIKI", "ZARA", "SANAL POS ALISVERIS")):
        return _result("Alışveriş", "Kısılabilir", 0.75, "rule", "shopping", normalized)

    if parsed_amount < 0:
        return _result("Diğer", "Belirsiz", 0.35, "fallback", "negative_unknown", normalized)

    return _result("Diğer", "Belirsiz", 0.20, "fallback", "unknown", normalized)


def classify_dataframe(
    df: pd.DataFrame,
    description_col: str = "Açıklama",
    amount_col: str = "Tutar",
    overwrite_category: bool = True,
) -> pd.DataFrame:
    result_df = df.copy()
    classifications = [
        classify_transaction(row.get(description_col, ""), row.get(amount_col, 0.0))
        for _, row in result_df.iterrows()
    ]

    if overwrite_category or "Kategori" not in result_df.columns:
        result_df["Kategori"] = [item.category for item in classifications]

    result_df["Gider Tipi"] = [item.expense_type for item in classifications]
    result_df["Sınıflandırma Güveni"] = [item.confidence for item in classifications]
    result_df["Sınıflandırma Yöntemi"] = [item.method for item in classifications]
    result_df["Sınıflandırma Kuralı"] = [item.rule for item in classifications]
    return result_df


def _result(
    category: str,
    expense_type: str,
    confidence: float,
    method: str,
    rule: str,
    normalized_description: str,
) -> ClassificationResult:
    return ClassificationResult(
        category=category,
        expense_type=expense_type,
        confidence=confidence,
        method=method,
        rule=rule,
        normalized_description=normalized_description,
    )
