"""
Optional embedding-based category suggestion.

This module is intentionally separate from the main Flask app. It is meant to
run in the embedding environment (`.venv_embed`) where sentence-transformers and
the local/Hugging Face embedding model are installed.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_NAME = "google/embeddinggemma-300m"

CATEGORY_PROTOTYPES = {
    "Gelir": [
        "salary payment income cash deposit scholarship refund transfer incoming money",
        "maas odemesi atm yatan kart kyk ogrenim kredisi gelir iade temettu freelance",
    ],
    "Banka Ücreti": [
        "bank fee commission tax transfer fee bsmv eft havale masraf",
        "komisyon tahsilat bsmv banka ucreti fast eft havale ucreti",
    ],
    "Gıda": [
        "food grocery restaurant meal delivery supermarket coffee",
        "getir yemeksepeti yemek pay market migros bim a101 restoran cig kofte kahve",
    ],
    "Konut/Fatura": [
        "rent mortgage utility bill electricity water natural gas internet dues",
        "kira ev kredisi elektrik su dogalgaz internet aidat fatura",
    ],
    "Ulaştırma": [
        "transport fuel taxi public transit highway car maintenance",
        "akaryakit petrol shell opet uber taksi iett hgs otoyol arac bakim",
    ],
    "Eğitim": [
        "education school university course stationery online learning",
        "okul universite kurs dershane kirtasiye egitim udemy coursera",
    ],
    "Borç/Kredi/Kart": [
        "credit card payment loan debt installment card statement payment",
        "kk otomatik odeme kredi karti kart odeme borc kredi taksit",
    ],
    "Alışveriş": [
        "shopping ecommerce clothing online purchase marketplace",
        "trendyol hepsiburada amazon n11 sanal pos alisveris magaza giyim",
    ],
    "İsteğe Bağlı": [
        "entertainment subscription holiday cinema game hobby music video",
        "netflix spotify steam sinema tatil hobi eglence abonelik",
    ],
    "Diğer": [
        "other uncategorized unknown transaction transfer note",
        "diger bilinmeyen aciklama net olmayan banka hareketi",
    ],
}


@dataclass(frozen=True)
class EmbeddingSuggestion:
    category: str
    confidence: float
    score: float
    method: str = "embedding"


@lru_cache(maxsize=1)
def _load_model() -> SentenceTransformer:
    return SentenceTransformer(MODEL_NAME)


@lru_cache(maxsize=1)
def _prototype_vectors() -> tuple[list[str], np.ndarray]:
    labels = list(CATEGORY_PROTOTYPES)
    texts = ["; ".join(CATEGORY_PROTOTYPES[label]) for label in labels]
    model = _load_model()
    vectors = model.encode(texts, normalize_embeddings=True)
    return labels, np.asarray(vectors)


def suggest_category(description: str) -> EmbeddingSuggestion:
    labels, prototypes = _prototype_vectors()
    model = _load_model()
    query = model.encode([description], normalize_embeddings=True)
    scores = np.asarray(query) @ prototypes.T
    best_idx = int(np.argmax(scores[0]))
    score = float(scores[0][best_idx])
    confidence = max(0.0, min(0.95, (score + 1.0) / 2.0))
    return EmbeddingSuggestion(labels[best_idx], round(confidence, 4), round(score, 4))


def _serve_json_lines() -> int:
    for line in sys.stdin:
        if not line.strip():
            continue
        payload = json.loads(line)
        suggestion = suggest_category(payload.get("description", ""))
        print(json.dumps(suggestion.__dict__, ensure_ascii=False), flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("description", nargs="?", help="Transaction description")
    parser.add_argument("--json-lines", action="store_true", help="Read JSON lines from stdin")
    args = parser.parse_args()

    if args.json_lines:
        return _serve_json_lines()
    if not args.description:
        parser.error("description is required unless --json-lines is used")

    suggestion = suggest_category(args.description)
    print(json.dumps(suggestion.__dict__, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
