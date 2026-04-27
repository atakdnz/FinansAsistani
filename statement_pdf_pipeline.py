"""
PDF bank statement pipeline.

Flow:
1. Render PDF pages to images with pypdfium2.
2. Extract OCR text boxes with PaddleOCR.
3. Parse statement-like rows into transactions.
4. Classify with deterministic rules, then ask embedding only for low-confidence
   rows when `.venv_embed` is available.

Run:
    PADDLE_PDX_CACHE_HOME=.paddlex_cache .venv_paddle/bin/python statement_pdf_pipeline.py Hesap_Hareketleri_26042026-1.pdf
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from transaction_classifier import ClassificationResult, classify_transaction, parse_amount


BASE_DIR = Path(__file__).resolve().parent
PADDLE_CACHE_DIR = BASE_DIR / ".paddlex_cache"
EMBED_PYTHON = BASE_DIR / ".venv_embed" / "bin" / "python"
EMBED_HELPER = BASE_DIR / "embedding_classifier.py"
AMOUNT_RE = re.compile(r"-?\d{1,3}(?:[.,]\d{3})*[.,]\d{2}|-?\d+[.,]\d{2}")
DATE_RE = re.compile(r"\b\d{2}\.\d{2}\.\d{4}\b")


@dataclass
class OcrToken:
    text: str
    score: float
    box: tuple[float, float, float, float]
    page: int

    @property
    def x1(self) -> float:
        return self.box[0]

    @property
    def y1(self) -> float:
        return self.box[1]

    @property
    def x2(self) -> float:
        return self.box[2]

    @property
    def y2(self) -> float:
        return self.box[3]

    @property
    def y_mid(self) -> float:
        return (self.y1 + self.y2) / 2


@dataclass
class ParsedTransaction:
    tarih: str
    aciklama: str
    tutar: float
    bakiye: float | None
    page: int
    raw_text: str


@dataclass
class ClassifiedTransaction:
    tarih: str
    aciklama: str
    tutar: float
    bakiye: float | None
    kategori: str
    gider_tipi: str
    siniflandirma_guveni: float
    siniflandirma_yontemi: str
    siniflandirma_kurali: str
    embedding_kategori: str | None
    embedding_guveni: float | None
    raw_text: str


def _configure_paddle_cache() -> None:
    os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(PADDLE_CACHE_DIR))
    PADDLE_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def render_pdf_pages(pdf_path: Path, dpi_scale: float = 3.0) -> list[Path]:
    import pypdfium2 as pdfium

    out_dir = Path(tempfile.mkdtemp(prefix="statement_pages_"))
    pdf = pdfium.PdfDocument(str(pdf_path))
    image_paths: list[Path] = []
    for index in range(len(pdf)):
        page = pdf[index]
        bitmap = page.render(scale=dpi_scale)
        image = bitmap.to_pil()
        image_path = out_dir / f"page_{index + 1}.png"
        image.save(image_path)
        image_paths.append(image_path)
    return image_paths


def create_ocr():
    _configure_paddle_cache()
    from paddleocr import PaddleOCR

    return PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        text_detection_model_name="PP-OCRv5_mobile_det",
        text_recognition_model_name="en_PP-OCRv5_mobile_rec",
        text_det_limit_side_len=1600,
        text_det_limit_type="max",
    )


def extract_ocr_tokens(pdf_path: Path) -> list[OcrToken]:
    ocr = create_ocr()
    tokens: list[OcrToken] = []
    for page_index, image_path in enumerate(render_pdf_pages(pdf_path), start=1):
        for page_result in ocr.predict(str(image_path)):
            texts = page_result.get("rec_texts", [])
            scores = page_result.get("rec_scores", [])
            boxes = page_result.get("rec_boxes", [])
            for text, score, box in zip(texts, scores, boxes):
                if not str(text).strip():
                    continue
                x1, y1, x2, y2 = [float(v) for v in box]
                tokens.append(OcrToken(str(text).strip(), float(score), (x1, y1, x2, y2), page_index))
    return tokens


def group_tokens_into_lines(tokens: list[OcrToken], y_threshold: float = 22.0) -> list[list[OcrToken]]:
    lines: list[list[OcrToken]] = []
    for page in sorted({token.page for token in tokens}):
        page_tokens = sorted([token for token in tokens if token.page == page], key=lambda item: (item.y_mid, item.x1))
        for token in page_tokens:
            if not lines or lines[-1][0].page != page or abs(lines[-1][0].y_mid - token.y_mid) > y_threshold:
                lines.append([token])
            else:
                lines[-1].append(token)
    return [sorted(line, key=lambda item: item.x1) for line in lines]


def parse_transactions(tokens: list[OcrToken]) -> list[ParsedTransaction]:
    transactions: list[ParsedTransaction] = []
    for line in group_tokens_into_lines(tokens):
        text = " ".join(token.text for token in line)
        date_match = DATE_RE.search(text)
        if not date_match:
            if transactions and not AMOUNT_RE.search(text) and line[0].x1 > 350:
                transactions[-1].aciklama = f"{transactions[-1].aciklama} {text}".strip()
                transactions[-1].raw_text = f"{transactions[-1].raw_text} {text}".strip()
            continue
        if text.upper().startswith(("DÖNEM", "DONEM")):
            continue

        amounts = list(AMOUNT_RE.finditer(text))
        if not amounts:
            continue

        amount_match = amounts[-2] if len(amounts) >= 2 else amounts[-1]
        balance_match = amounts[-1] if len(amounts) >= 2 else None
        description = text[date_match.end() : amount_match.start()].strip(" -|")

        transactions.append(
            ParsedTransaction(
                tarih=date_match.group(0),
                aciklama=description,
                tutar=parse_amount(amount_match.group(0)),
                bakiye=parse_amount(balance_match.group(0)) if balance_match else None,
                page=line[0].page,
                raw_text=text,
            )
        )
    return transactions


def _embedding_suggestions(descriptions: list[str]) -> list[dict[str, Any] | None]:
    if not descriptions or not EMBED_PYTHON.exists() or not EMBED_HELPER.exists():
        return [None for _ in descriptions]

    env = os.environ.copy()
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    payload = "".join(json.dumps({"description": item}, ensure_ascii=False) + "\n" for item in descriptions)
    try:
        proc = subprocess.run(
            [str(EMBED_PYTHON), str(EMBED_HELPER), "--json-lines"],
            input=payload,
            text=True,
            capture_output=True,
            check=True,
            cwd=str(BASE_DIR),
            env=env,
            timeout=180,
        )
    except (subprocess.SubprocessError, OSError):
        return [None for _ in descriptions]

    suggestions: list[dict[str, Any] | None] = []
    for line in proc.stdout.splitlines():
        try:
            suggestions.append(json.loads(line))
        except json.JSONDecodeError:
            suggestions.append(None)
    while len(suggestions) < len(descriptions):
        suggestions.append(None)
    return suggestions[: len(descriptions)]


def classify_transactions(transactions: list[ParsedTransaction], embedding_threshold: float = 0.50) -> list[ClassifiedTransaction]:
    deterministic: list[ClassificationResult] = [
        classify_transaction(item.aciklama, item.tutar) for item in transactions
    ]
    low_conf_descriptions = [
        item.aciklama for item, result in zip(transactions, deterministic) if result.confidence < embedding_threshold
    ]
    embedding_results = _embedding_suggestions(low_conf_descriptions)
    embedding_iter = iter(embedding_results)

    classified: list[ClassifiedTransaction] = []
    for item, result in zip(transactions, deterministic):
        embedding = next(embedding_iter) if result.confidence < embedding_threshold else None
        final_category = result.category
        final_method = result.method
        final_rule = result.rule
        final_confidence = result.confidence

        if embedding and embedding.get("category") != "Diğer" and embedding.get("confidence", 0) >= 0.62:
            final_category = str(embedding["category"])
            final_method = "deterministic+embedding"
            final_rule = f"{result.rule};embedding_override"
            final_confidence = min(0.80, float(embedding["confidence"]))

        classified.append(
            ClassifiedTransaction(
                tarih=item.tarih,
                aciklama=item.aciklama,
                tutar=item.tutar,
                bakiye=item.bakiye,
                kategori=final_category,
                gider_tipi=result.expense_type,
                siniflandirma_guveni=round(final_confidence, 4),
                siniflandirma_yontemi=final_method,
                siniflandirma_kurali=final_rule,
                embedding_kategori=embedding.get("category") if embedding else None,
                embedding_guveni=embedding.get("confidence") if embedding else None,
                raw_text=item.raw_text,
            )
        )
    return classified


def write_csv(rows: list[ClassifiedTransaction], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "Tarih",
        "Açıklama",
        "Kategori",
        "Gider Tipi",
        "Tutar",
        "Bakiye",
        "Sınıflandırma Güveni",
        "Sınıflandırma Yöntemi",
        "Sınıflandırma Kuralı",
        "Embedding Kategori",
        "Embedding Güveni",
        "Ham OCR Satırı",
    ]
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "Tarih": row.tarih,
                    "Açıklama": row.aciklama,
                    "Kategori": row.kategori,
                    "Gider Tipi": row.gider_tipi,
                    "Tutar": row.tutar,
                    "Bakiye": row.bakiye,
                    "Sınıflandırma Güveni": row.siniflandirma_guveni,
                    "Sınıflandırma Yöntemi": row.siniflandirma_yontemi,
                    "Sınıflandırma Kuralı": row.siniflandirma_kurali,
                    "Embedding Kategori": row.embedding_kategori or "",
                    "Embedding Güveni": row.embedding_guveni or "",
                    "Ham OCR Satırı": row.raw_text,
                }
            )


def process_pdf(pdf_path: Path, output_path: Path) -> list[ClassifiedTransaction]:
    tokens = extract_ocr_tokens(pdf_path)
    parsed = parse_transactions(tokens)
    classified = classify_transactions(parsed)
    write_csv(classified, output_path)
    return classified


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--output", type=Path, default=BASE_DIR / "extracted_transactions.csv")
    parser.add_argument("--json", action="store_true", help="Print classified rows as JSON")
    args = parser.parse_args()

    rows = process_pdf(args.pdf, args.output)
    if args.json:
        print(json.dumps([asdict(row) for row in rows], ensure_ascii=False, indent=2))
    else:
        print(f"{len(rows)} transactions written to {args.output}")
        for row in rows:
            print(f"{row.tarih} | {row.aciklama} | {row.tutar:.2f} | {row.kategori} | {row.gider_tipi}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
