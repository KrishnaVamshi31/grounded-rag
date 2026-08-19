"""
Step 1 — Ingestion.

Turns PDFs into chunks that carry enough metadata to cite later.

Usage:
    python ingest.py data/*.pdf --out chunks.jsonl
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import pymupdf

TARGET_CHARS = 900     # rough chunk size
OVERLAP_CHARS = 150    # tail of chunk N repeated at head of chunk N+1
HEADING_RATIO = 1.15   # a block this much bigger than body text is a heading
HEADING_MAX_CHARS = 120


def page_lines(page):
    """Return [(text, max_font_size, is_bold)] for each LINE on the page.

    We use 'dict' extraction rather than plain get_text() because we need the
    font size — that's the only cheap signal for what is a heading.

    Line level, not block level: PDF producers routinely glue a heading into
    the same block as the paragraph above it, and a block-level check then
    misses the heading entirely because the merged text is too long.
    """
    out = []
    for block in page.get_text("dict")["blocks"]:
        if block.get("type") != 0:      # 0 = text, 1 = image
            continue
        for line in block["lines"]:
            spans = line["spans"]
            if not spans:
                continue
            text = re.sub(r"\s+", " ", "".join(s["text"] for s in spans)).strip()
            if not text:
                continue
            size = max(round(s["size"], 1) for s in spans)
            bold = any("bold" in s["font"].lower() for s in spans)
            out.append((text, size, bold, line["bbox"][1]))
    return out


def normalize(text):
    """Strip digits so 'The Encoder | 5' and 'The Encoder | 9' collapse."""
    return re.sub(r"\d+", "#", text).strip()


def boilerplate(doc, margin=0.08, min_share=0.2):
    """Text that repeats in the top/bottom margin across pages = header/footer."""
    counter = Counter()
    for page in doc:
        height = page.rect.height
        for text, _, _, y in page_lines(page):
            if y < height * margin or y > height * (1 - margin):
                counter[normalize(text)] += 1
    threshold = max(3, int(len(doc) * min_share))
    return {k for k, v in counter.items() if v >= threshold}


def toc_sections(doc):
    """(page, title) from the PDF's own outline, if it has one."""
    return [(page, title) for _level, title, page in doc.get_toc() if page > 0]


def section_for_page(sections, page_number):
    current = None
    for page, title in sections:
        if page <= page_number:
            current = title
        else:
            break
    return current


def body_font_size(doc):
    """Most common font size, weighted by how much text uses it = body text."""
    counter = Counter()
    for page in doc:
        for text, size, _, _y in page_lines(page):
            counter[size] += len(text)
    return counter.most_common(1)[0][0] if counter else 10.0


def split_with_overlap(text, target=TARGET_CHARS, overlap=OVERLAP_CHARS):
    """Split on sentence boundaries, packing up to `target` chars per chunk."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks, current = [], ""
    for sentence in sentences:
        if current and len(current) + len(sentence) + 1 > target:
            chunks.append(current.strip())
            current = current[-overlap:] + " " + sentence
        else:
            current = f"{current} {sentence}".strip()
    if current.strip():
        chunks.append(current.strip())
    return chunks


def ingest_pdf(path):
    """Yield chunk records for one PDF."""
    doc = pymupdf.open(path)
    body = body_font_size(doc)
    source = Path(path).name
    junk = boilerplate(doc)              # running headers/footers to discard
    sections = toc_sections(doc)         # trust the PDF's outline when it has one
    heading = None                       # carries forward until the next heading

    for page_index, page in enumerate(doc):
        page_number = page_index + 1     # 1-based, matches what a human sees
        records, buffer = [], []
        if sections:
            heading = section_for_page(sections, page_number)

        def flush(buffer=buffer, records=records, page_number=page_number):
            """Chunk whatever prose has piled up under the current heading."""
            if not buffer:
                return
            joined = " ".join(buffer)
            buffer.clear()
            for piece in split_with_overlap(joined):
                records.append({
                    "source": source,
                    "page": page_number,
                    "heading": heading,
                    "text": piece,
                    "chars": len(piece),
                })

        for text, size, bold, _y in page_lines(page):
            if normalize(text) in junk:  # running header/footer — never content
                continue
            is_heading = (
                not sections
                and len(text) <= HEADING_MAX_CHARS
                and len(re.findall(r"[A-Za-z]", text)) >= 4
                and (size >= body * HEADING_RATIO or (bold and size > body))
            )
            if is_heading:
                flush()                  # close out the previous section first
                heading = text
            else:
                buffer.append(text)
        flush()

        for record in records:
            yield record

    doc.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pdfs", nargs="+")
    parser.add_argument("--out", default="chunks.jsonl")
    args = parser.parse_args()

    count = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for pdf in args.pdfs:
            for record in ingest_pdf(pdf):
                record["chunk_id"] = f"c{count:05d}"
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
            print(f"  {pdf}")
    print(f"{count} chunks -> {args.out}")


if __name__ == "__main__":
    main()
