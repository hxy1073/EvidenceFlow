"""Build a page-aware BM25 corpus from completed MinerU Markdown sidecars.

The JSONL is built beside the live file and replaced only after all source
PDFs have matching Markdown and block metadata. The original FAISS index stays
available as a rollback until a new vector index is built.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "pdf-data" / "mineru-output"
MARKDOWN = OUTPUT / "markdown"
TARGET = ROOT / "rag-service" / "rag" / "mineru_corpus.jsonl"
PDF_DIR = ROOT / "pdf-data" / "论文"
MAX_TEXT = 780


def plain_text(block: dict) -> str:
    text = block.get("text") or block.get("table_body") or ""
    text = re.sub(r"<[^>]*>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def emit_chunks(source: str, page: int, blocks: list[dict], image_names: list[str], parser: str = "mineru"):
    title = Path(source).stem
    prefix = f"# {title}\n## 第 {page} 页\n\n"
    current: list[tuple[str, list[float] | None]] = []
    length = 0

    def result():
        if not current:
            return None
        return {"page_content": prefix + "\n".join(text for text, _ in current),
                "metadata": {"source_file": f"{title}.md", "source_pdf": source,
                             "page": page, "parser": parser,
                             "headers": [[1, title], [2, f"第 {page} 页"]],
                             "bboxes": [bbox for _, bbox in current if bbox is not None],
                             "img_path": " <DELIMITER> ".join(image_names) if image_names
                             else "NO_IMAGE_AVAILABLE", "img_count": len(image_names)}}

    for block in blocks:
        text = plain_text(block)
        if not text:
            continue
        bbox = block["bbox"] if block.get("type") == "text" else None
        for start in range(0, len(text), MAX_TEXT):
            part = text[start:start + MAX_TEXT]
            if current and length + len(part) > MAX_TEXT:
                yield result()
                current, length = [], 0
            current.append((part, bbox))
            length += len(part)
    if current:
        yield result()


def build(expected: int | None = None) -> None:
    sources = sorted(PDF_DIR.glob("*.pdf"))
    if expected is None:
        expected = len(sources)
    if expected == 0:
        raise RuntimeError("No source PDFs found")
    if len(sources) != expected:
        raise RuntimeError(f"Expected {expected} PDFs; found {len(sources)}")
    missing = [p.name for p in sources if not (MARKDOWN / f"{p.stem}.md").exists()
               or not (MARKDOWN / f"{p.stem}.blocks.json").exists()]
    if missing:
        raise RuntimeError(f"MinerU corpus is incomplete: {len(missing)} PDFs missing parsed files")

    TARGET.parent.mkdir(parents=True, exist_ok=True)
    staging = TARGET.with_suffix(".jsonl.new")
    total = image_count = 0
    with staging.open("w", encoding="utf-8") as stream:
        for pdf in sources:
            sidecar = MARKDOWN / f"{pdf.stem}.blocks.json"
            data = json.loads(sidecar.read_text(encoding="utf-8"))
            if data.get("source_pdf") != pdf.name:
                raise RuntimeError(f"Source mismatch: {sidecar}")
            grouped: dict[int, list[dict]] = defaultdict(list)
            for block in data["blocks"]:
                grouped[int(block["page"])].append(block)
            if not grouped:
                raise RuntimeError(f"No located text blocks: {sidecar}")
            for page, blocks in sorted(grouped.items()):
                images = []
                for block in blocks:
                    raw_ref = block.get("image_path")
                    if not raw_ref:
                        continue
                    raw_dir = (OUTPUT / block["raw"]).resolve()
                    image = (raw_dir / raw_ref).resolve()
                    if not image.is_relative_to(raw_dir) or not image.is_file():
                        continue
                    # MinerU image names are content hashes; avoid ambiguity if a
                    # different export happens to use the same basename.
                    image_name = f"{Path(block['raw']).name}-{image.name}"
                    image_count += 1
                    images.append(image_name)
                images = list(dict.fromkeys(images))
                for chunk in emit_chunks(pdf.name, page, blocks, images,
                                         data.get("parser", "mineru")):
                    stream.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                    total += 1
    if total < expected:
        staging.unlink(missing_ok=True)
        raise RuntimeError(f"Too few retrieval chunks: {total}")
    staging.replace(TARGET)
    print(f"Installed {total} chunks from {len(sources)} PDFs; linked {image_count} image blocks")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected", type=int)
    build(parser.parse_args().expected)
