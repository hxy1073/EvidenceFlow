"""Resumable MinerU parsing of the canonical PDFs in pdf-data/论文."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import httpx
import pymupdf

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "论文"
OUT = ROOT / "mineru-output"
MANIFEST = OUT / "manifest.json"
API = "https://mineru.net/api/v4"


def api_key() -> str:
    for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
        if line.startswith("MINERU_API_KEY="):
            return line.split("=", 1)[1].strip()
    raise RuntimeError("MINERU_API_KEY is missing")


def save(manifest: dict) -> None:
    OUT.mkdir(exist_ok=True)
    temporary = MANIFEST.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(MANIFEST)


def register_sources(manifest: dict) -> dict:
    changed = False
    for path in sorted(SOURCE.glob("*.pdf")):
        doc_id = hashlib.sha256(path.name.encode("utf-8")).hexdigest()[:16]
        existing = manifest["documents"].get(doc_id)
        if existing:
            if existing["source"] != path.name:
                raise RuntimeError(f"Manifest ID collision: {path.name}")
            continue
        with pymupdf.open(path) as pdf:
            pages = len(pdf)
        ids = []
        for start in range(1, pages + 1, 200):
            end = min(start + 199, pages)
            job_id = f"{doc_id}-{start:04d}-{end:04d}"
            manifest["jobs"][job_id] = {
                "id": job_id, "doc_id": doc_id, "source": path.name,
                "start": start, "end": end, "status": "pending",
            }
            ids.append(job_id)
        manifest["documents"][doc_id] = {"source": path.name, "pages": pages, "jobs": ids}
        changed = True
    if changed:
        save(manifest)
    return manifest


def load() -> dict:
    if MANIFEST.exists():
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    else:
        manifest = {"documents": {}, "jobs": {}, "batches": {}}
    return register_sources(manifest)


def client() -> httpx.Client:
    return httpx.Client(
        headers={"Authorization": f"Bearer {api_key()}"},
        timeout=httpx.Timeout(90, connect=15),
        follow_redirects=True,
    )


def check(response: httpx.Response) -> dict:
    response.raise_for_status()
    data = response.json()
    if data.get("code") != 0:
        raise RuntimeError(f"MinerU API error {data.get('code')}: {data.get('msg')}")
    return data["data"]


def submit(manifest: dict, limit: int | None = None) -> None:
    # A process can stop after URLs are issued but before the upload finishes.
    # Signed URLs are not persisted, so issue a fresh URL for these jobs.
    pending = [job for job in manifest["jobs"].values()
               if (job["status"] == "pending" and not job.get("batch_id"))
               or job["status"] in ("issued", "upload_failed")]
    if limit is not None:
        pending = pending[:limit]
    if not pending:
        print("No pending jobs", flush=True)
        return
    with client() as api:
        for offset in range(0, len(pending), 30):
            group = pending[offset:offset + 30]
            payload = {
                "files": [{"name": f"{job['id']}.pdf", "data_id": job["id"],
                           "page_ranges": f"{job['start']}-{job['end']}"} for job in group],
                "model_version": "vlm", "enable_table": True, "enable_formula": True,
                "language": "ch",
            }
            result = check(api.post(f"{API}/file-urls/batch", json=payload))
            batch_id = result["batch_id"]
            urls = result["file_urls"]
            if len(urls) != len(group):
                raise RuntimeError("MinerU returned a mismatched number of upload URLs")
            manifest["batches"][batch_id] = [job["id"] for job in group]
            for job in group:
                job["batch_id"] = batch_id
                job["status"] = "issued"
            save(manifest)
            print(f"Batch {batch_id}: {len(group)} files", flush=True)
            for job, url in zip(group, urls):
                try:
                    with (SOURCE / job["source"]).open("rb") as stream:
                        with httpx.Client(timeout=httpx.Timeout(300, connect=20), follow_redirects=True) as uploader:
                            response = uploader.put(url, content=stream)
                            response.raise_for_status()
                    job["status"] = "uploaded"
                    print(f"Uploaded {job['id']} ({job['start']}-{job['end']})", flush=True)
                except Exception as exc:
                    job["status"] = "upload_failed"
                    job["error"] = str(exc)[:250]
                    print(f"Upload failed {job['id']}: {type(exc).__name__}", flush=True)
                save(manifest)


def safe_extract(data: bytes, target: Path) -> Path:
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        for member in archive.infolist():
            name = Path(member.filename)
            if name.is_absolute() or ".." in name.parts:
                raise ValueError("Unsafe path in MinerU archive")
            output = target / name
            if member.is_dir():
                output.mkdir(parents=True, exist_ok=True)
            else:
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_bytes(archive.read(member))
    markdowns = list(target.rglob("full.md"))
    if not markdowns:
        markdowns = list(target.rglob("*.md"))
    if len(markdowns) != 1:
        raise ValueError(f"Expected one Markdown file; found {len(markdowns)}")
    return markdowns[0]


def assemble(manifest: dict, doc_id: str) -> None:
    document = manifest["documents"][doc_id]
    if document.get("assembled"):
        return
    jobs = [manifest["jobs"][job_id] for job_id in document["jobs"]]
    if not all(job["status"] == "done" for job in jobs):
        return
    output = OUT / "markdown" / f"{Path(document['source']).stem}.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    parts = [f"# {Path(document['source']).stem}\n\n"]
    ranges = []
    blocks = []
    offset = len(parts[0])
    for job in jobs:
        raw_dir = OUT / "raw" / job["id"]
        md_path = raw_dir / job["markdown_path"]
        body = md_path.read_text(encoding="utf-8").strip()
        # MinerU heading levels become children of the stable filename heading.
        body = re.sub(r"(?m)^(#{1,5})([ \t]+)", lambda m: "#" + m.group(1) + m.group(2), body)
        # Keep images and rewrite their links relative to the assembled Markdown.
        def rewrite_image(match: re.Match) -> str:
            original = match.group(2)
            if re.match(r"^[a-z]+://", original, flags=re.I):
                return match.group(0)
            resolved = (md_path.parent / original).resolve()
            if not resolved.is_relative_to(raw_dir.resolve()) or not resolved.exists():
                return match.group(0)
            relative = resolved.relative_to(OUT).as_posix()
            return f"![{match.group(1)}](../{relative})"
        body = re.sub(r"!\[([^]]*)\]\(([^)]+)\)", rewrite_image, body)
        prefix = f"<!-- PDF pages {job['start']}-{job['end']} -->\n\n"
        segment = prefix + body + "\n\n"
        ranges.append({"start_page": job["start"], "end_page": job["end"],
                       "markdown_start": offset + len(prefix),
                       "markdown_end": offset + len(prefix) + len(body),
                       "raw": str(raw_dir.relative_to(OUT)).replace("\\", "/")})
        content_lists = list(raw_dir.rglob("*content_list.json"))
        if len(content_lists) == 1:
            content_list = json.loads(content_lists[0].read_text(encoding="utf-8"))
            page_indices = [item.get("page_idx") for item in content_list if isinstance(item.get("page_idx"), int)]
            # Some MinerU versions number a selected range from zero; others retain original indices.
            local_pages = job["start"] > 1 and page_indices and min(page_indices) == 0
            for item in content_list:
                page_idx = item.get("page_idx")
                bbox = item.get("bbox")
                if not isinstance(page_idx, int) or not isinstance(bbox, list) or len(bbox) != 4:
                    continue
                page = job["start"] + page_idx if local_pages else page_idx + 1
                if not job["start"] <= page <= job["end"]:
                    continue
                blocks.append({"page": page, "bbox": bbox, "type": item.get("type", ""),
                               "text": item.get("text", ""),
                               "table_body": item.get("table_body", ""),
                               "image_path": item.get("img_path", ""),
                               "raw": str(raw_dir.relative_to(OUT)).replace("\\", "/")})
        parts.append(segment)
        offset += len(segment)
    output.write_text("".join(parts), encoding="utf-8")
    sidecar = output.with_suffix(".source.json")
    sidecar.write_text(json.dumps({"source_pdf": document["source"],
                                   "source_pdf_path": f"论文/{document['source']}",
                                   "pages": document["pages"], "ranges": ranges},
                                  ensure_ascii=False, indent=2), encoding="utf-8")
    output.with_suffix(".blocks.json").write_text(
        json.dumps({"source_pdf": document["source"], "coordinate_system": "mineru_normalized_1000",
                    "blocks": blocks}, ensure_ascii=False), encoding="utf-8")
    document["assembled"] = True
    save(manifest)
    print(f"Assembled {output.name}", flush=True)


def poll(manifest: dict) -> None:
    with client() as api:
        for batch_id, job_ids in list(manifest["batches"].items()):
            if all(manifest["jobs"][job_id]["status"] in ("done", "failed", "upload_failed")
                   for job_id in job_ids):
                continue
            data = check(api.get(f"{API}/extract-results/batch/{batch_id}"))
            for item in data.get("extract_result", []):
                job_id = item.get("data_id") or Path(item.get("file_name", "")).stem
                job = manifest["jobs"].get(job_id)
                if not job or job.get("batch_id") != batch_id or job["status"] == "done":
                    continue
                state = item.get("state", "")
                if state == "done":
                    url = item.get("full_zip_url")
                    if not url:
                        continue
                    try:
                        with httpx.Client(timeout=httpx.Timeout(180, connect=15), follow_redirects=True) as downloader:
                            response = downloader.get(url)
                            response.raise_for_status()
                            content = response.content
                        md_path = safe_extract(content, OUT / "raw" / job_id)
                        job["markdown_path"] = str(md_path.relative_to(OUT / "raw" / job_id)).replace("\\", "/")
                        job["status"] = "done"
                        print(f"Done {job_id}", flush=True)
                    except Exception as exc:
                        job["error"] = str(exc)[:250]
                        print(f"Download failed {job_id}: {type(exc).__name__}", flush=True)
                elif state == "failed":
                    job["status"] = "failed"
                    job["error"] = item.get("err_msg", "")[:250]
                    print(f"Parse failed {job_id}: {job['error']}", flush=True)
                else:
                    job["status"] = state or "uploaded"
                save(manifest)
    for doc_id in manifest["documents"]:
        assemble(manifest, doc_id)


def counts(manifest: dict) -> dict[str, int]:
    result = {}
    for job in manifest["jobs"].values():
        result[job["status"]] = result.get(job["status"], 0) + 1
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("plan", "submit", "poll", "run"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--interval", type=int, default=30)
    args = parser.parse_args()
    manifest = load()
    if args.action in ("submit", "run"):
        submit(manifest, args.limit)
    if args.action in ("poll", "run"):
        while True:
            try:
                poll(manifest)
            except httpx.HTTPError as exc:
                if args.action != "run":
                    raise
                print(f"MinerU polling network error: {type(exc).__name__}; retrying", flush=True)
            print("Status", counts(manifest), flush=True)
            if args.action != "run" or not any(job["status"] not in ("done", "failed", "upload_failed")
                                                for job in manifest["jobs"].values() if job.get("batch_id")):
                break
            time.sleep(args.interval)
    if args.action == "run" and all(job["status"] == "done" for job in manifest["jobs"].values()):
        if not all(doc.get("assembled") for doc in manifest["documents"].values()):
            raise RuntimeError("All MinerU jobs finished but some Markdown files were not assembled")
        builder = ROOT.parent / "rag-service" / "build" / "build_mineru_corpus.py"
        subprocess.run([sys.executable, str(builder), "--expected", str(len(manifest["documents"]))], check=True)
        print("MinerU corpus replaced all temporary extraction results", flush=True)
    if args.action == "plan":
        print("Documents", len(manifest["documents"]), "jobs", len(manifest["jobs"]), counts(manifest))


if __name__ == "__main__":
    main()
