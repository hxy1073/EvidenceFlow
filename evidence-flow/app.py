"""EvidenceFlow: project conversations, RAG evidence review, and final findings."""

from __future__ import annotations

import json
import os
import re
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx
import pymupdf
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
STATIC = ROOT / "static"
DATA = ROOT / "data"
DATABASE = DATA / "evidenceflow.sqlite"
MARKDOWN_DIR = WORKSPACE / "rag-service" / "data" / "index"
MINERU_MARKDOWN_DIR = WORKSPACE / "pdf-data" / "mineru-output" / "markdown"
PDF_DIR = WORKSPACE / "pdf-data" / "论文"


def load_local_env() -> None:
    env_file = ROOT / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        if not line or line.lstrip().startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        os.environ.setdefault(name.strip(), value.strip().strip('"').strip("'"))


load_local_env()
RAG_URL = os.getenv("EVIDENCEFLOW_RAG_URL", "http://127.0.0.1:8021").rstrip("/")
DEEPSEEK_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_PROXY = os.getenv("DEEPSEEK_PROXY") or None
OLLAMA_URL = os.getenv("EVIDENCEFLOW_OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
EMBEDDING_MODEL = os.getenv("EVIDENCEFLOW_EMBEDDING_MODEL", "bge-m3")

app = FastAPI(title="EvidenceFlow")
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/api/papers/{filename}")
def original_pdf(filename: str) -> FileResponse:
    path = (PDF_DIR / filename).resolve()
    if path.parent != PDF_DIR.resolve() or path.suffix.lower() != ".pdf" or not path.is_file():
        raise HTTPException(404, "原始 PDF 不存在")
    return FileResponse(path, media_type="application/pdf", filename=path.name,
                        content_disposition_type="inline")


def connection() -> sqlite3.Connection:
    DATA.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(DATABASE, timeout=30)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA foreign_keys=ON")
    return db


def initialize() -> None:
    with connection() as db:
        db.executescript(
            """
            CREATE TABLE IF NOT EXISTS projects (
              id TEXT PRIMARY KEY, name TEXT NOT NULL, created_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS conversations (
              id TEXT PRIMARY KEY, project_id TEXT NOT NULL REFERENCES projects(id),
              question TEXT NOT NULL, status TEXT NOT NULL, state_json TEXT NOT NULL,
              error TEXT, created_at REAL NOT NULL, updated_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS conversations_project ON conversations(project_id, created_at DESC);
            """
        )
        db.execute(
            "UPDATE conversations SET status='error', error='服务重启中断了分析，请重新提问' "
            "WHERE status IN ('retrieving', 'analyzing')"
        )


initialize()


def conversation_data(row: sqlite3.Row) -> dict[str, Any]:
    result = dict(row)
    result["state"] = json.loads(result.pop("state_json"))
    return result


def get_conversation(conversation_id: str) -> dict[str, Any]:
    with connection() as db:
        row = db.execute("SELECT * FROM conversations WHERE id=?", (conversation_id,)).fetchone()
    if not row:
        raise HTTPException(404, "对话不存在")
    return conversation_data(row)


def save_state(conversation_id: str, status: str, state: dict[str, Any], error: str | None = None) -> None:
    with connection() as db:
        db.execute(
            "UPDATE conversations SET status=?, state_json=?, error=?, updated_at=? WHERE id=?",
            (status, json.dumps(state, ensure_ascii=False), error, time.time(), conversation_id),
        )


class ProjectInput(BaseModel):
    name: str = Field(min_length=1, max_length=80)


class QuestionInput(BaseModel):
    project_id: str | None = None
    question: str = Field(min_length=2, max_length=500)


class SelectionInput(BaseModel):
    conclusion_id: str
    evidence_ids: list[str]


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/api/config")
def config() -> dict[str, Any]:
    return {"rag_url": RAG_URL, "model": DEEPSEEK_MODEL, "key_configured": bool(DEEPSEEK_API_KEY)}


@app.get("/api/images/{image_name}")
def evidence_image(image_name: str) -> Response:
    if (not image_name or image_name in (".", "..") or "/" in image_name
            or "\\" in image_name or Path(image_name).suffix.lower() not in
            {".png", ".jpg", ".jpeg", ".webp", ".gif"}):
        raise HTTPException(400, "图片名称无效")
    with httpx.Client(timeout=15.0) as client:
        result = client.get(f"{RAG_URL}/images/{quote(image_name)}")
    if result.status_code != 200:
        raise HTTPException(404, "图片不存在")
    return Response(result.content, media_type=result.headers.get("content-type", "image/jpeg"))


@app.get("/api/projects")
def projects() -> list[dict[str, Any]]:
    with connection() as db:
        rows = db.execute("SELECT * FROM projects ORDER BY created_at DESC").fetchall()
        conversations = db.execute(
            "SELECT id, project_id, question, status, created_at FROM conversations ORDER BY created_at DESC"
        ).fetchall()
    by_project: dict[str, list[dict[str, Any]]] = {}
    for row in conversations:
        by_project.setdefault(row["project_id"], []).append(dict(row))
    return [{**dict(row), "conversations": by_project.get(row["id"], [])} for row in rows]


@app.post("/api/projects")
def create_project(payload: ProjectInput) -> dict[str, Any]:
    project_id = uuid.uuid4().hex
    with connection() as db:
        db.execute(
            "INSERT INTO projects(id,name,created_at) VALUES(?,?,?)",
            (project_id, payload.name.strip(), time.time()),
        )
    return {"id": project_id, "name": payload.name.strip(), "conversations": []}


@app.post("/api/conversations")
def create_conversation(payload: QuestionInput, tasks: BackgroundTasks) -> dict[str, Any]:
    with connection() as db:
        project_id = payload.project_id
        if project_id:
            if not db.execute("SELECT 1 FROM projects WHERE id=?", (project_id,)).fetchone():
                raise HTTPException(404, "项目不存在")
        else:
            existing = db.execute("SELECT id FROM projects ORDER BY created_at ASC LIMIT 1").fetchone()
            if existing:
                project_id = existing["id"]
            else:
                project_id = uuid.uuid4().hex
                db.execute(
                    "INSERT INTO projects(id,name,created_at) VALUES(?,?,?)",
                    (project_id, "我的研究项目", time.time()),
                )
        conversation_id = uuid.uuid4().hex
        now = time.time()
        db.execute(
            "INSERT INTO conversations VALUES(?,?,?,?,?,?,?,?)",
            (conversation_id, project_id, payload.question.strip(), "retrieving", "{}", None, now, now),
        )
    tasks.add_task(analyze_conversation, conversation_id)
    return get_conversation(conversation_id)


@app.get("/api/conversations/{conversation_id}")
def conversation(conversation_id: str) -> dict[str, Any]:
    return get_conversation(conversation_id)


@app.delete("/api/conversations/{conversation_id}")
def delete_conversation(conversation_id: str) -> dict[str, bool]:
    with connection() as db:
        deleted = db.execute("DELETE FROM conversations WHERE id=?", (conversation_id,)).rowcount
    if not deleted:
        raise HTTPException(404, "对话不存在")
    return {"deleted": True}


@app.post("/api/conversations/{conversation_id}/selection")
def select_evidence(conversation_id: str, payload: SelectionInput) -> dict[str, Any]:
    conversation = get_conversation(conversation_id)
    if conversation["status"] not in ("ready", "final"):
        raise HTTPException(409, "分析尚未完成")
    state = conversation["state"]
    conclusion = next((x for x in state["conclusions"] if x["id"] == payload.conclusion_id), None)
    if conclusion is None:
        raise HTTPException(404, "结论不存在")
    allowed = set(conclusion["evidence_ids"])
    if not set(payload.evidence_ids).issubset(allowed):
        raise HTTPException(422, "所选证据不属于此结论")
    conclusion["selected_evidence_ids"] = list(dict.fromkeys(payload.evidence_ids))
    save_state(conversation_id, "ready", state)
    return get_conversation(conversation_id)


@app.post("/api/conversations/{conversation_id}/finalize")
def finalize(conversation_id: str) -> dict[str, Any]:
    conversation = get_conversation(conversation_id)
    if conversation["status"] not in ("ready", "final"):
        raise HTTPException(409, "分析尚未完成")
    save_state(conversation_id, "final", conversation["state"])
    return get_conversation(conversation_id)


@app.get("/api/conversations/{conversation_id}/source/{evidence_id}")
def source(conversation_id: str, evidence_id: str) -> dict[str, Any]:
    conversation = get_conversation(conversation_id)
    candidate = next((x for x in conversation["state"].get("candidates", []) if x["id"] == evidence_id), None)
    if candidate is None:
        raise HTTPException(404, "证据不存在")
    pdf_name = candidate.get("source_pdf")
    if pdf_name:
        pdf_path = (PDF_DIR / pdf_name).resolve()
        if pdf_path.parent != PDF_DIR.resolve() or not pdf_path.is_file():
            raise HTTPException(404, "原始 PDF 不存在")
        with pymupdf.open(pdf_path) as pdf:
            page_count = len(pdf)
        return {"name": pdf_name, "pdf": True, "pdf_url": f"/api/papers/{quote(pdf_name)}",
                "page": candidate.get("page") or 1,
                "page_count": page_count, "evidence_id": evidence_id}
    source_file = candidate.get("source_file")
    if source_file:
        path = (MARKDOWN_DIR / source_file).resolve()
        if path.parent == MARKDOWN_DIR.resolve() and path.exists():
            markdown = path.read_text(encoding="utf-8")
            start, end = candidate.get("source_start"), candidate.get("source_end")
            located = isinstance(start, int) and isinstance(end, int) and 0 <= start < end <= len(markdown)
            return {
                "name": source_file,
                "markdown": markdown,
                "start": start if located else None,
                "end": end if located else None,
                "located": located,
            }
    return {"name": "RAG 检索片段", "markdown": candidate["content"], "start": 0,
            "end": len(candidate["content"]), "located": False}


@app.get("/api/conversations/{conversation_id}/pdf-page/{evidence_id}/{page_number}")
def pdf_page(conversation_id: str, evidence_id: str, page_number: int) -> Response:
    conversation = get_conversation(conversation_id)
    candidate = next((x for x in conversation["state"].get("candidates", []) if x["id"] == evidence_id), None)
    if candidate is None or not candidate.get("source_pdf"):
        raise HTTPException(404, "原始 PDF 不存在")
    path = (PDF_DIR / candidate["source_pdf"]).resolve()
    if path.parent != PDF_DIR.resolve() or not path.is_file():
        raise HTTPException(404, "原始 PDF 不存在")
    with pymupdf.open(path) as pdf:
        if page_number < 1 or page_number > len(pdf):
            raise HTTPException(404, "页码不存在")
        page = pdf[page_number - 1]
        # MinerU content_list bboxes are normalized to 0..1000. Overlay only
        # text blocks; figures and images remain untouched.
        if page_number == candidate.get("page"):
            for bbox in candidate.get("bboxes") or []:
                if len(bbox) != 4:
                    continue
                rect = pymupdf.Rect(bbox[0] * page.rect.width / 1000,
                                    bbox[1] * page.rect.height / 1000,
                                    bbox[2] * page.rect.width / 1000,
                                    bbox[3] * page.rect.height / 1000)
                if not rect.is_empty and rect.intersects(page.rect):
                    page.draw_rect(rect, color=None, fill=(1, 0.84, 0.2), fill_opacity=0.27,
                                   overlay=True)
        pixmap = page.get_pixmap(matrix=pymupdf.Matrix(3, 3), alpha=False)
        return Response(content=pixmap.tobytes("png"), media_type="image/png",
                        headers={"Cache-Control": "private, max-age=300"})


def normalized_with_offsets(value: str) -> tuple[str, list[int]]:
    chars: list[str] = []
    offsets: list[int] = []
    for i, char in enumerate(value):
        if not char.isspace():
            chars.append(char)
            offsets.append(i)
    return "".join(chars), offsets


def header_title(headers: Any) -> str:
    return " / ".join(
        str(part[1] if isinstance(part, (tuple, list)) and len(part) > 1 else part)
        for part in (headers or [])
    )


def locate_source(content: str) -> tuple[str | None, int | None, int | None]:
    body = content.split("\n\n", 1)[-1].strip()
    compact_body, _ = normalized_with_offsets(body)
    if len(compact_body) < 24:
        return None, None, None
    snippets = [compact_body[: min(len(compact_body), 100)]]
    if len(compact_body) > 120:
        snippets.append(compact_body[80:160])
    for path in MARKDOWN_DIR.glob("*.md"):
        raw = path.read_text(encoding="utf-8", errors="replace")
        compact, offsets = normalized_with_offsets(raw)
        for snippet in snippets:
            found = compact.find(snippet)
            if found >= 0:
                end_compact = compact.find(compact_body, found)
                length = len(compact_body) if end_compact == found else len(snippet)
                return path.name, offsets[found], offsets[min(found + length - 1, len(offsets) - 1)] + 1
    return None, None, None


def retrieve_candidates(question: str) -> list[dict[str, Any]]:
    # Ollama may have large generation models occupying its available memory.
    # Use the RAG service's BM25 path until the embedding model is loaded.
    retrieval_mode = "bm25"
    try:
        with httpx.Client(timeout=3.0) as client:
            loaded = client.get(f"{OLLAMA_URL}/api/ps").json().get("models", [])
        if any(item.get("name", "").split(":", 1)[0] == EMBEDDING_MODEL for item in loaded):
            retrieval_mode = "hybrid"
    except (httpx.HTTPError, ValueError):
        pass
    with httpx.Client(timeout=httpx.Timeout(150.0, connect=5.0)) as client:
        response = client.post(
            f"{RAG_URL}/ask",
            json={"message": question, "top_k": 30, "candidate_k": 60,
                  "long_context_reorder": False, "retrieval_mode": retrieval_mode},
        )
        response.raise_for_status()
        data = response.json()
    if data.get("code") != 0:
        raise RuntimeError(f"RAG 检索失败：{data.get('message', '未知错误')}")
    candidates = []
    seen = set()
    for item in data.get("data", []):
        content = item.get("content", "").strip()
        if not content or content in seen:
            continue
        seen.add(content)
        metadata = item.get("metadata") or {}
        source_file = metadata.get("source_file")
        start = end = None
        if not metadata.get("source_pdf") and (not source_file or metadata.get("source_start") is None or metadata.get("source_end") is None):
            source_file, start, end = locate_source(content)
        elif metadata.get("source_start") is not None:
            start, end = metadata["source_start"], metadata["source_end"]
        candidates.append({
            "id": f"E{len(candidates)+1:02d}", "content": content,
            "excerpt": content.split("\n\n", 1)[-1][:500],
            "title": header_title(metadata.get("headers")) or (source_file or "文献片段"),
            "source_file": source_file, "source_start": start, "source_end": end,
            "source_pdf": metadata.get("source_pdf"), "page": metadata.get("page"),
            "bboxes": metadata.get("bboxes") or [],
            "images": [ref["name"] for ref in item.get("image_urls", [])
                       if isinstance(ref, dict) and isinstance(ref.get("name"), str)],
        })
    if not candidates:
        raise RuntimeError("RAG 没有返回候选片段")
    return candidates


def call_deepseek(question: str, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not DEEPSEEK_API_KEY:
        raise RuntimeError("未配置 DEEPSEEK_API_KEY")
    evidence = [{"id": c["id"], "title": c["title"], "text": c["excerpt"][:180]} for c in candidates]
    prompt = (
        "你是严谨的研究证据分析员。仅依据给定候选片段回答问题，不得补造事实、论文、页码或引文。"
        "先判断相关性，排除无关片段，再归纳2到5条相互区别的结论。每条结论必须绑定1到5个给定证据ID，"
        "summary要说明支持结论的具体内容。若证据不足，返回空的conclusions数组。"
        "仅返回JSON对象，结构为{\"conclusions\":[{\"title\":\"短标题\",\"summary\":\"解释\","
        "\"evidence_ids\":[\"E01\"]}]}。"
    )
    request_body = {"model": DEEPSEEK_MODEL, "temperature": 0.2,
                    "response_format": {"type": "json_object"},
                    "messages": [{"role": "system", "content": prompt},
                                 {"role": "user", "content": json.dumps({"question": question, "candidates": evidence,}, ensure_ascii=False)}]}
    proxies = [None, DEEPSEEK_PROXY] if DEEPSEEK_PROXY else [None]
    data = None
    last_error = None
    for proxy in proxies:
        for attempt in range(3 if proxy else 2):
            try:
                with httpx.Client(timeout=httpx.Timeout(120.0, connect=10.0), proxy=proxy,
                                  trust_env=False) as client:
                    response = client.post(
                        f"{DEEPSEEK_URL}/chat/completions",
                        headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}"},
                        json=request_body,
                    )
                    response.raise_for_status()
                    data = response.json()
                break
            except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout) as exc:
                last_error = exc
                time.sleep(min(2 ** attempt, 8))
        if data is not None:
            break
    if data is None:
        raise RuntimeError(f"DeepSeek 连接失败：{last_error}")
    raw = data["choices"][0]["message"]["content"]
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
    parsed = json.loads(raw)
    allowed = {c["id"] for c in candidates}
    conclusions = []
    for item in parsed.get("conclusions", [])[:5]:
        ids = [value for value in item.get("evidence_ids", []) if value in allowed]
        ids = list(dict.fromkeys(ids))[:5]
        if not ids or not str(item.get("title", "")).strip():
            continue
        conclusions.append({
            "id": f"C{len(conclusions)+1:02d}", "title": str(item["title"]).strip(),
            "summary": str(item.get("summary", "")).strip(), "evidence_ids": ids,
            "selected_evidence_ids": ids.copy(),
        })
    return conclusions


def analyze_conversation(conversation_id: str) -> None:
    try:
        conversation = get_conversation(conversation_id)
        candidates = retrieve_candidates(conversation["question"])
        state = {"candidates": candidates, "conclusions": []}
        save_state(conversation_id, "analyzing", state)
        conclusions = call_deepseek(conversation["question"], candidates)
        state["conclusions"] = conclusions
        save_state(conversation_id, "ready", state)
    except Exception as exc:
        try:
            previous = get_conversation(conversation_id)["state"]
            save_state(conversation_id, "error", previous, str(exc)[:400])
        except Exception:
            pass


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=os.getenv("EVIDENCEFLOW_HOST", "127.0.0.1"),
                port=int(os.getenv("EVIDENCEFLOW_PORT", "8031")))
