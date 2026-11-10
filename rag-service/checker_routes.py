"""RAG 检查网页、双范围检索评测及相关静态资源路由。"""

import asyncio
import json
import logging
import re
import time
from collections.abc import Callable
from pathlib import Path

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import FileResponse, RedirectResponse
from langchain_ollama import ChatOllama
from ollama import ResponseError
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class EvaluationRequest(BaseModel):
    question: str = Field(min_length=1)
    reference_answer: str = Field(min_length=1)
    top_contexts: list[str]
    candidate_contexts: list[str]


def create_checker_router(
    *,
    project_root: Path,
    settings: dict,
    ask_func: Callable[[str, str, int | None, int | None], list[dict]],
    get_domain: Callable[[], str],
) -> APIRouter:
    """创建检查台路由；检索实现和当前领域由主服务注入。"""
    router = APIRouter(tags=["RAG checker"])
    max_top_k = settings["retrieval"]["max_top_k"]
    max_candidate_k = settings["retrieval"]["max_candidate_k"]
    default_top_k = settings["retrieval"]["top_k"]
    default_candidate_k = settings["retrieval"]["candidate_k"]

    class RagCheckRequest(BaseModel):
        question: str = Field(min_length=1)
        reference_answer: str = Field(min_length=1)
        top_k: int = Field(default=default_top_k, ge=1, le=max_top_k)
        candidate_k: int = Field(
            default=default_candidate_k,
            ge=1,
            le=min(max_top_k, max_candidate_k),
        )

    class RagRetrieveRequest(BaseModel):
        question: str = Field(min_length=1)
        top_k: int = Field(default=default_top_k, ge=1, le=max_top_k)
        candidate_k: int = Field(
            default=default_candidate_k,
            ge=1,
            le=min(max_top_k, max_candidate_k),
        )

    def evaluate_both_contexts(
        question: str,
        reference_answer: str,
        top_contexts: list[str],
        candidate_contexts: list[str],
    ) -> dict:
        max_chars = int(settings["llm"]["evaluation_context_max_chars"])

        def format_contexts(contexts: list[str]) -> str:
            return "\n\n".join(
                f"[{index + 1}] {content[:max_chars]}"
                for index, content in enumerate(contexts)
            ) or "（无上下文）"

        prompt = f"""你是严格的 RAG 检索评测器。一次评估最终 Top K 和 Candidate K 两组上下文。

问题：{question}
标准答案/期望要点：{reference_answer}

最终 Top K 上下文：
{format_contexts(top_contexts)}

Candidate K 上下文：
{format_contexts(candidate_contexts)}

任务：
1. 把标准答案拆成最少数量的独立事实要点。
2. 对每个要点分别判断是否被 Top K、Candidate K 明确支持。
3. 对两组中的每个上下文块分别判断是否有助于回答问题。
4. 只做布尔判断，不计算分数；分数由程序固定计算。
只返回合法 JSON，不要 Markdown：
{{"claims":["事实要点"],"top_supported":[true],"candidate_supported":[true],"top_relevant_indices":[1],"candidate_relevant_indices":[1],"top_reason":"简短说明","candidate_reason":"简短说明"}}
三个数组 claims、top_supported、candidate_supported 必须等长。相关块编号从 1 开始。"""
        model_name = settings["llm"]["evaluation_model"]
        llm = ChatOllama(
            model=model_name,
            base_url=settings["llm"]["llm_host"],
            temperature=0,
            format="json",
        )
        response = None
        for attempt in range(2):
            try:
                response = llm.invoke(prompt)
                break
            except ResponseError as exc:
                if exc.status_code != 503 or attempt == 1:
                    detail = (
                        f"Ollama 评测服务不可用（HTTP {exc.status_code}）。"
                        f"请确认模型 {model_name} 已安装、Ollama 正常运行且显存/内存充足。"
                    )
                    raise HTTPException(status_code=503, detail=detail) from exc
                logger.warning("Ollama returned 503, retry combined evaluation once")
                time.sleep(1)
        raw = response.content if isinstance(response.content, str) else str(response.content)
        try:
            result = json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.error("RAG evaluation returned invalid JSON: %s", raw)
            raise HTTPException(status_code=502, detail=f"评测模型未返回有效 JSON: {exc}")
        claims = [str(item) for item in result.get("claims", [])]
        top_supported = result.get("top_supported", [])
        candidate_supported = result.get("candidate_supported", [])
        if not claims or len(top_supported) != len(claims) or len(candidate_supported) != len(claims):
            raise HTTPException(status_code=502, detail="评测模型返回的事实要点和判断数组长度不一致")

        def build_metrics(supported: list, relevant_indices: list, contexts: list[str], reason: str) -> dict:
            flags = [bool(value) for value in supported]
            valid_indices = sorted({
                int(index) for index in relevant_indices
                if isinstance(index, (int, float)) and 1 <= int(index) <= len(contexts)
            })
            return {
                "retrieval_recall": sum(flags) / len(flags),
                "covered_points": [claim for claim, flag in zip(claims, flags) if flag],
                "missed_points": [claim for claim, flag in zip(claims, flags) if not flag],
                "relevant_context_indices": valid_indices,
                "reason": reason,
                "model": model_name,
                "context_count": len(contexts),
            }

        return {
            "top": build_metrics(top_supported, result.get("top_relevant_indices", []), top_contexts,
                                 str(result.get("top_reason", ""))),
            "candidate": build_metrics(candidate_supported, result.get("candidate_relevant_indices", []),
                                       candidate_contexts, str(result.get("candidate_reason", ""))),
        }

    @router.get("/checker", include_in_schema=False)
    async def rag_checker():
        return FileResponse(project_root / "static" / "rag-checker.html")

    @router.get("/gui", include_in_schema=False)
    async def rag_gui():
        return RedirectResponse(url="/checker", status_code=status.HTTP_307_TEMPORARY_REDIRECT)

    @router.get("/images/{image_name}", include_in_schema=False)
    async def rag_image(image_name: str):
        if image_name in {"", "NO_IMAGE_AVAILABLE"} or "/" in image_name or "\\" in image_name:
            raise HTTPException(status_code=404, detail="图片不存在")
        mineru_match = re.fullmatch(r"([0-9a-f]{16}-[0-9]{4}-[0-9]{4})-([A-Za-z0-9._-]+\.(?:png|jpe?g|gif|webp))", image_name, re.I)
        if mineru_match:
            raw_dir = project_root.parent / "pdf-data" / "mineru-output" / "raw" / mineru_match.group(1)
            if raw_dir.is_dir():
                for img_path in raw_dir.rglob(mineru_match.group(2)):
                    if img_path.is_file():
                        return FileResponse(img_path)
        for img_path in (project_root / "img").rglob(image_name):
            if any(part.startswith("_archive") for part in img_path.parts):
                continue
            if img_path.is_file():
                return FileResponse(img_path)
        raise HTTPException(status_code=404, detail="图片不存在")

    @router.post("/evaluate", response_description="RAG 双范围评测结果")
    async def evaluate_endpoint(request: EvaluationRequest):
        try:
            evaluation = await asyncio.to_thread(
                evaluate_both_contexts, request.question, request.reference_answer,
                request.top_contexts, request.candidate_contexts
            )
            return {"code": 0, "message": "success", "data": evaluation}
        except HTTPException as exc:
            return {"code": exc.status_code, "message": exc.detail, "data": {}}
        except Exception as exc:
            logger.error("RAG evaluation failed: %s", exc, exc_info=True)
            return {"code": 500, "message": f"RAG 评测失败: {exc}", "data": {}}

    @router.post("/rag_retrieve", response_description="RAG 双范围检索结果")
    async def rag_retrieve_endpoint(request: RagRetrieveRequest):
        """只执行检索，不等待评测模型，供页面优先展示文档。"""
        if request.candidate_k < request.top_k:
            raise HTTPException(status_code=422, detail="candidate_k 不能小于 top_k")
        started_at = time.time()
        domain = get_domain()
        try:
            top_docs = await asyncio.to_thread(
                ask_func, request.question, domain, request.top_k, request.candidate_k
            )
            candidate_docs = await asyncio.to_thread(
                ask_func, request.question, domain, request.candidate_k, request.candidate_k
            )
            return {
                "code": 0,
                "message": "success",
                "data": {
                    "question": request.question,
                    "domain": domain,
                    "model": settings["llm"]["evaluation_model"],
                    "top": {"k": request.top_k, "documents": top_docs},
                    "candidate": {"k": request.candidate_k, "documents": candidate_docs},
                },
                "elapsed_time": time.time() - started_at,
            }
        except HTTPException:
            raise
        except Exception as exc:
            logger.error("RAG retrieve failed: %s", exc, exc_info=True)
            raise HTTPException(status_code=500, detail=f"RAG 检索失败: {exc}")

    @router.post("/rag_check", response_description="检索并完成 RAG 双范围评测")
    async def rag_check_endpoint(request: RagCheckRequest):
        if request.candidate_k < request.top_k:
            raise HTTPException(status_code=422, detail="candidate_k 不能小于 top_k")
        started_at = time.time()
        try:
            domain = get_domain()
            top_docs = await asyncio.to_thread(
                ask_func, request.question, domain, request.top_k, request.candidate_k
            )
            candidate_docs = await asyncio.to_thread(
                ask_func, request.question, domain, request.candidate_k, request.candidate_k
            )
            evaluation = await asyncio.to_thread(
                evaluate_both_contexts,
                request.question,
                request.reference_answer,
                [item.get("content", "") for item in top_docs],
                [item.get("content", "") for item in candidate_docs],
            )
            return {
                "code": 0,
                "message": "success",
                "data": {
                    "question": request.question,
                    "reference_answer": request.reference_answer,
                    "domain": domain,
                    "model": settings["llm"]["evaluation_model"],
                    "top": {"k": request.top_k, "documents": top_docs, "metrics": evaluation["top"]},
                    "candidate": {"k": request.candidate_k, "documents": candidate_docs,
                                  "metrics": evaluation["candidate"]},
                },
                "elapsed_time": time.time() - started_at,
            }
        except HTTPException:
            raise
        except Exception as exc:
            logger.error("RAG check failed: %s", exc, exc_info=True)
            raise HTTPException(status_code=500, detail=f"RAG 检索评测失败: {exc}")

    return router
