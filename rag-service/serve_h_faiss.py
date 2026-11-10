import concurrent
import hashlib
import json
import logging
import os
import re
import pickle
import pkgutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Sequence
from logging.handlers import TimedRotatingFileHandler
from urllib.parse import urlparse

import faiss
import numpy as np
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.vectorstores import VectorStoreRetriever
from tqdm import tqdm

if not hasattr(pkgutil, 'ImpImporter'):
    class ImpImporter:
        def __init__(self, path):
            self.path = path

    pkgutil.ImpImporter = ImpImporter

import jieba
import yaml
from fastapi import FastAPI, HTTPException, status
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_transformers import LongContextReorder
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS  # 替换为FAISS
from pydantic import BaseModel, Field
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

run_time = time.time()
# ===================== 全局配置与初始化 =====================
# 1. 关闭不必要的日志
# FAISS 线程数在加载全局配置（SETTINGS）后再设置，见 load_settings 之后

# 2. 路径处理
PROJECT_ROOT = Path(__file__).parent
CONFIG_PATH = PROJECT_ROOT.parent / "rag-config" / "config.yaml"
# CONFIG_PATH = PROJECT_ROOT / "config" / "rag_settings.yaml"
QUERY_INTENT_CONFIG_PATH = PROJECT_ROOT / "config" / "query_intents.yaml"
RAG_SETTINGS_PATH = PROJECT_ROOT.parent / "rag-config" / "config.yaml"


# 3. 日志配置
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True, parents=True)

log_formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

file_handler = TimedRotatingFileHandler(
    filename=LOG_DIR / "serve_faiss.log",
    when="midnight",
    interval=1,
    backupCount=30,
    encoding='utf-8',
    delay=False
)
file_handler.setFormatter(log_formatter)
file_handler.suffix = "%Y-%m-%d.log"

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(log_formatter)

root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
root_logger.addHandler(file_handler)
root_logger.addHandler(console_handler)

logger = logging.getLogger(__name__)

# 4. 关闭第三方库冗余日志
logging.getLogger('jieba').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('requests').setLevel(logging.WARNING)
logging.getLogger('faiss').setLevel(logging.WARNING)  # 关闭FAISS调试日志


# 5. 全局检索参数配置
def _require_section(settings: dict, section: str) -> dict:
    value = settings.get(section)
    if not isinstance(value, dict):
        raise RuntimeError(f"配置缺失或格式错误: {section}")
    return value


def _require_key(section: dict, section_name: str, key: str):
    if key not in section:
        raise RuntimeError(f"配置缺失: {section_name}.{key}")
    return section[key]


def _require_int(section: dict, section_name: str, key: str) -> int:
    value = _require_key(section, section_name, key)
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"配置必须是整数: {section_name}.{key}={value!r}") from exc


def _parse_host_port(url: str) -> tuple[str, int]:
    parsed = urlparse(url)
    if not parsed.hostname or parsed.port is None:
        raise RuntimeError(f"配置格式错误: llm.rag_host={url!r}，需要类似 http://127.0.0.1:8021")
    return parsed.hostname, parsed.port


def load_settings(path: Path = RAG_SETTINGS_PATH):
    if not path.exists():
        raise RuntimeError(f"配置文件不存在: {path}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            settings = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        raise RuntimeError(f"配置文件解析失败: {path}: {exc}") from exc

    if not isinstance(settings, dict):
        raise RuntimeError(f"配置文件格式错误，应为 YAML 对象: {path}")

    llm = _require_section(settings, "llm")
    retrieval = _require_section(settings, "retrieval")
    faiss_settings = _require_section(settings, "faiss")

    retrieval["top_k"] = _require_int(retrieval, "retrieval", "top_k")
    retrieval["candidate_k"] = _require_int(retrieval, "retrieval", "candidate_k")
    retrieval["max_top_k"] = _require_int(retrieval, "retrieval", "max_top_k")
    retrieval["max_candidate_k"] = _require_int(retrieval, "retrieval", "max_candidate_k")

    weights = _require_key(retrieval, "retrieval", "hybrid_weights")
    if (
        not isinstance(weights, list)
        or len(weights) != 2
        or not all(isinstance(v, (int, float)) for v in weights)
    ):
        raise RuntimeError("配置必须是两个数字: retrieval.hybrid_weights")

    embedding_model = _require_key(llm, "llm", "embedding_model")
    embedding_base_url = _require_key(llm, "llm", "llm_host")
    rag_host = _require_key(llm, "llm", "rag_host")
    server_host, server_port = _parse_host_port(rag_host)

    faiss_settings["threads"] = _require_int(faiss_settings, "faiss", "threads")

    settings["embedding"] = {
        "model": embedding_model,
        "base_url": embedding_base_url,
    }
    settings["server"] = {
        "host": server_host,
        "port": server_port,
    }

    logger.info(
        "RAG配置: top_k=%s candidate_k=%s max_top_k=%s max_candidate_k=%s "
        "embedding_model=%s embedding_base_url=%s faiss_threads=%s",
        retrieval["top_k"],
        retrieval["candidate_k"],
        retrieval["max_top_k"],
        retrieval["max_candidate_k"],
        embedding_model,
        embedding_base_url,
        faiss_settings["threads"],
    )

    return settings


SETTINGS = load_settings()
DEBUG_LINE_SIMILARITY = SETTINGS["debug"]["line_similarity"]

# 服务启动时从配置读取一次的全局检索参数。
# 请求未显式传入 top_k/candidate_k 时直接使用这两个值。
TOP_K = SETTINGS["retrieval"]["top_k"]
CANDIDATE_K = SETTINGS["retrieval"]["candidate_k"]

# FAISS 线程数（来自全局配置，重启生效）
faiss.omp_set_num_threads(SETTINGS["faiss"]["threads"])

# 5. 加载配置文件
class GlobalConfig:
    def __init__(self):
        self._domain = None
        self.initialized = False

    @property
    def domain(self):
        return self._domain

    @domain.setter
    def domain(self, value):
        self._domain = value


global_config = GlobalConfig()

try:
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    global_config.domain = config['llm']['rag']
    global_config.initialized = True
    if __name__ == "__main__" or (os.getpid() == os.getppid()):
        logger.info(f"成功加载初始配置，当前RAG领域：{global_config.domain}")
except FileNotFoundError:
    logger.error(f"配置文件不存在：{CONFIG_PATH}")
    sys.exit(1)
except KeyError:
    logger.error("配置文件中未找到 'llm.rag' 配置项")
    sys.exit(1)

# 6. FastAPI 实例
app = FastAPI(
    title="RAG 检索服务 (FAISS)",
    version="1.0.0",
    description="基于LangChain 1.x的混合检索（FAISS向量+BM25）服务"
)


# ===================== 数据模型与缓存 =====================
class QueryRequest(BaseModel):
    message: str
    retrieval_mode: str = Field(default="hybrid", pattern="^(hybrid|bm25)$")
    top_k: int | None = Field(default=TOP_K, ge=1, le=SETTINGS["retrieval"]["max_top_k"])
    candidate_k: int | None = Field(
        default=CANDIDATE_K,
        ge=1,
        le=SETTINGS["retrieval"]["max_candidate_k"],
    )
    long_context_reorder: bool = Field(default=True, description="是否启用长上下文重排序（将最相关文档放在中间位置）")


class DomainSwitchRequest(BaseModel):
    new_domain: str
    clear_cache: bool = True


retriever_cache: dict[tuple[str, int], EnsembleRetriever] = {}


# ===================== 核心工具函数 =====================
def get_embedding() -> OllamaEmbeddings:
    """获取Ollama嵌入模型"""
    if not hasattr(get_embedding, "instance"):
        get_embedding.instance = OllamaEmbeddings(
            model=SETTINGS["embedding"]["model"],
            base_url=SETTINGS["embedding"]["base_url"]
        )
    return get_embedding.instance


model = get_embedding()


def jieba_cut_doc(doc: Document) -> list[str]:
    return list(jieba.cut(doc.page_content)) if doc.page_content.strip() else []


def init_worker():
    """子进程初始化"""
    import logging
    import sys
    import os

    logger = logging.getLogger()
    logger.handlers = []
    logger.setLevel(logging.CRITICAL + 1)

    sys.stdout = open(os.devnull, 'w')
    sys.stderr = open(os.devnull, 'w')

    import jieba
    jieba.initialize()


def validate_domain(domain: str) -> str:
    """校验领域名称，只允许字母、数字、下划线，防止路径穿越。"""
    if not isinstance(domain, str) or not re.fullmatch(r"[A-Za-z0-9_]+", domain or ""):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"非法领域名称：{domain}（只允许字母、数字、下划线）"
        )
    return domain


def get_faiss_retriever(domain: str, k: int = 3) -> tuple[BM25Retriever, VectorStoreRetriever]:
    """
    获取BM25检索器和FAISS向量库检索器
    """
    # FAISS 存储路径
    validate_domain(domain)
    mineru_corpus = PROJECT_ROOT / "rag" / "mineru_corpus.jsonl"
    if domain == "index" and mineru_corpus.is_file():
        import json
        docs = []
        with mineru_corpus.open(encoding="utf-8") as stream:
            for line in stream:
                item = json.loads(line)
                docs.append(Document(page_content=item["page_content"], metadata=item["metadata"]))
        if not docs:
            raise HTTPException(status_code=500, detail="MinerU corpus is empty")
        tokenized = load_or_create_tokenized_docs(domain, docs)
        bm25 = load_or_create_bm25_retriever(domain, docs, tokenized, k=k)
        return bm25, None
    faiss_dir = PROJECT_ROOT / f"rag/faiss_langchain_db_{domain}"
    # 无需预先创建目录：索引文件检查通过即说明目录已存在，否则抛404

    # 检查FAISS索引文件是否存在
    index_file = faiss_dir / "index.faiss"
    pkl_file = faiss_dir / "index.pkl"

    if index_file.exists() and pkl_file.exists():
        try:
            # 加载FAISS向量库
            vectordb = FAISS.load_local(
                folder_path=str(faiss_dir),
                embeddings=model,
                allow_dangerous_deserialization=True
            )
            logger.info(f"成功加载FAISS向量库：{faiss_dir}")
        except Exception as e:
            logger.error(f"加载FAISS向量库失败：{e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"FAISS向量库加载失败：{str(e)}"
            )
    else:
        logger.warning(f"FAISS向量库文件不存在：{faiss_dir}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"未找到{domain}领域的FAISS向量库，请先创建索引"
        )

    # 生成向量库检索器
    vectordb_retriever = vectordb.as_retriever(search_kwargs={'k': k})

    # 从FAISS获取文档用于BM25
    try:
        # 获取所有文档
        doc_list = []
        # FAISS 存储文档在 docstore 中
        if hasattr(vectordb, 'docstore'):
            # 获取所有文档ID
            if hasattr(vectordb, 'index_to_docstore_id'):
                for idx in vectordb.index_to_docstore_id:
                    doc_id = vectordb.index_to_docstore_id[idx]
                    doc = vectordb.docstore.search(doc_id)
                    if doc and doc.page_content.strip():
                        doc_list.append(doc)
            else:
                # 备用方法：直接访问 docstore
                if hasattr(vectordb.docstore, '_dict'):
                    for doc in vectordb.docstore._dict.values():
                        if doc.page_content.strip():
                            doc_list.append(doc)

        if not doc_list:
            logger.warning(f"FAISS向量库中没有找到文档")
            # 如果没有文档，创建空的BM25检索器
            empty_docs = [Document(page_content="", metadata={})]
            tokenized_docs = load_or_create_tokenized_docs(domain, empty_docs)
            bm25_retriever = load_or_create_bm25_retriever(
                domain=domain,
                docs=empty_docs,
                tokenized_docs=tokenized_docs,
                k=k
            )
        else:
            logger.info(f"从FAISS加载了{len(doc_list)}条文档")
            s = time.time()
            tokenized_docs = load_or_create_tokenized_docs(domain, doc_list)
             # 初始化BM25Retriever，传入预分词结果（禁用内部预处理）
            bm25_retriever = load_or_create_bm25_retriever(
                domain=domain,
                docs=doc_list,
                tokenized_docs=tokenized_docs,
                k=k
            )
            end = time.time()
            logger.info(f"数据加载耗时：{end - s}s")

        return bm25_retriever, vectordb_retriever
    except Exception as e:
        logger.error(f"创建BM25检索器失败：{e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"BM25检索器创建失败：{str(e)}"
        )


def get_ensemble_retriever(domain: str, k: int = 3, weights=None) -> EnsembleRetriever:
    """创建混合检索器（FAISS向量+BM25）"""
    if weights is None:
        weights = SETTINGS["retrieval"]["hybrid_weights"]
    bm25_retriever, faiss_retriever = get_faiss_retriever(domain, k)
    if faiss_retriever is None:
        return bm25_retriever
    ensemble_retriever = EnsembleRetriever(
        retrievers=[faiss_retriever, bm25_retriever],
        weights=weights,
        search_kwargs={"k": k}
    )
    logger.debug(f"创建混合检索器完成，权重：{weights}")
    return ensemble_retriever


def get_retriever(domain: str, k: int = 3) -> EnsembleRetriever:
    """获取检索器（带缓存）"""
    if domain == "index":
        corpus = PROJECT_ROOT / "rag" / "mineru_corpus.jsonl"
        revision = corpus.stat().st_mtime_ns if corpus.is_file() else None
        if getattr(get_retriever, "_corpus_revision", None) != revision:
            for key in [key for key in retriever_cache if key[0] == domain]:
                retriever_cache.pop(key, None)
            get_retriever._corpus_revision = revision
    # 缓存超时清理
    # current_time = time.time()
    # if domain in cache_last_update:
    #     if current_time - cache_last_update[domain] > CACHE_TIMEOUT:
    #         retriever_cache.pop((domain, k), None)
    #         logger.info(f"检索器缓存超时，已清理：{domain}")
    # 从缓存获取或创建
    cache_key = (domain, k)
    if cache_key not in retriever_cache:
        retriever_cache[cache_key] = get_ensemble_retriever(domain, k)
        logger.info(f"创建并缓存检索器：{domain}，top_k={k}")
    else:
        logger.debug(f"从缓存获取检索器：{domain}，top_k={k}")

    return retriever_cache[cache_key]


def identity_preprocess(text: str) -> str:
    return text


# 结巴分词预处理
def jieba_cut(text: str) -> list[str]:
    return list(jieba.cut(text)) if text.strip() else []


def corpus_fingerprint(docs: list[Document]) -> str:
    digest = hashlib.sha256()
    for doc in docs:
        digest.update(doc.page_content.encode("utf-8"))
        digest.update(b"\0")
        digest.update(json.dumps(doc.metadata, ensure_ascii=False, sort_keys=True).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def load_or_create_bm25_retriever(domain: str, docs: list[Document], tokenized_docs: list[list[str]] = None,
                                  k: int = 3) -> BM25Retriever:
    BM25_CACHE_PATH = PROJECT_ROOT / "cache" / "faiss" / f"bm25_retriever_{domain}.pkl"
    fingerprint_path = BM25_CACHE_PATH.with_suffix(".sha256")
    fingerprint = corpus_fingerprint(docs)
    if (BM25_CACHE_PATH.exists() and BM25_CACHE_PATH.stat().st_size > 0
            and fingerprint_path.exists() and fingerprint_path.read_text() == fingerprint):
        try:
            with open(BM25_CACHE_PATH, "rb") as f:
                retriever = pickle.load(f)
            if (hasattr(retriever, 'vectorizer') and retriever.vectorizer is not None
                    and len(retriever.docs) == len(docs)):
                retriever.k = k
                retriever.preprocess_func = jieba_cut
                return retriever
            else:
                logger.warning("BM25缓存与当前FAISS文档不一致，重新生成")
        except (pickle.UnpicklingError, EOFError, AttributeError) as e:
            logger.warning(f"BM25缓存文件损坏，重新生成：{e}")
        BM25_CACHE_PATH.unlink(missing_ok=True)

    if tokenized_docs is None or len(tokenized_docs) != len(docs):
        logger.error("tokenized_docs不能为空且需与文档数量匹配")
        raise ValueError("tokenized_docs must be non-empty and match the length of docs")

    from rank_bm25 import BM25Okapi
    bm25_okapi = BM25Okapi(tokenized_docs)

    bm25_retriever = BM25Retriever(
        docs=docs,
        preprocess_func=jieba_cut,
        tokenized_documents=tokenized_docs,
        k=k
    )
    bm25_retriever.vectorizer = bm25_okapi

    BM25_CACHE_PATH.parent.mkdir(exist_ok=True, parents=True)
    with open(BM25_CACHE_PATH, "wb") as f:
        pickle.dump(bm25_retriever, f, protocol=pickle.HIGHEST_PROTOCOL)
    fingerprint_path.write_text(fingerprint)

    return bm25_retriever


def load_or_create_tokenized_docs(domain: str, docs: list[Document]) -> list[list[str]]:
    """优先加载缓存的分词结果，不存在则执行并行分词并保存缓存"""
    TOKENIZED_CACHE_PATH = PROJECT_ROOT / "cache" / "faiss" / f"tokenized_docs_{domain}.pkl"
    fingerprint_path = TOKENIZED_CACHE_PATH.with_suffix(".sha256")
    fingerprint = corpus_fingerprint(docs)

    # 缓存存在且有效则直接加载
    if (TOKENIZED_CACHE_PATH.exists() and TOKENIZED_CACHE_PATH.stat().st_size > 0
            and fingerprint_path.exists() and fingerprint_path.read_text() == fingerprint):
        try:
            with open(TOKENIZED_CACHE_PATH, "rb") as f:
                tokenized_docs = pickle.load(f)
            # 校验分词结果与文档数量匹配
            if len(tokenized_docs) == len(docs):
                logger.info(f"成功加载{domain}领域的分词结果缓存")
                return tokenized_docs
            else:
                logger.warning("分词缓存与文档数量不匹配，重新分词")
        except (pickle.UnpicklingError, EOFError) as e:
            logger.warning(f"分词缓存文件损坏，重新分词：{e}")
        TOKENIZED_CACHE_PATH.unlink(missing_ok=True)

    # 多线程并行分词（带进度条）
    logger.info(f"{domain}领域无分词缓存，开始并行分词...")
    max_workers = min(32, (os.cpu_count() or 4) * 2)
    doc_texts = [doc.page_content for doc in docs]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(jieba_cut, text) for text in doc_texts]
        # 进度条显示
        for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                desc="多线程并行分词中",
                unit="文档",
                ncols=80
        ):
            try:
                future.result()
            except Exception as e:
                logger.warning(f"单个文档分词失败：{e}")
                # 失败时返回空列表，保证后续结果结构一致
                future.set_result([])

    # 还原分词结果顺序
    tokenized_docs = [future.result() for future in futures]

    # 保存分词缓存
    TOKENIZED_CACHE_PATH.parent.mkdir(exist_ok=True, parents=True)
    try:
        with open(TOKENIZED_CACHE_PATH, "wb") as f:
            pickle.dump(tokenized_docs, f, protocol=pickle.HIGHEST_PROTOCOL)
        fingerprint_path.write_text(fingerprint)
        logger.info(f"成功保存{domain}领域的分词结果缓存")
    except Exception as e:
        logger.error(f"保存分词缓存失败：{e}")
        if TOKENIZED_CACHE_PATH.exists():
            TOKENIZED_CACHE_PATH.unlink(missing_ok=True)

    return tokenized_docs


def switch_domain(new_domain: str, clear_old_cache: bool = True) -> dict:
    """
    动态切换RAG领域
    :param new_domain: 新领域名称
    :param clear_old_cache: 是否清理旧领域的检索器缓存
    :return: 切换结果字典
    """
    # 1. 校验领域名称（防止路径穿越）
    validate_domain(new_domain)

    # 检查FAISS向量库是否存在
    faiss_dir = PROJECT_ROOT / f"rag/faiss_langchain_db_{new_domain}"
    index_file = faiss_dir / "index.faiss"
    pkl_file = faiss_dir / "index.pkl"

    if not (index_file.exists() and pkl_file.exists()):
        logger.error(f"新领域{new_domain}的FAISS向量库不存在：{faiss_dir}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"未找到{new_domain}领域的FAISS向量库，请先初始化该领域数据"
        )

    # 2. 记录旧领域
    old_domain = global_config.domain

    # 3. 清理旧领域缓存（若需要）
    if clear_old_cache:
        old_cache_keys = [key for key in retriever_cache if key[0] == old_domain]
        for cache_key in old_cache_keys:
            retriever_cache.pop(cache_key, None)
        logger.info(f"\n已清理旧领域{old_domain}的{len(old_cache_keys)}个检索器缓存")

    # 4. 更新全局领域配置
    global_config.domain = new_domain
    logger.info(f"领域已成功切换：{old_domain} → {new_domain}")

    # 5. 预加载新领域检索器（可选，提升首次查询速度）
    try:
        get_retriever(new_domain, SETTINGS["retrieval"]["candidate_k"])
        logger.info(f"已预加载新领域{new_domain}的检索器")
    except Exception as e:
        logger.error(f"预加载新领域{new_domain}检索器失败：{e}")
        # 预加载失败不回滚领域切换，仅提示
        return {
            "old_domain": old_domain,
            "new_domain": new_domain,
            "status": "domain_switched_but_preload_failed",
            "message": f"领域切换成功，但预加载检索器失败：{str(e)}"
        }

    return {
        "old_domain": old_domain,
        "new_domain": new_domain,
        "status": "success",
        "message": "领域切换成功且已预加载检索器"
    }


# ===================== 文档处理函数 =====================
# def remove_duplicates(
#         docs: list[Document],
#         query_embedding: list[float],
#         embedding_model: OllamaEmbeddings
# ) -> tuple[list[Document], list]:
#     """文档去重并计算相似度"""
#     seen_contents = set()
#     similarity_scores = []
#     unique_docs = []
#
#     for doc in docs:
#         if not doc.page_content.strip():
#             logger.warning("空文档，跳过")
#             continue
#
#         entries = doc.page_content.split('\n')
#         unique_entries = []
#         for entry in entries:
#             entry_stripped = entry.strip()
#             if entry_stripped and entry_stripped not in seen_contents:
#                 seen_contents.add(entry_stripped)
#                 unique_entries.append(entry_stripped)
#
#         if not unique_entries:
#             continue
#
#         # 计算相似度
#         try:
#             doc_embeddings = embedding_model.embed_documents(unique_entries)
#             similarities = cosine_similarity([query_embedding], doc_embeddings)[0]
#         except Exception as e:
#             logger.error(f"计算相似度失败：{e}")
#             similarities = [0.0] * len(unique_entries)
#
#         similarity_scores.append(list(zip(unique_entries, similarities)))
#         new_content = '\n'.join(unique_entries)
#         unique_docs.append(Document(metadata=doc.metadata, page_content=new_content))
#
#     return unique_docs, similarity_scores


def reorder_documents(docs: list[Document]) -> Sequence[Document]:
    """长上下文文档排序"""
    if not docs:
        return []
    try:
        reorderer = LongContextReorder()
        return reorderer.transform_documents(docs)
    except Exception as e:
        logger.error(f"文档排序失败：{e}")
        return docs


def build_image_refs(img_path: str) -> list[dict]:
    """将 metadata 中的图片路径字段转换为前端可展示的图片引用。"""
    if not img_path or img_path == "NO_IMAGE_AVAILABLE":
        return []

    refs = []
    for image_name in img_path.split("<DELIMITER>"):
        image_name = image_name.strip()
        if not image_name:
            continue
        refs.append({
            "name": image_name,
            "url": f"/images/{image_name}"
        })
    return refs


def expand_query(query: str) -> str:
    """补充常见质量规范问法的同义表达，提升召回。"""
    additions = []
    if "公差" in query:
        additions.extend(["允差", "偏差", "+/-", "±", "管控", "控制", "判定基准", "标准"])
    if "线宽" in query:
        additions.extend(["线宽线距", "线宽测量", "线宽量测"])
    if "外层" in query:
        additions.extend(["外层蚀刻", "外层检验项目"])

    if not additions:
        return query
    return f"{query} {' '.join(additions)}"


def doc_header_text(doc: Document) -> str:
    headers = doc.metadata.get("headers", [])
    return " ".join(str(item[1]) for item in headers if isinstance(item, (list, tuple)) and len(item) > 1)


def significant_query_terms(query: str) -> list[str]:
    stop_words = {"的", "了", "是", "多少", "什么", "如何", "怎么", "请问"}
    terms = [term.strip() for term in jieba.cut(query) if term.strip() and term.strip() not in stop_words]

    for phrase in ("外层", "线宽", "线距", "公差", "管控", "判定基准"):
        if phrase in query and phrase not in terms:
            terms.append(phrase)

    return [term for term in terms if len(term) > 1]


def rerank_docs(query: str, docs: list[Document]) -> list[Document]:
    """对混合检索候选做轻量重排，避免答案块被 Top K 截断。"""
    terms = significant_query_terms(query)
    scored_docs = []

    for index, doc in enumerate(docs):
        header = doc_header_text(doc)
        content = doc.page_content or ""
        combined = f"{header}\n{content}"
        score = 0.0

        for term in terms:
            if term in header:
                score += 4.0
            if term in content:
                score += 2.0

        if "公差" in query:
            tolerance_markers = ("+/-", "±", "公差", "允差", "偏差", "管控", "控制")
            if any(marker in combined for marker in tolerance_markers):
                score += 6.0
            if "线宽" in query and "线宽" in combined and any(marker in combined for marker in ("+/-", "±", "管控", "控制")):
                score += 10.0
            if any(marker in combined for marker in ("自主件", "批量")):
                score += 3.0

        if "线宽" in query and "线宽线距" in combined:
            score += 5.0
        if "外层" in query and "外层" in combined:
            score += 3.0
        if "记录" in header and "记录" not in query:
            score -= 4.0

        scored_docs.append((score, -index, doc))

    scored_docs.sort(key=lambda item: (item[0], item[1]), reverse=True)
    logger.info("重排结果: %s", [(round(score, 2), doc_header_text(doc).split()[-1:] or [""]) for score, _, doc in scored_docs])
    return [doc for _, _, doc in scored_docs]


# ===================== 核心业务逻辑 =====================
def rank_text(doc: Document) -> str:
    return f"{doc_header_text(doc)}\n{doc.page_content or ''}"


def detect_query_language(text: str) -> str:
    """返回用于日志展示的查询语言，不参与检索和排序。"""
    if re.search(r"[\u4e00-\u9fff]", text):
        return "中文"
    if re.search(r"[A-Za-z]", text):
        return "英文"
    return "其他/无法识别"


def normalized_ratio(value: float, values: Sequence[float]) -> float:
    max_value = max(values) if values else 0.0
    if max_value <= 0:
        return 0.0
    return value / max_value


def ranker_query_terms(query: str) -> list[str]:
    stop_words = {"的", "了", "是", "有", "和", "与", "及", "或", "多少", "什么", "如何", "怎么", "请问"}
    terms = [term.strip() for term in jieba.cut(query) if term.strip() and term.strip() not in stop_words]
    return [term for term in terms if len(term) > 1]


def symbol_intent_score(query: str, text: str) -> float:
    query_markers = set(re.findall(r"[+\-±%≤≥<>/=]+|[A-Za-z]+|\d+(?:\.\d+)?", query))
    text_markers = set(re.findall(r"[+\-±%≤≥<>/=]+|[A-Za-z]+|\d+(?:\.\d+)?", text))
    score = len(query_markers & text_markers) * 0.6

    spec_words = ("公差", "允差", "偏差", "管控", "控制", "标准", "判定", "要求", "范围", "规格")
    if any(word in query for word in spec_words):
        score += sum(0.8 for marker in ("+/-", "±", "%", "≤", ">=", "<=", "管控", "控制", "要求") if marker in text)
    return score


def load_query_intent_config() -> dict:
    try:
        mtime = QUERY_INTENT_CONFIG_PATH.stat().st_mtime
    except FileNotFoundError:
        mtime = None

    if (
        not hasattr(load_query_intent_config, "cache")
        or getattr(load_query_intent_config, "mtime", None) != mtime
    ):
        try:
            with open(QUERY_INTENT_CONFIG_PATH, "r", encoding="utf-8") as f:
                load_query_intent_config.cache = yaml.safe_load(f) or {}
            load_query_intent_config.mtime = mtime
        except FileNotFoundError:
            logger.warning(f"query intent config not found: {QUERY_INTENT_CONFIG_PATH}")
            load_query_intent_config.cache = {}
            load_query_intent_config.mtime = mtime
        except Exception as e:
            logger.warning(f"query intent config load failed: {e}")
            load_query_intent_config.cache = {}
            load_query_intent_config.mtime = mtime
    return load_query_intent_config.cache


def detect_query_intents(query: str, config: dict) -> list[str]:
    matched = []
    intents = config.get("intents", {}) if isinstance(config, dict) else {}
    for intent_name, intent_config in intents.items():
        triggers = intent_config.get("query_triggers", {}) or {}
        terms = triggers.get("terms", []) or []
        patterns = triggers.get("patterns", []) or []
        if any(term and term in query for term in terms):
            matched.append(intent_name)
            continue
        for pattern in patterns:
            try:
                if re.search(pattern, query, flags=re.IGNORECASE):
                    matched.append(intent_name)
                    break
            except re.error:
                logger.warning(f"invalid query intent regex: {pattern}")
    return matched or ["general"]


def active_rank_weights(intents: list[str], config: dict) -> dict:
    defaults = (config.get("ranker_defaults", {}) or {}).get("weights", {}) or {}
    weights = {
        "tfidf_score": float(defaults.get("tfidf_score", 0.52)),
        "lexical_score": float(defaults.get("lexical_score", 0.34)),
        "symbol_score": float(defaults.get("symbol_score", 0.04)),
        "original_rank_score": float(defaults.get("original_rank_score", 0.10)),
    }
    configured_intents = config.get("intents", {}) if isinstance(config, dict) else {}
    for intent in intents:
        rank_features = (configured_intents.get(intent, {}) or {}).get("rank_features", {}) or {}
        if rank_features:
            weights.update({name: float(value) for name, value in rank_features.items()})
            break
    return weights


def active_doc_markers(intents: list[str], config: dict) -> list[str]:
    markers = []
    configured_intents = config.get("intents", {}) if isinstance(config, dict) else {}
    for intent in intents:
        markers.extend((configured_intents.get(intent, {}) or {}).get("doc_markers", []) or [])
    return list(dict.fromkeys(marker for marker in markers if marker))


def ranker_query_terms(query: str) -> list[str]:
    config = load_query_intent_config()
    stop_words = set((config.get("ranker_defaults", {}) or {}).get("stop_words", []) or [])
    terms = [term.strip() for term in jieba.cut(query) if term.strip() and term.strip() not in stop_words]
    return [term for term in terms if len(term) > 1]


def symbol_intent_score(query: str, text: str, doc_markers: Sequence[str] | None = None) -> float:
    query_markers = set(re.findall(r"[+\-±%≤≥<>/=]+|[A-Za-z]+|\d+(?:\.\d+)?", query))
    text_markers = set(re.findall(r"[+\-±%≤≥<>/=]+|[A-Za-z]+|\d+(?:\.\d+)?", text))
    score = len(query_markers & text_markers) * 0.6
    score += sum(0.8 for marker in (doc_markers or []) if marker in text)
    return score


def rank_candidates(query: str, docs: list[Document]) -> list[Document]:
    """Fast local reranker over candidates returned by FAISS + BM25."""
    if len(docs) <= 1:
        logger.info("候选文档数量=%s，无需执行重排", len(docs))
        return docs

    texts = [rank_text(doc) for doc in docs]
    terms = ranker_query_terms(query)
    intent_config = load_query_intent_config()
    intents = detect_query_intents(query, intent_config)
    rank_weights = active_rank_weights(intents, intent_config)
    doc_markers = active_doc_markers(intents, intent_config)
    logger.info("查询意图：%s；重排权重：%s", intents, rank_weights)

    try:
        tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), max_features=20000)
        matrix = tfidf.fit_transform([query] + texts)
        tfidf_scores = cosine_similarity(matrix[0], matrix[1:]).ravel().tolist()
    except Exception as e:
        logger.warning(f"TF-IDF 重排失败，将使用词项匹配分数：{e}")
        tfidf_scores = [0.0] * len(docs)

    lexical_scores = []
    symbol_scores = []
    for doc, text in zip(docs, texts):
        header = doc_header_text(doc)
        lexical = 0.0
        for term in terms:
            if term in header:
                lexical += 3.0
            if term in doc.page_content:
                lexical += 1.0
        lexical_scores.append(lexical)
        symbol_scores.append(symbol_intent_score(query, text, doc_markers))

    scored_docs = []
    for index, doc in enumerate(docs):
        original_rank_score = 1.0 / (index + 1)
        score = (
            rank_weights["tfidf_score"] * normalized_ratio(tfidf_scores[index], tfidf_scores)
            + rank_weights["lexical_score"] * normalized_ratio(lexical_scores[index], lexical_scores)
            + rank_weights["symbol_score"] * normalized_ratio(symbol_scores[index], symbol_scores)
            + rank_weights["original_rank_score"] * original_rank_score
        )
        scored_docs.append((score, -index, doc))

    scored_docs.sort(key=lambda item: (item[0], item[1]), reverse=True)
    score_details = []
    for rank, (score, negative_index, doc) in enumerate(scored_docs, start=1):
        original_index = -negative_index
        title = " / ".join(doc_header_text(doc).split()[-2:]) or "无标题"
        content_preview = re.sub(r"\s+", " ", doc.page_content or "")[:60]
        score_details.append({
            "排名": rank,
            "标题": title,
            "TF-IDF余弦相似度": round(float(tfidf_scores[original_index]), 4),
            "词项匹配分": round(float(lexical_scores[original_index]), 4),
            "符号意图分": round(float(symbol_scores[original_index]), 4),
            "综合得分": round(float(score), 4),
            "内容预览": content_preview,
        })
    logger.info("候选文档相似度及重排结果：%s", score_details)
    return [doc for _, _, doc in scored_docs]


def ask(
        query: str,
        domain: str,
        k: int | None = None,
        candidate_k: int | None = None,
        long_context_reorder: bool = True,
        retrieval_mode: str = "hybrid",
) -> list[dict]:
    """
    核心查询函数
    """

    # 纠正 query

    if not query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="查询内容不能为空"
        )

    query_str = query.strip()
    if domain == "index" and (PROJECT_ROOT / "rag" / "mineru_corpus.jsonl").is_file():
        retrieval_mode = "bm25"
    k = TOP_K if k is None else k
    candidate_k = CANDIDATE_K if candidate_k is None else candidate_k
    candidate_k = max(k, candidate_k)
    retriever = get_retriever(domain, candidate_k)
    logger.info(
        "开始处理查询：%s；查询语言：%s；知识库领域：%s；top_k=%s；candidate_k=%s",
        query_str,
        detect_query_language(query_str),
        domain,
        k,
        candidate_k,
    )

    try:
        start = time.time()
        if retrieval_mode == "bm25":
            if domain == "index" and (PROJECT_ROOT / "rag" / "mineru_corpus.jsonl").is_file():
                mixed_docs = retriever.invoke(query_str)[:candidate_k]
            else:
                bm25_retriever, _ = get_faiss_retriever(domain, candidate_k)
                mixed_docs = bm25_retriever.invoke(query_str)[:candidate_k]
        else:
            try:
                mixed_docs = retriever.invoke(query_str)[:candidate_k]
            except Exception as e:
                logger.warning(f"混合检索失败，降级为仅使用 BM25：{e}")
                bm25_retriever, _ = get_faiss_retriever(domain, candidate_k)
                mixed_docs = bm25_retriever.invoke(query_str)[:candidate_k]
                retrieval_mode = "bm25"
        mixed_docs = rank_candidates(query_str, mixed_docs)[:k]
        retrieval_mode_name = "FAISS + BM25 混合检索" if retrieval_mode == "hybrid" else "BM25 降级检索"
        logger.info("检索模式：%s", retrieval_mode_name)
        logger.info(
            f"混合检索耗时：{time.time() - start:.4f}s，top_k={k}，candidate_k={candidate_k}，返回{len(mixed_docs)}条"
        )

        if not mixed_docs:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="未找到相关文档"
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"检索失败：{e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"检索服务异常：{str(e)}"
        )

    if not DEBUG_LINE_SIMILARITY:
        start = time.time()
        reordered_docs = reorder_documents(mixed_docs) if long_context_reorder else mixed_docs
        logger.info(f"document reorder cost: {time.time() - start:.4f}s")

        result = []
        for doc in reordered_docs:
            img_path = doc.metadata.get("img_path", "NO_IMAGE_AVAILABLE")
            result.append({
                "content": doc.page_content,
                "img_path": img_path,
                "image_urls": build_image_refs(img_path),
                "metadata": doc.metadata
            })

        logger.info(f"查询处理完成：{query_str}，返回 {len(result)} 个文本块")
        return result

    start = time.time()
    query_embedding = model.embed_query(query_str)
    # unique_docs, similarity_scores = remove_duplicates(mixed_docs, query_embedding, get_embedding())
    # logger.info(f"文档去重耗时：{time.time() - start:.4f}s，剩余{len(unique_docs)}条")

    # 直接计算每个文档的相似度分数（不去重）    
    similarity_scores = []   
    for doc in mixed_docs:        
        if not doc.page_content.strip():         
             continue       
    
        # 按行分割文档内容        
        entries = doc.page_content.split('\n')        
        entries = [entry.strip() for entry in entries if entry.strip()]                
        
        if entries:            
            try:                
                # 获取文档条目的嵌入向量                
                doc_embeddings = model.embed_documents(entries)               
                # 计算余弦相似度                
                similarities = cosine_similarity([query_embedding], doc_embeddings)[0]            
            except Exception as e:                
                logger.error(f"计算相似度失败：{e}")                
                similarities = [0.0] * len(entries)                        
            
            # 记录相似度分数            
            doc_scores = []            
            for entry, score in zip(entries, similarities):                
                # 截取内容前80字符，避免日志过长                
                short_entry = entry[:80] + "..." if len(entry) > 80 else entry               
                doc_scores.append((short_entry, float(score)))            
            similarity_scores.append(doc_scores)    
    
    # 输出相似度分数日志    
    logger.info(f"查询全部（未过滤）每条文档的相似度分数: {similarity_scores}")    
    logger.info(f"相似度计算耗时：{time.time() - start:.4f}s")

    start = time.time()
    reordered_docs = reorder_documents(mixed_docs) if long_context_reorder else mixed_docs
    logger.info(f"文档排序耗时：{time.time() - start:.4f}s")

    page_contents = [doc.page_content for doc in reordered_docs]
    if len(set(page_contents)) < len(page_contents):
        logger.warning(f"查询[{query_str}]返回重复文档，清理缓存")
        retriever_cache.pop((domain, k), None)

    result = []
    for doc in reordered_docs:
        img_path = doc.metadata.get("img_path", "NO_IMAGE_AVAILABLE")
        result.append({
            "content": doc.page_content,
            "img_path": img_path,
            "image_urls": build_image_refs(img_path),
            "metadata": doc.metadata
        })

    logger.info(f"查询处理完成：{query_str}，返回{len(result)}条结果")
    content_list = [re.sub(r'^#\s*(\d+\.)?|^#\s*Instruction:\s*|[\n\s]+(?=\.)|\s+\.$', '', re.sub(r'\n+', ' ', item['content'])) for item in result]
    logger.info(f"提取后的content结果（前50字）: {[item[:50] for item in content_list]} ")

    return result


# ===================== API 接口 =====================
@app.post("/ask", response_description="检索结果")
async def ask_endpoint(request: QueryRequest):
    """
    RAG检索接口
    """
    try:
        start_time = time.time()
        result = ask(
            request.message,
            global_config.domain,
            request.top_k,
            request.candidate_k,
            request.long_context_reorder,
            request.retrieval_mode,
        )
        total_time = time.time() - start_time
        logger.info(f"接口响应耗时：{total_time:.4f}s")
        return {
            "code": 0,
            "message": "success",
            "data": result,
            "elapsed_time": total_time
        }
    except HTTPException as e:
        logger.error(f"接口异常：{e.detail}")
        return {
            "code": e.status_code,
            "message": e.detail,
            "data": []
        }
    except Exception as e:
        logger.error(f"接口未知异常：{e}", exc_info=True)
        return {
            "code": status.HTTP_500_INTERNAL_SERVER_ERROR,
            "message": "服务器内部错误",
            "data": []
        }


@app.get("/health", response_description="健康检查")
async def health_check():
    """服务健康检查接口"""
    return {
        "status": "healthy",
        "domain": global_config.domain,
        "cache_count": len(retriever_cache),
        "timestamp": time.time()
    }


@app.post("/switch_domain", response_description="领域切换结果")
async def switch_domain_endpoint(request: DomainSwitchRequest):
    """
    动态切换RAG检索领域
    """
    try:
        result = switch_domain(request.new_domain, request.clear_cache)
        return {
            "code": 0,
            "message": "领域切换请求处理完成",
            "data": result
        }
    except HTTPException as e:
        logger.error(f"领域切换失败：{e.detail}")
        return {
            "code": e.status_code,
            "message": e.detail,
            "data": {}
        }
    except Exception as e:
        logger.error(f"领域切换未知异常：{e}", exc_info=True)
        return {
            "code": status.HTTP_500_INTERNAL_SERVER_ERROR,
            "message": f"领域切换失败：{str(e)}",
            "data": {}
        }


@app.get("/current_domain", response_description="当前领域信息")
async def get_current_domain():
    """查询当前生效的RAG领域"""
    return {
        "code": 0,
        "message": "success",
        "data": {
            "current_domain": global_config.domain,
            "initialized": global_config.initialized
        }
    }


# 检查网页及双范围评测功能独立维护，主服务只注入检索函数和当前领域。
from checker_routes import create_checker_router

app.include_router(
    create_checker_router(
        project_root=PROJECT_ROOT,
        settings=SETTINGS,
        ask_func=ask,
        get_domain=lambda: global_config.domain,
    )
)


# ===================== 启动服务 =====================
if __name__ == "__main__":
    # 预加载检索器
    try:
        get_retriever(global_config.domain, SETTINGS["retrieval"]["candidate_k"])
        logger.info("检索器预加载完成，服务启动中...")
    except HTTPException as e:
        if e.status_code == 404:
            logger.warning("尚未导入文献索引；服务已启动，导入论文后可检索")
        else:
            logger.error(f"检索器预加载失败：{e}")
            sys.exit(1)
    except HTTPException as e:
        if e.status_code == 404:
            logger.warning("尚未导入文献索引；服务已启动，导入论文后可检索")
        else:
            logger.error(f"检索器预加载失败：{e}")
            sys.exit(1)
    except Exception as e:
        logger.error(f"检索器预加载失败：{e}")
        sys.exit(1)

    # 启动FastAPI服务
    import asyncio
    from uvicorn import Config, Server


    async def main():
        server_host = os.getenv("EVIDENCEFLOW_RAG_HOST", SETTINGS["server"]["host"])
        server_port = int(os.getenv("EVIDENCEFLOW_RAG_PORT", SETTINGS["server"]["port"]))
        server_url = f"http://{server_host}:{server_port}"
        config = Config(
            app,
            host=server_host,
            port=server_port,
            log_level="warning"
        )
        server = Server(config)
        # 启动前打印提示
        end = time.time()
        logger.info(f"服务启动总耗时：{end - run_time:.4f}s")
        print(f"\n服务已成功启动！耗时：{end - run_time:.4f}s")
        print(f"接口地址: {server_url}")
        print(f"GUI地址: {server_url}/gui")
        print(f"文档地址: {server_url}/docs")
        print(f"健康检查: {server_url}/health")
        print(f"查看当前知识库: {server_url}/current_domain")
        print(f"当前检索领域: {global_config.domain}\n")
        print("当前数据库类型:faiss")
        await server.serve()

    asyncio.run(main())
