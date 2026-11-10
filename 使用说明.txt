EvidenceFlow 项目工程使用说明
创作者：hxy
版本：不含文献库 / 源码版

一、版本内容
本版只包含源代码和配置模板，不包含 PDF、Markdown、检索片段、证据图片或 Python 运行环境。
本项目用于将研究问题关联到论文证据，生成结论后允许用户回查原文、筛选证据并汇总结果。

二、主要文件
- start.bat / start.ps1：在 Windows 上启动检索服务和网页服务；默认端口分别为 8021、8031。
- stop.bat / stop.ps1：停止由本目录启动的服务。
- evidence-flow/app.py：网页后端、项目/对话记录、RAG 候选调用、DeepSeek 结论归纳和 PDF 定位。
- evidence-flow/static/：网页前端文件。
- rag-service/serve_h_faiss.py、checker_routes.py：RAG 检索接口及辅助路由。
- rag-config/config.yaml：RAG 服务配置。
- pdf-data/mineru_parse.py：可选的 MinerU PDF 解析脚本。
- rag-service/build/build_mineru_corpus.py：把解析结果构建为可检索语料。
- requirements.txt：源码版安装依赖用。
- pdf-data/论文/、pdf-data/mineru-output/、rag-service/rag/：文献及检索数据目录，仅含文献库版预置数据。
- evidence-flow/data/：运行后自动创建本地项目与对话数据库，不随提交包提供。

三、运行方法
1. 安装 Windows 64 位 Python 3.11 或 3.12，并确保 py 或 python 命令可用。
2. 解压后保持 evidenceflow 目录结构不变，双击 start.bat。首次启动会建立 .venv 并从网络安装 requirements.txt。
3. 浏览器访问 http://127.0.0.1:8031/，用完双击 stop.bat。
4. 此版没有文献数据，页面可以启动，但提问检索在导入并解析论文前无法返回证据。

如 8021 或 8031 被占用，可在 PowerShell 的项目目录运行：
  .\start.ps1 -RagPort 8121 -WebPort 8131
然后访问 http://127.0.0.1:8131/。本项目默认只监听本机 127.0.0.1。

四、API 配置与安全
1. 将 evidence-flow/.env.example 复制为 evidence-flow/.env，在其中的 DEEPSEEK_API_KEY= 后填写你自己的 Key。
2. 默认模型和地址写在同一配置文件中：DEEPSEEK_MODEL 与 DEEPSEEK_BASE_URL；如网络需要代理，可设置 DEEPSEEK_PROXY。
3. 只有解析新 PDF 时才需要 MinerU：将 pdf-data/.env.example 复制为 pdf-data/.env，并填写 MINERU_API_KEY。
4. 本提交包不含任何真实 API Key。不要把 .env、日志、个人对话数据库或自己的密钥上传到公开仓库。

五、功能与交互
打开首页后输入研究问题并点击“开始检索与分析”。系统检索候选论文片段，调用 DeepSeek 归纳结论，
在证据核验页展示结论、来源和片段。点击证据可查看 PDF 原页及定位区域；勾选要保留的证据后进入最终结论页。
左侧可创建多个研究项目，按问题保存独立对话，也可删除不需要的对话。
如需启用检索：将有权使用的 PDF 放入 pdf-data/论文/，先在 pdf-data/.env 填入自己的 MinerU Key，运行 `python pdf-data/mineru_parse.py run`。脚本按每批最多 200 页提交解析任务，完成后自动生成 rag-service/rag/mineru_corpus.jsonl。该过程需要联网和 MinerU 额度。

六、运行限制
- DeepSeek 结论生成依赖外网、有效 API Key 和相应的服务额度；没有 Key 时网页和本地检索仍可启动，但无法完成结论生成。
- 未导入文献时，检索与 PDF 回查不可用；这属于源码版的预期状态。
- 当前 RAG 对已解析语料默认使用 BM25；如需 FAISS 向量检索，还需自行准备 Ollama、bge-m3 与对应索引。
- Windows 运行环境仅随含文献库版提供；macOS/Linux 需自行创建 Python 虚拟环境并安装 requirements.txt，两个服务也需分别启动。
- 原论文可能受版权约束，请仅在获准范围内使用和分发含文献库版。
