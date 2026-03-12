# Code Review Report

> 项目：LangGraph RAG Research Agent
> 审查日期：2026-03-13
> 审查人：Senior Python Engineer

---

## 严重问题（需立即修复）

### [app/tools/vector_search.py:24] `collection.count()` 为 0 时 `min()` 参数导致 Chroma 崩溃

- **问题描述**：`n_results=min(k, max(collection.count(), 1))` 的意图是避免 `n_results > 文档数` 报错，但当 collection 为空（`count() == 0`）时，`max(0, 1) == 1`，Chroma 的 `query()` 仍然会因为 collection 里没有任何向量而抛出异常（`ValueError: n_results (1) > number of elements in index (0)`）。应在 count==0 时提前返回空列表。

- **建议修复**：
  ```python
  count = collection.count()
  if count == 0:
      return []
  results = collection.query(
      query_embeddings=[query_embedding],
      n_results=min(k, count),
      include=["documents", "metadatas", "distances"],
  )
  ```

---

### [app/memory/conversation.py:9-10] 路径穿越（Path Traversal）漏洞

- **问题描述**：`_session_path` 直接将用户提供的 `session_id` 拼接成文件路径，`_CONV_DIR / f"{session_id}.json"`，而 `session_id` 来自客户端（HTTP body 或 WebSocket payload）。恶意请求者可传入 `../../etc/passwd` 等路径，读写或覆盖服务器上任意文件。

- **建议修复**：对 `session_id` 校验格式（仅允许 UUID 格式），并在使用前解析为绝对路径后验证其是否在 `_CONV_DIR` 内：
  ```python
  import re
  _UUID_RE = re.compile(r'^[0-9a-f-]{36}$', re.IGNORECASE)

  def _session_path(session_id: str) -> Path:
      if not _UUID_RE.match(session_id):
          raise ValueError(f"Invalid session_id: {session_id}")
      path = (_CONV_DIR / f"{session_id}.json").resolve()
      if not path.is_relative_to(_CONV_DIR.resolve()):
          raise ValueError("Path traversal detected")
      return path
  ```

---

### [app/api/routes/upload.py:39] 文件名注入 / 路径穿越风险

- **问题描述**：`save_path = str(upload_dir / f"{uuid4()}_{file.filename}")`，`file.filename` 来自客户端 multipart 表单，可以是 `../../../etc/cron.d/evil` 等含有路径分隔符的字符串。即使前缀了 UUID，路径拼接仍然危险，`Path` 对象在 Windows/POSIX 下的表现不一致。

- **建议修复**：使用 `Path(file.filename).name` 提取纯文件名部分（去掉任何目录组件）：
  ```python
  safe_name = Path(file.filename).name  # 仅保留文件名部分
  save_path = upload_dir / f"{uuid4()}_{safe_name}"
  ```

---

### [app/api/routes/upload.py:44] 上传文件路径直接传给 `load_document`（MIME 类型可伪造）

- **问题描述**：MIME 类型检查使用的是 `file.content_type`，这是由客户端在 HTTP 请求头中声明的，而非服务端实际探测的，攻击者可以将任意文件（如脚本）声明为 `text/plain` 绕过校验后交给 `TextLoader` 处理。整个接口缺少文件大小限制，可被用于 DoS（上传超大文件耗尽内存）。

- **建议修复**：
  1. 服务端使用 `python-magic` 或 `filetype` 检测真实 MIME 类型。
  2. 在 `await file.read()` 前加大小限制：
  ```python
  MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB
  contents = await file.read(MAX_UPLOAD_BYTES + 1)
  if len(contents) > MAX_UPLOAD_BYTES:
      raise HTTPException(status_code=413, detail="File too large (max 50 MB).")
  ```

---

### [app/ingestion/pipeline.py:27-29] `ingest_url` 缺少 SSRF 防护

- **问题描述**：`ingest_url` 直接使用用户提供的 URL 发起 HTTP 请求，没有任何限制。攻击者可传入内网地址（`http://169.254.169.254/metadata`、`http://localhost:5432`）来探测或攻击内网服务（SSRF）。

- **建议修复**：在发起请求前解析目标 IP，拒绝 RFC-1918 私有地址、loopback 地址及 link-local 地址：
  ```python
  import ipaddress, socket
  def _is_safe_url(url: str) -> bool:
      from urllib.parse import urlparse
      host = urlparse(url).hostname
      try:
          ip = ipaddress.ip_address(socket.gethostbyname(host))
          return ip.is_global and not ip.is_loopback and not ip.is_private
      except Exception:
          return False
  ```
  同样的问题存在于 `app/tools/url_fetch.py`。

---

### [app/agent/nodes.py:75 / 84 / 102] 并行 tool node 的 `tools_used` 存在竞态条件（Race Condition）

- **问题描述**：三个 tool node（`vector_search_node`、`tavily_search_node`、`url_fetch_node`）在 graph 中被并行执行（`route_after_planner` 返回多个目标节点），每个 node 都通过 `state.get("tools_used", []) + ["xxx"]` 追加自己的工具名，然后写回 `tools_used`。

  由于 `AgentState` 中 `tools_used` 字段是 `list[str]`，没有声明任何 reducer（不同于 `messages` 使用了 `add_messages`），LangGraph 在合并并行节点的输出时，将对该字段执行 **最后写入覆盖（last-write-wins）**，导致最终 `tools_used` 只包含最后一个完成的工具名，另外两个工具的记录丢失。

- **建议修复**：在 `AgentState` 中为 `tools_used` 添加 `add` reducer，或改用 `Annotated[list[str], operator.add]`：
  ```python
  # state.py
  import operator
  from typing import Annotated

  class AgentState(TypedDict):
      ...
      tools_used: Annotated[list[str], operator.add]
  ```
  对 `vector_results`、`web_results`、`url_results` 也同理验证——但这三个字段每次由不同节点独立写入互不重叠，仅最后写入覆盖问题影响 `tools_used`，且 `planner` 在每次迭代开始时会将 `tools_used` 重置为 `[]`，故实际影响有限；但仍应显式声明 reducer 以确保语义正确。

---

### [app/agent/nodes.py:116] `grader` 中 `iteration_count` 比较逻辑与 `planner` 不一致（Off-by-one）

- **问题描述**：`planner` 在返回时将 `iteration_count` 从 0 递增为 1（第一次调用后为 1）；`grader` 判断条件为 `iteration_count < max_iterations`（默认 3），意味着当 `iteration_count == 3` 时不再循环，即最多执行 3 次 planner。但 `route_after_grader`（`router.py:23`）也做了同样的 `< max_iterations` 判断，造成双重检查，两处逻辑必须保持同步否则容易引入 bug。更重要的是：`planner` 在第一次被调用时已将 `iteration_count` 置为 1，而 `grader` 紧随其后检查 `1 < 3` 为真，则允许再次循环，这本身逻辑正确；但若 LLM JSON 解析失败（fallback 路径）且 `needs_more_research` 恒为 True，则循环次数正好是 `max_iterations` 次，而不是文档注释里描述的"更多"。应将循环控制逻辑集中在一处。

- **建议修复**：在 `grader` 中移除重复的 `iteration_count` 判断，让 `route_after_grader` 单独负责循环控制，避免两处各判断一次：
  ```python
  # grader 仅判断内容是否足够
  needs_more = total == 0
  return {"needs_more_research": needs_more}

  # route_after_grader 统一控制迭代上限
  def route_after_grader(state):
      if state.get("needs_more_research") and state.get("iteration_count", 1) < state.get("max_iterations", 3):
          return "planner"
      return "synthesizer"
  ```

---

## 中等问题（建议修复）

### [app/agent/nodes.py:13-20] 每次 LLM 调用都实例化新的 `ChatAnthropic` 对象

- **问题描述**：`_get_llm()` 每次被调用（`planner`、`synthesizer` 各调用一次，且每次 agent 运行都会重新创建）都新建一个 `ChatAnthropic` 实例。HTTP 连接池、SDK 初始化开销被白白浪费。

- **建议修复**：使用 `@lru_cache(maxsize=1)` 缓存实例，或在模块级别创建单例：
  ```python
  from functools import lru_cache

  @lru_cache(maxsize=1)
  def _get_llm() -> ChatAnthropic:
      return ChatAnthropic(...)
  ```
  注意：`ChatAnthropic` 是否线程安全需确认；通常 LangChain LLM 客户端是无状态的，缓存是安全的。

---

### [app/ingestion/embedder.py:28] `embed_documents` 在异步上下文中是同步阻塞调用

- **问题描述**：`embed_and_store` 是同步函数，被 `pipeline.py` 中的 `asyncio.to_thread` 包裹，本身没有问题。但 `embed_documents` 调用的是 `HuggingFaceEmbeddings`，其内部使用 sentence-transformers，可能消耗大量 CPU 时间（对大批量 chunks 而言），虽然已被 `to_thread` 包裹，但需要确保线程池大小能应对并发上传，否则大文件上传会因线程池饱和而变慢。此外 `embed_and_store` 没有批量大小（batch size）控制，对超大文件（数千个 chunks）的单次 `embed_documents` 调用可能导致内存峰值。

- **建议修复**：增加批量处理逻辑（每批 64-128 chunks），并在文档中说明并发上传时的 CPU 瓶颈限制。

---

### [app/memory/conversation.py:36-43] `save_conversation_turn` 存在文件并发写入竞态

- **问题描述**：`save_conversation_turn` 先 `load_conversation_history`（读文件），再修改，再写回（写文件），两步之间没有锁。若同一 `session_id` 并发发起两个请求，可能导致其中一个请求的写入被另一个覆盖（read-modify-write race）。

- **建议修复**：使用文件锁（`fcntl.flock` on Linux 或 `msvcrt.locking` on Windows，或跨平台的 `filelock` 库），或将会话存储改为 SQLite（已有 Chroma 依赖 SQLite，可直接复用）。

---

### [app/agent/nodes.py:27] `_format_history` 的截断逻辑有误

- **问题描述**：`history[-settings.MAX_CONVERSATION_HISTORY_TURNS * 2:]` 按字典条目数截断，但注释说的是"N turns"（每 turn = user + assistant = 2 条）。当 `MAX_CONVERSATION_HISTORY_TURNS=10` 时，`* 2` 是正确的。但 `load_conversation_history` 本身已经做了同样截断（`memory/conversation.py:22-23`），导致历史被截断**两次**——虽然结果相同，但存在冗余逻辑，容易在修改其中一处时产生不一致。

- **建议修复**：`_format_history` 直接遍历传入的 `history`，无需再次截断（截断职责已在 `load_conversation_history` 完成）：
  ```python
  def _format_history(history: list[dict]) -> str:
      if not history:
          return ""
      lines = ["## Prior conversation\n"]
      for turn in history:
          role = "User" if turn["role"] == "user" else "Assistant"
          lines.append(f"**{role}**: {turn['content']}\n")
      return "\n".join(lines)
  ```

---

### [app/agent/graph.py:41-50] `route_after_planner` 返回 `list[str]` 但 graph 使用 `add_conditional_edges` 映射字典

- **问题描述**：`add_conditional_edges` 的第三个参数（路由映射字典）在 LangGraph 中当路由函数返回 `list` 时用于将列表中的每个值映射到节点名。当前代码的 key 和 value 完全相同（`"vector_search_node": "vector_search_node"`），映射字典是冗余的——LangGraph 支持在返回 `list` 时直接传入节点名列表，无需此字典。但更重要的是：若 `route_after_planner` 返回 `["synthesizer"]`（无工具时直接跳 synthesizer），`"synthesizer"` 也需要出现在映射字典的 key 中（当前已有），这部分是正确的。整体没有严重 bug，但冗余的映射字典增加了维护成本，未来新增 tool node 时容易忘记更新字典。

- **建议修复**：验证 LangGraph 版本是否支持省略映射字典（直接返回节点名列表），如支持则简化为：
  ```python
  builder.add_conditional_edges("planner", route_after_planner)
  ```

---

### [app/api/routes/stream.py:74-79] WebSocket 从 `responder` 事件中获取 `citations` 失败

- **问题描述**：`responder` 节点返回的是 `{"messages": [AIMessage(content=answer)]}`，**不包含 `citations`**（`citations` 在 `synthesizer` 节点中写入 state，`responder` 节点并未再次返回它）。因此 `output.get("citations", [])` 永远取到空列表 `[]`，WebSocket 客户端永远收不到引用信息。

- **建议修复**：有两种方案：
  1. 在 `responder` 节点返回值中加入 `citations`（从 state 读取后回传）：
     ```python
     # nodes.py responder
     return {
         "messages": [AIMessage(content=answer)],
         "citations": state.get("citations", []),
     }
     ```
  2. 改为监听 `synthesizer` 的 `on_chain_end` 事件获取 citations，因为 citations 是由 `synthesizer` 写入 state 的。

---

### [app/api/routes/upload.py] `ingest_file` 作为 `BackgroundTasks` 运行时异常无法被客户端感知

- **问题描述**：文件上传后立即返回 `status="ingesting"`，实际 ingestion 在 FastAPI `BackgroundTasks` 中异步运行。若 ingestion 失败（文件解析错误、Chroma 写入失败等），错误会被静默丢弃，客户端无从得知。同时临时文件（`save_path`）在 ingestion 失败时也不会被清理，导致磁盘泄露。

- **建议修复**：
  1. 增加 ingestion 结果的回调通知机制（WebSocket 推送、轮询接口等）。
  2. 在 `ingest_file` 中包裹 try/finally 清理临时文件：
     ```python
     async def ingest_file(file_path, original_filename, mime_type):
         try:
             ...
         finally:
             Path(file_path).unlink(missing_ok=True)
     ```

---

### [app/agent/nodes.py:90-92] `url_fetch_node` 使用原始 `query` 提取 URL，而非 `refined_query`

- **问题描述**：`url_fetch_node` 从 `state["query"]`（原始问题）中提取 URL，而 `vector_search_node` 和 `tavily_search_node` 都优先使用 `refined_query`。如果 planner 对 query 进行了改写，URL 提取逻辑没有跟上，行为不一致。此外 URL 提取正则 `r"https?://[^\s]+"` 会贪婪匹配到句末标点（如 `http://example.com.`），可能产生无效 URL。

- **建议修复**：
  ```python
  query = state.get("refined_query") or state["query"]
  urls = re.findall(r"https?://[^\s,;，。\)）]+", query)
  ```

---

### [gradio_app.py:27-35] `_run` 函数在事件循环已运行时使用 `ThreadPoolExecutor + asyncio.run` 不安全

- **问题描述**：当 Gradio 在有事件循环的环境中运行时，`_run` 通过 `concurrent.futures.ThreadPoolExecutor` 起新线程再 `asyncio.run(coro)`。但同一个协程对象（`coro`）不能被多次 `await`，且此写法完全绕过了调用方的事件循环，可能导致：
  1. `coro` 中的异步上下文（如 `httpx.AsyncClient`）在错误的 loop 中运行。
  2. `asyncio.run` 内部创建新 loop，无法复用父 loop 的连接池等资源。

  `_run` 实际上只在 `upload_docs` 和 `ingest_url_handler` 中使用，这两个都是同步 Gradio 回调。正确做法是使用 `asyncio.run` 直接运行（Gradio 同步回调本身不在事件循环内执行），不需要复杂的 try/except 检测逻辑。

- **建议修复**：
  ```python
  def _run(coro):
      return asyncio.run(coro)
  ```

---

### [gradio_app.py:101-109] `_stream_query` 中 token 过滤逻辑不可靠

- **问题描述**：
  ```python
  if "synthesizer" in str(event.get("metadata", {})).lower() or not partial:
  ```
  这行代码意图是只流式输出 synthesizer 的 token，但判断条件 `"synthesizer" in str(metadata).lower()` 非常脆弱：若 metadata 字典的字符串化结果恰好包含 "synthesizer" 子串（例如某个 key 名），则 planner 的 token 也会被错误地输出。`or not partial` 条件更是会在 `partial` 为空字符串时接受来自任何节点（包括 planner）的第一个 token。

- **建议修复**：通过 `event.get("name")` 或 `event.get("tags")` 来精确判断事件所属节点：
  ```python
  # LangGraph v2 events 中节点信息在 metadata.langgraph_node
  node = event.get("metadata", {}).get("langgraph_node", "")
  if node == "synthesizer":
      partial += chunk.content
      ...
  ```

---

## 轻微问题 / 改进建议

### [app/config.py:17] `CHROMA_PERSIST_DIR` 使用相对路径，行为依赖 CWD

- 相对路径 `"./data/chroma_db"` 在不同启动目录下行为不同（直接 `python app/main.py` vs `uvicorn app.main:app`）。建议改为基于项目根目录的绝对路径，或在 `get_chroma_client()` 中做路径规范化（`Path(settings.CHROMA_PERSIST_DIR).resolve()`）。同理 `UPLOAD_DIR`、`_CONV_DIR`（`memory/conversation.py:6`）也有同样问题，尤其 `_CONV_DIR` 硬编码为 `Path("data/conversations")` 而非读取 `settings`，两者不一致。

---

### [app/ingestion/chunker.py:21] `chunk_id` 使用全局递增索引 `i`，跨文件不唯一

- **问题描述**：`chunk_id = f"{source}::chunk_{i}"` 中的 `i` 是当前批次内的索引，同一文件多次上传（Chroma upsert 幂等）没问题，但如果两个不同来源文件碰巧有相同的 `source` 路径（例如临时文件复用 UUID 极低概率碰撞），chunk_id 会冲突导致错误覆盖。建议使用 UUID 或文件内容哈希作为 `chunk_id` 的一部分。

---

### [app/tools/vector_search.py:22-25] `get_or_create_collection` 在高并发时存在潜在争用

- Chroma 的 `get_or_create_collection` 在多线程/多协程并发首次调用时，SQLite WAL 模式下通常安全，但在多进程部署（多 uvicorn workers）场景下 `lru_cache` 的单例保证会失效，建议文档中明确说明只支持单进程部署，或使用 Chroma HTTP server 模式。

---

### [app/api/schemas.py:8-10] `QueryRequest.query` 缺少长度和内容校验

- `query: str` 没有 `min_length`、`max_length` 约束，用户可以提交空字符串（`"  "`，通过 strip 后为空）或超长字符串（消耗大量 token）。建议：
  ```python
  from pydantic import field_validator
  query: str = Field(..., min_length=1, max_length=2000)

  @field_validator("query")
  @classmethod
  def strip_query(cls, v):
      v = v.strip()
      if not v:
          raise ValueError("query cannot be blank")
      return v
  ```

---

### [app/api/routes/stream.py:33] WebSocket 中 `max_iterations` 未做范围校验

- `max_iterations = int(data.get("max_iterations", 3))` 未校验范围，恶意客户端可传入 `999` 导致大量 LLM 调用消耗 API 费用。建议添加 `max_iterations = min(max(int(data.get("max_iterations", 3)), 1), 5)`。

---

### [app/agent/prompts.py] PLANNER_SYSTEM_PROMPT 使用反斜杠续行拼接字符串，可读性差

- 多处使用 `\` 续行（如第 1、8、21 行），建议统一改用括号包裹的隐式字符串拼接或三引号字符串，提升可读性和编辑友好度。

---

### [app/ingestion/loaders.py:15-16] `docx2txt` 为 `try` 导入，但未在依赖列表中明确声明

- `docx2txt` 在 `try/except ImportError` 外层直接 `import`（实际上是在 try 块内），若未安装则抛出 `ImportError` 被捕获后抛出 `ValueError`，错误信息不清晰。建议在项目 `requirements.txt` / `pyproject.toml` 中明确列出 `docx2txt` 为依赖，并在错误信息中提示安装命令。

---

### [gradio_app.py:133-161] `chat` 函数每次调用都创建新 event loop，loop 不被销毁

- `loop = asyncio.new_event_loop()` 后 `asyncio.set_event_loop(loop)`，但 `loop` 从未被关闭（`loop.close()`），可能导致资源泄漏（文件描述符、线程等）。`_run_gen` 内部的 `asyncio.run` 会创建并关闭自己的 loop，但外层显式创建的 `loop` 对象悬空未关闭。

---

### [app/agent/nodes.py:63-64] Planner 每次迭代重置 `tools_used = []`，导致历史工具记录丢失

- 每次 planner 被重新调用（循环迭代）时，`tools_used` 被置为 `[]`，之前迭代中使用过的工具记录清空。如果后续逻辑（如 grader 或 prompt）需要知道"本次完整研究过程中用了哪些工具"，该信息已丢失。可以考虑区分"本轮工具列表"和"全局工具历史"两个字段。

---

## 总体评价

该项目整体架构清晰，LangGraph 的 StateGraph 使用思路正确，将 planner/tool/grader/synthesizer/responder 各自分离成独立节点，符合 agent 设计最佳实践。异步处理（`asyncio.to_thread` 包裹阻塞 IO、`ainvoke` 调用 LLM）总体到位。FastAPI + LangGraph + Chroma 的技术栈组合合理。

**主要关注点**：

1. **安全性是最紧迫的短板**：路径穿越（session_id、文件名）、MIME 伪造、SSRF 三个问题需立即修复，特别是面向公网部署时风险极高。

2. **并行节点的 State Reducer 缺失**：`tools_used` 在并行执行场景下的 last-write-wins 问题，是 LangGraph 使用中的典型陷阱，需要显式声明 reducer。

3. **WebSocket citations 永远为空**：这是明显的功能性 bug，`responder` 节点不返回 `citations`，导致 streaming 接口的引用功能完全失效。

4. **Gradio 的 token 过滤逻辑**：字符串模糊匹配 metadata 不可靠，应使用 `langgraph_node` 元数据字段精确过滤。

修复优先级建议：安全漏洞 (P0) → WebSocket citations bug (P1) → State Reducer / Chroma 空集合崩溃 (P1) → 其余中等问题 (P2) → 改进建议 (P3)。
