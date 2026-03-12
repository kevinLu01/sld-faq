"""
Gradio UI for the LangGraph Research Agent.
Run: python gradio_app.py
"""

import asyncio
import uuid
from pathlib import Path

import gradio as gr
from langchain_core.messages import HumanMessage

from app.agent.graph import build_graph
from app.config import settings
from app.ingestion.pipeline import ingest_file, ingest_url
from app.memory.conversation import load_conversation_history

# Build graph once at startup
graph = build_graph()


# ── helpers ──────────────────────────────────────────────────────────────────

def _run(coro):
    """Run an async coroutine from sync Gradio handlers."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result()
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


def _format_citations(citations: list) -> str:
    if not citations:
        return ""
    lines = ["\n\n---\n**参考来源**"]
    for i, c in enumerate(citations, 1):
        title = c.get("title", "未知来源")
        url = c.get("url")
        src_type = c.get("source_type", "")
        icon = {"vector": "📄", "web_search": "🌐", "url_fetch": "🔗"}.get(src_type, "📎")
        if url:
            lines.append(f"{icon} [{i}] [{title}]({url})")
        else:
            lines.append(f"{icon} [{i}] {title}")
    return "\n".join(lines)


# ── chat handler ──────────────────────────────────────────────────────────────

async def _stream_query(query: str, session_id: str, history: list, max_iter: int):
    """Generator: yields (history, status) tuples as tokens arrive."""
    initial_state = {
        "messages": [HumanMessage(content=query)],
        "query": query,
        "session_id": session_id,
        "tools_to_use": [],
        "tools_used": [],
        "vector_results": [],
        "web_results": [],
        "url_results": [],
        "citations": [],
        "needs_more_research": False,
        "iteration_count": 0,
        "max_iterations": max_iter,
        "conversation_history": load_conversation_history(session_id),
        "refined_query": None,
        "final_answer": None,
    }
    config = {"configurable": {"thread_id": session_id}}

    partial = ""
    citations = []
    status = "🔍 思考中..."

    async for event in graph.astream_events(initial_state, config=config, version="v2"):
        etype = event.get("event")
        name = event.get("name", "")

        if etype == "on_chain_start" and name == "planner":
            status = "🗂 规划检索策略..."

        elif etype == "on_chain_start" and name in (
            "vector_search_node", "tavily_search_node", "url_fetch_node"
        ):
            labels = {
                "vector_search_node": "📄 检索本地文档...",
                "tavily_search_node": "🌐 实时网络搜索...",
                "url_fetch_node": "🔗 抓取网页内容...",
            }
            status = labels[name]

        elif etype == "on_chain_start" and name == "synthesizer":
            status = "✍️ 生成答案..."

        elif etype == "on_chat_model_stream":
            chunk = event["data"].get("chunk")
            if chunk and chunk.content:
                # Only stream tokens from synthesizer (skip planner tokens)
                node_tags = event.get("tags", [])
                if "synthesizer" in str(event.get("metadata", {})).lower() or not partial:
                    partial += chunk.content
                    cur_history = history + [[query, partial]]
                    yield cur_history, status

        elif etype == "on_chain_end" and name == "responder":
            output = event["data"].get("output", {})
            citations = output.get("citations", [])

    # Final message with citations appended
    final_text = partial + _format_citations(citations)
    yield history + [[query, final_text]], "✅ 完成"


def chat(query: str, history: list, session_id: str, max_iter: int):
    """Sync wrapper for Gradio — streams via generator."""
    if not query.strip():
        yield history, session_id, ""
        return

    async def _gen():
        results = []
        async for h, s in _stream_query(query, session_id, history, max_iter):
            results.append((h, s))
        return results

    # Stream synchronously
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    async def _run_gen():
        async for h, s in _stream_query(query, session_id, history, int(max_iter)):
            yield h, s

    import queue
    q: queue.Queue = queue.Queue()

    def _thread():
        async def _collect():
            async for item in _run_gen():
                q.put(item)
            q.put(None)  # sentinel
        asyncio.run(_collect())

    import threading
    t = threading.Thread(target=_thread, daemon=True)
    t.start()

    while True:
        item = q.get()
        if item is None:
            break
        h, s = item
        yield h, session_id, s

    t.join()


# ── upload handler ────────────────────────────────────────────────────────────

def upload_docs(files):
    if not files:
        return "请选择文件。"

    import mimetypes
    results = []
    for f in files:
        path = f.name
        filename = Path(path).name
        mime, _ = mimetypes.guess_type(path)
        mime = mime or "application/octet-stream"
        try:
            result = _run(ingest_file(path, filename, mime))
            results.append(f"✅ {filename} — 摄入 {result['chunks_ingested']} 个片段")
        except Exception as e:
            results.append(f"❌ {filename} — 错误: {e}")

    return "\n".join(results)


def ingest_url_handler(url: str):
    url = url.strip()
    if not url:
        return "请输入 URL。"
    try:
        result = _run(ingest_url(url))
        return f"✅ 摄入完成 — {result['chunks_ingested']} 个片段\n{url}"
    except Exception as e:
        return f"❌ 失败: {e}"


# ── Gradio UI ─────────────────────────────────────────────────────────────────

def new_session():
    return str(uuid.uuid4())


with gr.Blocks(title="Research Agent", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🔬 Research Agent\n基于 LangGraph + Claude 的研究助手")

    session_id = gr.State(value=str(uuid.uuid4()))

    with gr.Tabs():

        # ── Chat Tab ──
        with gr.Tab("💬 对话"):
            chatbot = gr.Chatbot(
                label="对话",
                height=520,
                bubble_full_width=False,
                show_copy_button=True,
            )
            with gr.Row():
                query_box = gr.Textbox(
                    placeholder="输入问题，按 Enter 发送...",
                    show_label=False,
                    scale=9,
                    lines=1,
                )
                send_btn = gr.Button("发送", variant="primary", scale=1)

            with gr.Row():
                status_box = gr.Textbox(
                    label="状态",
                    value="就绪",
                    interactive=False,
                    scale=8,
                )
                max_iter = gr.Slider(
                    minimum=1, maximum=5, value=3, step=1,
                    label="最大检索轮次",
                    scale=2,
                )

            clear_btn = gr.Button("🗑 清空对话 / 新会话", size="sm")

            def clear_chat():
                return [], new_session(), "就绪"

            send_btn.click(
                chat,
                inputs=[query_box, chatbot, session_id, max_iter],
                outputs=[chatbot, session_id, status_box],
            ).then(lambda: "", outputs=query_box)

            query_box.submit(
                chat,
                inputs=[query_box, chatbot, session_id, max_iter],
                outputs=[chatbot, session_id, status_box],
            ).then(lambda: "", outputs=query_box)

            clear_btn.click(clear_chat, outputs=[chatbot, session_id, status_box])

        # ── Knowledge Base Tab ──
        with gr.Tab("📚 知识库"):
            gr.Markdown("### 上传文档\n支持 PDF、DOCX、TXT 格式")
            file_input = gr.File(
                file_count="multiple",
                file_types=[".pdf", ".docx", ".doc", ".txt", ".md"],
                label="选择文件",
            )
            upload_btn = gr.Button("上传并摄入", variant="primary")
            upload_result = gr.Textbox(label="结果", lines=5, interactive=False)

            upload_btn.click(upload_docs, inputs=file_input, outputs=upload_result)

            gr.Markdown("---\n### 摄入网页 URL")
            url_input = gr.Textbox(placeholder="https://example.com/article", label="URL")
            ingest_btn = gr.Button("摄入", variant="primary")
            ingest_result = gr.Textbox(label="结果", lines=3, interactive=False)

            ingest_btn.click(ingest_url_handler, inputs=url_input, outputs=ingest_result)

        # ── Settings Tab ──
        with gr.Tab("⚙️ 配置信息"):
            gr.Markdown(f"""
| 配置项 | 值 |
|---|---|
| 模型 | `{settings.CLAUDE_MODEL}` |
| Base URL | `{settings.ANTHROPIC_BASE_URL}` |
| Embedding | `{settings.EMBEDDING_MODEL}` |
| 向量库路径 | `{settings.CHROMA_PERSIST_DIR}` |
| Tavily | `{"已配置 ✅" if settings.TAVILY_API_KEY else "未配置 ❌"}` |
""")


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True,
    )
