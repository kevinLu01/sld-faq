import json
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from langchain_core.messages import HumanMessage

from app.memory.conversation import load_conversation_history

router = APIRouter(tags=["stream"])


@router.websocket("/ws/query")
async def websocket_query(websocket: WebSocket):
    """
    WebSocket streaming endpoint.

    Protocol:
      Client → {"query": "...", "session_id": "abc", "max_iterations": 3}
      Server → {"type": "token", "content": "..."}   (repeated per token)
      Server → {"type": "citations", "data": [...]}
      Server → {"type": "done"}
      Server → {"type": "error", "message": "..."}   (on error)
    """
    await websocket.accept()

    try:
        data = await websocket.receive_json()
    except Exception:
        await websocket.send_json({"type": "error", "message": "Invalid JSON payload."})
        await websocket.close()
        return

    query = data.get("query", "").strip()
    session_id = data.get("session_id", "default")
    max_iterations = int(data.get("max_iterations", 3))

    if not query:
        await websocket.send_json({"type": "error", "message": "Query cannot be empty."})
        await websocket.close()
        return

    graph = websocket.app.state.graph

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
        "max_iterations": max_iterations,
        "conversation_history": load_conversation_history(session_id),
        "refined_query": None,
        "final_answer": None,
    }

    config = {"configurable": {"thread_id": session_id}}
    citations_sent = False

    try:
        async for event in graph.astream_events(initial_state, config=config, version="v2"):
            event_type = event.get("event")

            # Stream LLM tokens (only from synthesizer / planner nodes)
            if event_type == "on_chat_model_stream":
                chunk = event["data"].get("chunk")
                if chunk and chunk.content:
                    await websocket.send_json({"type": "token", "content": chunk.content})

            # Send citations when responder node completes
            elif event_type == "on_chain_end" and event.get("name") == "responder":
                output = event["data"].get("output", {})
                if not citations_sent:
                    citations = output.get("citations", [])
                    await websocket.send_json({"type": "citations", "data": citations})
                    citations_sent = True

        await websocket.send_json({"type": "done"})

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass
