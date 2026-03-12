import json
import os
from pathlib import Path
from app.config import settings

_CONV_DIR = Path("data/conversations")


def _session_path(session_id: str) -> Path:
    return _CONV_DIR / f"{session_id}.json"


def load_conversation_history(session_id: str) -> list[dict]:
    """Load prior turns for context injection. Returns [] if no history."""
    path = _session_path(session_id)
    if not path.exists():
        return []
    try:
        with open(path, encoding="utf-8") as f:
            history = json.load(f)
        # Return only the last N turns (each turn = 2 entries: user + assistant)
        max_entries = settings.MAX_CONVERSATION_HISTORY_TURNS * 2
        return history[-max_entries:]
    except (json.JSONDecodeError, OSError):
        return []


def save_conversation_turn(
    session_id: str,
    query: str,
    answer: str,
    citations: list,
) -> None:
    """Append a turn to the session's conversation log."""
    _CONV_DIR.mkdir(parents=True, exist_ok=True)
    history = load_conversation_history(session_id)

    history.append({"role": "user", "content": query})
    history.append({"role": "assistant", "content": answer, "citations": citations})

    # Trim to last MAX_TURNS
    max_entries = settings.MAX_CONVERSATION_HISTORY_TURNS * 2
    history = history[-max_entries:]

    path = _session_path(session_id)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
