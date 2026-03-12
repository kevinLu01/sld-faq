# ==============================================================================
# tests/test_memory.py
#
# Unit tests for app.memory.conversation:
#   - load_conversation_history
#   - save_conversation_turn
#
# Tests use pytest's tmp_path fixture to redirect file I/O to a temporary
# directory, keeping the real data/conversations directory untouched.
# ==============================================================================

import json
import pytest
from pathlib import Path

# Pre-defined valid UUIDs for use as session_ids (P0: _session_path now requires UUID format)
_UUID_MISSING   = "00000000-0000-0000-0000-000000000001"
_UUID_EXISTING  = "00000000-0000-0000-0000-000000000002"
_UUID_TRUNCATE  = "00000000-0000-0000-0000-000000000003"
_UUID_CORRUPT   = "00000000-0000-0000-0000-000000000004"
_UUID_EMPTY     = "00000000-0000-0000-0000-000000000005"
_UUID_EXACT     = "00000000-0000-0000-0000-000000000006"
_UUID_NEW       = "00000000-0000-0000-0000-000000000007"
_UUID_CONTENT   = "00000000-0000-0000-0000-000000000008"
_UUID_APPEND_S2 = "00000000-0000-0000-0000-000000000009"
_UUID_CITE      = "00000000-0000-0000-0000-000000000010"
_UUID_TRIM      = "00000000-0000-0000-0000-000000000011"
_UUID_AUTODIR   = "00000000-0000-0000-0000-000000000012"
_UUID_SESS_A    = "00000000-0000-0000-0000-000000000013"
_UUID_SESS_B    = "00000000-0000-0000-0000-000000000014"
_UUID_UNICODE   = "00000000-0000-0000-0000-000000000015"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _redirect_conv_dir(tmp_path: Path, monkeypatch):
    """
    Patch the module-level _CONV_DIR inside app.memory.conversation to point
    at a temporary directory so tests never touch the real filesystem.
    Returns the patched directory path.
    """
    import app.memory.conversation as conv_mod
    conv_dir = tmp_path / "conversations"
    monkeypatch.setattr(conv_mod, "_CONV_DIR", conv_dir)
    return conv_dir


def _write_session(conv_dir: Path, session_id: str, data: list) -> Path:
    """Write a JSON session file directly, bypassing save_conversation_turn."""
    conv_dir.mkdir(parents=True, exist_ok=True)
    path = conv_dir / f"{session_id}.json"
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return path


# ===========================================================================
# load_conversation_history
# ===========================================================================

class TestLoadConversationHistory:
    """Tests for reading serialised conversation turns from disk."""

    def test_returns_empty_list_when_file_missing(self, tmp_path, monkeypatch):
        """Missing session file should return an empty list, not raise."""
        _redirect_conv_dir(tmp_path, monkeypatch)
        from app.memory.conversation import load_conversation_history
        result = load_conversation_history(_UUID_MISSING)
        assert result == []

    def test_loads_existing_session(self, tmp_path, monkeypatch):
        """Saved turns are returned as a list of dicts."""
        conv_dir = _redirect_conv_dir(tmp_path, monkeypatch)
        turns = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!", "citations": []},
        ]
        _write_session(conv_dir, _UUID_EXISTING, turns)

        from app.memory.conversation import load_conversation_history
        result = load_conversation_history(_UUID_EXISTING)
        assert result == turns

    def test_truncates_to_max_history_turns(self, tmp_path, monkeypatch, mock_settings):
        """
        Only the last MAX_CONVERSATION_HISTORY_TURNS * 2 entries are returned.
        mock_settings sets MAX_CONVERSATION_HISTORY_TURNS=3, so max 6 entries.
        """
        conv_dir = _redirect_conv_dir(tmp_path, monkeypatch)
        # 5 turns = 10 entries (exceeds the limit of 6)
        turns = []
        for i in range(5):
            turns.append({"role": "user", "content": f"q{i}"})
            turns.append({"role": "assistant", "content": f"a{i}", "citations": []})
        _write_session(conv_dir, _UUID_TRUNCATE, turns)

        from app.memory.conversation import load_conversation_history
        result = load_conversation_history(_UUID_TRUNCATE)

        max_entries = mock_settings.MAX_CONVERSATION_HISTORY_TURNS * 2  # = 6
        assert len(result) == max_entries
        # Should be the LAST 6 entries
        assert result == turns[-max_entries:]

    def test_returns_empty_list_on_corrupt_json(self, tmp_path, monkeypatch):
        """A corrupt JSON file should return [] rather than raise."""
        conv_dir = _redirect_conv_dir(tmp_path, monkeypatch)
        conv_dir.mkdir(parents=True, exist_ok=True)
        bad_file = conv_dir / f"{_UUID_CORRUPT}.json"
        bad_file.write_text("{not valid json", encoding="utf-8")

        from app.memory.conversation import load_conversation_history
        result = load_conversation_history(_UUID_CORRUPT)
        assert result == []

    def test_returns_empty_list_on_empty_json_array(self, tmp_path, monkeypatch):
        """An empty JSON array in the file returns an empty list."""
        conv_dir = _redirect_conv_dir(tmp_path, monkeypatch)
        _write_session(conv_dir, _UUID_EMPTY, [])

        from app.memory.conversation import load_conversation_history
        result = load_conversation_history(_UUID_EMPTY)
        assert result == []

    def test_exactly_at_limit_returns_all_entries(self, tmp_path, monkeypatch, mock_settings):
        """When entry count equals the limit exactly, all entries are returned."""
        conv_dir = _redirect_conv_dir(tmp_path, monkeypatch)
        max_entries = mock_settings.MAX_CONVERSATION_HISTORY_TURNS * 2  # = 6
        turns = [{"role": "user", "content": f"x{i}"} for i in range(max_entries)]
        _write_session(conv_dir, _UUID_EXACT, turns)

        from app.memory.conversation import load_conversation_history
        result = load_conversation_history(_UUID_EXACT)
        assert len(result) == max_entries


# ===========================================================================
# save_conversation_turn
# ===========================================================================

class TestSaveConversationTurn:
    """Tests for appending a turn and persisting it to disk."""

    def test_creates_file_if_not_exists(self, tmp_path, monkeypatch):
        """Saving to a new session creates the JSON file."""
        conv_dir = _redirect_conv_dir(tmp_path, monkeypatch)

        from app.memory.conversation import save_conversation_turn
        save_conversation_turn(_UUID_NEW, "hello?", "hi there", [])

        path = conv_dir / f"{_UUID_NEW}.json"
        assert path.exists()

    def test_saved_content_contains_query_and_answer(self, tmp_path, monkeypatch):
        """The saved file holds the user query and assistant answer."""
        conv_dir = _redirect_conv_dir(tmp_path, monkeypatch)

        from app.memory.conversation import save_conversation_turn
        save_conversation_turn(_UUID_CONTENT, "What is AI?", "AI is ...", [])

        data = json.loads((conv_dir / f"{_UUID_CONTENT}.json").read_text(encoding="utf-8"))
        assert {"role": "user", "content": "What is AI?"} in data
        assert any(
            e["role"] == "assistant" and "AI is ..." in e["content"] for e in data
        )

    def test_appends_to_existing_session(self, tmp_path, monkeypatch):
        """Subsequent saves append new turns to the existing session."""
        conv_dir = _redirect_conv_dir(tmp_path, monkeypatch)
        _write_session(conv_dir, _UUID_APPEND_S2, [
            {"role": "user", "content": "first q"},
            {"role": "assistant", "content": "first a", "citations": []},
        ])

        from app.memory.conversation import save_conversation_turn
        save_conversation_turn(_UUID_APPEND_S2, "second q", "second a", [])

        data = json.loads((conv_dir / f"{_UUID_APPEND_S2}.json").read_text(encoding="utf-8"))
        user_messages = [e["content"] for e in data if e["role"] == "user"]
        assert "first q" in user_messages
        assert "second q" in user_messages

    def test_citations_are_persisted(self, tmp_path, monkeypatch):
        """Citations list is stored on the assistant entry."""
        conv_dir = _redirect_conv_dir(tmp_path, monkeypatch)
        citations = [{"source_type": "vector", "title": "Doc A", "excerpt": "..."}]

        from app.memory.conversation import save_conversation_turn
        save_conversation_turn(_UUID_CITE, "q", "a", citations)

        data = json.loads((conv_dir / f"{_UUID_CITE}.json").read_text(encoding="utf-8"))
        assistant_entry = next(e for e in data if e["role"] == "assistant")
        assert assistant_entry["citations"] == citations

    def test_trimmed_to_max_history_turns(self, tmp_path, monkeypatch, mock_settings):
        """
        After saving, the persisted file must not exceed MAX_TURNS * 2 entries.
        We pre-fill with the maximum, then add one more turn; the oldest is dropped.
        """
        conv_dir = _redirect_conv_dir(tmp_path, monkeypatch)
        max_entries = mock_settings.MAX_CONVERSATION_HISTORY_TURNS * 2  # = 6

        # Pre-fill with exactly max_entries
        existing = []
        for i in range(mock_settings.MAX_CONVERSATION_HISTORY_TURNS):
            existing.append({"role": "user", "content": f"q{i}"})
            existing.append({"role": "assistant", "content": f"a{i}", "citations": []})
        _write_session(conv_dir, _UUID_TRIM, existing)

        from app.memory.conversation import save_conversation_turn
        save_conversation_turn(_UUID_TRIM, "new q", "new a", [])

        data = json.loads((conv_dir / f"{_UUID_TRIM}.json").read_text(encoding="utf-8"))
        assert len(data) == max_entries

    def test_creates_conversations_directory_if_missing(self, tmp_path, monkeypatch):
        """The conversations directory is created automatically if absent."""
        conv_dir = _redirect_conv_dir(tmp_path, monkeypatch)
        assert not conv_dir.exists()

        from app.memory.conversation import save_conversation_turn
        save_conversation_turn(_UUID_AUTODIR, "q", "a", [])

        assert conv_dir.exists()

    def test_session_ids_are_isolated(self, tmp_path, monkeypatch):
        """Turns from different sessions are stored in separate files."""
        conv_dir = _redirect_conv_dir(tmp_path, monkeypatch)

        from app.memory.conversation import save_conversation_turn
        save_conversation_turn(_UUID_SESS_A, "qa", "aa", [])
        save_conversation_turn(_UUID_SESS_B, "qb", "ab", [])

        data_a = json.loads((conv_dir / f"{_UUID_SESS_A}.json").read_text(encoding="utf-8"))
        data_b = json.loads((conv_dir / f"{_UUID_SESS_B}.json").read_text(encoding="utf-8"))

        assert all(e["content"] in ("qa", "aa") for e in data_a)
        assert all(e["content"] in ("qb", "ab") for e in data_b)

    def test_non_ascii_content_is_preserved(self, tmp_path, monkeypatch):
        """Unicode / non-ASCII characters survive the JSON round-trip."""
        conv_dir = _redirect_conv_dir(tmp_path, monkeypatch)

        from app.memory.conversation import save_conversation_turn
        save_conversation_turn(_UUID_UNICODE, "你好吗？", "我很好，谢谢！", [])

        data = json.loads((conv_dir / f"{_UUID_UNICODE}.json").read_text(encoding="utf-8"))
        assert any("你好吗？" in e.get("content", "") for e in data)
        assert any("我很好，谢谢！" in e.get("content", "") for e in data)
