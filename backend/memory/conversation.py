from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ConversationMemory:
    history: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    pending_permission: dict[str, Any] | None = None

    def add_turn(self, user_query: str, answer: str, actions_taken: list[str]) -> None:
        self.history.append(
            {
                "user_query": user_query,
                "answer": answer,
                "actions_taken": actions_taken,
            }
        )

    def add_tool_call(self, tool_name: str, tool_input: dict[str, Any], output: Any) -> None:
        self.tool_calls.append(
            {"tool": tool_name, "input": tool_input, "output": output}
        )

    def set_pending_permission(self, permission_type: str, payload: dict[str, Any]) -> None:
        self.pending_permission = {"type": permission_type, "payload": payload}

    def clear_pending_permission(self) -> None:
        self.pending_permission = None

    def last_turns(self, n: int = 3) -> list[dict[str, Any]]:
        if n <= 0:
            return []
        return self.history[-n:]
