from __future__ import annotations

import logging
from typing import Any

from backend.memory.conversation import ConversationMemory
from backend.tools.news import NewsTool
from backend.tools.weather import WeatherTool

logger = logging.getLogger(__name__)


class ToolExecutor:
    def __init__(self, memory: ConversationMemory) -> None:
        self.weather_tool = WeatherTool()
        self.news_tool = NewsTool()
        self.memory = memory

    def execute(self, action: str, location: str) -> dict[str, Any]:
        logger.info("Executing tool action=%s location=%s", action, location)
        if action == "weather_tool":
            result = self.weather_tool.run(location)
            self.memory.add_tool_call("weather", {"location": location}, result)
            return {"tool": "weather", "result": result}
        if action == "news_tool":
            result = self.news_tool.run(location)
            self.memory.add_tool_call("news", {"location": location}, result)
            return {"tool": "news", "result": result}
        return {"tool": "none", "result": {}}
