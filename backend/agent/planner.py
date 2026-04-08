from __future__ import annotations

import logging
import re
from difflib import SequenceMatcher
from typing import Any, TypedDict

from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from backend.agent.executor import ToolExecutor
from backend.agent.prompts import FINAL_ANSWER_PROMPT
from backend.config import settings
from backend.memory.conversation import ConversationMemory
from backend.rag.retriever import TravelRetriever

logger = logging.getLogger(__name__)

YES_WORDS = {"yes", "y", "sure", "ok", "okay", "please", "go ahead", "do it"}
NO_WORDS = {"no", "n", "skip", "nope", "not now"}
INLINE_WEATHER_CONSENT_PHRASES = {
    "you can check weather",
    "check weather",
    "use weather",
    "weather allowed",
    "i allow weather",
    "i approve weather",
    "with weather",
    "include weather forecast",
}
SOURCE_NAME_TO_URL = {
    "wikivoyage": "https://en.wikivoyage.org/",
    "wiki": "https://en.wikipedia.org/",
    "wikipedia": "https://en.wikipedia.org/",
    "blog": "https://www.nomadicmatt.com/",
}


class AgentState(TypedDict, total=False):
    query: str
    location: str
    thought: str
    next_action: str
    observation: str
    tool_output: dict[str, Any]
    answer: str
    actions_taken: list[str]
    sources: list[str]
    suggested_followups: list[str]
    permission_question: str
    weather_consent: bool | None


class TravelPlannerAgent:
    def __init__(self, retriever: TravelRetriever, memory: ConversationMemory) -> None:
        self.retriever = retriever
        self.memory = memory
        self.executor = ToolExecutor(memory=memory)
        self.llm = ChatOpenAI(model=settings.model_name, api_key=settings.openai_api_key)
        self.graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(AgentState)
        graph.add_node("planner", self._planner_node)
        graph.add_node("tool_executor", self._tool_node)
        graph.add_node("responder", self._responder_node)
        graph.set_entry_point("planner")
        graph.add_conditional_edges(
            "planner",
            self._route_from_planner,
            {"tool": "tool_executor", "respond": "responder"},
        )
        graph.add_edge("tool_executor", "responder")
        graph.add_edge("responder", END)
        return graph.compile()

    def _infer_location(self, query: str) -> str:
        match = re.search(r"\bto ([A-Za-z][A-Za-z\\s-]{1,40})", query, flags=re.IGNORECASE)
        if not match:
            return ""
        return match.group(1).strip(" .?!,")

    def _planner_node(self, state: AgentState) -> AgentState:
        query = state["query"].strip()
        location = state.get("location", "").strip()
        weather_consent = state.get("weather_consent")
        if not location:
            location = self._infer_location(query)
        lowered = query.lower()
        tokens = re.findall(r"[a-z]+", lowered)
        actions = state.get("actions_taken", [])
        is_first_turn = len(self.memory.history) == 0

        # Handle explicit permission follow-up first.
        if self.memory.pending_permission and self.memory.pending_permission["type"] == "weather":
            if lowered in YES_WORDS or any(word in lowered for word in YES_WORDS):
                payload_location = self.memory.pending_permission["payload"].get("location") or location
                self.memory.clear_pending_permission()
                actions.append("permission_granted:weather")
                return {
                    "location": payload_location,
                    "thought": "User granted weather permission, so I should fetch live weather.",
                    "next_action": "weather_tool",
                    "actions_taken": actions,
                }
            if lowered in NO_WORDS or any(word in lowered for word in NO_WORDS):
                self.memory.clear_pending_permission()
                actions.append("permission_denied:weather")
                return {
                    "location": location,
                    "thought": "User denied weather lookup, so I will continue with RAG only.",
                    "next_action": "rag_only",
                    "actions_taken": actions,
                }

        weather_intent = self._intent_match(
            lowered=lowered,
            tokens=tokens,
            keywords=["weather", "temperature", "forecast", "climate", "wether", "temprature"],
            phrases=["climate now", "weather now", "weather forecast"],
        ) or any(e in query for e in ["🌦", "☀", "🌤", "🌧", "⛅"])
        news_intent = self._intent_match(
            lowered=lowered,
            tokens=tokens,
            keywords=["news", "safety", "issue", "issues", "alert", "risk", "unsafe"],
            phrases=["travel news", "safety issues"],
        ) or any(e in query for e in ["📰", "⚠", "🚨"])
        inline_weather_consent = any(phrase in lowered for phrase in INLINE_WEATHER_CONSENT_PHRASES)

        # User experience boost: include weather in the very first answer when location is available.
        if is_first_turn and location:
            actions.append("weather_tool:auto_first_answer")
            return {
                "location": location,
                "thought": "For the first response, I should proactively include current weather context.",
                "next_action": "weather_tool",
                "actions_taken": actions,
            }

        if weather_intent:
            if not location:
                actions.append("needs_location_for_weather")
                return {
                    "location": "",
                    "thought": "Weather lookup needs a destination location.",
                    "next_action": "rag_only",
                    "observation": "No location provided for weather lookup.",
                    "actions_taken": actions,
                }
            if weather_consent is True:
                actions.append("permission_granted:weather_inline")
                return {
                    "location": location,
                    "thought": "User already granted weather consent in this request.",
                    "next_action": "weather_tool",
                    "actions_taken": actions,
                }
            if inline_weather_consent:
                actions.append("permission_granted:weather_in_query_text")
                return {
                    "location": location,
                    "thought": "User explicitly approved weather lookup in their query text.",
                    "next_action": "weather_tool",
                    "actions_taken": actions,
                }
            if weather_consent is False:
                actions.append("permission_denied:weather_inline")
                return {
                    "location": location,
                    "thought": "User denied weather consent in this request.",
                    "next_action": "rag_only",
                    "actions_taken": actions,
                }
            permission_question = "Do you want me to check current weather?"
            self.memory.set_pending_permission("weather", {"location": location})
            actions.append("permission_requested:weather")
            logger.info("Agent decision: ask weather permission")
            return {
                "location": location,
                "thought": "The user asked about weather, which requires live data and consent.",
                "next_action": "ask_weather_permission",
                "permission_question": permission_question,
                "actions_taken": actions,
            }

        if news_intent:
            if not location:
                actions.append("needs_location_for_news")
                return {
                    "location": "",
                    "thought": "News lookup needs a destination location.",
                    "next_action": "rag_only",
                    "observation": "No location provided for travel news lookup.",
                    "actions_taken": actions,
                }
            actions.append("news_tool")
            logger.info("Agent decision: call news tool")
            return {
                "location": location,
                "thought": "The query involves safety/news, so recent travel news is needed.",
                "next_action": "news_tool",
                "actions_taken": actions,
            }

        actions.append("rag_only")
        logger.info("Agent decision: rag only")
        return {
            "location": location,
            "thought": "The query can be answered from travel knowledge base.",
            "next_action": "rag_only",
            "actions_taken": actions,
        }

    def _route_from_planner(self, state: AgentState) -> str:
        if state.get("next_action") in {"weather_tool", "news_tool"}:
            return "tool"
        return "respond"

    def _tool_node(self, state: AgentState) -> AgentState:
        action = state.get("next_action", "")
        location = state.get("location", "").strip()
        if not location:
            return {
                "observation": "Cannot run tool without a location.",
                "tool_output": {},
                "actions_taken": state.get("actions_taken", []) + ["tool_skipped:no_location"],
            }
        try:
            tool_response = self.executor.execute(action=action, location=location)
            return {
                "tool_output": tool_response,
                "observation": f"{tool_response['tool']} tool returned data.",
            }
        except Exception as exc:  # noqa: BLE001
            logger.exception("Tool execution failed")
            return {
                "tool_output": {"tool": "error", "result": {"error": str(exc)}},
                "observation": f"Tool execution failed: {exc}",
                "actions_taken": state.get("actions_taken", []) + ["tool_error"],
            }

    def _responder_node(self, state: AgentState) -> AgentState:
        query = state["query"]
        location = state.get("location", "")
        actions_taken = state.get("actions_taken", [])
        recent_turns = self.memory.last_turns(3)

        if state.get("next_action") == "ask_weather_permission":
            answer = (
                f"Thought: {state.get('thought')}\n"
                "Action: Request user permission for live weather lookup.\n"
                "Observation: Weather tool requires explicit consent.\n"
                f"Final Answer: {state.get('permission_question')}"
            )
            return {"answer": answer, "sources": []}

        retrieved_docs = self.retriever.retrieve(query=query, location=location or None, k=5)
        rag_context = "\n\n".join(doc.page_content for doc in retrieved_docs)
        citation_candidates: list[str] = []
        rag_page_urls: list[str] = []
        rag_source_names: set[str] = set()
        for doc in retrieved_docs:
            src_url = str(doc.metadata.get("source_url", "")).strip()
            src_name = str(doc.metadata.get("source", "")).strip()
            if src_url:
                rag_page_urls.append(src_url)
            if src_name:
                rag_source_names.add(src_name.lower())

        if rag_page_urls:
            citation_candidates.extend(rag_page_urls)
        else:
            for src_name in sorted(rag_source_names):
                mapped = SOURCE_NAME_TO_URL.get(src_name, "")
                if mapped:
                    citation_candidates.append(mapped)

        tool_output = state.get("tool_output", {})
        news_tool_used = tool_output.get("tool") == "news"
        extra_sources: list[str] = []
        if tool_output.get("tool") == "news":
            for article in tool_output.get("result", {}).get("articles", []):
                url = article.get("url")
                if url:
                    extra_sources.append(url)
            if not extra_sources:
                extra_sources.append("https://newsapi.org/")
        if tool_output.get("tool") == "weather":
            extra_sources.append("https://openweathermap.org/")

        citation_candidates.extend(extra_sources)
        source_names = list(dict.fromkeys([s for s in citation_candidates if s]))
        citation_list_text = "\n".join(
            f"[{i}] {src}" for i, src in enumerate(source_names, start=1)
        ) or "[1] No external source available."

        prompt = (
            f"{FINAL_ANSWER_PROMPT}\n\n"
            f"User Query: {query}\n"
            f"Location: {location or 'unknown'}\n"
            f"Recent Conversation: {recent_turns}\n"
            f"Planner Thought: {state.get('thought', '')}\n"
            f"Actions Taken: {actions_taken}\n"
            f"news_tool_used: {news_tool_used}\n"
            f"Tool Observation: {state.get('observation', 'No tool used')}\n"
            f"Tool Data: {tool_output}\n\n"
            f"RAG Context:\n{rag_context if rag_context else 'No retrieved context found.'}\n\n"
            f"Citation List:\n{citation_list_text}\n\n"
            "Write a useful answer for travel planning."
        )
        model_response = self.llm.invoke(prompt)
        answer = model_response.content if hasattr(model_response, "content") else str(model_response)
        if not news_tool_used:
            lines = answer.splitlines()
            filtered_lines: list[str] = []
            blocked = (
                "politically unstable",
                "political unrest",
                "demonstration",
                "violence",
                "line of control",
                "current instability",
            )
            for line in lines:
                if any(token in line.lower() for token in blocked):
                    continue
                filtered_lines.append(line)
            answer = "\n".join(filtered_lines).strip()
        citations_block = "\n".join(
            f"[{i}] {src}" for i, src in enumerate(source_names, start=1)
        ) or "[1] No external source available."
        if "Citations:" not in answer:
            answer = f"{answer}\n\nCitations:\n{citations_block}"
        elif "[" not in answer:
            answer = f"{answer}\n{citations_block}"
        followups = self._suggest_followups(query=query, answer=answer, location=location)
        return {"answer": answer, "sources": source_names, "suggested_followups": followups}

    @staticmethod
    def _intent_match(
        lowered: str,
        tokens: list[str],
        keywords: list[str],
        phrases: list[str],
    ) -> bool:
        if any(phrase in lowered for phrase in phrases):
            return True
        for token in tokens:
            for keyword in keywords:
                if token == keyword:
                    return True
                if SequenceMatcher(a=token, b=keyword).ratio() >= 0.82:
                    return True
        return False

    def _suggest_followups(self, query: str, answer: str, location: str) -> list[str]:
        prompt = (
            "Suggest exactly 3 short follow-up questions a traveler might ask next. "
            "Keep them practical and specific. Return one question per line, no numbering.\n\n"
            f"Location: {location}\n"
            f"User Query: {query}\n"
            f"Assistant Answer: {answer}\n"
        )
        try:
            resp = self.llm.invoke(prompt)
            text = resp.content if hasattr(resp, "content") else str(resp)
            lines = [line.strip(" -\t") for line in text.splitlines() if line.strip()]
            cleaned: list[str] = []
            for line in lines:
                if line and line not in cleaned:
                    cleaned.append(line.rstrip("?") + "?")
                if len(cleaned) == 3:
                    break
            return cleaned
        except Exception:  # noqa: BLE001
            fallback_location = location or "the destination"
            return [
                f"What is the best day-by-day itinerary for {fallback_location}?",
                f"How can I reduce costs further while traveling in {fallback_location}?",
                "What local transport and safety tips should I keep in mind?",
            ]

    def run(
        self,
        query: str,
        location: str | None = None,
        weather_consent: bool | None = None,
    ) -> dict[str, Any]:
        initial_state: AgentState = {
            "query": query,
            "location": location or "",
            "weather_consent": weather_consent,
            "actions_taken": [],
        }
        result = self.graph.invoke(initial_state)
        answer = result.get("answer", "")
        sources = result.get("sources", [])
        actions_taken = result.get("actions_taken", [])
        suggested_followups = result.get("suggested_followups", [])
        self.memory.add_turn(query, answer, actions_taken)
        return {
            "answer": answer,
            "sources": sources,
            "actions_taken": actions_taken,
            "suggested_followups": suggested_followups,
        }
