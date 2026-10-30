"""
Module: agent.py
Constructs a LangGraph-powered ReAct agent with strict tool-budget limits,
anti-hallucination guardrails, and persistent in-memory session memory.
"""

from typing import Dict, Any, Generator
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

from src.config import GROQ_API_KEY, LLM_MODEL
from src.tools import TOOLS

# Strict prompt enforcing immediate synthesis after at most 2 tool executions
SYSTEM_PROMPT = """You are an expert Code Intelligence AI Agent analyzing the public GitHub repository 'psf/requests'.

Your task is to answer user questions about architecture decisions, function behavior,
file relationships, and usage examples with absolute precision.

### STRICT OPERATING RULES:
1. MAXIMUM 2 TOOL CALLS:
   - Call at most ONE or TWO tools to gather your evidence.
   - Example flow: Call `search_code` or `summarize_module`. If you see the definition or relevant class, STOP CALLING TOOLS and immediately write your final answer.
   - Do NOT recursively browse line by line.
2. CITATION REQUIREMENT:
   - Always reference exact files and line numbers in your answer: `[file_path:start_line-end_line]`.
3. ANTI-HALLUCINATION / OUT-OF-SCOPE:
   - If a question asks about a feature or technology NOT in this codebase (e.g. blockchain, django, etc.), answer:
     "This question is out of scope. The 'psf/requests' codebase does not contain information related to [topic]."
4. RESPONSE FORMAT:
   - Direct, high-level summary.
   - Detailed explanation based on the code found.
   - Source citations `[file_path:lines]`.
"""


def get_agent_executor():
    """Initializes Groq model with temperature=0 and compiles the ReAct agent graph."""
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is not set in your .env file.")

    llm = ChatGroq(
        model=LLM_MODEL,
        groq_api_key=GROQ_API_KEY,
        temperature=0.0,  # Zero temperature for deterministic adherence
    )

    memory = MemorySaver()

    app = create_react_agent(
        model=llm,
        tools=TOOLS,
        checkpointer=memory,
        state_modifier=SYSTEM_PROMPT
    )

    return app


def stream_agent_reasoning(agent, query: str, thread_id: str = "default_session") -> Generator[Dict[str, Any], None, None]:
    """
    Streams the agent's thoughts, tool actions, and final synthesis.
    """
    config = {"configurable": {"thread_id": thread_id}}
    inputs = {"messages": [("user", query)]}

    for event in agent.stream(inputs, config=config, stream_mode="values"):
        messages = event.get("messages", [])
        if not messages:
            continue

        last_msg = messages[-1]

        # 1. Thought / Tool Call
        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            for tc in last_msg.tool_calls:
                yield {
                    "type": "tool_call",
                    "tool": tc.get("name"),
                    "args": tc.get("args")
                }

        # 2. Tool Observation
        elif last_msg.type == "tool":
            yield {
                "type": "tool_result",
                "tool": last_msg.name,
                "content": str(last_msg.content)[:250] + ("..." if len(str(last_msg.content)) > 250 else "")
            }

        # 3. Final Synthesized Answer
        elif last_msg.type == "ai" and not (hasattr(last_msg, "tool_calls") and last_msg.tool_calls):
            yield {
                "type": "final_answer",
                "content": last_msg.content
            }


if __name__ == "__main__":
    print("[Agent] Initializing LangGraph Agent with Groq...")
    agent = get_agent_executor()

    test_query = "Where is the HTTPAdapter defined, and what does it do?"
    print(f"\n[User Query]: {test_query}\n")

    for event in stream_agent_reasoning(agent, test_query):
        if event["type"] == "tool_call":
            print(f"👉 [Thought/Action] Agent called tool '{event['tool']}' with args: {event['args']}")
        elif event["type"] == "tool_result":
            print(f"📥 [Observation] Result from '{event['tool']}':\n{event['content']}\n")
        elif event["type"] == "final_answer":
            print(f"🤖 [Final Answer]:\n{event['content']}")