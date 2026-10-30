"""
Module: agent.py
Production-grade, low-latency Code Intelligence Agent.
Guarantees clean Markdown synthesis with exact citations in under 3 seconds.
"""

from typing import Dict, Any, Generator
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage, ToolMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

from src.config import GROQ_API_KEY, LLM_MODEL
from src.tools import TOOLS

DECISION_SYSTEM_PROMPT = """You are an expert Code Intelligence AI Agent analyzing the public GitHub repository 'psf/requests'.

Your task is to analyze the user's question and invoke the appropriate tool:
- `search_code`: For semantic retrieval of classes, functions, methods, or architectural concepts.
- `read_file`: For inspecting a specific file path.
- `list_directory`: For exploring the repository directory structure.
- `summarize_module`: For an architectural overview of a module.

OUT-OF-SCOPE RULE:
If the user asks about something completely unrelated to this codebase (e.g. blockchain, ethereum, smart contracts, django),
do NOT call any tool. Answer immediately stating clearly that it is out of scope.
"""

SYNTHESIS_SYSTEM_PROMPT = """You are an expert software engineer explaining the 'psf/requests' codebase.
Your goal is to answer the user's question thoroughly and accurately using ONLY the provided code evidence.

RULES:
1. Format your response cleanly using Markdown with headings, bullet points, and code blocks.
2. ALWAYS cite the exact file paths and line ranges from the evidence, e.g., `[src/requests/adapters.py:307-363]`.
3. Do NOT output any XML tags, <tool_call> tags, or function signatures. Write plain human-readable Markdown.
"""


def call_decision_model(state: MessagesState):
    """Decides which tool to call or answers directly if out of scope."""
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is not set.")

    llm = ChatGroq(
        model=LLM_MODEL,
        groq_api_key=GROQ_API_KEY,
        temperature=0.0,
        max_tokens=300,
    )
    llm_with_tools = llm.bind_tools(TOOLS)
    messages = [SystemMessage(content=DECISION_SYSTEM_PROMPT)] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


def should_continue(state: MessagesState) -> str:
    """Routes to 'tools' if tool calls are requested; otherwise finishes."""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END


def synthesize_answer(state: MessagesState):
    """
    Takes the retrieved tool evidence and writes a clean, cited Markdown response.
    """
    llm = ChatGroq(
        model=LLM_MODEL,
        groq_api_key=GROQ_API_KEY,
        temperature=0.0,
        max_tokens=650,
    )

    # Extract user query
    user_query = ""
    for msg in state["messages"]:
        if isinstance(msg, HumanMessage):
            user_query = msg.content

    # Extract all tool observation contents
    tool_evidences = []
    for msg in state["messages"]:
        if isinstance(msg, ToolMessage):
            tool_evidences.append(str(msg.content))

    evidence_text = "\n\n".join(tool_evidences)

    synthesis_prompt = f"""User Question:
{user_query}

Retrieved Code Evidence from Repository:
{evidence_text}

Please provide a comprehensive, clear explanation answering the question based on the code evidence above.
Include exact file paths and line numbers: `[file_path:start_line-end_line]`."""

    messages = [
        SystemMessage(content=SYNTHESIS_SYSTEM_PROMPT),
        HumanMessage(content=synthesis_prompt)
    ]

    response = llm.invoke(messages)
    return {"messages": [response]}


def get_agent_executor():
    """Compiles the custom LangGraph state machine with memory."""
    workflow = StateGraph(MessagesState)

    # 1. Add Nodes
    workflow.add_node("agent", call_decision_model)
    workflow.add_node("tools", ToolNode(TOOLS))
    workflow.add_node("synthesize", synthesize_answer)

    # 2. Add Edges
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", should_continue, ["tools", END])
    workflow.add_edge("tools", "synthesize")
    workflow.add_edge("synthesize", END)

    # 3. Attach in-memory checkpointer
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    return app


def stream_agent_reasoning(agent, query: str, thread_id: str = "default_session") -> Generator[Dict[str, Any], None, None]:
    """
    Streams the agent's thought process, tool actions, and final synthesis.
    """
    config = {"configurable": {"thread_id": thread_id}}
    inputs = {"messages": [HumanMessage(content=query)]}

    for event in agent.stream(inputs, config=config, stream_mode="values"):
        messages = event.get("messages", [])
        if not messages:
            continue

        last_msg = messages[-1]

        # 1. Action (Tool Call)
        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            for tc in last_msg.tool_calls:
                yield {
                    "type": "tool_call",
                    "tool": tc.get("name"),
                    "args": tc.get("args")
                }

        # 2. Observation (Tool Result)
        elif isinstance(last_msg, ToolMessage):
            yield {
                "type": "tool_result",
                "tool": getattr(last_msg, "name", "tool"),
                "content": str(last_msg.content)[:180] + ("..." if len(str(last_msg.content)) > 180 else "")
            }

        # 3. Final Synthesized Answer
        elif isinstance(last_msg, AIMessage) and not (hasattr(last_msg, "tool_calls") and last_msg.tool_calls):
            yield {
                "type": "final_answer",
                "content": last_msg.content
            }