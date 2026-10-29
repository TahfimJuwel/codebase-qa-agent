"""
Module: tools.py
Defines the 4 core tools required by the assessment specification for agentic code intelligence.
Each tool is wrapped as a LangChain Tool with clear docstrings so the LLM understands
when and why to invoke it.
"""

import os
from pathlib import Path
from typing import Optional
from langchain_core.tools import tool

from src.config import REPO_PATH
from src.vectorstore import CodebaseVectorStore

# Initialize our singleton vector store client for fast queries
_vector_store: Optional[CodebaseVectorStore] = None


def get_vector_store() -> CodebaseVectorStore:
    """Lazy initializer for the vector store to avoid loading models on import."""
    global _vector_store
    if _vector_store is None:
        _vector_store = CodebaseVectorStore()
    return _vector_store


@tool
def search_code(query: str) -> str:
    """
    Semantic search over the indexed codebase.
    Use this tool to find relevant functions, classes, methods, or documentation
    matching a natural language concept or technical question.
    Returns the top matching code chunks with file paths and line ranges.
    """
    store = get_vector_store()
    results = store.search(query, top_k=4)

    if not results:
        return "No relevant code chunks found for this query in the repository."

    output = []
    for i, res in enumerate(results, start=1):
        output.append(
            f"--- Match {i} (Relevance Score: {res['score']}) ---\n"
            f"File: {res['file_path']} (Lines {res['start_line']}-{res['end_line']})\n"
            f"Symbol: {res['chunk_name']} ({res['chunk_type']})\n"
            f"Code:\n{res['content']}\n"
        )

    return "\n".join(output)


@tool
def read_file(path: str, start_line: Optional[int] = None, end_line: Optional[int] = None) -> str:
    """
    Read the raw content of a file by its path relative to the repo root.
    Optionally specify start_line and end_line to inspect a specific section.
    Use this tool when you need full file context or to inspect code around a known line.
    """
    # Sanitize path to prevent directory traversal outside repo root
    clean_path = path.strip().replace("\\", "/").lstrip("/")
    target_file = (REPO_PATH / clean_path).resolve()

    # Security check: must reside inside REPO_PATH
    if not str(target_file).startswith(str(REPO_PATH.resolve())):
        return f"Error: Access denied. Cannot read files outside repository root: {path}"

    if not target_file.exists() or not target_file.is_file():
        return f"Error: File '{path}' does not exist in the repository."

    try:
        with open(target_file, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()

        total_lines = len(lines)
        s_line = max(1, start_line or 1)
        e_line = min(total_lines, end_line or total_lines)

        if s_line > total_lines:
            return f"Error: start_line {s_line} exceeds total lines ({total_lines}) in '{path}'."

        selected_lines = lines[s_line - 1 : e_line]
        numbered_lines = [f"{s_line + idx}: {line}" for idx, line in enumerate(selected_lines)]

        return (
            f"File: {clean_path} (Lines {s_line} to {e_line} of {total_lines})\n"
            + "".join(numbered_lines)
        )
    except Exception as e:
        return f"Error reading file '{path}': {str(e)}"


@tool
def list_directory(path: str = ".") -> str:
    """
    List files and subdirectories at the given path relative to the repo root.
    Use this tool to explore the folder structure and discover what files exist.
    """
    clean_path = path.strip().replace("\\", "/").lstrip("/")
    target_dir = (REPO_PATH / clean_path).resolve()

    if not str(target_dir).startswith(str(REPO_PATH.resolve())):
        return f"Error: Access denied. Path outside repository root: {path}"

    if not target_dir.exists() or not target_dir.is_dir():
        return f"Error: Directory '{path}' does not exist."

    entries = []
    try:
        for item in sorted(target_dir.iterdir()):
            # Skip hidden git or cache folders
            if item.name.startswith(".") or item.name == "__pycache__":
                continue
            if item.is_dir():
                entries.append(f"[DIR]  {item.name}/")
            else:
                entries.append(f"[FILE] {item.name}")

        rel_path = clean_path if clean_path else "."
        return f"Directory listing for '{rel_path}':\n" + "\n".join(entries)
    except Exception as e:
        return f"Error listing directory '{path}': {str(e)}"


@tool
def summarize_module(module_name: str) -> str:
    """
    Generate an architectural overview of a module or class by name.
    Inspects docstrings, exported classes, and functions to summarize its responsibility.
    Example module_name: 'sessions', 'models', 'adapters', or 'src/requests/api.py'.
    """
    import ast

    # Normalize module_name
    query = module_name.strip().replace("\\", "/")
    if not query.endswith(".py") and not query.endswith(".rst") and not query.endswith(".md"):
        candidate_paths = [
            REPO_PATH / f"src/requests/{query}.py",
            REPO_PATH / f"requests/{query}.py",
            REPO_PATH / f"{query}.py",
        ]
    else:
        candidate_paths = [REPO_PATH / query, REPO_PATH / f"src/{query}"]

    target_file = None
    for cand in candidate_paths:
        if cand.exists() and cand.is_file():
            target_file = cand
            break

    if not target_file:
        return (
            f"Module '{module_name}' could not be located directly on disk. "
            f"Try using 'search_code' or 'list_directory' to find its exact path."
        )

    rel_path = str(target_file.relative_to(REPO_PATH)).replace("\\", "/")

    try:
        with open(target_file, "r", encoding="utf-8", errors="replace") as f:
            code = f.read()

        tree = ast.parse(code)
        docstring = ast.get_docstring(tree) or "No module docstring available."

        classes = []
        functions = []

        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                c_doc = ast.get_docstring(node) or "No docstring."
                first_line = c_doc.strip().splitlines()[0] if c_doc else ""
                classes.append(f"- Class `{node.name}` (Line {node.lineno}): {first_line}")
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                f_doc = ast.get_docstring(node) or "No docstring."
                first_line = f_doc.strip().splitlines()[0] if f_doc else ""
                functions.append(f"- Function `{node.name}` (Line {node.lineno}): {first_line}")

        summary = [
            f"=== Module Summary: {rel_path} ===",
            f"Module Overview:\n{docstring}\n",
            f"Exported Classes ({len(classes)}):",
            "\n".join(classes) if classes else "  None",
            f"\nExported Functions ({len(functions)}):",
            "\n".join(functions) if functions else "  None",
        ]
        return "\n".join(summary)

    except Exception as e:
        return f"Error analyzing module '{module_name}': {str(e)}"


# Export list of all tools for the LangGraph agent
TOOLS = [search_code, read_file, list_directory, summarize_module]