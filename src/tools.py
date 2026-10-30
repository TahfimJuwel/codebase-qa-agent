"""
Module: tools.py
Defines the 4 core tools with strict Pydantic schemas and token-safety limits
to prevent context-window overflow and stay well within free-tier API rate limits.
"""

from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field
from langchain_core.tools import tool

from src.config import REPO_PATH
from src.vectorstore import CodebaseVectorStore

_vector_store: Optional[CodebaseVectorStore] = None

# Maximum lines read_file is allowed to return in a single call to protect token budget
MAX_READ_LINES = 50


def get_vector_store() -> CodebaseVectorStore:
    global _vector_store
    if _vector_store is None:
        _vector_store = CodebaseVectorStore()
    return _vector_store


class SearchCodeInput(BaseModel):
    query: str = Field(description="The semantic search query for functions, classes, or code concepts.")


class ReadFileInput(BaseModel):
    path: str = Field(description="The relative path to the file inside the repo, e.g. 'src/requests/adapters.py'.")
    start_line: Optional[int] = Field(default=1, description="Starting line number (1-indexed).")
    end_line: Optional[int] = Field(default=None, description="Ending line number (max 80 lines per call).")


class ListDirectoryInput(BaseModel):
    path: str = Field(default=".", description="The relative directory path to list, e.g. 'src/requests' or '.'.")


class SummarizeModuleInput(BaseModel):
    module_name: str = Field(description="The name or path of the module to summarize, e.g. 'adapters' or 'sessions'.")


@tool(args_schema=SearchCodeInput)
def search_code(query: str) -> str:
    """
    Semantic search over the indexed codebase.
    Returns the top-3 matching code chunks with file paths and line ranges.
    """
    store = get_vector_store()
    results = store.search(query, top_k=2)

    if not results:
        return "No relevant code chunks found for this query in the repository."

    output = []
    for i, res in enumerate(results, start=1):
        # Truncate chunk content if exceedingly long
        content_lines = res["content"].splitlines()
        truncated_content = "\n".join(content_lines[:35])
        if len(content_lines) > 35:
            truncated_content += "\n... [truncated for token brevity]"

        output.append(
            f"--- Match {i} (Score: {res['score']}) ---\n"
            f"File: {res['file_path']} (Lines {res['start_line']}-{res['end_line']})\n"
            f"Symbol: {res['chunk_name']} ({res['chunk_type']})\n"
            f"Code:\n{truncated_content}\n"
        )

    return "\n".join(output)


@tool(args_schema=ReadFileInput)
def read_file(path: str, start_line: Optional[int] = 1, end_line: Optional[int] = None) -> str:
    """
    Read targeted lines of a file by relative path.
    Returns at most 80 lines per call to stay within token budgets.
    """
    clean_path = path.strip().replace("\\", "/").lstrip("/")
    target_file = (REPO_PATH / clean_path).resolve()

    if not str(target_file).startswith(str(REPO_PATH.resolve())):
        return f"Error: Access denied. Cannot read outside repo root: {path}"

    if not target_file.exists() or not target_file.is_file():
        return f"Error: File '{path}' does not exist in repository."

    try:
        with open(target_file, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()

        total_lines = len(lines)
        s_line = max(1, start_line or 1)
        
        # Enforce maximum 80 lines per call
        requested_end = end_line if end_line is not None else (s_line + MAX_READ_LINES - 1)
        e_line = min(total_lines, requested_end, s_line + MAX_READ_LINES - 1)

        if s_line > total_lines:
            return f"Error: start_line {s_line} exceeds total lines ({total_lines})."

        selected_lines = lines[s_line - 1 : e_line]
        numbered_lines = [f"{s_line + idx}: {line}" for idx, line in enumerate(selected_lines)]

        header = f"File: {clean_path} (Lines {s_line} to {e_line} of {total_lines})\n"
        if (e_line - s_line + 1) >= MAX_READ_LINES:
            header += "[Note: Capped at 80 lines. Specify higher start_line for remaining code.]\n"

        return header + "".join(numbered_lines)
    except Exception as e:
        return f"Error reading file '{path}': {str(e)}"


@tool(args_schema=ListDirectoryInput)
def list_directory(path: str = ".") -> str:
    """
    List files and subdirectories at the given path relative to the repo root.
    """
    clean_path = path.strip().replace("\\", "/").lstrip("/")
    target_dir = (REPO_PATH / clean_path).resolve()

    if not str(target_dir).startswith(str(REPO_PATH.resolve())):
        return f"Error: Access denied. Path outside repo root: {path}"

    if not target_dir.exists() or not target_dir.is_dir():
        return f"Error: Directory '{path}' does not exist."

    entries = []
    try:
        for item in sorted(target_dir.iterdir()):
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


@tool(args_schema=SummarizeModuleInput)
def summarize_module(module_name: str) -> str:
    """
    Generate an architectural overview of a module or class by name.
    """
    import ast

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
        return f"Module '{module_name}' could not be located directly on disk."

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
            "\n".join(classes[:10]) if classes else "  None",
            f"\nExported Functions ({len(functions)}):",
            "\n".join(functions[:10]) if functions else "  None",
        ]
        return "\n".join(summary)

    except Exception as e:
        return f"Error analyzing module '{module_name}': {str(e)}"


TOOLS = [search_code, read_file, list_directory, summarize_module]