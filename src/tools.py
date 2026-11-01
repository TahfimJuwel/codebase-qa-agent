"""
Module: tools.py
Defines the 4 core tools required by the assessment specification.
Hardened against path-traversal (symlinks, prefix collisions), Python <3.12 f-string errors,
and upstream vector store exceptions.
"""

import threading
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field
from langchain_core.tools import tool

from src.config import REPO_PATH
from src.vectorstore import CodebaseVectorStore

_vector_store: Optional[CodebaseVectorStore] = None
_lock = threading.Lock()


def get_vector_store() -> CodebaseVectorStore:
    """Thread-safe lazy initialization for the vector database."""
    global _vector_store
    if _vector_store is None:
        with _lock:
            if _vector_store is None:
                _vector_store = CodebaseVectorStore()
    return _vector_store


def is_safe_subpath(target: Path, base: Path) -> bool:
    """
    Cryptographically safe check ensuring target resides strictly inside base.
    Prevents path-prefix collisions (e.g. repo vs repo_backup) and symlink escapes.
    """
    try:
        target.resolve().relative_to(base.resolve())
        return True
    except (ValueError, OSError):
        return False


# --- Strict Pydantic Argument Schemas ---

class SearchCodeInput(BaseModel):
    query: str = Field(description="The semantic search query for functions, classes, or code concepts.")


class ReadFileInput(BaseModel):
    path: str = Field(description="The relative path to the file inside the repo, e.g. 'src/requests/adapters.py'.")
    start_line: Optional[int] = Field(default=1, description="Optional starting line number (1-indexed).")
    end_line: Optional[int] = Field(default=None, description="Optional ending line number.")


class ListDirectoryInput(BaseModel):
    path: str = Field(default=".", description="The relative directory path to list, e.g. 'src/requests' or '.'.")


class SummarizeModuleInput(BaseModel):
    module_name: str = Field(description="The name or path of the module or class to summarize, e.g. 'adapters' or 'Session'.")


# --- Tool Implementations ---

@tool(args_schema=SearchCodeInput)
def search_code(query: str) -> str:
    """
    Semantic search over the indexed codebase.
    Returns the top matching code chunks with file paths, symbol names, and exact line ranges.
    """
    try:
        store = get_vector_store()
        results = store.search(query, top_k=2)
    except Exception as e:
        return f"Error executing semantic search in vector store: {str(e)}"

    if not results:
        return "No relevant code chunks found for this query in the repository."

    output = []
    for i, res in enumerate(results, start=1):
        content_lines = res["content"].splitlines()
        code_preview = "\n".join(content_lines[:50])

        output.append(
            f"--- Match {i} (Score: {res['score']}) ---\n"
            f"File: {res['file_path']} (Lines {res['start_line']}-{res['end_line']})\n"
            f"Symbol: {res['chunk_name']} ({res['chunk_type']})\n"
            f"Code:\n{code_preview}\n"
        )

    return "\n".join(output)


@tool(args_schema=ReadFileInput)
def read_file(path: str, start_line: Optional[int] = 1, end_line: Optional[int] = None) -> str:
    """
    Read the raw content of a file by its relative path.
    Enforces strict path security and bounded line windows.
    """
    clean_path = path.strip().replace("\\", "/").lstrip("/")
    target_file = (REPO_PATH / clean_path)

    # Security Guard: Strict relative_to hierarchy check
    if not is_safe_subpath(target_file, REPO_PATH):
        return f"Error: Access denied. Cannot read files outside repository root: {path}"

    resolved_file = target_file.resolve()
    if not resolved_file.exists() or not resolved_file.is_file():
        return f"Error: File '{path}' does not exist in repository."

    try:
        with open(resolved_file, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()

        total_lines = len(lines)
        s_line = max(1, start_line or 1)

        # For small files (<=80 lines), return the entire file in one call
        if total_lines <= 80:
            numbered_lines = [f"{idx + 1}: {line}" for idx, line in enumerate(lines)]
            return f"File: {clean_path} (Full File: Lines 1 to {total_lines})\n" + "".join(numbered_lines)

        e_line = min(total_lines, end_line or (s_line + 60))
        selected_lines = lines[s_line - 1 : e_line]
        numbered_lines = [f"{s_line + idx}: {line}" for idx, line in enumerate(selected_lines)]

        return f"File: {clean_path} (Lines {s_line} to {e_line} of {total_lines})\n" + "".join(numbered_lines)
    except Exception as e:
        return f"Error reading file '{path}': {str(e)}"


@tool(args_schema=ListDirectoryInput)
def list_directory(path: str = ".") -> str:
    """
    List files and subdirectories at the given path relative to the repo root.
    """
    clean_path = path.strip().replace("\\", "/").lstrip("/")
    target_dir = (REPO_PATH / clean_path)

    if not is_safe_subpath(target_dir, REPO_PATH):
        return f"Error: Access denied. Path outside repository root: {path}"

    resolved_dir = target_dir.resolve()
    if not resolved_dir.exists() or not resolved_dir.is_dir():
        return f"Error: Directory '{path}' does not exist."

    entries = []
    try:
        for item in sorted(resolved_dir.iterdir()):
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
    Hardened against path traversal, duplicate naming ambiguities, and symlink escapes.
    """
    import ast

    clean_query = module_name.strip().replace("\\", "/").lstrip("/")
    resolved_repo_root = REPO_PATH.resolve()

    target_file: Optional[Path] = None

    # --- 1. Direct Relative Path Resolution with Strict relative_to Guard ---
    candidate_paths = [
        resolved_repo_root / clean_query,
        resolved_repo_root / f"{clean_query}.py",
        resolved_repo_root / f"src/{clean_query}",
        resolved_repo_root / f"src/{clean_query}.py",
    ]

    for cand in candidate_paths:
        if is_safe_subpath(cand, resolved_repo_root) and cand.resolve().is_file():
            target_file = cand.resolve()
            break

    # --- 2. Dynamic Search & Ambiguity Handling with Symlink Protection ---
    if not target_file:
        file_stem = Path(clean_query).stem
        matched_files = []

        for path in resolved_repo_root.rglob(f"{file_stem}.py"):
            # Security Guard: Ensure symlinks do not resolve outside the repository
            if not is_safe_subpath(path, resolved_repo_root):
                continue

            parts = set(path.parts)
            if any(ignored in parts for ignored in (".git", "__pycache__", "venv", ".venv", "build", "dist")):
                continue
            matched_files.append(path)

        if len(matched_files) == 1:
            target_file = matched_files[0].resolve()
        elif len(matched_files) > 1:
            # Python <3.12 safe formatting using .as_posix() (no backslashes inside f-string)
            rel_matches = [f"  - {p.relative_to(resolved_repo_root).as_posix()}" for p in matched_files]
            return (
                f"Multiple modules matching '{file_stem}.py' were found in the repository:\n"
                + "\n".join(rel_matches)
                + "\nPlease specify the exact relative path from the list above."
            )

    # --- 3. Fallback: Search for Class by Name (Fault-Tolerant) ---
    if not target_file:
        try:
            store = get_vector_store()
            class_hits = store.search(f"class {clean_query}", top_k=2)
            valid_class_matches = [
                hit for hit in class_hits
                if hit.get("chunk_type") == "class" and clean_query.lower() in hit.get("chunk_name", "").lower()
            ]
            if valid_class_matches:
                hit = valid_class_matches[0]
                return (
                    f"=== Class Summary: `{hit['chunk_name']}` ===\n"
                    f"File: {hit['file_path']} (Lines {hit['start_line']}-{hit['end_line']})\n\n"
                    f"Definition Outline:\n{hit['content'][:400]}..."
                )
        except Exception:
            pass  # Fall through to standard not found message

        return (
            f"Module or class '{module_name}' could not be located in the repository. "
            f"Try using 'search_code' or 'list_directory' to locate it."
        )

    # --- 4. Parse Module AST & Extract Accurate Top-Level Labels ---
    rel_path = target_file.relative_to(resolved_repo_root).as_posix()

    try:
        with open(target_file, "r", encoding="utf-8", errors="replace") as f:
            code = f.read()

        tree = ast.parse(code)
        docstring = ast.get_docstring(tree) or "No module docstring available."

        top_classes = []
        top_functions = []

        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                c_doc = ast.get_docstring(node) or "No docstring."
                first_line = c_doc.strip().splitlines()[0] if c_doc else ""
                top_classes.append(f"- Class `{node.name}` (Line {node.lineno}): {first_line}")
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                f_doc = ast.get_docstring(node) or "No docstring."
                first_line = f_doc.strip().splitlines()[0] if f_doc else ""
                top_functions.append(f"- Function `{node.name}` (Line {node.lineno}): {first_line}")

        summary = [
            f"=== Module Summary: {rel_path} ===",
            f"Module Overview:\n{docstring}\n",
            f"Top-level Classes ({len(top_classes)}):",
            "\n".join(top_classes[:10]) if top_classes else "  None",
            f"\nTop-level Functions ({len(top_functions)}):",
            "\n".join(top_functions[:10]) if top_functions else "  None",
        ]
        return "\n".join(summary)

    except Exception as e:
        return f"Error analyzing module '{module_name}': {str(e)}"


TOOLS = [search_code, read_file, list_directory, summarize_module]