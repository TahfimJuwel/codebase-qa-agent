"""
Module: parser.py
Implements AST-aware code parsing and chunking for Python source files,
and structural markdown/text chunking for non-Python documentation.
Strictly adheres to: 'Do NOT split in the middle of a function or class.'
"""

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class CodeChunk:
    """Represents a syntactically self-contained chunk of code or documentation."""
    file_path: str         # Relative path to repo root (e.g., "src/requests/models.py")
    chunk_name: str        # e.g., "PreparedRequest.prepare_url" or "global_scope"
    chunk_type: str        # "class", "function", "method", "module_header", or "text_section"
    start_line: int        # 1-indexed start line
    end_line: int          # 1-indexed end line
    content: str           # The actual code or text block

    def to_dict(self):
        return {
            "file_path": self.file_path,
            "chunk_name": self.chunk_name,
            "chunk_type": self.chunk_type,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "content": self.content
        }


class ASTCodeParser:
    """
    Parses Python source code files into semantic AST blocks (Classes, Methods, Functions).
    """

    def __init__(self, repo_root: Path):
        self.repo_root = repo_root

    def parse_python_file(self, file_path: Path) -> List[CodeChunk]:
        """
        Parses a single Python file using Python's native AST module.
        Extracts:
        1. Module-level header (imports + module docstring).
        2. Top-level functions.
        3. Classes and their individual methods.
        """
        relative_path = str(file_path.relative_to(self.repo_root)).replace("\\", "/")
        
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                source_code = f.read()
        except Exception as e:
            print(f"[Parser] Error reading {relative_path}: {e}")
            return []

        lines = source_code.splitlines(keepends=True)
        if not lines:
            return []

        try:
            tree = ast.parse(source_code, filename=str(file_path))
        except SyntaxError:
            # If the file has invalid syntax, fallback to parsing as plain text
            return self._fallback_plain_text_chunk(relative_path, lines)

        chunks: List[CodeChunk] = []

        # 1. Extract Module Header (docstring + top-level imports)
        header_lines: List[str] = []
        header_end = 0

        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom, ast.Expr)):
                end = getattr(node, "end_lineno", node.lineno)
                if end > header_end:
                    header_end = end
            else:
                # Stop when hitting the first class or function definition
                break

        if header_end > 0:
            header_content = "".join(lines[0:header_end]).strip()
            if header_content:
                chunks.append(
                    CodeChunk(
                        file_path=relative_path,
                        chunk_name=f"{relative_path}:header",
                        chunk_type="module_header",
                        start_line=1,
                        end_line=header_end,
                        content=header_content
                    )
                )

        # 2. Extract Top-level classes, methods, and standalone functions
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                chunks.extend(self._parse_class_node(node, lines, relative_path))
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                chunks.append(self._parse_function_node(node, lines, relative_path))

        # If AST found no classes or functions (e.g. only constants/config), chunk whole file
        if not chunks:
            return self._fallback_plain_text_chunk(relative_path, lines)

        return chunks

    def _parse_class_node(self, node: ast.ClassDef, lines: List[str], relative_path: str) -> List[CodeChunk]:
        """
        Extracts a class and its individual methods so large classes do not exceed token limits,
        while maintaining complete class-level context.
        """
        chunks: List[CodeChunk] = []
        class_name = node.name
        class_start = node.lineno
        class_end = getattr(node, "end_lineno", len(lines))

        # Identify all methods in this class
        method_nodes = [
            n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]

        if not method_nodes:
            # Class has no methods (e.g. data class or exception class), take the whole class
            class_content = "".join(lines[class_start - 1 : class_end]).strip()
            chunks.append(
                CodeChunk(
                    file_path=relative_path,
                    chunk_name=class_name,
                    chunk_type="class",
                    start_line=class_start,
                    end_line=class_end,
                    content=class_content
                )
            )
            return chunks

        # First: create a chunk for Class definition header & class-level docstring/attributes
        first_method_start = method_nodes[0].lineno
        class_header_content = "".join(lines[class_start - 1 : first_method_start - 1]).strip()
        if class_header_content:
            chunks.append(
                CodeChunk(
                    file_path=relative_path,
                    chunk_name=f"{class_name} (class definition)",
                    chunk_type="class",
                    start_line=class_start,
                    end_line=first_method_start - 1,
                    content=class_header_content
                )
            )

        # Second: create complete, unbroken chunks for each method inside the class
        for method in method_nodes:
            m_start = method.lineno
            m_end = getattr(method, "end_lineno", m_start)
            method_content = "".join(lines[m_start - 1 : m_end]).strip()
            
            # Prepend class context so embedding knows what class this method belongs to
            scoped_content = f"# Class: {class_name}\n{method_content}"

            chunks.append(
                CodeChunk(
                    file_path=relative_path,
                    chunk_name=f"{class_name}.{method.name}",
                    chunk_type="method",
                    start_line=m_start,
                    end_line=m_end,
                    content=scoped_content
                )
            )

        return chunks

    def _parse_function_node(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef, lines: List[str], relative_path: str
    ) -> CodeChunk:
        """Extracts a top-level standalone function without splitting."""
        fn_start = node.lineno
        fn_end = getattr(node, "end_lineno", len(lines))
        fn_content = "".join(lines[fn_start - 1 : fn_end]).strip()

        return CodeChunk(
            file_path=relative_path,
            chunk_name=node.name,
            chunk_type="function",
            start_line=fn_start,
            end_line=fn_end,
            content=fn_content
        )

    def _fallback_plain_text_chunk(self, relative_path: str, lines: List[str]) -> List[CodeChunk]:
        """Fallback for non-python files or small configuration files."""
        content = "".join(lines).strip()
        if not content:
            return []
        return [
            CodeChunk(
                file_path=relative_path,
                chunk_name=Path(relative_path).name,
                chunk_type="text_section",
                start_line=1,
                end_line=len(lines),
                content=content
            )
        ]

    def parse_file(self, file_path: Path) -> List[CodeChunk]:
        """Main dispatcher for any file based on its extension."""
        if file_path.suffix.lower() == ".py":
            return self.parse_python_file(file_path)
        
        # For non-python files (e.g. .md, .rst, .toml), parse line by line without breaking
        relative_path = str(file_path.relative_to(self.repo_root)).replace("\\", "/")
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
            return self._fallback_plain_text_chunk(relative_path, lines)
        except Exception:
            return []










if __name__ == "__main__":
    from src.config import REPO_PATH
    from src.ingest import collect_repo_files

    parser = ASTCodeParser(repo_root=REPO_PATH)
    all_files = collect_repo_files(REPO_PATH)
    
    total_chunks = 0
    all_chunks = []
    for f in all_files:
        chunks = parser.parse_file(f)
        total_chunks += len(chunks)
        all_chunks.extend(chunks)

    print(f"\n[Parser] Successfully parsed {len(all_files)} files into {total_chunks} structured AST chunks.")
    
    # Inspect 3 sample chunks
    print("\n--- Sample Extracted Chunks ---")
    for chunk in all_chunks[10:13]:
        print(f"\nFile: {chunk.file_path}")
        print(f"Name: {chunk.chunk_name} ({chunk.chunk_type})")
        print(f"Lines: L{chunk.start_line} - L{chunk.end_line}")
        print("Preview:")
        print("\n".join(chunk.content.splitlines()[:5]))
        print("...")