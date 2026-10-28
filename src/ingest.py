"""
Module: ingest.py
Handles cloning/fetching of the target Git repository and collects only valid
text-based source files while strictly skipping binaries and cache artifacts.
"""

import os
from pathlib import Path
from typing import List
import git
from src.config import REPO_URL, REPO_PATH


# Set of extensions considered safe text/source code for indexing
ALLOWED_EXTENSIONS = {
    ".py",    # Python source files
    ".md",    # Documentation
    ".rst",   # Python doc format commonly used in requests
    ".txt",   # Text files / license
    ".toml",  # pyproject.toml configuration
    ".yaml",  # CI/CD configs
    ".yml",
    ".json"   # Configurations
}

# Directories that must ALWAYS be ignored to prevent noise, bloat, or loops
IGNORED_DIRECTORIES = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".tox",
    "venv",
    ".venv",
    "build",
    "dist",
    "htmlcov",
    ".eggs",
}


def clone_or_update_repo(repo_url: str = REPO_URL, target_dir: Path = REPO_PATH) -> Path:
    """
    Clones the repository if it does not exist locally; otherwise pulls latest changes.
    """
    if target_dir.exists() and (target_dir / ".git").exists():
        print(f"[Ingest] Repository already exists at {target_dir}. Fetching latest...")
        repo = git.Repo(target_dir)
        try:
            repo.remotes.origin.pull()
            print("[Ingest] Successfully updated to latest commit.")
        except Exception as e:
            print(f"[Ingest] Warning: Could not pull latest changes ({e}). Using existing local copy.")
    else:
        print(f"[Ingest] Cloning repository from {repo_url} into {target_dir}...")
        git.Repo.clone_from(repo_url, target_dir, depth=1)  # depth=1 for shallow, fast clone
        print("[Ingest] Clone completed.")

    return target_dir


def is_binary_file(file_path: Path) -> bool:
    """
    Determines if a file is binary by inspecting its initial byte chunk.
    If null bytes (0x00) exist, it's considered a binary file.
    """
    try:
        with open(file_path, "rb") as f:
            chunk = f.read(1024)
            if b"\x00" in chunk:
                return True
        return False
    except Exception:
        return True


def collect_repo_files(repo_path: Path = REPO_PATH) -> List[Path]:
    """
    Recursively scans the cloned repo directory and returns a list of valid
    text-based source files, skipping binaries and ignored directories.
    """
    collected_files: List[Path] = []

    for root, dirs, files in os.walk(repo_path):
        # Prune ignored directories in-place so os.walk doesn't descend into them
        dirs[:] = [d for d in dirs if d not in IGNORED_DIRECTORIES]

        for file_name in files:
            file_path = Path(root) / file_name

            # 1. Filter by allowed file extension
            if file_path.suffix.lower() not in ALLOWED_EXTENSIONS:
                continue

            # 2. Skip binary files defensively
            if is_binary_file(file_path):
                continue

            collected_files.append(file_path)

    print(f"[Ingest] Scanned codebase. Collected {len(collected_files)} valid text/code files for indexing.")
    return collected_files


if __name__ == "__main__":
    # Test script run directly
    repo_dir = clone_or_update_repo()
    files = collect_repo_files(repo_dir)
    print("\nSample collected files:")
    for f in files[:5]:
        print(" -", f.relative_to(repo_dir))