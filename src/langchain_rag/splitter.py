"""LangChain text-splitter wrapper.

Step 2 hand-wrote a word/token-aware chunker; LangChain ships
``RecursiveCharacterTextSplitter`` which handles separators and overlap for us.
This thin wrapper keeps the splitter configurable while exposing the same
"return chunks with metadata" shape the rest of the pipeline expects.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class LangChainSplitter:
    """Wraps LangChain's recursive splitter with a stable config and helpers."""

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        separators: list[str] | None = None,
    ) -> None:
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators
            or ["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )

    def split_text(self, text: str) -> list[str]:
        """Split plain text into overlapping chunks (strings)."""
        return self.splitter.split_text(text)

    def split_documents(
        self, docs: list[Document], source: str | None = None
    ) -> list[Document]:
        """Split pre-built Documents, tagging each with the source filename."""
        out = self.splitter.split_documents(docs)
        if source:
            for d in out:
                d.metadata.setdefault("source", source)
        return out

    def load_and_split(self, file_path: str | Path) -> list[dict[str, Any]]:
        """Read a text file and return chunk dicts: {text, metadata{source, index}}."""
        path = Path(file_path)
        text = path.read_text(encoding="utf-8")
        chunks = self.split_text(text)
        return [
            {"text": chunk, "metadata": {"source": path.name, "index": i}}
            for i, chunk in enumerate(chunks)
        ]

    def load_directory(self, data_dir: str | Path) -> list[dict[str, Any]]:
        """Read all text files in a directory and return chunk dicts."""
        dir_path = Path(data_dir)
        if not dir_path.exists():
            return []
        all_chunks: list[dict[str, Any]] = []
        for file_path in sorted(dir_path.glob("*.txt")):
            all_chunks.extend(self.load_and_split(file_path))
        return all_chunks
