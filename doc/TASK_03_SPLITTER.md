# Task 3: Implement Splitter (LangChain `RecursiveCharacterTextSplitter`)

## Objective
Create `src/langchain_rag/splitter.py` that splits documents into overlapping chunks using LangChain's `RecursiveCharacterTextSplitter`. This **replaces** Step 2's hand-written `chunker.py` — and is your first taste of the framework abstraction gap.

## File to Create
`/home/dipak/agentic/step3_langchain_rag/src/langchain_rag/splitter.py`

## The comparison (hand-written vs LangChain)
Step 2's `chunker.py` was ~70 lines of manual word-splitting + overlap carry-over. LangChain expresses the splitter in a few config lines:
```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " ", ""],
    length_function=len,
)
chunks = splitter.split_text(text)
```
> **What's happening inside?** It tries each separator in order, splitting on the first match offset <= `chunk_size`, recursively, then merges with `chunk_overlap`. Test what breaks if you drop `separators=["",]` (empty-char fallback).

## What You Need To Do

### Class: `LangChainSplitter`
Design it as a thin, stable wrapper so the rest of the pipeline has a consistent shape (same `{text, metadata}` dicts as Step 2). Something like:
```python
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

class LangChainSplitter:
    def __init__(self, chunk_size=512, chunk_overlap=50, separators=None) -> None:
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap,
            separators=separators or ["\n\n", "\n", ". ", " ", ""],
        )

    def split_text(self, text: str) -> list[str]: ...

    def split_documents(self, docs, source=None) -> list[Document]: ...

    def load_and_split(self, file_path) -> list[dict]: ...
        # read .txt, split_text, return [{"text":..., "metadata":{"source":<basename>,"index":i}}]

    def load_directory(self, data_dir) -> list[dict]: ...
        # iterate sorted *.txt in data_dir, concat load_and_split results
```

### Step 3.1: `split_text`
Wrap `self.splitter.split_text(text)`. Return `list[str]`.

### Step 3.2: `load_and_split` — metadata `source` = **basename** ⭐
Same rule as Step 2: `metadata["source"]` must be the **basename** (`Path(file_path).name`), so `context_precision` can match `golden.jsonl`. Build dicts:
```python
{"text": chunk, "metadata": {"source": path.name, "index": i}}
```

### Step 3.3: `load_directory`
Iterate `data_dir`'s `.txt` files (sorted), call `load_and_split`, concatenate. If the dir is missing, return `[]` (no crash).

## Testing
```bash
cd /home/dipak/agentic/step3_langchain_rag
uv run python -c "
from langchain_rag.splitter import LangChainSplitter
s = LangChainSplitter()
chunks = s.load_directory('data/documents')
print('Total chunks:', len(chunks))
print('Sources:', {c['metadata']['source'] for c in chunks})
print('First chunk length:', len(chunks[0]['text']))
#print(chunks)
"
```
**Expected:** > 0 chunks; sources = {python_basics.txt, machine_learning.txt, rag_concepts.txt, api_design.txt} (**basenames, no paths**). Chunk count will differ slightly from Step 2 because `RecursiveCharacterTextSplitter` splits on separators, not raw word counts — document this difference in `notes.md`.

## Completion Criteria
- [ ] `splitter.py` created with `LangChainSplitter`
- [ ] `split_text()` returns chunks
- [ ] `load_and_split()` returns `{text, metadata}` dicts
- [ ] `metadata["source"]` is the **basename**
- [ ] `load_directory("data/documents")` returns > 0 chunks from 4 sources, no crash if dir missing

## Mocking (learner note) — no mocks here
`RecursiveCharacterTextSplitter` is pure local logic over strings (no network, no model). Test it directly on real text. The **filesystem** reads are local + fast — fine to use directly, same as Step 2.

## Verify
```bash
cd /home/dipak/agentic/step3_langchain_rag
uv run ruff check src/langchain_rag/splitter.py
uv run mypy src/langchain_rag/splitter.py
uv run pytest tests/test_splitter.py -v
uv run pytest tests/test_splitter.py -v --cov=langchain_rag.splitter --cov-report=term-missing
uv run ruff check tests/test_splitter.py
uv run mypy tests/test_splitter.py
```
> You'll need to write `tests/test_splitter.py` (Task 15 covers the full suite, but add tests for *this* file now as you go — mirror Step 2's convention of testing each module).

## Report Back
When done, tell me:
1. Total chunks produced (and how it differs from Step 2's count)
2. The set of sources returned (confirm basenames)
3. What happened when you set `separators=[""]` — did chunking change? (framework-internals lesson)
4. Paste your `splitter.py` for me to review
