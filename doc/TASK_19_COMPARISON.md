# Task 19: Step 2 vs Step 3 Gap Analysis ⭐ NEW (the point of this milestone)

## Objective
Perform the deep-dive **framework gap analysis** that this milestone exists for: run the *same* eval on Step 2 (hand-written) and Step 3 (LangChain) and analyze **code volume, debugging, flexibility, and eval scores**. This turns the whole step from "I reimplemented it in LangChain" into "I understand the tradeoff."

## What You Need To Do

### Step 16.1: Code volume comparison
Count lines of the parallel components and fill the table.

| Component | Step 2 (hand-written) | Step 3 (LangChain) | Reduction |
|-----------|----------------------|--------------------|-----------|
| Chunking / splitting | `chunker.py` ___ lines | `splitter.py` ___ lines | ___% |
| Embedding | `embedder.py` ___ lines | `embeddings.py` ___ lines | ___% |
| Vector store | `store.py` ___ lines | `vectorstore.py` ___ lines | ___% |
| Retriever | `retriever.py` ___ lines | `retriever.py` ___ lines | ___% |
| Generator | `generator.py` ___ lines | `generator.py` + `prompts.py` ___ lines | ___% |
| Pipeline | `pipeline.py` ___ lines | `pipeline.py` ___ lines | ___% |
| Prompt mgmt | `prompt_registry.py` ___ lines | `prompt_registry.py` + adapter ___ lines | ___% |

Use `wc -l` per file. Fill actual numbers.

> ⭐ Mentor note: the honest picture is not just "LangChain = fewer lines." In Step 3 the registry lives in a **shared module** (`prompt_registry.py` + the new `prompts.py` adapter). LangChain's `PromptTemplate` alone has **no** versioning — so the adapter is the thing that keeps the registry meaningful. Count that honestly.

### Step 16.2: Fill the "what did we lose?" table
| Feature | Step 2 registry | LangChain `PromptTemplate` |
|---------|-----------------|---------------------------|
| Version numbers | ✅ | ❌ |
| Status lifecycle (draft→approved→retired) | ✅ | ❌ |
| No-overwrite guard | ✅ | ❌ |
| Audit history | ✅ | ❌ |
| Eval-score linking | ✅ | ❌ |
| Compare versions | ✅ | ❌ |
| Share prompts | ❌ | ✅ (Prompt Hub) |
> Verdict: the **registry + adapter** pattern is what makes Step 3 production-ready; LangChain alone isn't.

### Step 16.3: Eval score comparison
From `eval/results/` on both repos, record the four keyword metrics AND the DeepEval (Groq) scores for each. Then analyze any diffs:

| Metric | Step 2 (hand-written) | Step 3 (LangChain) | Same-ish? | Why? |
|--------|----------------------|--------------------|-----------|------|
| Context Recall | ___ | ___ | ? | |
| Context Precision | ___ | ___ | ? | |
| Faithfulness | ___ | ___ | ? | |
| Answer Relevance | ___ | ___ | ? | |

Investigate discrepancies (see the questions below): chunk-boundary differences from `RecursiveCharacterTextSplitter` vs your word-count chunker, Chroma cosine vs numpy cosine, prompt template rendering differences.

### Step 16.4: Debugging experience
Try to answer: "Why is retrieval returning the wrong chunks?" in each codebase.
- **Step 2:** you can inspect `embedder.embed_query`, `store.search` similarity scores, and chunk/index directly.
- **Step 3:** `retriever.store.search` reaches into Chroma's `_collection`; the scores are one distance→similarity conversion away. Try `langchain.debug = True` and see the framework-internals noise.

### Step 16.5: Decision matrix
Fill in when you'd use which (adapt Step 2/Step 3 README's matrix):
```
START
 ├─ Quick prototype/demo?              → LangChain
 ├─ Need custom prompt versioning?     → Hand-written registry (+ LangChain adapter)
 ├─ Debugging-critical?                → Hand-written (see everything)
 ├─ Production with standard patterns? → LangChain (+ registry adapter)
 ├─ Multi-provider LLM switching?      → LangChain (ChatOpenAI/ChatGroq/ChatGemini)
 └─ Learning RAG mechanics?            → Hand-written first, then LangChain
```

## Deliverables
- `doc/notes.md` — write up all findings (code volume, debugging notes, the `openai_api_base` gotcha, the `_collection` score-recovery hack, the registry-adapter verdict)
- Update `doc/SUMMARY.md`/this doc with the completed tables

## Completion Criteria
- [ ] Code-volume table filled with real `wc -l` numbers
- [ ] "What we lost" table verified
- [ ] Step 2 vs Step 3 eval scores recorded and analyzed
- [ ] Debugging comparison noted
- [ ] Decision matrix filled
- [ ] Findings written into `doc/notes.md`

## Report Back
When done, tell me:
1. The code-volume numbers (Step 2 vs Step 3, total)
2. The eval score deltas (keyword AND DeepEval) with your explanation
3. Three things LangChain saved you, and three things it cost you
4. Your final decision: for **this** project, hand-written vs LangChain vs hybrid?
