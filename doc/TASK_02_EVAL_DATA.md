# Task 2: Evaluation Data + Register Prompt V1

## Objective
Review the golden dataset (identical to Step 2) and register the first prompt version in the shared registry. This is the **ground truth** the evaluator measures against — and it must be byte-for-byte the same as Step 2 for a fair A/B.

## What You Need To Do

### Step 2.1: Review the Golden Dataset
Location: `/home/dipak/agentic/step3_langchain_rag/eval/golden.jsonl`

Format: JSON Lines — one JSON object per line:
```json
{"question": "What is Python?", "answer": "Python is a high-level, interpreted programming language...", "source": "python_basics.txt", "difficulty": "easy"}
```

**Fields:**
| Field | Purpose |
|-------|---------|
| `question` | Input question |
| `answer` | Gold standard answer |
| `source` | Which document contains the answer (basename such as `python_basics.txt`) |
| `difficulty` | easy / medium / hard |

> This dataset is **the same file as Step 2**. Because the data, model, and metrics are identical, any score difference between Step 2 and Step 3 is attributable to the **framework**, not the data.

Verify:
```bash
cd /home/dipak/agentic/step3_langchain_rag
uv run python -c "
import json
from collections import Counter
rows = [json.loads(l) for l in open('eval/golden.jsonl') if l.strip()]
print('Total pairs:', len(rows))
print('By source:', Counter(r['source'] for r in rows))
print('By difficulty:', Counter(r['difficulty'] for r in rows))
"
```
**Expected:** 20 pairs, 4 sources, mix of easy/medium/hard.

### Step 2.2: Understand the shared PromptRegistry (preview)
Full build in Task 10. You only need the schema you'll seed now:
- `prompt_id`: `RAG_ANSWER`
- `version`: `1.0.0`
- `template`: the answer prompt with `{context}` and `{question}` variables
- `input_variables`: `["context", "question"]`
- `model`, `temperature`: from the registry Policy (default the Step 2 Qwen model + `0.0`)
- `status`: `draft → testing → approved → retired`
- `eval_scores`, `run_count`: Evidence, filled by Tasks 11/12

> ⭐ Step 3 takeaway: in LangChain you'd normally define the prompt with `PromptTemplate.from_template(...)`. Here you register it in the **registry** and LangChain's `PromptTemplate` is built **from the registry** in Task 10 (`LangChainPromptAdapter`). Registry first, LangChain second.

### Step 2.3: Verify data references
The 4 source documents must exist under `data/documents/`:
```bash
ls data/documents/
```
**Expected:** api_design.txt, machine_learning.txt, python_basics.txt, rag_concepts.txt (identical to Step 2).

## Completion Criteria
- [ ] Golden dataset loads: 20 valid JSON lines
- [ ] Confirmed `source` values match document basenames
- [ ] Understand registry fields (Content / Policy / Output / Evidence)
- [ ] All 4 documents present under `data/documents/`

## Mocking (learner note) — no code yet
No testable module in this task (just data + registry schema). It fixes the *contract* the evaluator will mock later — the `source` basename is what `context_precision` compares.

## Report Back
When done, tell me:
1. Total Q&A pairs
2. Count by source and difficulty
3. Any `source` in golden.jsonl that does NOT match a document filename
