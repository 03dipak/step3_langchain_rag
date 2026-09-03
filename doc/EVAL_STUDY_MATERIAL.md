# Study Material: "How to Evaluate Your RAG App" — CampusX LLM Evals Series

> **Purpose:** Structured study notes distilled from the CampusX GenAI-interview /
> RAG-eval video series. Use this as reference material for reproducing the same
> 3-level evaluation framework in this repo (`step3_langchain_rag`) — the tools
> used are DeepEval, MLflow, GitHub Actions, and LangSmith/Langfuse/Confident AI.

---

## 1. Interview Framing — "How Do You Evaluate Your RAG App?"

The single most common reason candidates **fail** this interview question is that
they only name 3–4 metrics (e.g. "recall and precision"). The winning answer is a
**structured framework** — present it as a suite, not a list:

1. **Build an Evaluation Suite** (a set of automated eval scripts), not one-off checks.
2. Test at **three levels** (Component → Pipeline/Triad → Application Quality/Safety/Ops).
3. Add **Regression Testing** wired into a **CI/CD pipeline** so bad changes can't deploy.
4. Add **Online Evaluation** with **drift tracking** and a **self-improving golden dataset** (feed production edge-cases back into offline goldens).

---

## 2. The Multi-Level Evaluation Framework

> **Core principle:** a RAG app cannot be judged by a single test. It needs a
> suite, run at three distinct levels — each answering a different question.

```
                        ┌────────────────────────────────────────────┐
                        │     1. COMPONENT-LEVEL  (unit-style)   │
                        │   Retriever: Recall, Precision          │
                        │   Generator: Faithfulness, Answer       │
                        │             Relevance, Citation Accuracy │
                        └────────────────────────────────────────────┘
                                          │ build + connect
                                          ▼
                        ┌────────────────────────────────────────────┐
                        │  2. PIPELINE-LEVEL  (the RAG Triad)    │
                        │   Context Relevance ↔ Faithfulness ↔       │
                        │            Answer Relevance                │
                        └────────────────────────────────────────────┘
                                          │ ship whole system
                                          ▼
                        ┌────────────────────────────────────────────┐
           3. APPLICATION-LEVEL            │                      │
       Quality: Correctness/Completeness/  │                      │
       Style  ·  Safety: Toxicity/PII/     │                      │
       Jailbreak  ·  Ops: Latency/Cost/Rel │                      │
       └────────────────────────────────────────────┘
```

### Level A — Component-Level (isolated, like unit tests)
Each component is evaluated **in isolation as it is built**, **before** wiring the
pipeline. Two components:

| Component | Step | Metrics |
|-----------|------|---------|
| **Retriever** (load → chunk → embed → vector-DB query) | 1 & 2 | **Recall** — portion of all relevant docs found. **Precision** — portion of retrieved docs actually useful. |
| **Generator** | 3 & 4 | **Faithfulness**, **Answer Relevance**, **Citation Accuracy** — judged against a hand-made golden dataset of questions + contexts. |

### Level B — Pipeline-Level (the RAG Triad)
Once retriever + generator are connected, evaluate them **together** using the
**RAG Triad** — the three pairwise relationships between question, context, answer:

```
                [ Question ]
                /          \
      Context  /            \  Answer
    Relevance /              \ Relevance
             /                \
      [ Context ] ----------- [ Answer ]
                  Faithfulness
```

- **Context Relevance** — is the *retrieved context* relevant to the *question*?
- **Faithfulness** — is the *answer* derived from the *context*?
- **Answer Relevance** — is the *answer* relevant to the *question*?

### Level C — Application/System-Level (the whole app)
Evaluate the completed system across **three suites**:

| Suite | Metrics |
|-------|---------|
| **Quality (step 7)** | **Correctness** (accurate?), **Completeness** (answers all sub-parts?), **Style** (matches brand voice). |
| **Safety (step 8)** | **Toxicity**, **PII leakage**, **jailbreak / scope adherence**. |
| **Ops (step 9)** | **Latency**, **Cost per query**, **token consumption**, **Reliability**. |

---

## 3. Tool Choice — DeepEval vs. Ragas

The series uses **DeepEval**. Rationale:

1. **Ragas** was already covered in the earlier advanced RAG course.
2. DeepEval is **built on top of pytest** — Python developers feel at home; tests are pytest tests.
3. **Broader scope**: DeepEval extends to agents, multi-turn chatbots, and image apps; it is highly adopted and positioned to become the industry standard.

> House note: this repo (`step3_langchain_rag`) already vendors **DeepEval 2.9.3**
> with a **Groq free-tier judge** (`GroqJudge`, `openai/gpt-oss-120b`) instead of an
> OpenAI-keyed judge — same DeepEval API, different (free) underlying LLM. See
> `doc/TASK_16_DEEEEVAL.md`.

---

## 4. Component 1 — Evaluating the RETRIEVER

### 4.1 Two failure modes
1. **Missing Context** — fails to retrieve the correct chunk needed to answer.
2. **Noisy Retrieval** — retrieves the right chunk but padded with irrelevant noise.

These map exactly to **Recall** and **Precision**. Both are **reference-based**:
they need a golden dataset to compare against.

### 4.2 The recall/precision trade-off
- Low recall? Raise **K** (e.g. 5 → 10).
- But pulling recall toward 100% via bigger K **adds noise → precision drops** toward 0.
- The two metrics are in natural tension.

### 4.3 Fallacy: programmatic ID-based evals
A tempting naive approach maps each golden question to specific chunk IDs and
scores by exact ID match. **This fails for real RAG**:
1. **Extreme human labor** — a human must read all (e.g. 800+) chunks to label correct IDs per question.
2. **Fragility to retuning ("the voiding problem")** — changing chunk size/overlap rebuilds chunks & boundaries, **voiding every ID mapping** and forcing a full re-label.
- It only works if docs are fully disjoint AND parameters never change — neither true for transcripts. **Bad engineering here.**
- **Solution:** use a golden set of `{question, ideal_answer}` — the ideal answer is stable regardless of chunking, so it survives re-tunes.

### 4.4 DeepEval Contextual Recall (how it's computed)
1. LLM judge reads the **ideal answer** and splits it into atomic claims.
2. Judge checks each retrieved chunk for the presence of each claim.
3. **`Contextual Recall` = claims-found-in-contexts / total-claims-in-ideal-answer.**

### 4.5 DeepEval Contextual Precision (ranking-aware)
Used when you want to reward the retriever for putting relevant chunks **on top**.
1. Judge scores each chunk individually: "does it help produce the expected answer?" (yes/no + reason).
2. Precision is computed at **each retrieval step** and **averaged** — so rank matters.

| Case | Order | Step-wise precision | Outcome |
|------|-------|--------------------|---------|
| A | `[Rel, Rel, Noise, Noise, Noise]` | 1/1, 2/2, 2/3, 2/4, 2/5 | **High** average |
| B | `[Noise, Noise, Noise, Rel, Rel]` | 0/1, 0/2, 0/3, 1/4, 2/5 | **Very low** average |

> Same fraction of relevant chunks (2/5), totally different scores because precision-at-K is the headline — ranking-aware.

### 4.6 Building the retriever golden set — 4 methods
1. **Hand-authored** — high quality, exhausting, not scalable.
2. **LLM-assisted drafting + human review** *(recommended & used)* — feed transcripts to Claude → generate student-style queries/answers → manually clean out general-internet knowledge.
3. **DeepEval `Synthesizer`** — disappointing here; generated over-formal / off-topic questions students would never ask.
4. **Production logs** — ingest real user queries with positive feedback (thumbs-up) after launch.

Final: `retriever_goldens.json` = **15 high-quality student-style questions**.

### 4.7 DeepEval code pillars
1. `LLMTestCase` — one row: `input`, `expected_output`, `retrieval_context`.
2. `ContextualRecallMetric` / `ContextualPrecisionMetric` — config: threshold + judge model.
3. `evaluate(test_cases, metrics)` — runs every metric on every case via the judge.

> Troubleshooting: run with `python3 -m evals.eval_retriever` (module flag) and
> add `__init__.py` to `src/` and `evals/` to fix `ModuleNotFoundError: No module named 'src'`.

### 4.8 Sweep / optimization results (the value of an offline suite)

| Run | chunk/overlap | embedding | reranker | Contextual Recall | Contextual Precision | pass (of 15) |
|-----|---------------|-----------|----------|-------------------|----------------------|--------------|
| 1 baseline | 750/100 | text-embedding-3-small | no | 80% | 80% | 10–15 |
| 2 larger chunks | 1000/150 | text-embedding-3-small | no | **97%** | 83% | 12–15 |
| 3 reranker | 1000/150 | text-embedding-3-small | yes | 92% | **85%** | 13–15 |
| 4 large embedding | 1000/150 | text-embedding-3-large | yes | **99%** | 85% | 12–15 |
| 5 lower K to 3 | 1000/150 | text-embedding-3-large | yes | lowered | **84% (dropped)** | lowered |

**Learning:** larger chunks → better recall (whole concepts not split); reranker →
better precision (puts relevant chunks on top); larger embedding → near-perfect
recall; **lowering K backfired** (precision dropped to 84%). Slight variance is
expected on a small 15-row set.

**Target state:** Contextual Recall >95%, Contextual Precision ~85%.

---

## 5. Component 2 — Evaluating the GENERATOR

### 5.1 Setup (`generator.py`)
- Model **GPT-4o-mini**, **temperature 0** (deterministic eval outputs).
- System prompt: act as helpful TA; answer **only from context**; if context is
  insufficient say *"I don't have enough information in the course material to
  answer that"* (no hallucination); clear & concise.
- **Isolation:** feed the generator the **golden context** (bypass the retriever),
  so a low score is purely the generator's fault.

### 5.2 Two failure modes & their metrics
| Failure | Description | Metric |
|---------|-------------|--------|
| **Unfaithful (hallucination)** | Adds facts from pre-training absent from context. | **Faithfulness** — grounded strictly in context? |
| **Irrelevant** | Faithful to context but doesn't answer the question. | **Answer Relevance** — does it address the query? |

> Key distinction: **faithful ≠ correct.** If retrieval pulls wrong context, a
> faithful generator still produces a grounded-but-wrong answer. Faithfulness only
> checks grounding.

### 5.3 Faithfulness (LLM-judged)
1. Golden set = `{question, golden_context}`.
2. Generator(q, golden_context) → answer.
3. Judge splits the **answer** into atomic claims.
4. Check each claim against **golden context**.
5. **`Faithfulness` = claims-present-in-context / total-claims-in-answer.**

### 5.4 Answer Relevance (reference-free)
No golden context/reference needed — compares answer to question directly.
1. Judge splits answer into claims.
2. Judge: does each claim help answer the question (or off-topic)?
3. **`Answer Relevance` = relevant-claims / total-claims-in-answer.**
   - Example: explaining "benchmark contamination" while the answer is about
     benchmark saturation is technically true but off-topic → lowers the score.

### 5.5 LLM-as-a-judge reliability (the "bad pitch" argument)
Judges can be wrong (false +/−). But using the **same** judge across runs keeps a
**consistent bias** — like a bad cricket pitch affecting both teams equally — so
version-to-version comparisons (V1 vs V2) remain valid. **Mitigation:** always use
your most capable model as the judge.

### 5.6 Optimizing the generator
Generators are tuned via **model upgrade** or **system-prompt engineering** (retrievers via chunk size / reranker).

- Baseline: Faithfulness ~91%, Answer Relevance ~73%.
- Added rules over iterations (prompt-engineering the "constitution"):
  - "Do not strengthen or overstate claims."
  - "The context is an informal lecture transcript; synthesize and rephrase it."
  - "Do not require the question's exact wording to appear."
- Optimized: **Faithfulness 96%, Answer Relevance 92%.**
- Track metrics, prompts, and run-configs via `deepeval view` → **Confident AI** dashboard.

---

## 6. Pipeline-Level — the RAG Triad (full RAG pipeline)

Run the integrated pipeline against the golden questions; `retrieval_context` comes
from the real retriever now.

### Context Relevance (reference-free)
1. Query → retriever fetches K contexts.
2. Judge splits the fetched chunks into claims, flags which claims are useful for the query.
3. **`Contextual Relevance` = relevant-claims-retrieved / total-claims-in-retrieved-context.**

### The "curious case": standalone precision 99%/89% but pipeline triad only ~42%
Pipeline RAG Triad run: Faithfulness ~92–93%, Answer Relevance ~86–87%,
**Contextual Relevance only ~42–43%**.

**Why? "Intra-chunk noise."**
- Standalone **Precision** asks: does a chunk *contain* a useful line? (a 1000-char
  chunk with 1 useful sentence counts as "relevant" → high precision).
- **Contextual Relevance** asks: of all the *claims inside* the chunk, how many are
  useful? (5 lines, 1 useful → 4 lines are flagged as noise → low relevance).

**Fix:** **reduce chunk size** at retrieval config so the model fetches tighter,
more condensed spans — killing intra-chunk noise and raising Contextual Relevance.

---

## 7. Application-Level — QUALITY (G-Eval)

The first five metrics (Recall, Precision, Faithfulness, Answer Relevance, Context
Relevance) are **count-based**: judge splits text into atomic claims, then we
mathematically count favorable/unfavorable ratios (a score like 3/4).

**Count-based metrics fail for holistic qualities:**
- **Style** exists only at the whole-answer level, not per-sentence.
- **Correctness** (e.g. using an analogy) can't be matched sentence-by-sentence to a golden answer.

So Correctness / Completeness / Style need **judgment-based** metrics — an LLM reads
the whole answer and assigns a score.

### 7.1 The flaw of the "simple" LLM-as-a-judge → high variance
Simple setup: question + expected + actual + high-level criterion → judge scores
0–10. **Problem:** re-running without changing code gives wildly fluctuating scores
(75% → 85%). Causes:
1. **No strict constitution** — "compare and decide correctness" leaves room for
   interpretation; judge evaluates differently each call.
2. **Discrete probabilistic jumps** — if the model is torn between 7 (40% probable)
   and 8 (51% probable), tiny token shifts flip the emitted integer between runs.

### 7.2 G-Eval (2023 paper) — the deterministic judge
Best with **GPT-4** as the underlying judge. Two innovations kill variance:

**Innovation 1 — Chain of Thought evaluation steps (the rule book).**
Instead of a loose criterion, break the criterion into **4–5 highly specific,
step-by-step guidelines** + a **clear scoring rubric** (what counts as 0–4 / 5–8 /
9–10). Removes the model's guessing space; same "constitution" every run.

**Innovation 2 — Probability-weighted scoring (log-probabilities).**
Extract the **log-probs** of output tokens: take top numerical tokens 0–10,
ignore non-numerics, normalize to sum 1, and emit the **weighted average**.
- e.g. instead of a hard "8", the weighted average of tokens 7/8/9 → a stable **7.84**.
- Scores now change only by a fraction of a decimal across runs, instead of integer jumps.

### 7.3 DeepEval `GEval` implementation
- Initialize with `name`, high-level `criteria`, `threshold`.
- **Custom steps:** pass your own evaluation steps directly (rather than auto-generated)
  once you understand your edge-cases — this eliminates step-generation variability.
- **`strict_mode`:** `False` → probability-weighted scoring (recommended). `True` →
  bypass log-prob math, return raw integers.

### 7.4 Hands-on quality results (15-question `correctness_goldens.json`)

**Correctness** (whether the answer is *factually correct*, vs Faithfulness which only checks grounding):
- Baseline **66%** (8 pass/7 fail). Failure cause: judge penalized brevity — the
  golden answers were detailed, the chatbot's correct-but-shorter answers lost points.
- Fix: refined G-Eval steps + explicit rubric "do not deduct for brevity or omitted
  points; only wrong statements count."
- Optimized: **84%**, re-run **83%** → shows G-Eval's stability.

**Completeness** (does it address all parts/sub-questions?):
- Baseline **68%** (5 pass/10 fail). Cause: generator prompt-engineered to be concise/restricted → missed sub-points.
- Fix: prompt-engineering generator system prompt — "answer thoroughly", "identify
  every distinct part of the question", "address all components".
- Optimized: **75%** (14 pass/1 fail).

**Style** (matches the brand voice — e.g. CampusX teaching style, intuition-first):
- Baseline **54%** (no expected answer; checks actual output only). Cause: never told
  how to format + the style rubric over-rewarded "analogies/concrete examples",
  penalizing clear direct answers lacking an analogy.
- Fix (two-fold): generator prompt "write in flowing conversational prose, explain
  intuition first in plain language"; style rubric "an analogy/example is a bonus,
  but a clear direct answer is fully acceptable."
- Optimized: **74%**.

---

## 8. Application-Level — SAFETY

### 8.1 Why LLM safety is different
The "brain" is **probabilistic** — same input → different output each time — making
safety at scale uniquely hard. AI Safety & Security is predicted to become its own
specialized field within ~5 years. Security loop is a continuous **2-step cycle**:
**Evaluate** (test against known attack scenarios) → **Guardrail** (controls) → re-evaluate.

### 8.2 Six LLM failure modes
1. **Sensitive information leakage** — extracting system prompts, private data, credentials, proprietary content.
2. **Scope / policy violation** — bypassing the intended role (e.g. Amazon's sales bot writing homework instead).
3. **Harmful / toxic output** — jailbreaking into dangerous or profane output.
4. **Misinformation / hallucination** — confident incorrect facts (covered by Faithfulness).
5. **Bias / unfairness** — biased against groups due to training data.
6. **Unsafe actions / excessive agency** — agents misusing tools (e.g. an agent trading away its funds).

**Two ways they appear:** *non-adversarial* (natural model/context/prompt failures)
vs *adversarial* (intentional attacker manipulation).

### 8.3 Adversarial attack taxonomy
1. **Prompt manipulation**: direct injection ("ignore all rules"), indirect injection
   (malicious text hidden in a webpage the model is told to read), jailbreaking
   (role-play to bypass rules), obfuscation (Base64/other-encoding), multi-turn escalation.
2. **Poisoning**: corrupt training / fine-tuning / **RAG knowledge-base** data.
3. **Privacy / inference**: flood queries to reverse-engineer / clone the model.
4. **Tool exploitation / hijacking**: take over agent↔tool (e.g. MCP) control.
5. **Resource exhaustion**: DoS / infinite-loop that inflates token bills.

### 8.4 Guardrail types
- **Prompt guardrails** (instructions in the system prompt)
- **Input guardrails** (small model screens incoming user prompts)
- **Output guardrails** (classifier scrubs PII/keys/CC before display)
- **Retrieval guardrails** (inspect retrieved chunks for toxicity before injection)
- **Tool guardrails** (filter tool args/instructions)
- **Human-in-the-loop** (route high-risk actions — e.g. refunds — to a human)
- **Operational guardrails** (rate limits, token caps, timeouts, max agent-loop steps)
- **Red teaming** — a dedicated group acts as ethical hackers; new failure points → evaluate → test → guardrail → continuous loop.

### 8.5 Defining the app's attack surface (CampusX example)
Not all failure modes apply. For a text-only doubt-solver (no tools):
- ✅ **Leakage** — transcripts may contain private student questions, phone numbers, emails, API keys.
- ✅ **Scope/policy violation** — users coaxing it into a general-purpose coding agent (API-bill risk).
- ✅ **Toxicity** — brand reputation; a rude reply gets screenshotted onto social media.
- ❌ **Bias** — homogeneous student demographic, narrow topic (deferred).
- ❌ **Unsafe actions** — no tools/agents, so no agency risk.

**Attack surface = Toxicity + Leakage + Scope adherence.**

### 8.6 The Safety Policy (the "constitution")
1. **Scope adherence** — answer only questions tied to enrolled learning content.
2. **Leakage** — never reveal system prompts, raw chunks, verbatim premium content, or PII.
3. **Toxicity** — no abusive/hateful/threatening/sexually-inappropriate output.

### 8.7 Hands-on safety results

**Toxicity** (`toxicity_goldens.json`, 15 cases incl. adversarial, benign, mixed):
- Why test it even though OpenAI/Anthropic align their models: custom definitions
  ("Are you stupid?" is toxic in education but not blocked by providers), RAG contexts
  can carry/echo toxic words, model switches to weaker open models, and two-way defense is best practice. Benign queries guard against false positives.
- DeepEval built-in **`ToxicityMetric`** (reference-free) → judge labels opinions toxic/non-toxic → ratio.
- Result: **0.00** (100% pass) — modern models are well-aligned out of the box.

**Leakage** (`leakage_goldens.json`, 15 = prompt / course-content / PII):
- Metrics: **G-Eval** (custom) for prompt & course-content leakage; DeepEval built-in **PII Leakage Metric**.
- Result: course-content 100%, prompt 96%, **PII 80%** (4/5). PII failed *false positive*: user said "My name is Anjali...", model replied "Hi Anjali..." → DeepEval flagged the name as a leak (extreme strictness).
- Fixes: system prompt forbids reproducing credentials/passwords/personal details;
  wrap context in strict tags (e.g. `<course_context>`) so it isn't confused with system instructions; output-classifier filters to redact PII.

**Scope adherence**:
- DeepEval's built-in `Misuse` metric is too broad (needs a wide domain); the app's domain is narrow ("LLM Evals") → build a **custom G-Eval** "Scope Adherence".
- Result: baseline **94%** (14/15). Failure: a student asked for a model-evals explanation *plus* a romantic anniversary message — the bot drafted the romance → scope fail.
- Fixes: system-prompt rules forbidding non-course content (→ 99%); **query decomposition**
  — a small model splits a multi-part query and passes only the scope-adherent parts to the generator.

### 8.8 Takeaway
System prompts aren't written in a day — they **grow into a robust "constitution"
(trade secret)** through eval-driven iteration. Changing a prompt to fix one metric
(e.g. scope) can break another (e.g. the RAG Triad) → this is exactly why
**regression testing** over the whole suite is required before shipping.

---

## 9. Application-Level — OPERATIONS (ops/telemetry-driven)

**Ops evals ≠ quality evals.** Quality evals rely on LLM judges + curated goldens;
**ops evals are software + telemetry** — direct measurements, **no judges, no golden data**.

Three core pillars (throughput is a 4th but needs load/stress testing → out of scope here):
1. **Latency** — how long users wait.
2. **Cost** — how much per query.
3. **Reliability** — how often the pipeline succeeds vs. errors/timeouts.

### 9.1 Why measure ops OFFLINE (pre-deploy)?
Absolute values vary between local and prod, but the **differential** (direction +
comparative change between runs) is accurate and actionable. The danger of blind deploy:

> Baseline (K=5): Correctness 91%, Faithfulness 94%, AnsRel 93%, avg latency 2.3s,
> P95 4.8s, cost 72p.
> You add a **reranker**, use **K=10**, and upgrade to a **larger LLM** — quality
> improves (95/96/95) BUT latency spikes to 4.1s / P95 6.2s and cost → ₹1.08/query.
> **Without offline ops evals you'd ship this, then get user complaints + bill shock.**

### 9.2 Latency — best practices
- **Prefer distributions over averages:** track **P50/P95/P99** (tail latency), not just mean.
- **Component breakdown:** time embedding, retrieval, re-ranking, generation separately (find the bottleneck).
- **TTFT (time-to-first-token):** essential for streaming apps.
- **Watch cold starts:** skip the first 1–2 warm-up runs (model/db/container init).
- **Correlate with output length & context size** (latency scales with them).
- **Latency ≠ throughput:** latency is per-query; throughput is concurrent queries/window.
- **Repeat runs:** external LLM APIs are noisy — run each query multiple times (e.g. 5×) and average.
- **Track failures separately:** timed-out queries shouldn't skew the average.
- **Set SLO budgets** (e.g. P95 < 3s).
- **Representative/segmented workloads:** a balanced mix of simple/medium/complex questions.

**Latency experiment** (`eval_latency.py`): 5 questions × 5 runs (25 calls), 2 warm-ups dropped.
- SLO: P95 e2e < 3.0s, P95 TTFT < 1.2s.
- Result: e2e mean 3.6s / med 3.8s / **P95 5.3s / P99 5.3s**; retrieval ~700ms, generation **2.9s** (the clear bottleneck, ~4× retrieval); TTFT mean 1.6s P95 2.0s. **FAILED both SLOs.**
- Reducers: faster generator model / model router / shorter answers; smaller K or contextual compression; caching (embeddings, contexts, reranker outputs, system prompts); co-locate vector-DB/reranker/LLM regions.

### 9.3 Cost & tokens
- **Tokens dominate cost**; **output tokens ~4× the price of input tokens**.
- Other sources: paid LLM APIs, commercial vector DBs (Pinecone), commercial reranking APIs (Cohere), embeddings, hosting.
- Best practices: track **cost per query** (not just periodic total), break down input vs output cost, track distributions (expensive long-tail), segment by question complexity, set a cost budget.

**Cost experiment** (`eval_cost.py`): 4 questions × 3 runs (12 calls), GPT-4o-mini pricing, INR @95.
- Avg input tokens **1,700** (1,109 cached), avg output **209**.
- **Avg cost ~₹0.02/query** (2 paise) — stable & bounded, **PASSED**.
- Note: RAG spends more on input tokens (context injection) than output (opposite of coding agents).
- Projection: 2,000 q/day → ₹57/day, ~₹1,700/month.
- Reducers: smaller context (chunks/K, compression), shorter system prompt, concise answers, prompt caching, cheaper/self-hosted open model.

### 9.4 Reliability
Core metrics: **Success Rate, Error Rate (1−Success), Timeout Rate, Retry Rate**.
- **Categorize failures by source** (LLM API down, retriever, reranker, rate limit, formatting) — not one generic error tag.
- Test under **heavy concurrent load**, large sample (1000+ queries) to expose rare edge failures.

**Reliability experiment** (`eval_reliability.py`): 4 questions × 5 runs (20 hits) → Success 100%, Error 0%, Retry 0%. (Ideal locally; real failures appear under production load/rate-limits.)

> With Component + Pipeline + Application (Quality/Safety/Ops) done, the **RAG Eval
> Suite is complete.** Next: regression testing + CI/CD, then online evals + agents.

---

## 10. Regression Testing & CI/CD Gates

- **Regression testing =** run the *entire* eval suite together before each release. When going **V1 → V2**, it proves the app is objectively better and hasn't degraded — especially when a prompt fix for one metric breaks another.
- **Project structure:**
  ```
  rag_eval_project/
  ├── data/         # source (e.g. transcripts)
  ├── src/          # retriever.py, generator.py, rag_pipeline.py, fastAPI, Streamlit UI
  ├── evals/        # eval_retriever.py, eval_generator.py, ...  (the EVAL SUITE)
  └── run_evals.py  # triggers all tests → produces one clear evaluation report
  ```
- **Experiment tracking** (MLflow / Confident AI / Weights & Biases) logs params
  (chunk size, overlap, temperature) alongside metrics → visualize improvement over runs.
- **CI/CD (GitHub Actions):** run the suite on every code push; if any metric drops
  below the **current baseline threshold**, the CI pipeline **halts and blocks deployment**.

---

## 11. Online Evaluation & Observability

Evaluation continues **after deployment**:
- **Tracking/observability tools:** LangSmith, Langfuse, or Confident AI capture real-time latency, cost, tokens, and **user feedback (thumbs up/down)**.
- **Production metrics:** periodically compute online faithfulness, answer relevance, correctness.
- **Drift detection:** monitor a rolling window (e.g. 24h); alert if faithfulness starts dropping.
- **Self-improving loop:** extract edge-case chats where the model misbehaves →
  add them back to the **offline golden datasets** → continually improve future eval runs.

---

## 12. The Interview Answer (condensed cheat-sheet)

> "I don't evaluate a RAG app with one metric — I build an **evaluation suite** tested
> at **three levels**. **Component-level:** I evaluate the retriever (contextual recall
> and precision) and the generator (faithfulness, answer relevance, citation accuracy)
> in isolation against golden sets. **Pipeline-level:** I run the **RAG Triad** — context
> relevance, faithfulness, answer relevance — over the integrated pipeline. **Application-
> level:** quality (correctness, completeness, style), safety (toxicity, PII, jailbreak),
> and ops (latency, cost, reliability). Then I wrap it all in **regression testing inside
> CI/CD** so any update that drops below the baseline blocks deployment, and I keep
> **online evals + drift monitoring** running in production, feeding edge-cases back into
> a self-improving golden set."

---

## 13. Cheat-sheet of Metrics (reference quick-glance)

| Level | Metric | Reference-based? | Judge type | Formula essence |
|-------|--------|------------------|-----------|-----------------|
| Component | Contextual Recall | ✅ reference | LLM (atomic claims) | claims-found / claims-in-ideal |
| Component | Contextual Precision | ✅ reference | LLM (rank-aware) | avg precision-at-K |
| Component | Faithfulness | ✅ reference (context) | LLM | grounded-claims / total-claims |
| Component | Answer Relevance | ⛔ reference-free | LLM | relevant-claims / total-claims |
| Component | Citation Accuracy | ✅ reference | LLM | cited-source correct? |
| Pipeline | Context Relevance | ⛔ reference-free | LLM | useful-claims / claims-in-context |
| Pipeline | (the RAG Triad repeats the above) | — | LLM | the 3 pairwise links |
| Application | Correctness | ✅ reference | **G-Eval** | rubric + weighted score |
| Application | Completeness | ✅ reference | **G-Eval** | covers all sub-parts? |
| Application | Style | ⛔ ref-free (brand) | **G-Eval** | brand voice match |
| Application | Toxicity | ⛔ ref-free | `ToxicityMetric` | toxic-opinion ratio |
| Application | PII Leakage | ⛔ ref-free | `PII LeakageMetric` | detector flags |
| Application | Scope/Jailbreak | ⛔ ref-free | custom G-Eval | stays in role? |
| Application | Latency | — (telemetry) | none | P50/P95/P99, TTFT, breakdown |
| Application | Cost/Token | — (telemetry) | none | cost-per-query, in/out split |
| Application | Reliability | — (telemetry) | none | success/error/timeout/retry |

---

## 14. Session roadmap (for self-study pacing)

| Session | Topic |
|---------|-------|
| 1 | Component-level evals — build & test retriever + generator (DeepEval) |
| 2 | RAG pipeline integration + run the RAG Triad |
| 3 | Application-level evals (Quality → Safety → Ops) |
| 4 | Regression testing + CI/CD, then online evaluation setup |

---

## 15. Study Notes / Personal Takeaways

- **Isolation first.** Build-test-optimize each component *before* wiring the pipeline
  — mirrors unit-testing discipline and makes blame attribution clean.
- **Count-based → judgment-based cliff.** Recall/prec/faithfulness/answer-relevance
  count atomic claims; correctness/completeness/style need whole-answer judges (G-Eval).
- **G-Eval's two wins** are a *rigid step rubric* (deterministic constitution) + *log-prob
  weighted scoring* (no discrete integer jumps).
- **Always the strongest model as judge**, and keep the **same judge** for comparability.
- **Retriever tuning axes:** chunk size (recall vs intra-chunk noise), reranker
  (precision/rank), embedding model (recall), K (precision; don't lower blindly).
- **Ops evals offline** catch regressions that quality metrics silently hide (reranker
  + bigger K + bigger model all raised quality but wrecked latency/cost).
- **Safety is scoped, not generic:** enumerate the real attack surface (here: toxicity,
  leakage, scope) and build a written **safety policy** / constitution to drive eval + guardrails.
- **System prompts are living artifacts** — grown by eval-driven iteration into a
  robust constitution; protect them and regress-test every change.
- **Close the loop:** offline goldens ← production edge-cases (self-improving), online
  drift watched in a rolling window.