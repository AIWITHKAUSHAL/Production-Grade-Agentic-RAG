# Code flow, storage and RAG concepts

Based on the repository implementation. Examples below are illustrative, not measurements from a live run. Paths are relative to the repository root; database names are defaults, not verified live deployment settings.

## 1. Which file calls which?

| Entry / caller | Calls | Purpose / example |
|---|---|---|
| `backend/app/main.py` | Registers `api/routes/ingest.py`, `query.py`, `health.py`; startup calls `agents/graph.py:get_graph()` | Serves frontend and API; prepares the graph. |
| `frontend/index.html` → upload API | `api/routes/ingest.py:upload_and_ingest()` | Save `policy.pdf` to `uploads/policy.pdf`, then ingest it. |
| `ingest.py` | `ingestion/pipeline.py:get_pipeline().ingest_file()` | Coordinate extraction, splitting and indexing. |
| `pipeline.py` | `document_loader.py:DocumentLoader.load()` | PDF: Docling, with PyMuPDF fallback; DOCX: python-docx; text/Markdown: text reader. |
| `pipeline.py` | `chunking.py:get_chunker()` → `chunk()` | Create shorter passages; default 1,000 characters with 200 overlap. Add file hash, document ID and source metadata. |
| `pipeline.py` | `retrieval/vector_store.py:get_vector_store()` → `aadd_documents()` | Store chunks and vectors in Qdrant. |
| `vector_store.py` | `core/embeddings.py:get_embeddings()` | EURI embeddings represent document/query meaning numerically. |
| `pipeline.py` | `BM25Index.add_documents()` | Build an ingestion-owned keyword index in memory. See implementation note below. |
| Browser → query API | `api/routes/query.py:query()` or `query_stream()` | Initialize `AgentState`; set `thread_id` to the session ID; invoke graph or stream graph events. |
| `query.py` | `agents/graph.py:get_graph()` | Obtain graph compiled with a checkpointer when available. |
| `graph.py` | Functions in `agents/nodes.py` | Defines node order, conditional branches and retry limits. |
| `nodes.py:understand_query()` | `core/llm.py:get_grader_llm()` | Analyze intent and key concepts. |
| `nodes.py:route_query()` | `rag/self_rag.py:should_retrieve()` when appropriate | Choose document search, web, calculator or general generation. |
| `nodes.py:retrieve()` | `retrieval/hybrid_search.py:search()` | Vector search + keyword search, then fusion and reranking. |
| `hybrid_search.py` | `vector_store.py:similarity_search_with_score()`, `bm25_retriever.py:get_bm25_index().search()`, `rrf.py:reciprocal_rank_fusion()`, `reranker.py:CrossEncoderReranker.rerank()` | Retrieve candidates, combine ranks, prioritize the best passages. |
| `nodes.py:web_search_node()` | `agents/tools.py:web_search()` | Tavily if configured; DuckDuckGo fallback. |
| `nodes.py:calculator_node()` | `agents/tools.py:calculator.invoke()` | Evaluate an expression such as `sqrt(144)`; then go directly to final answer. Natural-language math is not translated into an expression here. |
| `nodes.py:grade_documents()` | `rag/crag.py:evaluate()` and `rag/self_rag.py:filter_relevant_docs()` | Keep useful evidence; add web results if CRAG requests correction. |
| `nodes.py:generate()` | `core/llm.py:get_llm()` | Build context and citations; general route without documents uses a direct prompt. |
| `nodes.py:evaluate_generation()` | `rag/grader.py` and `rag/self_rag.py` | Check support, answer relevance and utility. |
| `nodes.py:rewrite_query()` | `core/llm.py:get_grader_llm()` | Reformulate the search; graph sends it back to retrieval, or web for a web route. |
| `nodes.py:final_answer()` | API response | Return answer, sources, scores and steps; calculator uses its result. |

## 2. Definitions with one worked example

Evidence: **policy.pdf, page 2: “Customers may request a refund within 30 days of purchase.”** Question: **“What is the refund period?”**

| Concept | Definition | Example |
|---|---|---|
| RAG | Retrieval-augmented generation: find evidence and give it to the model before answering. | Retrieve the policy passage, then answer “30 days.” |
| Agentic RAG | A workflow chooses tools and corrective steps based on the question and intermediate results. | Search documents for policy questions; use web search for current news. |
| Chunk | A shorter passage created from a document. | One refund paragraph with filename and page metadata. |
| Overlap | Repeated text between neighboring chunks to preserve boundary context. | The last 200 characters may appear at the start of the next default chunk. |
| Metadata | Information attached to a passage. | `source=policy.pdf`, `page=2`, `file_hash=...`. |
| Embedding | A numeric representation of text used to compare meaning. | “Money back” can match “refund” without exact wording. |
| Semantic search | Find passages whose vector representations are similar to the query. | “How long do I have to get my money back?” finds the refund paragraph. |
| BM25 | Keyword ranking based on term matches, rarity and document length. | A passage containing “refund period” ranks for those terms. |
| Hybrid search | Combine semantic and keyword retrieval. | Recover both paraphrases and exact policy terms. |
| RRF | Reciprocal Rank Fusion combines ranked lists using rank positions, with weights here. | A passage ranked highly by both retrievers rises in the combined list. |
| Cross-encoder reranking | Score query and candidate passage together to reorder retrieved results. | A refund deadline paragraph outranks a generic refund overview. |
| CRAG | Corrective RAG: assess retrieval quality and choose whether to use documents or supplement/replace them with web results. | Of 5 retrieved passages, 3 relevant means use; 1 relevant means hybrid; 0 relevant means web. |
| Self-RAG | Model-driven retrieval and reflection checks used in this project. | Decide whether to retrieve, then assess passage relevance, support and usefulness. |
| IsRET | Teaching shorthand for “should we retrieve?”; this code implements `should_retrieve()`. | A policy-specific question needs evidence; a creative writing request can use the general route. |
| IsREL | Is a passage relevant to answering this question? | Refund deadline passage: relevant. Shipping-only passage: irrelevant. |
| Grounding | Are the factual claims supported by the evidence supplied to the checker? | “30 days” is grounded; “60 days” is unsupported by the example passage. |
| Hallucination | A claim that the supplied evidence does not support in this grading setup. | “Refunds are allowed for 60 days.” This checker does not establish truth across all possible external sources. |
| IsSUP | Is the generated response supported by the retrieved passages? | “30 days from purchase” is supported; “30 days plus a free replacement” adds an unsupported claim. |
| Answer relevance | Does the response answer the original question? | “30 days” answers the period question. “Refunds go to the original payment method” misses the requested period, even if another passage supports it. |
| IsUSE | LLM rating of how useful the answer is, from 1 to 5. | “You have 30 days from purchase [policy.pdf, p.2]” is more useful than “Read the policy.” Ratings are illustrative. |
| Query rewrite | Reformulate the search after weak results or a poor answer. | “Refund period” → “customer refund deadline days after purchase.” |
| Checkpoint | A saved graph execution snapshot, with state and execution metadata. | A session snapshot includes the query, retrieved passages, scores, retry count and steps. |
| Trace | A record of execution for inspection, distinct from document storage or checkpointing. | LangSmith shows which nodes and model calls ran. |

## 3. Exact CRAG and quality decisions

`rag/crag.py` counts passages the document grader marks relevant:

- Relevant fraction **≥ 0.6**: `use`.
- Fraction **≥ 0.2 and < 0.6**: `hybrid` (keep relevant evidence and request web supplementation).
- Fraction **< 0.2**, or no documents: `web_search`.

For kept passages, CRAG also applies a relevance score threshold of 0.5. `grade_documents()` runs Self-RAG relevance filtering, but retains the CRAG-filtered passages if Self-RAG rejects all of them. Web evidence added inside this node is appended without another grading pass.

`nodes.py:evaluate_generation()` declares quality good exactly when:

```python
grounded and relevant and utility >= 3
```

Here `grounded` comes from `HallucinationGrader`, `relevant` from `AnswerRelevanceGrader`, and `utility` from `is_useful()`. The separate `is_supported()` score is saved as `grounding_score`; its boolean is discarded. `self_rag_issup` is assigned the hallucination grader's `grounded` boolean. `self_rag_isrel` exists in state but is not updated by `grade_documents()`.

Despite its name, `hallucination_score` is a **support/grounding score** from the grounding prompt: higher means more support, not more hallucination. All of these scores are LLM judgments, not calibrated probabilities or guarantees.

After grading: no graded documents and retries remaining → rewrite. After evaluation: good quality **or retry limit reached** → final answer; otherwise rewrite. Default `MAX_RETRIES=3`, so the graph can make the initial attempt plus three rewrites. Calculator skips generation/evaluation. General generation still goes through evaluation; with no documents the grounding grader returns false, so it can enter retrieval retries.

## 4. Where checkpoints are stored

Code path:

```text
api/routes/query.py
  session_id → config["configurable"]["thread_id"]
agents/graph.py:get_graph()
  checkpoints/mongodb_saver.py:get_checkpoint_saver()
    MongoDBCheckpointSaver(uri, db_name, collection_name)
  compile_graph(checkpointer=saver)
  graph.compile(checkpointer=checkpointer)
```

| Setting | Default / meaning |
|---|---|
| `MONGODB_URI` | Required connection URI; selects the MongoDB deployment. Credentials are not included here. |
| `MONGODB_DB` | `agentic_rag_db` |
| `MONGODB_CHECKPOINT_COLLECTION` | `checkpoints` |
| Logical checkpoint location | `agentic_rag_db.checkpoints` under default settings. The saver library also manages supporting writes; this app explicitly names only the checkpoint collection. |
| Session key | `thread_id`, supplied from the request's `session_id`, or a generated UUID. |
| History endpoint | `GET /api/v1/query/history/{session_id}` reads MongoDB snapshots through `get_thread_history()`. |
| Fallback | MongoDB unavailable → in-memory `MemorySaver`; if unavailable → compile without a checkpointer. A checkpoint compilation error also triggers compilation without one. |

Example request body:

```json
{"query": "What is the refund period?", "session_id": "class-demo-01"}
```

Illustrative inspection in mongosh, after connecting to the configured deployment:

```javascript
use agentic_rag_db
db.checkpoints.find({thread_id: "class-demo-01"})
```

This is database storage, not a checkpoint file in the repository. In this project's Docker Compose deployment, the named volume `mongo_data` is mounted at `/data/db` in the MongoDB container. Compose may prefix the volume name with the project name. The host storage location belongs to Docker; the code does not specify a normal host folder for MongoDB. Atlas storage lives in the configured remote deployment instead.

Checkpoints preserve graph execution state. They do **not** automatically turn previous answers into conversational context: this query API initializes fresh fields, and the generation prompts do not explicitly inject previous chat turns. The history endpoint reads MongoDB directly, so memory-fallback history is not exposed by that endpoint.

## 5. Other storage locations and implementation details

| Data | Location | Example |
|---|---|---|
| Original uploads | `uploads/` relative to the server working directory | `uploads/policy.pdf`; Docker maps host `./uploads` to `/app/uploads`. |
| Document vectors + content + metadata | Qdrant configured by `QDRANT_URL`; default collection `agentic_rag_docs` | Refund paragraph vector, text, source and page. |
| BM25 documents | Python process memory | Restart loses the index. Ingestion currently constructs a private `BM25Index`; search uses the separate global `get_bm25_index()` instance, so uploaded chunks do not populate the retrieval singleton. |
| Graph state | MongoDB checkpoints, with memory/no-checkpointer fallback | Query, documents, answer and intermediate control values. |
| Optional tracing | Configured LangSmith project | Node executions and LLM calls. |

The vector store currently creates 1,536-dimensional collections. The local embedding fallback uses a model with a different output size, so that fallback is not automatically compatible with that collection. This guide documents the current code; it does not modify application behavior.

## 6. Suggested classroom walkthrough

1. Upload `policy.pdf`: follow loader → chunks → embeddings → Qdrant.
2. Ask the refund-period question: follow API → graph → router → retrieval.
3. Explain keyword versus semantic matching, then RRF and reranking.
4. Use the five-passage CRAG examples to explain corrective retrieval.
5. Compare a supported answer, an unsupported answer and an answer that misses the question.
6. Show the good-quality condition and bounded rewrite loop.
7. Use `class-demo-01` to explain execution state and MongoDB history.

Image generation: built-in image_gen. The companion prompt files record the generation specifications.
