---
name: customer-support-rag-chatbot
description: Build an end-to-end RAG customer support chatbot on Databricks using medallion architecture (Bronze → Silver → Gold), AI Search (Vector Search) for semantic retrieval, and Foundation Model LLMs for response generation. Use when the user asks to create a customer support chatbot, RAG-based conversational agent, support knowledge base search, or any context-aware 24/7 customer query resolution system. Also load when the user mentions AI Search Delta Sync indexes, managed embeddings with databricks-gte-large-en, or multi-turn RAG with conversation history.
---

# Customer Support RAG Chatbot — Medallion Architecture

## Overview
This skill covers the complete implementation of a **24/7 context-aware customer support chatbot** using Retrieval-Augmented Generation (RAG) on Databricks. It uses the medallion architecture (Bronze → Silver → Gold) to process support knowledge base content (FAQs, resolved tickets, KB articles) and powers a conversational agent with AI Search semantic retrieval and LLM generation.

Validated against the production notebook "Customer Support RAG Chatbot" (ID: 384535899474742), running on Serverless compute, Unity Catalog `main.customer_support`, October 2026 API versions.

## When to Load

Load this skill when the user wants to:
- Build a customer support chatbot or conversational agent
- Create a RAG system over support knowledge bases (FAQs, tickets, articles)
- Implement 24/7 context-aware query resolution
- Use AI Search Delta Sync with managed embeddings for support content
- Build multi-turn conversational RAG with conversation history

## Architecture

```
Source Data (FAQs / Resolved Tickets / KB Articles)
        │
        ▼
┌─────────────────────┐
│  BRONZE LAYER       │  Raw ingest — verbatim, no parsing
│  bronze_support     │  Columns: record_type, question_or_title, answer_or_body,
│                     │          category, product, severity, ticket_id
└────────┬────────────┘
         │ parse + structure
         ▼
┌─────────────────────┐
│  SILVER LAYER       │  Structured + typed + quality filtered
│  silver_support     │  Columns: content_hash (PK), title, body, source_label,
│                     │          category, product, severity (normalized)
└────────┬────────────┘
         │ enrich for embedding
         ▼
┌─────────────────────┐
│  GOLD LAYER         │  Embed-ready + governed
│  gold_support       │  Columns: doc_id (PK), content (embed), record_type,
│                     │          source_label, severity_level, category, product
│                     │  CDF=true, PRIMARY KEY, NOT NULL
└────────┬────────────┘
         │ Delta Sync
         ▼
┌─────────────────────┐
│  VECTOR INDEX       │  AI Search Delta Sync (TRIGGERED or CONTINUOUS)
│  gold_support_      │  Managed embeddings via databricks-gte-large-en
│  vs_index           │  ANN + HYBRID similarity search
└────────┬────────────┘
         │ similarity_search()
         ▼
┌─────────────────────┐
│  RAG PIPELINE       │  Retrieve → Augment → Generate
│                     │  LLM: databricks-meta-llama-3-3-70b-instruct
│                     │  Multi-turn conversation support
└─────────────────────┘
```

## Critical API Notes (October 2026)

| Rule | Detail |
|---|---|
| Use `databricks-ai-search==0.78` | `databricks-vectorsearch` is DEPRECATED |
| Never reinstall `databricks-sdk` | Pre-installed on Serverless — reinstalling causes version conflicts |
| Always `restartPython()` after pip install | Required for the new package to load in the session |
| `get_index(index_name=...)` | No `endpoint_name` parameter needed |
| `similarity_search(filters=dict)` | NOT `filters_json=json.dumps(dict)` |
| `similarity_search(disable_notice=True)` | Suppress development auth notice |
| `w.serving_endpoints.query(messages=[ChatMessage(...)])` | Must use SDK objects, not plain dicts |
| PRIMARY KEY requires NOT NULL | `ALTER COLUMN c SET NOT NULL` before `ADD CONSTRAINT` |

## Quick-Reference: Global Configuration

```python
CATALOG_NAME      = "main"
SCHEMA_NAME       = "customer_support"
BRONZE_TABLE      = f"{CATALOG_NAME}.{SCHEMA_NAME}.bronze_support"
SILVER_TABLE      = f"{CATALOG_NAME}.{SCHEMA_NAME}.silver_support"
GOLD_TABLE        = f"{CATALOG_NAME}.{SCHEMA_NAME}.gold_support"
VS_ENDPOINT_NAME  = "ai-vector-compute"
VS_INDEX_NAME     = f"{CATALOG_NAME}.{SCHEMA_NAME}.gold_support_vs_index"
EMBEDDING_MODEL   = "databricks-gte-large-en"
LLM_ENDPOINT_NAME = "databricks-meta-llama-3-3-70b-instruct"
```

## Key Implementation Patterns

### 1. Setup & Client Initialization

```python
%pip install --upgrade databricks-ai-search
dbutils.library.restartPython()

from databricks.sdk import WorkspaceClient
from databricks.ai_search.client import AISearchClient

w = WorkspaceClient()
ai_search = AISearchClient(disable_notice=True)
```

### 2. Bronze Layer — Raw Ingestion

Raw customer support data (FAQs, resolved tickets, KB articles) ingested as-is.

- Schema: `record_type`, `question_or_title`, `answer_or_body`, `category`, `product`, `severity`, `ticket_id`
- Use `spark.createDataFrame()` for synthetic data or `spark.read.text()` / Auto Loader for files
- Add `source_file` and `ingested_at` columns for lineage
- Idempotent write: check `spark.catalog.tableExists()` before `saveAsTable()`

### 3. Silver Layer — Structured Parsing

- Trim and standardize text fields (`title`, `body`, `category`, `product`, `severity`)
- Add `source_label` (human-readable: "FAQ Entry" / "Resolved Ticket" / "Knowledge Base Article")
- Add `content_hash` (SHA-256) for deduplication
- Quality gate: drop records with null/empty title or body
- Deduplicate by `content_hash`

### 4. Gold Layer — Embed-Ready Enrichment

- **PRIMARY KEY**: `doc_id` (SHA-256 hash of content)
- **Content column**: Rich natural-language text combining title + type + category + product + body for embedding
- **Metadata filters**: `record_type`, `category`, `product`, `severity_level` (CRITICAL/HIGH/MEDIUM/LOW/INFO)
- **CDF**: `ALTER TABLE ... SET TBLPROPERTIES (delta.enableChangeDataFeed = true)`
- **PK setup**: `ALTER COLUMN doc_id SET NOT NULL` then `ADD CONSTRAINT gold_support_pk PRIMARY KEY (doc_id)`

Content column template (what gets embedded):
```
Title: <title>

Type: <source_label>

Category: <category>

Product: <product>

Content:
<body>
```

### 5. Vector Search — AI Search Delta Sync Index

```python
def get_or_create_vs_endpoint(client, endpoint_name, endpoint_type="STANDARD"):
    """Get existing endpoint or create and poll until ONLINE."""
    try:
        ep = client.get_endpoint(name=endpoint_name)
        return ep
    except Exception:
        client.create_endpoint(name=endpoint_name, endpoint_type=endpoint_type)
        # Poll until ONLINE...

def get_or_create_delta_sync_index(
    client, endpoint_name, index_name, source_table,
    primary_key, embedding_col, embedding_model, pipeline_type="TRIGGERED"
):
    """Get existing index or create Delta Sync with managed embeddings."""
    try:
        idx = client.get_index(index_name=index_name)
        return idx
    except Exception:
        client.create_delta_sync_index(
            endpoint_name=endpoint_name,
            index_name=index_name,
            source_table_name=source_table,
            pipeline_type=pipeline_type,
            primary_key=primary_key,
            embedding_source_column=embedding_col,
            embedding_model_endpoint_name=embedding_model,
            columns_to_sync=[
                "doc_id", "content", "record_type", "source_label",
                "title", "body", "category", "product",
                "severity", "severity_level", "ticket_id"
            ]
        )
        # Poll until ready...
```

### 6. Retrieval — Similarity Search

```python
def similarity_search(
    query: str, top_k: int = 5, filter_dict: dict = None,
    query_type: str = "ANN"  # or "HYBRID"
) -> list:
    idx = ai_search.get_index(index_name=VS_INDEX_NAME)
    kwargs = dict(
        query_text=query,
        columns=["doc_id", "content", "record_type", "source_label",
                 "title", "body", "category", "product",
                 "severity_level", "ticket_id"],
        num_results=top_k,
        query_type=query_type
    )
    if filter_dict:
        kwargs["filters"] = filter_dict  # dict directly
    kwargs["disable_notice"] = True
    raw = idx.similarity_search(**kwargs)
    col_names = [c["name"] for c in raw.get("manifest", {}).get("columns", [])]
    hits = [dict(zip(col_names, row))
            for row in raw.get("result", {}).get("data_array", [])]
    return hits
```

**Query Types:**
| Type | Mechanism | Best For |
|---|---|---|
| `ANN` | Approximate Nearest Neighbor (vector only) | Semantic queries: "password reset", "account security" |
| `HYBRID` | ANN + BM25 keyword matching | Queries with exact terms: error codes, ticket IDs, product names |

### 7. RAG Pipeline — Retrieve → Augment → Generate

```python
from databricks.sdk.service.serving import ChatMessage, ChatMessageRole

def rag_query(
    user_question: str, top_k: int = 5, filter_dict: dict = None,
    query_type: str = "ANN", conversation_history: list = None
) -> tuple:
    # 1. RETRIEVE
    docs = similarity_search(query=user_question, top_k=top_k,
                             filter_dict=filter_dict, query_type=query_type)
    # 2. AUGMENT — build context block
    context_lines = []
    for i, d in enumerate(docs, 1):
        context_lines.append(
            f"[{i}] Type: {d.get('source_label','?')} | Category: {d.get('category','?')} | "
            f"Product: {d.get('product','?')} | Severity: {d.get('severity_level','N/A')}\n"
            f"    Title: {d.get('title','?')}\n"
            f"    Content: {str(d.get('body','?'))[:500]}"
        )
    context = "\n\n".join(context_lines)
    # 3. GENERATE — call LLM with ChatMessage SDK objects
    system_prompt = (
        "You are an AI customer support agent..."
    )
    user_prompt = (
        f"SUPPORT KNOWLEDGE BASE CONTEXT:\n{context}\n\n"
        f"CUSTOMER QUESTION: {user_question}\n\n"
        "Provide a helpful, accurate response..."
    )
    messages = [ChatMessage(role=ChatMessageRole.SYSTEM, content=system_prompt)]
    if conversation_history:
        for role, content in conversation_history[-4:]:
            msg_role = ChatMessageRole.USER if role == "user" else ChatMessageRole.ASSISTANT
            messages.append(ChatMessage(role=msg_role, content=content))
    messages.append(ChatMessage(role=ChatMessageRole.USER, content=user_prompt))
    response = w.serving_endpoints.query(
        name=LLM_ENDPOINT_NAME, messages=messages,
        max_tokens=1024, temperature=0.3
    )
    return response.choices[0].message.content, docs
```

## System Prompt Guidelines for Customer Support

| Element | Recommendation |
|---|---|
| Persona | "AI customer support agent for a SaaS platform" |
| Grounding | "Ground every answer in the provided support knowledge base entries" |
| Honesty | "If the retrieved context does not contain the answer, say so honestly — do not hallucinate" |
| Formatting | "Provide step-by-step instructions when applicable" |
| Source attribution | "Reference the source type (FAQ, Resolved Ticket, or Knowledge Base Article)" |
| Escalation | "For critical issues, advise creating a support ticket" |
| Tone | "Be concise, friendly, and professional. Use bullet points for clarity" |
| Context | "If conversation history is provided, maintain context from previous exchanges" |

## LLM Parameters

| Parameter | Value | Rationale |
|---|---|---|
| `max_tokens` | 1024 | Sufficient for detailed support responses |
| `temperature` | 0.3 | Low for accuracy but slightly friendly |

## Metadata Filters for Similarity Search

| Filter | Values | Use Case |
|---|---|---|
| `category` | Account, Billing, Platform, API, Data, Integrations, Analytics, Mobile | Scope to support domain |
| `record_type` | faq, ticket, article | Scope to content type |
| `severity_level` | CRITICAL, HIGH, MEDIUM, LOW, INFO | Scope by urgency |
| `product` | Auth, Payments, API, Webhooks, etc. | Scope by product area |

## Score Interpretation

| Score Range | Meaning |
|---|---|
| 0.90–1.00 | Very high relevance (HYBRID exact keyword match) |
| 0.65–0.89 | Good semantic match |
| 0.45–0.64 | Moderate relevance |
| <0.45 | Weak match — consider broadening the query or removing filters |

## Common Errors & Fixes

| Error | Fix |
|---|---|
| `ModuleNotFoundError: databricks.vectorsearch` | Use `from databricks.ai_search.client import AISearchClient` |
| `TypeError: messages must be list of ChatMessage` | Use `ChatMessage(role=ChatMessageRole.SYSTEM, content=...)` |
| `Column X not found in index` | Add column to `columns_to_sync` when creating index |
| `PRIMARY KEY requires NOT NULL` | Run `ALTER COLUMN doc_id SET NOT NULL` before `ADD CONSTRAINT` |
| `CDF not enabled` | Run `ALTER TABLE SET TBLPROPERTIES (delta.enableChangeDataFeed = true)` |
| `filters_json is not a valid parameter` | Use `filters=dict` not `filters_json=json.dumps(dict)` |
| Quota exceeded: only 1 endpoint | Reuse existing endpoint; don't create a new one |
| Empty retrieved docs | Check index readiness; loosen or remove `filter_dict` |

## End-to-End Flow Checklist

- [ ] `databricks-ai-search` installed and Python restarted
- [ ] `WorkspaceClient` and `AISearchClient` initialized
- [ ] Bronze, Silver, Gold tables created and validated
- [ ] Gold table has CDF=true and PRIMARY KEY on `doc_id`
- [ ] Vector Search endpoint is ONLINE
- [ ] Delta Sync index exists with `columns_to_sync` including all filter/display columns
- [ ] `similarity_search()` function defined and tested
- [ ] `rag_query()` function defined and tested
- [ ] Multi-turn conversation history support validated

## Reference Notebook

- Notebook ID: 384535899474742
- Name: "Customer Support RAG Chatbot"
- Path: `/Users/vinodsavanapelly21@gmail.com/Drafts/Customer Support RAG Chatbot`
- Source: Created by user request, in conversation.
