# 🔎 Semantic Search

Semantic search allows Jimi to find information by meaning rather than exact keyword matching.
Instead of looking only for matching words, semantic search compares the conceptual similarity
between a user query and source text. This makes it useful when the same idea appears under
different terms, abbreviations, sentence structures, or technical phrasing.

In Jimi, semantic search is supported by embedding workflows in `gemini.py`, user interaction and
routing in `app.py`, configuration values in `config.py`, and exception logging through `boogr.py`.

 

## 🧭 Overview

Traditional keyword search answers the question:

```text
Where does this exact word or phrase appear?
```

Semantic search answers the question:

```text
Which passages are most related to this idea?
```

This difference matters when users are searching through technical documents, resumes, requirements,
policy material, source-code comments, logs, or generated documentation.

Example keyword query:

```text
Find every reference to internal controls.
```

Example semantic query:

```text
Find sections related to audit remediation, risk mitigation, corrective actions,
financial oversight, control testing, and accountability, even when the exact
phrase "internal controls" does not appear.
```

 

## 🧱 Architecture

Semantic search uses these parts of the Jimi application:

| Component           | Module      | Purpose                                                                    |
| ------------------- | ----------- | -------------------------------------------------------------------------- |
| Streamlit interface | `app.py`    | Captures the search query, source text, uploaded file, or selected corpus. |
| Embedding wrapper   | `gemini.py` | Generates vector representations of text through the `Embeddings` class.   |
| Storage workflow    | `gemini.py` | Supports file and object workflows through `Files` and `VectorStores`.     |
| Configuration       | `config.py` | Provides API keys, paths, defaults, and validation helpers.                |
| Logging             | `boogr.py`  | Records semantic-search failures and provider errors in SQLite.            |

Conceptual flow:

```text
User query
    │
    ▼
app.py validates search input
    │
    ▼
Text or document chunks are prepared
    │
    ▼
Embeddings.create(...) generates vectors
    │
    ▼
Similarity scores are computed
    │
    ▼
Most relevant passages are ranked
    │
    ▼
app.py displays results
```

The UI layer should not directly implement provider-specific embedding requests. That behavior
belongs in the `Embeddings` wrapper.

 

## 🧠 What Embeddings Do

An embedding is a numeric representation of text. Text with similar meaning should produce vectors
that are close to each other in vector space.

Example:

```text
"budget execution risk"
```

may be semantically close to:

```text
"funds control issue"
"obligation monitoring concern"
"financial management weakness"
"appropriation execution variance"
```

Even when exact words differ, the vector representation can help identify related content.

 

## 🔁 Semantic Search Workflow

A typical semantic-search workflow follows this sequence:

1. The user provides a query.
2. Jimi receives source text, uploaded content, or document chunks.
3. Jimi validates the required inputs.
4. The `Embeddings` wrapper generates vectors for the query and candidate text.
5. Similarity scores are calculated.
6. Results are sorted by relevance.
7. Jimi displays the highest-ranking matches.
8. The user refines the query or asks a follow-up question.

Conceptual diagram:

```text
Query Text
    │
    ▼
Query Embedding
    │
    ├──────────────┐
    │              │
    ▼              ▼
Candidate Text  Candidate Embeddings
    │              │
    └──────┬───────┘
           ▼
   Similarity Comparison
           │
           ▼
     Ranked Results
```

 

## 📄 Source Text Preparation

Semantic search performs best when source material is divided into useful chunks. Chunks should be
large enough to preserve context but small enough to produce precise matches.

Recommended chunking units:

| Source Type            | Recommended Chunk                              |
| ---------------------- | ---------------------------------------------- |
| Markdown documentation | Heading section or subsection.                 |
| Python source          | Function, class, or logical block.             |
| Policy document        | Paragraph, section, or requirement.            |
| Resume or announcement | Bullet, requirement, or qualification section. |
| CSV or tabular data    | Row summary or grouped record.                 |
| Logs                   | Error record or related event group.           |

Avoid overly large chunks because broad chunks can dilute relevance. Avoid tiny chunks because they
may lose context.

 

## 🧮 Similarity Scoring

Semantic search normally compares vectors using a similarity metric. The most common approach is
cosine similarity.

Conceptually:

```text
Higher score = stronger semantic relationship
Lower score  = weaker semantic relationship
```

Example result table:

| Rank | Score | Passage                                                                  |
| ---: | ----: | ------------------------------------------------------------------------ |
|    1 |  0.91 | The application logs exception metadata to SQLite for diagnostic review. |
|    2 |  0.84 | The logger captures module, cause, method, message, info, and traceback. |
|    3 |  0.78 | Runtime artifacts such as local databases should not be committed.       |

Scores are ranking aids, not absolute proof. Results should be reviewed for context and accuracy.

 

## 🧾 Embedding Workflow in Jimi

The `Embeddings` class in `gemini.py` provides the provider-facing embedding workflow.

Example conceptual use:

```python
from gemini import Embeddings

embeddings = Embeddings()
vector = embeddings.create(
	text="Find requirements related to audit remediation and internal controls.",
	model="gemini-embedding-001"
)
```

The wrapper is responsible for:

* Validating required text input.
* Setting the active embedding model.
* Creating the Gemini client.
* Calling the provider embedding endpoint.
* Returning a list of numeric vector values.
* Logging and raising provider failures when appropriate.

 

## 🔍 Search Use Cases

Semantic search is useful for several Jimi workflows.

### Documentation review

Use semantic search to find sections related to a topic across Markdown files.

Example:

```text
Find documentation sections related to configuration, environment variables,
logging, and runtime paths.
```

### Source-code review

Use semantic search to find related methods or comments across source files.

Example:

```text
Find functions related to file uploads, document processing, temporary paths,
and provider file APIs.
```

### Requirements extraction

Use semantic search to identify document sections related to requirements, even when the wording
varies.

Example:

```text
Find passages related to deadlines, deliverables, approvals, reporting,
responsible parties, and required documentation.
```

### Resume and announcement analysis

Use semantic search to compare experience against announcement language.

Example:

```text
Find experience bullets related to budget formulation, execution, audit,
internal controls, data analytics, and reporting.
```

### Exception-log review

Use semantic search to group related failures.

Example:

```text
Find exception records related to file uploads, missing API keys,
provider failures, and Streamlit rendering errors.
```
 

## ✍️ Prompting for Semantic Search

A good semantic-search prompt should define the concept being searched for and include related
terms.

Weak prompt:

```text
Search this.
```

Better prompt:

```text
Find sections conceptually related to exception handling, SQLite logging,
diagnostic traces, module metadata, workflow failures, and recoverable UI errors.
```

Strong prompt:

```text
Find the 10 passages most related to application diagnostics.

Include passages about:
- exception wrapping
- module/cause/method metadata
- SQLite logging
- tracebacks
- recoverable Streamlit failures
- provider API failures

Return:
Rank | Similarity Rationale | Passage Summary | Recommended Follow-Up
```

 

## 📊 Recommended Output Formats

Semantic search results are easier to review when returned in a structured format.

### Ranked table

```text
Return a Markdown table with:
Rank | Topic Match | Source | Reason | Suggested Action
```

### Review checklist

```text
Return:
- top matches
- near matches
- missing concepts
- ambiguous results
- recommended follow-up searches
```

### Source grouping

```text
Group results by:
- configuration
- provider wrappers
- Streamlit UI
- logging
- documentation
```

### Relevance explanation

```text
For each result, explain why it is relevant to the query in one sentence.
```

 

## 🧩 Query Expansion

Semantic search improves when the query includes related terms and synonyms.

Example base query:

```text
budget execution
```

Expanded query:

```text
budget execution, funds control, obligations, outlays, apportionment,
allotment, commitments, deobligations, spending plan variance, financial oversight
```

Example base query:

```text
logging
```

Expanded query:

```text
logging, exception handling, error records, SQLite diagnostics, traceback,
module metadata, failure capture, audit trail
```

Expanded queries help retrieve conceptually adjacent material.

 

## 🧪 Validation Checklist

When testing semantic search, verify:

```text
The query is not blank.
Source text or candidate chunks exist.
Chunks preserve enough context.
Embedding model is configured.
API key is available.
Embedding calls return vectors.
Similarity scoring produces ranked results.
Results are reviewable in the UI.
Exceptions are logged through boogr.py.
Recoverable UI errors do not crash the full application.
```

Run source checks:

```powershell
python -m py_compile .\app.py
python -m py_compile .\gemini.py
python -m py_compile .\config.py
python -m py_compile .\boogr.py
python -m compileall .
```

Run documentation checks:

```powershell
mkdocs build
```

 

## 🧾 Exception Handling

Semantic-search workflows can fail because of missing text, missing embeddings, unsupported models,
credential errors, provider failures, file parsing issues, or similarity-scoring problems.

Provider-level failures should log and raise:

```python
except Exception as e:
	ex = Error( e )
	ex.module = 'gemini'
	ex.cause = 'Embeddings'
	ex.method = 'create( self, text, model )'
	Logger( ).write( ex )
	raise ex
```

UI-level recoverable failures should log and preserve fallback behavior:

```python
except Exception as e:
	ex = Error( e )
	ex.module = 'app'
	ex.cause = 'SemanticSearch'
	ex.method = 'render_semantic_search_panel( )'
	Logger( ).write( ex )
	st.warning( str( ex ) )
	return None
```

The required rule is:

```text
One Error object.
One Logger write.
Same object logged and raised or handled.
Original control flow preserved.
```

 

## 🔐 Security and Privacy

Semantic search may process sensitive prompts, documents, logs, or extracted text. Treat all source
content and generated vectors as potentially sensitive.

Recommended practices:

* Do not commit uploaded documents.
* Do not commit local SQLite logs.
* Do not commit generated embedding files unless intentionally sanitized.
* Avoid logging full source text unless required.
* Keep API keys outside source control.
* Use small sanitized samples for documentation examples.
* Delete temporary files when no longer needed.
* Avoid exposing private file paths in generated documentation.

 

## 🧰 Troubleshooting

### Results are too broad

Use smaller source chunks and add more specific query terms.

### Results miss obvious matches

Expand the query with synonyms, related terms, acronyms, and domain-specific language.

### Scores seem close together

Review the top results manually. Similarity scores are ranking aids, not final judgments.

### The search fails before ranking

Check API keys, embedding model name, input text validation, and exception logs.

### The UI crashes

Inspect the SQLite exception log created through `boogr.py`. The `module`, `cause`, `method`,
`message`, and `trace` fields should identify the failing component.

### The results are not useful

Clarify whether the task needs keyword search, semantic search, document Q&A, or a hybrid approach.

 

## 📌 Best Practices

Use semantic search when wording may vary across sources. Use keyword search when exact terms
matter.

Best practices:

```text
Use meaningful chunks.
Include synonyms in the query.
Ask for ranked results.
Ask for relevance explanations.
Review high-ranking results manually.
Use follow-up prompts to refine scope.
Do not treat scores as absolute truth.
```

Avoid:

```text
Searching huge unchunked documents.
Using one-word vague queries.
Assuming embeddings from different models are comparable.
Committing generated vectors or logs without review.
Using semantic search when exact legal or technical wording is required.
```

 

## 🧭 Summary

Semantic search gives Jimi a meaning-based retrieval workflow for documents, source files, logs, and
user-provided text. It is most useful when related ideas may be expressed with different words.

The operational rule is:

```text
Use embeddings to find meaning, use structured prompts to control scope, and review ranked results before relying on them.
```
