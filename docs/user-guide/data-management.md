# 🗂️ Data Management

Jimi supports data-oriented workflows through a combination of Streamlit upload handling, Gemini
file operations, embedding generation, Google Cloud Storage-backed storage patterns, and
SQLite-backed exception logging. The data-management layer is designed to keep user inputs, uploaded
files, generated outputs, provider responses, and diagnostic records organized without placing
provider-specific logic directly in the Streamlit interface.

This page explains how Jimi manages files, documents, embeddings, vector-style storage,
configuration paths, and exception records.

 

## 🧭 Overview

Data management in Jimi is divided across several layers:

| Layer             | Module      | Responsibility                                                                          |
| ----------------- | ----------- | --------------------------------------------------------------------------------------- |
| User interface    | `app.py`    | Captures uploaded files, prompts, workflow selections, and display options.             |
| Provider wrappers | `gemini.py` | Uploads files, summarizes documents, generates embeddings, and manages storage objects. |
| Configuration     | `config.py` | Defines paths, keys, defaults, logging locations, and validation helpers.               |
| Logging           | `boogr.py`  | Writes structured exception records to SQLite.                                          |
| Documentation     | `docs/`     | Stores generated Markdown pages, images, CSS, and JavaScript assets.                    |

The guiding principle is separation of concerns. `app.py` should manage user interaction, while
`gemini.py` should manage provider APIs and storage operations.

 

## 🧱 Data Management Architecture

```text
User Input / Uploaded Files / Runtime Options
                    │
                    ▼
                app.py
                    │
                    ├── Validates required inputs
                    ├── Stores temporary UI state
                    ├── Routes workflow requests
                    └── Renders output
                    │
                    ▼
              gemini.py wrappers
                    │
                    ├── Files
                    ├── Embeddings
                    ├── VectorStores
                    ├── Chat
                    └── Images / Audio wrappers
                    │
                    ▼
External or Local Data Targets
                    │
                    ├── Gemini Files API
                    ├── Google Cloud Storage
                    ├── Google GenAI responses
                    ├── Local filesystem paths
                    └── SQLite logging database
```

This structure lets Jimi support multiple data workflows while keeping the UI code readable and
maintainable.

 

## 📁 Local File Handling

Jimi uses local files for workflows such as document summarization, document search, audio
transcription, audio translation, and image analysis.

Typical local-file responsibilities include:

* Accepting uploaded files from the Streamlit interface.
* Saving temporary files when a provider API requires a path.
* Passing file paths to wrapper methods.
* Preserving user-facing status messages.
* Removing or ignoring temporary files when no longer needed.
* Avoiding source control of user-provided files.

Recommended local file categories:

| Category                | Example                                   | Source Control           |
| ----------------------- | ----------------------------------------- | ------------------------ |
| Runtime uploads         | Uploaded PDFs, images, audio files        | Do not commit            |
| Generated logs          | SQLite exception database                 | Do not commit            |
| Generated documentation | `site/` build output                      | Usually do not commit    |
| Documentation assets    | Architecture diagrams, icons, screenshots | Commit when used by docs |
| Source files            | `.py`, `.md`, `.css`, `.js`               | Commit                   |

 

## 📄 Document Workflows

Document workflows usually begin in `app.py`, where the user uploads a file and provides a prompt.
The file path and prompt are then passed to the `Files` wrapper in `gemini.py`.

Common document workflows include:

| Workflow       | Wrapper Method         | Purpose                                          |
| -------------- | ---------------------- | ------------------------------------------------ |
| Upload file    | `Files.upload(...)`    | Sends a local file to Gemini file storage.       |
| Retrieve file  | `Files.retrieve(...)`  | Gets metadata for a previously uploaded file.    |
| List files     | `Files.list(...)`      | Lists available remote or configured files.      |
| Summarize file | `Files.summarize(...)` | Uses Gemini to summarize a PDF or text document. |
| Search file    | `Files.search(...)`    | Uses a prompt to query a document.               |
| Survey files   | `Files.survey(...)`    | Processes multiple documents as a group.         |
| Delete file    | `Files.delete(...)`    | Removes a remote file by identifier.             |

A document summarization flow typically looks like this:

```text
User uploads document
        │
        ▼
app.py saves or references file path
        │
        ▼
app.py passes prompt and path to Files.summarize(...)
        │
        ▼
Files wrapper uploads or attaches document
        │
        ▼
Gemini model generates summary
        │
        ▼
app.py renders summary to the user
```

 

## 🔎 Semantic Search and Embeddings

Semantic search depends on converting text into numeric vectors. Jimi uses the `Embeddings` wrapper
to generate those vectors.

A basic semantic workflow is:

```text
Text or document chunk
        │
        ▼
Embeddings.create(...)
        │
        ▼
Gemini embedding model
        │
        ▼
Vector representation
        │
        ▼
Similarity comparison or retrieval workflow
```

Embedding outputs can support:

* Semantic search.
* Similarity scoring.
* Document clustering.
* Retrieval workflows.
* Search result ranking.
* Future vector-store integration.

The `Embeddings` wrapper should own embedding-model calls. The Streamlit layer should collect text
and display results, not build provider-specific embedding requests.

---

## 🧮 Embedding Data Handling

Embedding vectors should be treated as derived data. They are generated from source text and can be
regenerated when the model or chunking strategy changes.

Recommended handling:

| Data Type         | Recommended Handling                                 |
| ----------------- | ---------------------------------------------------- |
| Source document   | Preserve only when needed and permitted.             |
| Extracted text    | Store only when needed for search or audit.          |
| Text chunks       | Recreate when chunking strategy changes.             |
| Embeddings        | Regenerate when model or source text changes.        |
| Similarity scores | Treat as runtime output unless needed for reporting. |

Do not assume embeddings are portable across models. If the embedding model changes, previously
generated vectors may no longer be comparable.

 

## 🧩 Vector-Style Storage

Jimi’s `VectorStores` wrapper uses Google Cloud Storage concepts to manage collection-like storage.
Buckets and prefixes can be treated as collections, while blobs can be treated as stored documents
or assets.

Supported storage-style operations include:

| Operation       | Purpose                                                              |
| --------------- | -------------------------------------------------------------------- |
| `create(...)`   | Creates or initializes storage behavior depending on implementation. |
| `upload(...)`   | Uploads a local file into a configured bucket.                       |
| `retrieve(...)` | Retrieves metadata or object reference for a stored blob.            |
| `list(...)`     | Lists available objects in a bucket.                                 |
| `delete(...)`   | Deletes a stored object.                                             |

Conceptual mapping:

```text
Collection
    │
    ▼
Google Cloud Storage bucket or prefix
    │
    ▼
Blob objects
    │
    ▼
Documents, data files, embeddings, or assets
```

This design gives Jimi a reusable storage abstraction without requiring the UI to interact directly
with Google Cloud Storage APIs.

 

## 🗃️ Runtime State

Streamlit session state is used for temporary runtime values such as selected modes, prompts,
uploaded file references, generated outputs, and interface toggles.

Session state is appropriate for:

* Selected model.
* Selected mode.
* Current prompt.
* Uploaded file metadata.
* Active workflow settings.
* Display toggles.
* Last generated response.
* Conversation history.

Session state is not appropriate for:

* API keys.
* Long-term document storage.
* Durable logs.
* Sensitive data persistence.
* Provider credentials.
* Large file archives.

Durable information should be stored through an intentional storage mechanism, not accidentally
retained in session state.

 

## ⚙️ Configuration-Driven Paths

Data paths should be centralized in `config.py`. This avoids scattered hard-coded paths and makes
the application easier to move between local development, hosted environments, and documentation
builds.

Examples of configuration-managed values include:

```text
LOG_DIR
LOG_PATH
LOG_FILE
MODEL_PATH
DB_PATH
GOOGLE_API_KEY
GEMINI_API_KEY
GOOGLE_CLOUD_PROJECT_ID
GOOGLE_CLOUD_LOCATION
```

Use configuration values instead of hard-coded paths whenever a path may differ across machines or
deployments.

Recommended pattern:

```python
self.project_id = cfg.GOOGLE_CLOUD_PROJECT_ID
self.google_api_key = cfg.GOOGLE_API_KEY
self.gemini_api_key = cfg.GEMINI_API_KEY
```

Avoid this pattern:

```python
self.api_key = "hard-coded-key"
self.file_path = "C:/Users/example/local-only/path/file.pdf"
```

 

## 🧾 Exception Log Data

Jimi records exception diagnostics through `boogr.py`. Exceptions are wrapped in an `Error` object
and written by `Logger` to the configured SQLite database.

A typical log record includes:

| Field     | Purpose                                                     |
| --------- | ----------------------------------------------------------- |
| `created` | Timestamp for the exception record.                         |
| `cause`   | Class, workflow, or component where the exception occurred. |
| `module`  | Source module name such as `app` or `gemini`.               |
| `method`  | Method or function signature associated with the failure.   |
| `message` | Exception message.                                          |
| `info`    | Optional additional diagnostic information.                 |
| `trace`   | Captured traceback details.                                 |

Standard pattern:

```python
except Exception as e:
	ex = Error( e )
	ex.module = 'gemini'
	ex.cause = 'Files'
	ex.method = 'upload( self, filepath, name )'
	Logger( ).write( ex )
	raise ex
```

For recoverable UI behavior:

```python
except Exception as e:
	ex = Error( e )
	ex.module = 'app'
	ex.cause = 'DataManagement'
	ex.method = 'render_data_panel( )'
	Logger( ).write( ex )
	return None
```

The original control flow should be preserved. Hard failures should raise. Recoverable workflows
should log and continue, return, or display an appropriate message.

 

## 🧹 Data Hygiene

Data workflows should avoid retaining unnecessary files, logs, or derived outputs.

Recommended practices:

* Do not commit user-uploaded files.
* Do not commit local SQLite logs.
* Do not commit `.env` files.
* Do not commit temporary data exports unless they are documentation examples.
* Keep sample data small and sanitized.
* Keep generated documentation output out of source control unless intentionally publishing it.
* Use clear folder names for runtime data.
* Clean old temporary files during development.
* Avoid logging sensitive prompt or document content unless required.

Recommended `.gitignore` entries:

```gitignore
.venv/
__pycache__/
*.pyc
.env
.env.*
logging/*.db
logging/*.sqlite
uploads/
tmp/
temp/
site/
.cache/
.mypy_cache/
.pytest_cache/
```

 

## 🔐 Security and Privacy

Data management workflows may handle prompts, documents, file paths, generated outputs, audio,
images, and provider responses. Treat those values as potentially sensitive.

Security practices:

* Load API keys from environment variables or local config excluded from Git.
* Do not print secrets to Streamlit or logs.
* Do not log full document text unless explicitly required.
* Do not expose uploaded files through public documentation.
* Do not commit local databases.
* Do not store unnecessary provider responses.
* Keep Google Cloud permissions scoped to the minimum required access.
* Delete temporary files when workflows no longer need them.
* Use separate test data for documentation examples.

 

## 🧪 Data Workflow Validation

Validate source files first:

```powershell
python -m py_compile .\app.py
python -m py_compile .\gemini.py
python -m py_compile .\config.py
python -m py_compile .\boogr.py
```

Then run:

```powershell
python -m compileall .
```

Validate documentation:

```powershell
mkdocs build
```

Validate logging coverage:

```powershell
Select-String -Path .\app.py -Pattern "except Exception as e:|Error\( e \)|Logger\( \)\.write"
Select-String -Path .\gemini.py -Pattern "except Exception as e:|Error\( e \)|Logger\( \)\.write"
```

Validate that local runtime artifacts are not staged:

```powershell
git status
```

If runtime files appear in `git status`, add them to `.gitignore` or remove them before committing.

 

## 🧰 Recommended Development Workflow

Use this sequence when adding or changing a data workflow:

1. Define the workflow in the Streamlit UI.
2. Validate required user input with `throw_if` or equivalent checks.
3. Add provider-specific logic to the appropriate wrapper class.
4. Keep file upload, embedding, or storage behavior out of UI-only helpers.
5. Add Google-style docstrings to new methods.
6. Add the required exception logging pattern.
7. Preserve fallback behavior where the UI should continue.
8. Run Python compile checks.
9. Run MkDocs build.
10. Test with small sample files before larger data.

 

## 📊 Data Workflow Examples

### Document summary workflow

```text
Uploaded PDF
    │
    ▼
Temporary local path
    │
    ▼
Files.summarize(prompt, filepath, model)
    │
    ▼
Gemini response
    │
    ▼
Streamlit summary output
```

### Embedding workflow

```text
Text input
    │
    ▼
Embeddings.create(text, model)
    │
    ▼
Embedding vector
    │
    ▼
Similarity or retrieval result
```

### Storage workflow

```text
Local file
    │
    ▼
VectorStores.upload(path, bucket, name)
    │
    ▼
Google Cloud Storage object
    │
    ▼
Object metadata returned to UI
```

### Logging workflow

```text
Exception
    │
    ▼
Error(e)
    │
    ▼
module / cause / method metadata
    │
    ▼
Logger().write(ex)
    │
    ▼
SQLite Exceptions table
```

 

## 🧭 Summary

Jimi’s data-management architecture separates UI state, provider file operations, embedding
generation, storage abstractions, runtime configuration, and exception logging. This makes the
application easier to debug, extend, document, and deploy.

The operational rule is:

```text
Keep user interaction in app.py, provider data operations in gemini.py, configuration in config.py, and durable diagnostics in boogr.py.
```

Following this rule helps prevent fragile UI code, scattered paths, missing logs, and undocumented
data workflows.
