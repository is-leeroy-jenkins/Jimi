###### Jimi

![](https://github.com/is-leeroy-jenkins/Jimi/blob/main/resources/images/jimi_project.png)

<p align="center">
  <a href="#-overview">Overview</a> |
  <a href="#-features">Features</a> |
  <a href="#-application-modes">Modes</a> |
  <a href="#-requirements">Requirements</a> |
  <a href="#-local-llm">LLM</a> |
  <a href="#-installation">Installation</a> |
  <a href="#-running-the-streamlit-application">Run</a> |
  <a href="#-configuration">Configuration</a> |
  <a href="#-design-and-architecture">Architecture</a> |
  <a href="#-capabilities">Capabilities</a> |
  <a href="#-data-management">Data</a> |
</p>

Jimi is a Python and Streamlit application for local language-model inference,
retrieval-augmented generation, semantic search, prompt engineering, and SQLite-backed data
management. It is designed for federal analysts, technical users, and data-science workflows that
benefit from local execution, durable prompt storage, document-grounded question answering, semantic
retrieval, and controlled analytical tooling.

Jimi uses an optional local GGUF model through `llama.cpp`, supports document retrieval with
`sentence-transformers`, persists chat history and prompts in SQLite, and provides a Streamlit user
interface for text generation, document Q&A, semantic indexing, prompt administration, and database
operations.

## 🎥 Demo

![](https://github.com/is-leeroy-jenkins/Jimi/blob/main/resources/images/Jimi-functionality.gif)

## 🧱 Databricks

[![Jimi](https://img.shields.io/badge/Databricks-Jimi-FF3621?logo=databricks\&logoColor=white)](https://dbc-a0c21f80-7bb3.cloud.databricks.com/browse/folders/254131913339635?o=7474645703081351)

* Databricks workspace repository for the Jimi codebase.
* Supports collaborative development, analytics, notebook execution, and application deployment.
* 
## 🧠 Custom LLM

[![HuggingFace](https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-sm.svg)](https://huggingface.co/leeroy-jankins/jimi)

Jimi is designed to run against a local GGUF model, commonly a Gemma-family or Llama-family model
quantized for efficient local inference. The repository contains application code; the model should be
stored separately and referenced through configuration.

Example model name:

```text
Gemma-4-E4B-it.gguf
```

## 🧰 Overview

Jimi is a local-first analytical assistant built around a Streamlit interface and a configurable
local GGUF model. The application checks whether the configured model file exists, lazily loads the
model only when available, and routes text-generation and document-question-answering turns through
a shared prompt builder.

The application is intentionally simple and durable:

* The model is stored outside the repository.
* The app reads the model path from `config.py`.
* Chat history, embeddings, and prompts are persisted in SQLite.
* Document Q&A uses local text extraction, chunking, embeddings, and retrieval.
* Semantic Search builds a reusable SQLite-backed embedding index.
* Data Management provides practical SQLite inspection, import, query, visualization, and schema tools.

## ✨ Features

* **Local LLM inference** through `llama.cpp` and a configurable GGUF model path.
* **Deferred dependency resolution** so optional local model and embedding dependencies do not prevent
  the Streamlit UI from loading.
* **Lazy model loading** with `st.cache_resource` for local model and embedding resources.
* **Text Generation mode** with streaming output and adjustable task, reasoning, coding, response,
  inference, and context controls.
* **Document Q&A mode** with upload-based retrieval, document actions, parsing controls, local RAG
  prompt construction, retrieved-chunk display, and document inventory diagnostics.
* **Semantic Search mode** for building, querying, maintaining, and reusing a local embedding index.
* **Prompt Engineering mode** backed by the local SQLite `Prompts` table, including prompt generation,
  cloning, metadata, category inference, and cascade actions into Text Generation or Document Q&A.
* **Data Management mode** for importing, browsing, editing, profiling, visualizing, and querying SQLite data.
* **System Instructions editor** with reusable prompt-template loading, XML-to-Markdown conversion,
  preset application, and effective-prompt preview.
* **SQLite persistence** for chat history, prompt records, embeddings, and imported tabular data.
* **Document text extraction** using PyMuPDF for PDFs and defensive text decoding for text-like files.
* **Vector retrieval** using `sentence-transformers`, `sqlite-vec` when available, and cosine-similarity fallback.
* **Interactive visualizations** through Plotly Express.
* **Fixed footer status bar** showing active mode, temperature, Top-P, Top-K, penalties, token limits,
  context size, thread count, semantic status, and loaded document count.

## 🧩 Application Modes

| Mode                 | Description                                                                                                                                                    |
| -------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Text Generation`    | Chat with the local Jimi model using streaming llama.cpp output and configurable task, reasoning, coding, response, inference, and context controls.           |
| `Document Q&A`       | Upload one or more documents, build a retrieval context, run document actions, and ask document-grounded questions.                                            |
| `Semantic Search`    | Upload files, chunk text, create sentence-transformer embeddings, query the semantic index, and reuse selected chunks as context.                              |
| `Prompt Engineering` | Search, sort, page, select, edit, clone, generate, create, update, delete, and cascade reusable prompt records.                                                |
| `Data Management`    | Import Excel workbooks, browse SQLite tables, run CRUD operations, profile data, filter, aggregate, visualize, administer schema, and run guarded SQL queries. |

## 🛠️ Requirements

| Requirement           | Purpose                                                                           |
| --------------------- | --------------------------------------------------------------------------------- |
| Python 3.10+          | Runtime environment                                                               |
| Streamlit             | Web application framework                                                         |
| llama-cpp-python      | Optional local GGUF model inference                                               |
| PyMuPDF / `fitz`      | PDF text extraction and document preview support                                  |
| numpy                 | Vector math and cosine similarity                                                 |
| pandas                | Table import, display, CRUD support, and SQL result handling                      |
| plotly.express        | Interactive charts and visualizations                                             |
| sentence-transformers | Local embedding model for semantic search and document retrieval                  |
| sqlite-vec            | Optional SQLite vector table support for Document Q&A                             |
| SQLite                | Local persistence for chat history, prompt records, embeddings, and imported data |
| openpyxl              | Excel workbook import support through pandas                                      |
| config.py             | Application constants, model path, UI labels, modes, and default runtime settings |



## 🧊 Local LLM

Jimi uses optional local LLM support. The model is loaded only when the configured GGUF file exists.
The `local_model_available()` function checks the configured model path before `llama_cpp.Llama` is
imported and initialized.

The relevant configuration pattern is:

```python
MODEL_PATH = "C:/path/to/Gemma-4-E4B-it.gguf"
DEFAULT_CTX = 4096
CORES = 4
MODES = [
    "Text Generation",
    "Document Q&A",
    "Semantic Search",
    "Prompt Engineering",
    "Data Management"
]
```

The model-loading path uses `llama_cpp.Llama` with:

| Parameter    | Purpose                     |
| ------------ | --------------------------- |
| `model_path` | Path to the GGUF model file |
| `n_ctx`      | Context window size         |
| `n_threads`  | CPU thread count            |
| `n_batch`    | Batch size for inference    |
| `verbose`    | Runtime logging control     |

Recommended model locations include:

* A local `models/` or `llm/` folder.
* An LM Studio model cache.
* An external drive.
* A shared read-only model directory.

Example Windows path:

```text
C:\Users\<you>\source\llm\lmstudio\lmstudio-community\leeroy-jankins\jimi\Gemma-4-E4B-it.gguf
```

## 📦 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/is-leeroy-jenkins/Jimi.git
cd Jimi
```

### 2. Create and Activate a Virtual Environment

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Command Prompt:

```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

macOS / Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Install llama-cpp-python

```bash
python -m pip install llama-cpp-python
```

## ⚙️ Configuration

Jimi is configured primarily through `config.py`.

Common configuration values include:

| Setting             | Description                                           |
| ------------------- | ----------------------------------------------------- |
| `MODEL_PATH`        | Local path to the GGUF model file                     |
| `DEFAULT_CTX`       | Default context window                                |
| `CORES`             | Default CPU thread count                              |
| `DB_PATH`           | SQLite database path                                  |
| `MODES`             | Sidebar application mode list                         |
| `APP_TITLE`         | Streamlit page title                                  |
| `APP_SUBTITLE`      | Streamlit app subtitle/caption                        |
| `FAVICON`           | Streamlit page icon                                   |
| `LOGO_PATH`         | Sidebar logo path                                     |
| `BLUE_DIVIDER`      | Shared divider markup                                 |
| `XML_BLOCK_PATTERN` | XML-like delimiter pattern used for prompt conversion |

Optional environment-variable pattern:

```powershell
setx JIMI_LLM_PATH "C:\path\to\Gemma-4-E4B-it.gguf"
```

Then bind that value inside `config.py` if the project uses environment-based model path resolution.

## 🚀 Running the Streamlit Application

From the project root:

```bash
streamlit run app.py
```

Once running, the application is available at:

```text
http://localhost:8501
```

## 💬 Text Generation

The `Text Generation` mode provides streaming local inference through the shared `run_llm_turn()`
pipeline.

Control groups include:

| Control Group       | Options                                                                                                          |
| ------------------- | ---------------------------------------------------------------------------------------------------------------- |
| Task Preset         | Chat, Reasoning, Coding, Translation, Summarization, Extraction                                                  |
| Reasoning Controls  | Reasoning depth, answer-only mode, self-check, deterministic reasoning                                           |
| Coding Controls     | Code language, coding task, comments, editor format, fenced-code output, translation target language             |
| Response Controls   | Temperature, Top-P, Top-K, grounding toggle                                                                      |
| Inference Settings  | Repeat window, repeat penalty, presence penalty, frequency penalty                                               |
| Context Controls    | Context window, CPU threads, max tokens, random seed                                                             |
| System Instructions | Instruction editor, template selector, clear button, XML/Markdown conversion, preset application, prompt preview |

The text-generation workflow is:

1. User enters a chat prompt.
2. The app stores the user message in SQLite.
3. The prompt builder combines system instructions, task-specific instructions, optional semantic
   context, optional document context, and chat history.
4. `llama.cpp` streams the response into the Streamlit chat message.
5. The assistant message is saved to SQLite.

## 📚 Document Q&A

The `Document Q&A` mode supports local retrieval-augmented generation.

Supported upload types include:

* `pdf`
* `txt`
* `docx`

Document Q&A controls include:

| Control Group       | Options                                                                                                                                                       |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Retrieval Controls  | Chunks to retrieve, chunk size, chunk overlap, retrieved-chunk display, grounding requirement, excerpt-only answering, sqlite-vec preference, cosine fallback |
| Document Actions    | Answer Question, Summarize Active Document, Extract Key Points, Generate Outline, Extract Entities, Extract Tables, Compare Active Documents                  |
| Document Parsing    | OCR toggle, native PDF text preference, page markers, diagnostics                                                                                             |
| Response Settings   | Temperature, Top-P, Top-K                                                                                                                                     |
| Inference Settings  | Repeat window, repeat penalty, presence penalty, frequency penalty                                                                                            |
| Context Controls    | Context window, CPU threads, max tokens, random seed                                                                                                          |
| System Instructions | Instruction editor, template selector, clear button, XML/Markdown conversion                                                                                  |

The workflow includes:

1. Upload one or more documents.
2. Store active document names and bytes in Streamlit session state.
3. Preview the first active document when supported.
4. Extract text from PDF bytes using PyMuPDF or decode text-like files defensively.
5. Build document inventory rows with size, extracted text length, chunk count, and loaded status.
6. Chunk extracted text using configurable chunk size and overlap.
7. Generate embeddings with `sentence-transformers`.
8. Store vectors in `sqlite-vec` if available and enabled.
9. Fall back to in-memory cosine similarity when vector-table support is unavailable and fallback is enabled.
10. Retrieve the top relevant document chunks.
11. Build a document-grounded prompt using the selected document action.
12. Stream the answer from the local model.

## 🔍 Semantic Search

The `Semantic Search` mode builds and queries a reusable local embedding index.

Semantic Search supports:

* Uploading multiple `pdf`, `txt`, or `docx` files.
* Configuring chunk size and chunk overlap.
* Clearing or appending to the existing index.
* Embedding chunks with `sentence-transformers`.
* Storing chunks and vector blobs in the SQLite `embeddings` table.
* Querying the embedding index with Top-K and minimum-similarity controls.
* Selecting result rows in a Streamlit data editor.
* Sending selected chunks to Text Generation.
* Sending selected chunks to Document Q&A.
* Saving selected chunks as shared prompt context.
* Deleting the index.
* Recomputing diagnostics.
* Clearing query results.

When semantic context is enabled, Text Generation can retrieve similar chunks from the embedding
store and inject them into the prompt.

## 📝 Prompt Engineering

The `Prompt Engineering` mode manages reusable prompt records in the SQLite `Prompts` table.

Prompt records include:

| Field       | Description                                |
| ----------- | ------------------------------------------ |
| `PromptsId` | Primary key                                |
| `Caption`   | Display caption used by template selectors |
| `Name`      | Prompt name                                |
| `Text`      | Prompt body                                |
| `Version`   | Prompt version                             |
| `ID`        | External or user-defined identifier        |

Prompt Engineering supports:

* Search by caption, name, or text.
* Prompt category inference.
* Prompt category filtering.
* Sort field and sort direction selection.
* Pagination.
* Jump-to-ID navigation.
* Prompt row selection.
* Prompt editing.
* Prompt creation.
* Prompt deletion.
* Prompt cloning.
* Starter prompt generation.
* Local-model prompt-template drafting.
* Cascading selected prompts into shared Text Generation settings.
* Cascading selected prompts into shared Document Q&A settings.

Supported prompt categories include:

* General Chat.
* Reasoning.
* Coding.
* Translation.
* Summarization.
* Extraction.
* Document Extraction.
* OCR.
* Audio.
* JSON Output.

## 🏛️ Data Management

The `Data Management` mode provides a SQLite administration and exploration interface.

Tabs include:

| Tab       | Purpose                                                                                 |
| --------- | --------------------------------------------------------------------------------------- |
| Import    | Import Excel workbook sheets into SQLite tables                                         |
| Browse    | Browse existing SQLite tables                                                           |
| CRUD      | Insert, update, and delete table rows                                                   |
| Explore   | Page through table records                                                              |
| Filter    | Filter rows by column text containment                                                  |
| Aggregate | Run simple numeric aggregations                                                         |
| Visualize | Render charts from table data                                                           |
| Admin     | Profile data, drop tables, create indexes, create tables, view schema, and alter tables |
| SQL       | Execute guarded read-only SQL and download query results as CSV                         |

Administrative operations include:

* Table profiling.
* Table dropping with confirmation.
* Index creation.
* Custom table creation.
* Schema viewing.
* Row-count metrics.
* Index listing.
* Add column.
* Rename column.
* Rename table.
* Drop column.

The SQL console allows read-only SQL workflows and blocks unsafe operations such as:

* `INSERT`
* `UPDATE`
* `DELETE`
* `DROP`
* `ALTER`
* `CREATE`
* `ATTACH`
* `DETACH`
* `VACUUM`
* `REPLACE`
* `TRIGGER`

## 🧩 Design and Architecture

Jimi uses a local-first Streamlit architecture:

| Layer                   | Description                                                                                                             |
| ----------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| UI Layer                | Streamlit sidebar, expanders, tabs, chat messages, uploaders, dataframes, and charts                                    |
| Mode Layer              | Text Generation, Document Q&A, Semantic Search, Prompt Engineering, and Data Management blocks                          |
| Local Model Layer       | Optional `llama.cpp` model loading through `llama-cpp-python`                                                           |
| Prompt Layer            | Shared prompt builder with system instructions, task instructions, semantic context, document context, and chat history |
| Retrieval Layer         | PyMuPDF extraction, chunking, sentence-transformer embeddings, sqlite-vec, and cosine fallback                          |
| Persistence Layer       | SQLite database for chat history, prompts, embeddings, and imported data                                                |
| Prompt Operations Layer | Prompt category inference, starter prompt generation, cloning, cascading, and local prompt drafting                     |
| Visualization Layer     | Plotly Express charts over SQLite-backed pandas DataFrames                                                              |
| Status Layer            | Fixed footer summarizing active mode and runtime parameters                                                             |

Architecture diagram:

```text
┌──────────────────────────────────────────────┐
│                 Jimi Streamlit App           │
│                                              │
│  - Text Generation                           │
│  - Document Q&A                              │
│  - Semantic Search                           │
│  - Prompt Engineering                        │
│  - Data Management                           │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│              Shared Prompt Pipeline          │
│                                              │
│  - System instructions                       │
│  - Task-specific instructions                │
│  - Chat history                              │
│  - Semantic context                          │
│  - Document excerpts                         │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│             Local llama.cpp Model            │
│                                              │
│  - GGUF model path                           │
│  - Context window                            │
│  - CPU threads                               │
│  - Streaming output                          │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│              SQLite Persistence              │
│                                              │
│  - chat_history                              │
│  - embeddings                                │
│  - Prompts                                   │
│  - imported data tables                      │
└──────────────────────────────────────────────┘
```

## 💻 Capabilities

| Capability               | Description                                                                                           |
| ------------------------ | ----------------------------------------------------------------------------------------------------- |
| Local Text Generation    | Streams responses from a local llama.cpp model                                                        |
| Task Presets             | Adds structured task behavior for chat, reasoning, coding, translation, summarization, and extraction |
| Reasoning Controls       | Adjusts reasoning depth, answer-only output, self-check behavior, and deterministic preference        |
| Coding Controls          | Guides language, coding task, comments, editor format, and fenced-code output                         |
| System Instructions      | Stores and applies global assistant behavior instructions                                             |
| Prompt Templates         | Loads reusable prompt text from SQLite                                                                |
| Effective Prompt Preview | Displays the assembled system/task/user prompt preview                                                |
| Chat Persistence         | Saves user and assistant messages to SQLite                                                           |
| Document Q&A             | Retrieves relevant document chunks and injects them into the prompt                                   |
| Document Actions         | Summarizes, extracts, outlines, compares, and answers against active documents                        |
| Semantic Search          | Builds, queries, and maintains embeddings for uploaded files                                          |
| SQLite Vector Retrieval  | Uses sqlite-vec where available for document retrieval                                                |
| Cosine Fallback          | Falls back to in-memory cosine similarity when sqlite-vec is unavailable                              |
| Prompt Engineering       | Manages prompt records, categories, cloning, generation, and cascading actions                        |
| Excel Import             | Imports workbook sheets into SQLite tables                                                            |
| Table Browsing           | Displays SQLite table contents in Streamlit                                                           |
| CRUD Operations          | Inserts, updates, and deletes table rows                                                              |
| Data Profiling           | Computes type, null, distinct, minimum, maximum, and mean summaries                                   |
| Visualization            | Builds histograms, bar charts, line charts, scatter plots, box plots, pies, and correlation heatmaps  |
| SQL Console              | Runs guarded read-only SQL and reports execution metrics                                              |
| Footer Status            | Shows active mode and runtime parameters                                                              |

## 📁 File Organization

| File / Folder                                                                              | Description                                                                    |
| ------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------ |
| [`app.py`](https://github.com/is-leeroy-jenkins/Jimi/blob/main/app.py)                     | Main Streamlit application                                                     |
| [`config.py`](https://github.com/is-leeroy-jenkins/Jimi/blob/main/config.py)               | Constants, model path, modes, help text, paths, and UI settings                |
| [`requirements.txt`](https://github.com/is-leeroy-jenkins/Jimi/blob/main/requirements.txt) | Python package requirements                                                    |
| `stores/sqlite/Data.db`                                                                    | Local SQLite database for chat history, prompts, embeddings, and imported data |
| `resources/images`                                                                         | Project images, logos, and README assets                                       |
| `models/` or external model path                                                           | Recommended location for local GGUF model files                                |

## 🧪 Example Usage

### Local LLM Loading Pattern

```python
from llama_cpp import Llama

llm = Llama(
    model_path="C:/path/to/Gemma-4-E4B-it.gguf",
    n_ctx=4096,
    n_threads=4,
    n_batch=512,
    verbose=False
)
```

### Sentence Embedding Pattern

```python
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer("all-MiniLM-L6-v2")
vectors = embedder.encode([
    "Federal analysts need reliable local retrieval over policy documents."
])

print(vectors.shape)
```

### SQLite Prompt Query

```python
import sqlite3

with sqlite3.connect("stores/sqlite/Data.db") as conn:
    rows = conn.execute(
        "SELECT PromptsId, Caption, Name, Version FROM Prompts ORDER BY PromptsId DESC"
    ).fetchall()

print(rows)
```

### Streamlit Launch

```bash
streamlit run app.py
```

## 🧮 Runtime Parameters

Jimi exposes the following active runtime parameters in the UI and footer:

| Parameter         | Purpose                                                    |
| ----------------- | ---------------------------------------------------------- |
| Temperature       | Sampling randomness                                        |
| Top-P             | Nucleus sampling probability                               |
| Top-K             | Token candidate limit                                      |
| Frequency Penalty | Penalizes repeated token frequency                         |
| Presence Penalty  | Penalizes already-present tokens                           |
| Repeat Penalty    | Penalizes repeated generation patterns                     |
| Repeat Window     | Token window used for repeat control                       |
| Max Tokens        | Maximum response length                                    |
| Context Window    | llama.cpp context length                                   |
| CPU Threads       | Number of CPU threads used by the model                    |
| Random Seed       | Seed control exposed by the UI for deterministic workflows |
| Semantic          | Indicates whether semantic context is enabled              |
| Docs              | Number of loaded basic document/context entries            |

## 🧰 Troubleshooting

| Issue                             | Resolution                                                                                          |
| --------------------------------- | --------------------------------------------------------------------------------------------------- |
| Local model unavailable           | Confirm `MODEL_PATH` points to an existing GGUF file.                                               |
| UI loads but generation fails     | Verify `llama-cpp-python` is installed and compatible with your Python version.                     |
| Slow generation                   | Increase CPU thread count, reduce context window, or reduce max tokens.                             |
| High memory usage                 | Lower `n_ctx`, reduce batch size, or use a smaller quantized model.                                 |
| Document Q&A returns weak answers | Confirm documents are loaded, extracted text is available, and semantic chunks are being generated. |
| Retrieved chunks do not display   | Confirm `Show Retrieved Chunks` is enabled in Document Q&A retrieval controls.                      |
| Semantic Search fails             | Confirm `sentence-transformers` can download or load `all-MiniLM-L6-v2`.                            |
| PDF extraction fails              | Confirm PyMuPDF is installed and the file is a valid PDF.                                           |
| sqlite-vec unavailable            | Disable `Use sqlite-vec` or enable cosine fallback in Document Q&A.                                 |
| SQL query blocked                 | Use a read-only `SELECT`, `WITH`, `EXPLAIN`, or safe `PRAGMA` query.                                |


## 🚀 Application Badges

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python\&logoColor=white)](#-requirements)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit\&logoColor=white)](#-running-the-streamlit-application)
[![llama.cpp](https://img.shields.io/badge/llama.cpp-Local%20LLM-111111?logo=meta\&logoColor=white)](#-local-llm)
[![GGUF](https://img.shields.io/badge/GGUF-Quantized%20Model-6A5ACD)](#-local-llm)
[![SQLite](https://img.shields.io/badge/SQLite-Data%20Store-003B57?logo=sqlite\&logoColor=white)](#-data-management)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Model%20Host-FFD21E?logo=huggingface\&logoColor=black)](#-custom-llm)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](#-license)


## 📝 License

Jimi is published under the [MIT License](https://github.com/is-leeroy-jenkins/Jimi/blob/main/LICENSE).

