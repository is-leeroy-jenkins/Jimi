# 🛠️ Development

This guide explains how to set up, run, validate, document, and maintain the Jimi application. Jimi
is a Streamlit-based AI assistant that integrates Google Gemini workflows, configurable runtime
settings, SQLite-backed exception logging, and MkDocs-generated technical documentation.

The development workflow should protect three priorities:

1. Keep the Streamlit application stable.
2. Keep provider wrappers isolated and testable.
3. Keep source comments compatible with MkDocs and mkdocstrings.

 

## 🧭 Development Overview

Jimi is organized around a small set of core Python modules:

| File               | Purpose                                                                                     |
| ------------------ | ------------------------------------------------------------------------------------------- |
| `app.py`           | Streamlit application entry point, UI orchestration, session state, and workflow routing.   |
| `gemini.py`        | Google Gemini wrapper layer for text, image, embedding, audio, file, and storage workflows. |
| `config.py`        | Runtime settings, paths, constants, environment values, and validation helpers.             |
| `boogr.py`         | Shared exception wrapper and SQLite logging implementation.                                 |
| `requirements.txt` | Python package dependencies.                                                                |
| `mkdocs.yml`       | MkDocs Material site configuration.                                                         |
| `docs/`            | Markdown documentation, generated API reference pages, CSS, JavaScript, and images.         |

The preferred development model is to keep `app.py` focused on presentation and workflow routing
while keeping API-specific logic inside wrapper classes.

 

## 📦 Project Layout

A recommended project structure is:

```text
Jimi/
├── app.py
├── gemini.py
├── boogr.py
├── config.py
├── requirements.txt
├── mkdocs.yml
├── README.md
├── docs/
│   ├── index.md
│   ├── architecture.md
│   ├── development.md
│   ├── github-pages.md
│   ├── api/
│   │   ├── index.md
│   │   ├── app.md
│   │   ├── gemini.md
│   │   ├── config.md
│   │   └── boogr.md
│   ├── user-guide/
│   │   ├── index.md
│   │   ├── text-generation.md
│   │   ├── document-qna.md
│   │   ├── semantic-search.md
│   │   ├── prompt-engineering.md
│   │   └── data-management.md
│   └── assets/
│       ├── css/
│       │   └── jimi.css
│       ├── js/
│       │   └── jimi.js
│       └── images/
│           ├── jimi-architecture.png
│           └── jimi-class-map.png
└── logging/
    └── Exceptions.db
```

The exact image filenames can be adjusted, but every file referenced in `mkdocs.yml` or Markdown
pages must exist under `docs/`.

 

## 🐍 Python Environment Setup

Create and activate a virtual environment from the repository root.

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Command Prompt

```bat
python -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### macOS / Linux

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

 

## 🔐 Environment Configuration

Jimi should load credentials and runtime settings from environment variables or local configuration
values that are not committed to the repository.

Common environment values include:

```powershell
$env:GEMINI_API_KEY="your-gemini-api-key"
$env:GOOGLE_API_KEY="your-google-api-key"
$env:GOOGLE_CLOUD_PROJECT_ID="your-google-cloud-project-id"
$env:GOOGLE_CLOUD_LOCATION="us-central1"
```

Do not commit secrets, `.env` files, local database files, generated cache files, or provider
credentials.

Recommended `.gitignore` entries include:

```gitignore
.venv/
__pycache__/
*.pyc
.env
.env.*
logging/*.db
logging/*.sqlite
site/
.cache/
.mypy_cache/
.pytest_cache/
.DS_Store
```

 

## ▶️ Running the Streamlit App

Run the application from the repository root:

```powershell
streamlit run app.py
```

The default Streamlit URL is usually:

```text
http://localhost:8501
```

During local development, use a clean terminal session with the virtual environment activated. This
reduces path issues and ensures the application uses the packages from `.venv`.

 

## 🧱 Development Responsibilities by Module

### `app.py`

`app.py` is responsible for the Streamlit runtime and should remain focused on the user interface.

Keep these responsibilities in `app.py`:

* Page configuration.
* Sidebar controls.
* Session-state initialization.
* User input widgets.
* Uploaded-file handling.
* Workflow routing.
* Result rendering.
* Streamlit status messages.
* UI fallback behavior.

Avoid placing provider-specific SDK logic directly in `app.py`. If a workflow requires new Gemini
behavior, add that behavior to `gemini.py` first and call the wrapper method from `app.py`.

 

### `gemini.py`

`gemini.py` is responsible for Google Gemini provider integration.

Keep these responsibilities in `gemini.py`:

* GenAI client creation.
* Gemini model configuration.
* Request payload construction.
* Tool configuration.
* Safety configuration.
* Response parsing.
* File upload and retrieval.
* Image extraction.
* Embedding generation.
* TTS audio conversion.
* Transcription and translation prompt construction.
* Google Cloud Storage interaction.

Each wrapper should expose high-level methods that `app.py` can call without knowing the internal
GenAI request schema.

 

### `config.py`

`config.py` is responsible for centralizing runtime settings.

Keep these responsibilities in `config.py`:

* API key lookup.
* Runtime path definitions.
* Logging path definitions.
* Default model names.
* Application constants.
* Common validation helpers.

Avoid scattering hard-coded paths, API keys, database names, or model defaults across the codebase.

 

### `boogr.py`

`boogr.py` is responsible for exception metadata and persistent logging.

Keep these responsibilities in `boogr.py`:

* Wrapping exceptions in `Error`.
* Capturing traceback details.
* Storing module, cause, method, message, and info fields.
* Creating the logging table when needed.
* Writing exception records to SQLite.

Do not duplicate logger implementation code in `app.py` or `gemini.py`. Use the shared `Error` and
`Logger` classes.

 

## 🧾 Required Exception Logging Pattern

Every `except Exception as e:` block should use the shared logging pattern.

For hard failures, use:

```python
except Exception as e:
	ex = Error( e )
	ex.module = 'module_name'
	ex.cause = 'ClassOrWorkflowName'
	ex.method = 'method_name( self, arg1, arg2 )'
	Logger( ).write( ex )
	raise ex
```

For recoverable UI fallbacks, log the exception but preserve the original fallback behavior:

```python
except Exception as e:
	ex = Error( e )
	ex.module = 'app'
	ex.cause = 'WorkflowName'
	ex.method = 'render_workflow_panel( )'
	Logger( ).write( ex )
	st.warning( str( ex ) )
	return None
```

For non-critical loop operations, preserve the original control flow:

```python
except Exception as e:
	ex = Error( e )
	ex.module = 'app'
	ex.cause = 'TableRendering'
	ex.method = 'render_results_table( records )'
	Logger( ).write( ex )
	continue
```

Do not create duplicate wrapper objects in the same handler.

Correct:

```python
except Exception as e:
	ex = Error( e )
	ex.module = 'gemini'
	ex.cause = 'VectorStores'
	ex.method = 'delete( self, bucket, name )'
	Logger( ).write( ex )
	raise ex
```

Incorrect:

```python
except Exception as e:
	exception = Error( e )
	Logger( ).write( exception )

	ex = Error( e )
	raise ex
```

The correct pattern creates one `Error` object, writes that same object once, and raises or returns
according to the original method behavior.

 

## 🧠 Google-Style Docstring Standard

Jimi source files should use Google-style docstrings that are compatible with MkDocs, mkdocstrings,
and griffe.

Use these section names:

* `Purpose:`
* `Args:`
* `Attributes:`
* `Returns:`
* `Raises:`
* `Notes:`
* `Examples:`

Do not use:

* `Parameters:`
* `Return:`
* `Purpose` with underline separators.
* `Args:` entries for parameters that do not exist.
* `Returns:` in constructors that return `None`.
* `self` or `cls` in `Args:` sections.

 

## 📝 Function Docstring Template

Use this template for normal functions and methods:

```python
def generate_text( self, prompt: str, model: str = 'gemini-2.5-flash-lite' ) -> str | None:
	"""
	Purpose:
		Generates a Gemini text response from validated user input and normalized
		model configuration.

	Args:
		prompt: User prompt submitted from the application interface.
		model: Gemini model identifier used for content generation.

	Returns:
		Generated response text when the provider returns text content; otherwise None.

	Raises:
		Error: Wraps and logs provider, validation, or response parsing failures.
	"""
```

Use concise, accurate descriptions. The `Purpose:` section should describe what the method does, not
narrate how the developer wrote it.

 

## 🧱 Class Docstring Template

Use this template for classes:

```python
class Chat( Gemini ):
	"""
	Purpose:
		Provides text, multimodal, and tool-grounded Gemini content generation
		operations for Streamlit application workflows.

	Attributes:
		client: Google GenAI client used to submit provider requests.
		contents: Request content payload built from prompts, context, and files.
		content_config: Gemini generation configuration for the current request.
		content_response: Raw Gemini response returned by the provider.
	"""
```

Class docstrings should document durable class responsibilities and meaningful instance attributes.
Avoid listing temporary loop variables unless they are intentionally part of the object state.

 

## 🚫 Constructor Docstring Rule

Constructors should document initialization behavior and attributes, but they should not include
`Returns:` when the return value is `None`.

Correct:

```python
def __init__( self, model: str = 'gemini-2.5-flash-lite' ):
	"""
	Purpose:
		Initializes the chat wrapper with default Gemini model settings, empty
		request state, and response placeholders.

	Args:
		model: Default Gemini model identifier assigned to the wrapper.

	Attributes:
		model: Active Gemini model identifier.
		client: Google GenAI client placeholder.
		content_response: Provider response placeholder.
	"""
```

Incorrect:

```python
def __init__( self ):
	"""
	Returns:
		None
	"""
```

MkDocs and griffe can warn about constructor return documentation when it is unnecessary or
inconsistent with annotations.

 

## 🔎 Docstring Review Checklist

Before accepting a regenerated source file, inspect it for these issues:

```powershell
Select-String -Path .\app.py -Pattern "Parameters:|Return:|Purpose\s*$|_{3,}|-{3,}"
Select-String -Path .\gemini.py -Pattern "Parameters:|Return:|Purpose\s*$|_{3,}|-{3,}"
```

Also inspect constructor return sections:

```powershell
Select-String -Path .\app.py -Pattern "def __init__|Returns:" -Context 0,8
Select-String -Path .\gemini.py -Pattern "def __init__|Returns:" -Context 0,8
```

These checks are not replacements for a real build, but they quickly catch common documentation
formatting failures.

 

## ✅ Source Validation

Run syntax checks after every generated source replacement.

```powershell
python -m py_compile .\app.py
python -m py_compile .\gemini.py
python -m py_compile .\config.py
python -m py_compile .\boogr.py
```

Then run a broader compilation pass:

```powershell
python -m compileall .
```

A clean `py_compile` result confirms Python syntax only. It does not confirm provider credentials,
API availability, Streamlit runtime behavior, or MkDocs rendering.

 

## 🧪 Logging Validation

Use targeted searches to confirm every broad exception handler logs exactly once.

```powershell
Select-String -Path .\app.py -Pattern "except Exception as e:|Error\( e \)|Logger\( \)\.write"
Select-String -Path .\gemini.py -Pattern "except Exception as e:|Error\( e \)|Logger\( \)\.write"
```

For a stronger validation, inspect each handler manually or use a short script that checks every
`except Exception` block for one `Error( e )` wrapper and one `Logger( ).write(...)` call.

The expected rule is:

```text
Every except Exception handler:
- creates one Error object
- assigns module/cause/method metadata
- writes that same object with Logger
- preserves the original raise, return, continue, pass, or fallback behavior
```

 

## 📚 MkDocs Setup

Install documentation dependencies from `requirements.txt` or explicitly install the required
packages:

```powershell
pip install mkdocs mkdocs-material "mkdocstrings[python]" mkdocs-autorefs pymdown-extensions
```

The documentation build depends on:

* `mkdocs.yml`
* Markdown files under `docs/`
* CSS and JavaScript assets under `docs/assets/`
* API reference pages that use mkdocstrings directives
* Google-style docstrings in the Python source files

 

## 🧾 API Reference Pages

A typical API page should use mkdocstrings:

```markdown
# App API

::: app
    options:
      show_root_heading: true
      show_source: true
      members_order: source
```

Recommended API pages:

```text
docs/api/app.md
docs/api/gemini.md
docs/api/config.md
docs/api/boogr.md
```

Each source module must be importable from the project root when `mkdocs build` runs. If MkDocs
cannot import a module, verify the `paths:` setting in `mkdocs.yml`.

 

## 🏗️ Building Documentation

Run:

```powershell
mkdocs build
```

For local preview:

```powershell
mkdocs serve
```

Open the local site at:

```text
http://127.0.0.1:8000/
```

If a page exists under `docs/` but is not included in the `nav:` section of `mkdocs.yml`, MkDocs may
warn that the page is not included in navigation. Add the page to `nav:` or remove the unused file.

 

## 🧰 Common MkDocs Fixes

### Missing `site_name`

If MkDocs reports:

```text
ERROR - Config value 'site_name': Required configuration not provided.
```

Ensure the top of `mkdocs.yml` includes:

```yaml
site_name: Jimi Documentation
```

### API reference only shows one file

Confirm each API page exists and is listed in `nav:`:

```yaml
nav:
  - API Reference:
      - Overview: api/index.md
      - App: api/app.md
      - Gemini: api/gemini.md
      - Configuration: api/config.md
      - Logging: api/boogr.md
```

Confirm each API page has the correct directive:

```markdown
::: gemini
```

### CSS or JavaScript not loading

Confirm paths are relative to the `docs/` directory:

```yaml
extra_css:
  - assets/css/jimi.css

extra_javascript:
  - assets/js/jimi.js
```

The files should be located at:

```text
docs/assets/css/jimi.css
docs/assets/js/jimi.js
```

### Image link not found

If Markdown references this:

```markdown
![Architecture](images/jimi-architecture.png)
```

The image must exist at:

```text
docs/images/jimi-architecture.png
```

If the image is in `docs/assets/images/`, reference it as:

```markdown
![Architecture](assets/images/jimi-architecture.png)
```

 

## 🌐 GitHub Pages Workflow

The recommended GitHub Pages workflow is to publish the generated documentation site from MkDocs.

Build locally first:

```powershell
mkdocs build
```

Then deploy with:

```powershell
mkdocs gh-deploy
```

This pushes the generated `site/` output to the `gh-pages` branch.

In GitHub repository settings:

1. Open the repository.
2. Go to **Settings**.
3. Select **Pages**.
4. Set the source to the `gh-pages` branch.
5. Save the configuration.

The documentation site should become available at the configured GitHub Pages URL.

 

## 🧪 Pre-Commit Checklist

Before committing changes, run:

```powershell
python -m py_compile .\app.py
python -m py_compile .\gemini.py
python -m py_compile .\config.py
python -m py_compile .\boogr.py
python -m compileall .
mkdocs build
```

Then verify:

```text
No syntax errors.
No missing imports.
No MkDocs nav warnings that matter.
No griffe warnings from malformed docstrings.
No missing CSS or JavaScript assets.
No broken image paths.
No missing logging in exception handlers.
No duplicate Error wrappers in exception handlers.
```

 

## 🧹 Development Hygiene

Follow these rules while editing the codebase:

* Keep UI orchestration in `app.py`.
* Keep Gemini SDK calls in `gemini.py`.
* Keep configuration in `config.py`.
* Keep logging infrastructure in `boogr.py`.
* Do not hard-code credentials.
* Do not commit local databases.
* Do not commit generated `site/` output unless the project intentionally tracks it.
* Do not introduce undocumented public functions.
* Do not use non-Google docstring headings.
* Do not add new exception handlers without logging.
* Do not change fallback behavior unless intentionally refactoring the workflow.

 

## 🧭 Recommended Development Sequence

When making significant changes, use this sequence:

1. Update or add the wrapper method in `gemini.py`.
2. Validate the method signature and docstring.
3. Add exception logging with the required pattern.
4. Add or update UI routing in `app.py`.
5. Preserve Streamlit session-state keys.
6. Compile all Python files.
7. Update Markdown documentation.
8. Run `mkdocs build`.
9. Test the Streamlit workflow manually.
10. Commit only after validation passes.

  

## 🚀 Release Checklist

Before publishing a release or pushing documentation to GitHub Pages:

```text
Source files compile.
Streamlit app starts.
Primary workflows run.
Exception logging writes to SQLite.
Documentation builds.
API reference renders all expected modules.
Dark theme CSS loads.
JavaScript enhancements load.
Images and icons resolve.
GitHub Pages points to the MkDocs site.
README links point to the documentation site.
```

A release should not be considered complete until both the application and documentation build
successfully.

 

## 🧭 Summary

The Jimi development workflow is built around controlled source changes, strict docstring
formatting, consistent exception logging, and repeatable documentation builds. The project remains
maintainable when the Streamlit interface, provider wrappers, configuration layer, and logging layer
stay separated.

The core development rule is simple:

```text
Preserve runtime behavior, document every method correctly, and log every handled exception exactly once.
```
