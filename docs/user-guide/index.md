# 📘 User Guide

The Jimi User Guide explains how to use the application’s major workflows, including text
generation, document Q&A, semantic search, prompt engineering, and data management. It is written
for users, developers, and maintainers who need to understand how the Streamlit interface connects
to the underlying Gemini wrappers, configuration settings, file workflows, and logging system.

Jimi is designed as a Streamlit-based AI assistant that provides a practical interface for working
with prompts, documents, search workflows, embeddings, media operations, and provider-backed AI
services.

 

## 🧭 Guide Overview

The user guide is organized by workflow.

| Section                                     | Purpose                                                                            |
| ------------------------------------------- | ---------------------------------------------------------------------------------- |
| [Text Generation](text-generation.md)       | Explains how to generate, refine, and manage text responses.                       |
| [Document Q&A](document-qna.md)             | Explains how to upload files and ask questions about documents.                    |
| [Semantic Search](semantic-search.md)       | Explains embedding-backed search and similarity workflows.                         |
| [Prompt Engineering](prompt-engineering.md) | Explains how to write effective prompts for Jimi workflows.                        |
| [Data Management](data-management.md)       | Explains files, embeddings, runtime state, storage, logs, and configuration paths. |

Each page focuses on practical use, recommended workflows, common mistakes, and troubleshooting.

 

## 🚀 Starting the Application

Run Jimi from the project root:

```powershell
streamlit run app.py
```

The application normally opens in a browser at:

```text
http://localhost:8501
```

The Streamlit interface provides the primary controls for:

* Selecting a model.
* Choosing a workflow mode.
* Entering prompts.
* Uploading files.
* Adjusting model options.
* Viewing generated outputs.
* Reviewing workflow messages and errors.

 

## 🧱 Primary Workflows

Jimi supports several major workflow categories.

### Text Generation

Use text generation for general prompting, drafting, summarization, analysis, reasoning,
transformation, and structured output.

Common tasks include:

* Drafting Markdown content.
* Summarizing technical material.
* Rewriting text.
* Generating examples.
* Creating tables.
* Producing implementation notes.
* Asking grounded questions when supported tools are enabled.

See: [Text Generation](text-generation.md)

 

### Document Q&A

Use Document Q&A when you want to upload a file and ask questions about it.

Common tasks include:

* Summarizing a PDF.
* Extracting requirements.
* Reviewing a job announcement.
* Comparing a document to a resume or project description.
* Identifying deadlines, responsible parties, risks, and actions.
* Creating documentation from source material.

See: [Document Q&A](document-qna.md)

 

### Semantic Search

Use semantic search when the goal is meaning-based retrieval rather than exact keyword matching.

Common tasks include:

* Comparing text chunks.
* Finding conceptually similar passages.
* Ranking document sections.
* Supporting retrieval-style workflows.
* Creating embedding-backed search features.

See: [Semantic Search](semantic-search.md)

 

### Prompt Engineering

Use the prompt engineering guide when a workflow needs better instructions, stronger output
formatting, tighter constraints, or more reliable results.

Common tasks include:

* Turning vague prompts into specific prompts.
* Requesting tables, bullets, checklists, or JSON-like structures.
* Asking for source-grounded extraction.
* Creating reusable prompt templates.
* Improving document analysis prompts.

See: [Prompt Engineering](prompt-engineering.md)

 

### Data Management

Use the data-management guide to understand how Jimi handles files, embeddings, runtime state,
storage objects, configuration paths, and exception logs.

Common topics include:

* Uploaded files.
* Temporary paths.
* Gemini file operations.
* Google Cloud Storage-backed storage patterns.
* SQLite exception logs.
* `.gitignore` hygiene.
* Security and privacy practices.

See: [Data Management](data-management.md)

 

## ⚙️ Basic Operating Pattern

Most Jimi workflows follow the same pattern:

```text
Choose workflow
      │
      ▼
Select model and options
      │
      ▼
Provide prompt or upload file
      │
      ▼
Submit request
      │
      ▼
Review output
      │
      ▼
Refine prompt or continue workflow
```

The interface is designed so that users can start with a simple prompt and then add more structure
as needed.

 

## 🧠 Choosing the Right Workflow

Use this table to choose the correct guide page.

| Goal                                 | Recommended Section                         |
| ------------------------------------ | ------------------------------------------- |
| Generate an answer from a prompt     | [Text Generation](text-generation.md)       |
| Summarize or question a document     | [Document Q&A](document-qna.md)             |
| Compare meaning across text passages | [Semantic Search](semantic-search.md)       |
| Improve weak prompts                 | [Prompt Engineering](prompt-engineering.md) |
| Understand files, logs, and storage  | [Data Management](data-management.md)       |
| Understand source-level APIs         | [API Reference](../api/index.md)            |
| Understand the application design    | [Architecture](../architecture.md)          |
| Set up or maintain the project       | [Development](../development.md)            |

 

## 🧾 Recommended Prompt Structure

Most workflows improve when the prompt includes four elements:

```text
Task
Context
Constraints
Output format
```

Example:

```text
Review the uploaded document for implementation requirements.

Context:
This document will be used to create project documentation.

Constraints:
Do not invent file names or unsupported features.
Preserve named systems, modules, and workflows.

Output format:
Return a Markdown table with Requirement, Source Area, Impact, and Recommended Action.
```

A structured prompt reduces ambiguity and helps produce a response that is easier to validate.

 

## 📄 Working With Files

When using file-based workflows:

1. Upload only the file needed for the current task.
2. Ask a specific question about the file.
3. Specify the desired output format.
4. Review the response for missing or unsupported details.
5. Avoid uploading sensitive material unless the environment is approved for that use.

Good document prompt:

```text
Extract every requirement, deadline, responsible office, and deliverable from the uploaded file.
Return the result as a Markdown table.
```

Weak document prompt:

```text
What is this?
```

 

## 🔎 Working With Search and Retrieval

Search workflows work best when the user distinguishes between keyword search and meaning-based
search.

Keyword-style request:

```text
Find every reference to internal controls.
```

Semantic-style request:

```text
Find sections that are conceptually related to audit remediation, even if the words audit remediation are not used.
```

Semantic search is useful when the same idea may appear under different terms.

 

## 🗣️ Working With Generated Output

Review generated output before using it as final work product. For technical and documentation
workflows, verify that the response:

* Uses the correct file names.
* Does not invent modules or workflows.
* Preserves project-specific terminology.
* Matches the requested output format.
* Does not omit required constraints.
* Does not include unsupported claims.
* Is consistent with the source files and documentation.

For high-value documentation, generate smaller sections and validate each one before proceeding.

---

## 🧪 Validation for Developers

Developers should validate source and documentation after significant changes.

```powershell
python -m py_compile .\app.py
python -m py_compile .\gemini.py
python -m py_compile .\config.py
python -m py_compile .\boogr.py
python -m compileall .
mkdocs build
```

These checks help confirm that source files compile and that MkDocs can generate the documentation
site.

 

## 🧾 Error Handling and Logs

Jimi uses a shared exception logging pattern through `boogr.py`. When a workflow fails, the
application should log the failure with module, cause, method, message, and traceback details.

The standard logging pattern is:

```python
except Exception as e:
	ex = Error( e )
	ex.module = 'app'
	ex.cause = 'WorkflowName'
	ex.method = 'method_name( ... )'
	Logger( ).write( ex )
	return None
```

Provider-level failures may raise after logging:

```python
except Exception as e:
	ex = Error( e )
	ex.module = 'gemini'
	ex.cause = 'Files'
	ex.method = 'search( self, prompt, filepath, model )'
	Logger( ).write( ex )
	raise ex
```

This gives developers a durable diagnostic record while preserving the intended behavior of
recoverable UI workflows.

---

## 🔐 Security and Privacy

Treat prompts, uploaded files, file paths, outputs, and logs as potentially sensitive.

Recommended practices:

* Do not commit uploaded files.
* Do not commit local SQLite logs.
* Do not commit `.env` files.
* Do not hard-code API keys.
* Do not expose private paths in public documentation.
* Use sanitized examples in screenshots and docs.
* Avoid logging full document content unless required.
* Keep generated documentation separate from runtime artifacts.

 

## 🧰 Troubleshooting

### The app does not start

Run:

```powershell
python -m py_compile .\app.py
```

Then confirm the virtual environment is active and dependencies are installed:

```powershell
pip install -r requirements.txt
```

### The model request fails

Check:

* API key availability.
* Selected model name.
* Provider configuration.
* Network access.
* Exception logs.

### The document workflow fails

Check:

* File type.
* File size.
* Temporary file path.
* Prompt presence.
* Provider support for the selected model.

### The output is too vague

Rewrite the prompt with:

* A clear task.
* A specific scope.
* Constraints.
* Required output format.

### The documentation build fails

Run:

```powershell
mkdocs build
```

Then check for:

* Missing pages in `nav:`.
* Broken image paths.
* Bad mkdocstrings directives.
* Source files that cannot be imported.
* Legacy docstring sections such as `Parameters:` or `Return:`.

 

## 📌 Best Practices

Use Jimi as an iterative workflow tool:

1. Start with a focused request.
2. Ask for a structured output.
3. Review the answer.
4. Refine the prompt.
5. Save or copy the useful result.
6. Validate technical output before committing it.

For documentation work, generate one page or section at a time. This makes it easier to inspect
content, correct terminology, and keep the documentation consistent with the source code.

 

## 🧭 Summary

The Jimi User Guide is organized around practical workflows. Use text generation for prompt-based
output, Document Q&A for uploaded-file analysis, semantic search for meaning-based retrieval, prompt
engineering for better instructions, and data management for files, logs, storage, and
configuration.

The operating rule is:

```text
Choose the workflow, provide specific inputs, request a clear output format, and validate the result before using it.
```
