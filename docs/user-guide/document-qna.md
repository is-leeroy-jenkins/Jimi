# 📄 Document Q&A

Document Q&A allows Jimi to answer questions about uploaded documents, summarize source material,
extract relevant details, compare sections, and support prompt-driven review workflows. The feature
combines Streamlit upload handling in `app.py` with Gemini file and content-generation workflows in
`gemini.py`.

The goal is to let a user provide a document and ask targeted questions without manually copying the
full document into the prompt box.

 

## 🧭 Overview

Document Q&A is designed around a simple flow:

```text
User uploads a document
        │
        ▼
Jimi captures the file and prompt
        │
        ▼
app.py validates required inputs
        │
        ▼
gemini.py uploads or attaches the document
        │
        ▼
Gemini processes the document and question
        │
        ▼
Jimi displays the answer in Streamlit
```

The workflow keeps document-specific provider logic inside the `Files` wrapper while allowing the
Streamlit interface to focus on file selection, prompt capture, progress messages, and response
rendering.

 

## 🧱 Architecture

Document Q&A uses these project components:

| Component           | Module      | Role                                                                                                         |
| ------------------- | ----------- | ------------------------------------------------------------------------------------------------------------ |
| Streamlit UI        | `app.py`    | Accepts document uploads, captures questions, displays responses, and preserves session state.               |
| Gemini file wrapper | `gemini.py` | Uploads files, submits document prompts, retrieves responses, and handles provider-specific file operations. |
| Configuration       | `config.py` | Provides API keys, paths, model defaults, and validation helpers.                                            |
| Logging             | `boogr.py`  | Records structured exception diagnostics in SQLite.                                                          |

The key separation is that `app.py` should not build low-level Gemini file requests directly. It
should call methods on the `Files` wrapper.

 

## 📥 Supported Inputs

Document Q&A workflows typically support files such as:

| File Type | Common Use                                                                       |
| --------- | -------------------------------------------------------------------------------- |
| PDF       | Reports, policies, resumes, manuals, contracts, statements of work.              |
| TXT       | Plain-text notes, extracted content, logs, transcripts.                          |
| CSV       | Tabular records, exports, summaries, financial data.                             |
| Markdown  | Project documentation, README files, generated notes.                            |
| DOCX      | Draft documents, job announcements, technical documentation, business documents. |

Actual supported file types depend on the active Streamlit upload configuration and the provider
capabilities used by the selected wrapper method.

 

## 🧾 Typical Questions

Document Q&A is useful for questions such as:

```text
Summarize this document in plain English.
```

```text
What are the key requirements in this file?
```

```text
Extract the deadlines, responsible parties, and deliverables.
```

```text
Compare the applicant qualifications against the announcement.
```

```text
Find the sections related to budget, audit, internal controls, and reporting.
```

```text
Create a list of risks, assumptions, dependencies, and open issues.
```

```text
Generate a user-guide section based on this document.
```

 

## ⚙️ Basic Workflow

A normal Document Q&A session follows this sequence:

1. Open the Jimi application.
2. Select the document or file-analysis mode.
3. Upload a supported document.
4. Enter a question, instruction, or analysis prompt.
5. Select the Gemini model and optional parameters.
6. Submit the request.
7. Review the generated answer.
8. Refine the prompt if the answer needs more detail or a different format.

The uploaded file and prompt are treated as a single analysis request.

 

## 🧠 Prompting Guidance

Good document prompts are specific. They tell the model what to find, how to reason over the
document, and what format to return.

### Weak prompt

```text
Tell me about this.
```

### Stronger prompt

```text
Summarize the document in 8 bullets. Focus on purpose, scope, required actions,
deadlines, named organizations, risks, and any compliance requirements.
```

### Strong extraction prompt

```text
Extract every deadline, deliverable, responsible office, required document,
approval step, and reporting requirement. Return the results as a Markdown table.
```

### Strong comparison prompt

```text
Compare the document requirements against the following experience summary.
Identify direct matches, partial matches, gaps, and evidence that could be used
in a resume or cover letter.
```

 

## 📊 Recommended Output Formats

The user can ask for different output structures depending on the document review task.

| Output Format     | Best For                                           |
| ----------------- | -------------------------------------------------- |
| Bullets           | Fast summaries and executive reviews.              |
| Markdown table    | Requirements, deadlines, owners, and comparisons.  |
| Narrative summary | Plain-language explanation.                        |
| Checklist         | Compliance review or implementation planning.      |
| Risk register     | Project, policy, and contract review.              |
| Action plan       | Follow-up work and task assignment.                |
| Q&A format        | Training, interview preparation, and study guides. |

Example:

```text
Return the answer as a Markdown table with these columns:
Requirement | Source Section | Impact | Recommended Action
```

 

## 📄 Summarization Workflow

The `Files.summarize(...)` method is used when the user wants a document-level summary.

Conceptual flow:

```text
Document path
    │
    ▼
Files.summarize(prompt, filepath, model)
    │
    ▼
Gemini receives uploaded or attached file
    │
    ▼
Gemini generates summary
    │
    ▼
app.py displays summary
```

Typical summarization prompts:

```text
Summarize this document for a senior executive in no more than 10 bullets.
```

```text
Create a technical summary that preserves named systems, data sources,
dependencies, limitations, and implementation details.
```

```text
Create a plain-language summary for a non-technical reader.
```
 

## 🔎 Search Workflow

The `Files.search(...)` method is used when the user asks a targeted question against a document.

Conceptual flow:

```text
Question + document path
        │
        ▼
Files.search(prompt, filepath, model)
        │
        ▼
Gemini analyzes document against the question
        │
        ▼
Answer returned to app.py
```

Useful search prompts:

```text
Find every reference to internal controls and summarize the associated requirement.
```

```text
Identify whether this file contains budget formulation, budget execution,
audit remediation, or appropriations language.
```

```text
Find all named offices, agencies, systems, statutes, and reports.
```

 

## 📚 Multi-Document Review

The `Files.survey(...)` workflow can support analysis across multiple files when the UI provides a
list of file paths.

Conceptual flow:

```text
Multiple documents
        │
        ▼
Files.survey(prompt, filepaths, model)
        │
        ▼
Gemini processes grouped document context
        │
        ▼
Combined answer returned to app.py
```

Useful multi-document prompts:

```text
Compare these documents and identify overlapping requirements, conflicts,
unique requirements, and missing information.
```

```text
Create a consolidated action register from all uploaded documents.
```

```text
Summarize the common themes across these files and identify document-specific
exceptions.
```

 

## 🧩 Recommended Prompt Templates

### Executive summary

```text
Create an executive summary of the uploaded document.

Return:
1. Purpose
2. Scope
3. Key findings
4. Required actions
5. Deadlines
6. Risks
7. Recommended next steps
```

### Requirements extraction

```text
Extract all requirements from the document.

Return a Markdown table with:
Requirement | Responsible Party | Deadline | Source Section | Notes
```

### Compliance review

```text
Review the document for compliance obligations.

Identify:
- mandatory actions
- reporting requirements
- approval requirements
- required documentation
- missing or ambiguous requirements
- recommended controls
```

### Technical documentation conversion

```text
Convert the uploaded source material into project documentation.

Use:
- clear Markdown headings
- concise technical explanations
- examples where helpful
- no invented file names
- no unsupported claims
```

### Resume/job-announcement analysis

```text
Analyze the announcement and identify:
- specialized experience requirements
- technical competencies
- keywords to include in a resume
- cover letter themes
- evidence needed to support qualifications
```

 

## 🧾 Error Handling

Document workflows can fail for several reasons:

| Failure Type     | Example                                                 |
| ---------------- | ------------------------------------------------------- |
| Missing file     | User submits without uploading a document.              |
| Missing prompt   | User uploads a file but does not ask a question.        |
| Unsupported file | File type cannot be processed by the selected provider. |
| Provider error   | Gemini rejects the request or returns an API error.     |
| Credential error | API key is missing or invalid.                          |
| File path error  | Temporary file path does not exist or cannot be read.   |
| Size limit       | Document exceeds provider or application size limits.   |

Every handled exception should use the shared logging pattern:

```python
except Exception as e:
	ex = Error( e )
	ex.module = 'app'
	ex.cause = 'DocumentQnA'
	ex.method = 'render_document_qna_panel( )'
	Logger( ).write( ex )
	return None
```

Provider wrapper failures should log and raise when the method cannot safely continue:

```python
except Exception as e:
	ex = Error( e )
	ex.module = 'gemini'
	ex.cause = 'Files'
	ex.method = 'search( self, prompt, filepath, model )'
	Logger( ).write( ex )
	raise ex
```

 

 ## 🔐 Security and Privacy

Document Q&A may involve sensitive uploaded files. Treat document content, prompts, generated
answers, and file paths as potentially sensitive.

Recommended practices:

* Do not commit uploaded files to the repository.
* Do not commit local temporary files.
* Do not commit SQLite exception databases.
* Do not include secrets or private file paths in public documentation.
* Do not log full document content unless explicitly required.
* Delete temporary files when they are no longer needed.
* Use sanitized sample files for screenshots and documentation examples.
* Keep provider credentials in environment variables or local config excluded from Git.

 

## 🧪 Validation Checklist

Before relying on Document Q&A in development, verify:

```text
The app starts with streamlit run app.py.
The document mode renders correctly.
File upload controls accept the expected file types.
Prompt validation catches blank questions.
Missing-file validation is user friendly.
Gemini API keys are available.
The selected model supports the requested document workflow.
Exceptions are logged through boogr.py.
Recoverable UI errors do not crash the full app.
mkdocs build succeeds after docstring updates.
```

Run basic source validation:

```powershell
python -m py_compile .\app.py
python -m py_compile .\gemini.py
python -m py_compile .\config.py
python -m py_compile .\boogr.py
python -m compileall .
```

 

## 🧰 Troubleshooting

### The app says no file was uploaded

Confirm that the uploaded file is assigned to the expected Streamlit state key and that the workflow
handler receives either the file object or a valid temporary file path.

### Gemini cannot process the file

Check the file type, file size, model selection, and upload method. Some workflows require a
provider-uploaded file, while others can use bytes or local file paths.

### The response is too generic

Use a more specific prompt. Ask for named fields, tables, source sections, or required output
columns.

### The answer misses details

Ask the model to extract every instance of a term or return a table with source references. For long
documents, consider chunking or asking section-specific questions.

### The workflow crashes

Check the SQLite exception log configured through `boogr.py`. The logged `module`, `cause`,
`method`, `message`, and `trace` fields should identify the failing component.

 

## 📌 Best Practices

Use Document Q&A for focused analysis rather than vague review. The best results come from prompts
that define the task, expected scope, and output format.

Recommended practice:

```text
Ask a specific question.
Specify the output format.
Request source sections when needed.
Ask for gaps or uncertainties.
Use follow-up prompts to refine the answer.
```

Avoid:

```text
Vague prompts.
Uploading unnecessary sensitive data.
Assuming the model saw every detail perfectly.
Treating generated answers as final without review.
Committing uploaded files or logs.
```

---

## 🧭 Summary

Document Q&A gives Jimi a practical workflow for analyzing uploaded files through Gemini-backed
document operations. The feature is built around a clean separation between the Streamlit UI,
provider wrapper methods, configuration settings, and durable exception logging.

The operational rule is:

```text
Use app.py for user interaction, gemini.py for document-provider operations, config.py for runtime settings, and boogr.py for diagnostics.
```
