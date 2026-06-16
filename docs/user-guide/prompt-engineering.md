# ✍️ Prompt Engineering

Prompt engineering in Jimi is the practice of writing clear, structured instructions that help the
application produce accurate, useful, and reviewable output. Because Jimi supports text generation,
document Q&A, semantic search, data workflows, and documentation generation, strong prompts are
essential for controlling scope, format, assumptions, and output quality.

A good prompt tells Jimi:

1. What task to perform.
2. What context to use.
3. What constraints to follow.
4. What format to return.
5. What not to invent.

 

## 🧭 Overview

Jimi works best when prompts are explicit and structured. Vague prompts can produce generic output,
while clear prompts produce responses that are easier to inspect, validate, and reuse.

Weak prompt:

```text
Tell me about this.
```

Stronger prompt:

```text
Summarize the uploaded document in 8 bullets. Focus on purpose, scope,
requirements, deadlines, risks, responsible parties, and recommended next steps.
Do not invent details that are not supported by the document.
```

The second prompt gives the model a defined task, source boundary, content priorities, and quality
constraint.

 

## 🧱 Prompt Structure

Use this structure for most Jimi workflows:

```text
Task:
Context:
Constraints:
Output format:
```

Example:

```text
Task:
Generate a technical summary of the uploaded source file.

Context:
The summary will be used in MkDocs documentation for a Streamlit AI application.

Constraints:
Do not invent files, classes, features, or external services.
Preserve project-specific terminology.
Mention limitations or assumptions when the source is unclear.

Output format:
Return Markdown with headings, bullets, and a short summary table.
```

This structure reduces ambiguity and helps produce consistent output across workflows.

 

## 🎯 Task

The task tells Jimi what to do. It should be specific and action-oriented.

Good task statements:

```text
Summarize the uploaded document.
```

```text
Extract all requirements from the source file.
```

```text
Generate a Markdown user-guide page.
```

```text
Compare this job announcement against my experience.
```

```text
Convert these comments into Google-style docstrings.
```

Weak task statements:

```text
Review this.
```

```text
Make it better.
```

```text
What do you think?
```

A vague task forces the model to guess the desired output.

 

## 📚 Context

Context explains how the result will be used and what source material matters.

Examples:

```text
Context:
This page will be added to a MkDocs Material documentation site.
```

```text
Context:
The source file belongs to a Streamlit application that uses Gemini wrappers,
SQLite logging, and Google-style docstrings.
```

```text
Context:
The output will be copied directly into docs/user-guide/document-qna.md.
```

Context helps the model select the correct tone, level of detail, terminology, and structure.

 

## 🚧 Constraints

Constraints define boundaries. They are especially important for documentation, code, and
file-analysis workflows.

Useful constraints:

```text
Do not invent file names.
```

```text
Do not introduce modules that do not exist.
```

```text
Preserve existing behavior.
```

```text
Use Google-style docstrings only.
```

```text
Do not use Parameters: or Return: headings.
```

```text
Use icons only at heading level.
```

```text
Keep examples specific to this project.
```

```text
If the source is unclear, state the limitation instead of guessing.
```

Constraints make the output safer to reuse.

 

## 📐 Output Format

The output format tells Jimi how to arrange the answer.

Common formats include:

| Format               | Best For                                                |
| -------------------- | ------------------------------------------------------- |
| Bullets              | Quick summaries and executive notes.                    |
| Markdown             | Documentation pages, README sections, guides.           |
| Tables               | Requirements, comparisons, checklists, mappings.        |
| Numbered steps       | Setup instructions and procedures.                      |
| Code blocks          | Source files, config files, command examples.           |
| JSON-like structures | Schemas, structured extraction, configuration examples. |
| Checklists           | Validation, release readiness, QA review.               |

Example:

```text
Output format:
Return a Markdown table with these columns:
Requirement | Source Evidence | Implementation Impact | Recommended Action
```

 

## 🧾 Documentation Prompts

Jimi is frequently used to generate MkDocs documentation. Documentation prompts should specify file
location, audience, scope, and formatting rules.

Example:

```text
Generate docs/user-guide/semantic-search.md for the Jimi MkDocs site.

Audience:
Users and developers who need to understand semantic search workflows.

Include:
- Overview
- Architecture
- Common use cases
- Prompt examples
- Validation checklist
- Troubleshooting
- Summary

Rules:
- Use Markdown.
- Use icons only at heading level.
- Do not invent modules or files.
- Keep terminology consistent with app.py, gemini.py, config.py, and boogr.py.
```

 

## 🧩 Source Comment Prompts

For source-comment generation, prompts must be strict. The goal is to produce comments compatible
with MkDocs and mkdocstrings without changing runtime behavior.

Recommended prompt:

```text
Regenerate the attached Python file with:
1. Google-style docstrings compatible with MkDocs and mkdocstrings.
2. Robust Purpose sections for every function and class.
3. No Parameters: headings.
4. No Return: headings.
5. No underline-style docstring sections.
6. No Returns: section in __init__ methods.
7. No self or cls in Args sections.
8. Existing runtime behavior preserved.
9. Existing signatures preserved.
10. Exception handlers updated to use one Error object, one Logger write,
    and the same object raised or handled according to the original control flow.
```

This prompt is useful when regenerating files such as `app.py`, `gemini.py`, `config.py`, or
`boogr.py`.

 

## 🧾 Required Docstring Style

Use this docstring style:

```python
def example_method( value: str, limit: int = 10 ) -> list[str]:
	"""
	Purpose:
		Builds a normalized list of values from validated input for downstream
		application workflows.

	Args:
		value: Source text used to create the returned values.
		limit: Maximum number of values to return.

	Returns:
		List of normalized values.

	Raises:
		Error: Wraps and logs validation or processing failures.
	"""
```

Avoid this style:

```python
def example_method( value: str ) -> list[str]:
	"""
	Purpose
	-------
	Does something.

	Parameters:
	-----------
	value: str

	Return:
	-------
	list[str]
	"""
```

The second format can produce poor rendering or warnings during documentation builds.

 

## 🧠 Document Q&A Prompts

Document Q&A prompts should tell Jimi what to extract, summarize, compare, or verify.

### Summary prompt

```text
Summarize the uploaded document in no more than 10 bullets.

Focus on:
- purpose
- scope
- requirements
- deadlines
- responsible parties
- risks
- recommended next steps

Do not include unsupported assumptions.
```

### Requirements extraction prompt

```text
Extract every requirement from the uploaded document.

Return a Markdown table with:
Requirement | Responsible Party | Deadline | Source Section | Notes
```

### Compliance review prompt

```text
Review the uploaded document for compliance obligations.

Identify:
- mandatory actions
- reporting requirements
- approvals
- required documentation
- ambiguous requirements
- recommended internal controls
```

### Comparison prompt

```text
Compare the uploaded document against the following experience summary.

Return:
- Direct matches
- Partial matches
- Gaps
- Strong keywords
- Evidence that can be used in a resume or cover letter
```

 

## 🔎 Semantic Search Prompts

Semantic search prompts should distinguish meaning-based search from exact keyword search.

Keyword prompt:

```text
Find every passage that contains the phrase "internal controls."
```

Semantic prompt:

```text
Find passages that are conceptually related to audit remediation,
financial oversight, risk mitigation, control testing, or corrective actions,
even when those exact words are not used.
```

Ranking prompt:

```text
Rank the following document sections by relevance to budget formulation,
budget execution, appropriations law, and financial reporting.
Return the top 10 sections with a short justification for each.
```

 

## 🧰 Data Workflow Prompts

Use data workflow prompts when working with files, embeddings, logs, storage, or records.

Example:

```text
Analyze the uploaded CSV.

Return:
- column names
- likely data types
- missing values
- duplicate rows
- candidate key fields
- quality issues
- recommended cleaning steps
```

Example:

```text
Review the exception log structure.

Identify:
- required columns
- diagnostic value of each field
- missing fields that would improve troubleshooting
- recommendations for indexing or cleanup
```

 

## 🧱 Architecture Prompts

Architecture prompts should specify the application boundaries and what should be included.

Example:

```text
Generate architecture.md for the Jimi documentation site.

Include:
- system overview
- module responsibilities
- request flow
- configuration flow
- logging architecture
- document workflow
- Gemini wrapper layer
- deployment considerations

Rules:
- Use Markdown.
- Use icons only at heading level.
- Do not invent components.
- Keep the description consistent with app.py, gemini.py, config.py, and boogr.py.
```

 

## 🖼️ Diagram Prompts

When generating architecture or class diagrams, specify the style, aspect ratio, and content
boundary.

Example:

```text
Generate a dark-mode 16:9 architecture diagram for Jimi.

Include only:
- Streamlit UI
- app.py
- gemini.py wrappers
- config.py
- boogr.py logging
- Google Gemini services
- Google Cloud Storage
- SQLite Exceptions database

Style:
- dark background
- blue accents
- clean boxes
- readable labels
- no screenshots
- no extra features not present in the project
```

 

## 🧪 Validation Prompts

Validation prompts are useful when reviewing generated source files or documentation.

Example:

```text
Audit this generated Python file.

Check:
- Python syntax
- Google-style docstrings
- no Parameters headings
- no Return headings
- no underline-style docstring sections
- no Returns section in __init__
- every except Exception handler logs exactly once
- no duplicate Error wrappers
- existing control flow preserved

Return a concise table of findings.
```

 

## 🧭 Troubleshooting Poor Output

### Output is too generic

Add context and constraints.

Weak:

```text
Write docs.
```

Better:

```text
Generate docs/user-guide/text-generation.md for the Jimi MkDocs site.
Focus on Streamlit text workflows, Gemini Chat wrapper usage, prompt examples,
settings, validation, troubleshooting, and best practices.
```

### Output invents files or features

Add a strict boundary.

```text
Do not invent modules, files, classes, workflows, APIs, or screenshots.
Use only the provided source files and project context.
```

### Output is not formatted correctly

Specify the exact format.

```text
Return only Markdown.
Use ## headings.
Use tables where helpful.
Use fenced code blocks for commands.
```

### Output misses validation

Ask for a validation section.

```text
Include a validation checklist with py_compile, compileall, and mkdocs build commands.
```

### Output changes code behavior

Explicitly prohibit behavior changes.

```text
Preserve runtime behavior exactly.
Do not change function signatures.
Do not change return behavior.
Do not convert fallback handlers into hard failures.
```

 

## ✅ Prompt Checklist

Before submitting a prompt, check whether it includes:

```text
Clear task
Relevant context
Source boundary
Output format
Constraints
Validation expectations
What not to invent
Whether runtime behavior must be preserved
```

A strong prompt does not need to be long, but it must be specific.

 

## 📌 Reusable Prompt Templates

### Generate a user-guide page

```text
Generate docs/user-guide/<page-name>.md for the Jimi MkDocs site.

Include:
- overview
- architecture or workflow
- usage examples
- prompt examples where relevant
- validation checklist
- troubleshooting
- best practices
- summary

Rules:
- Use Markdown.
- Use icons only at heading level.
- Do not invent files, classes, or workflows.
- Keep terminology consistent with app.py, gemini.py, config.py, and boogr.py.
```

### Generate an API page

```text
Generate docs/api/<module-name>.md for the Jimi MkDocs site.

Include:
- short module overview
- mkdocstrings directive
- notes on how the module is used
- validation notes

Use this directive:
::: <module-name>
    options:
      show_root_heading: true
      show_source: true
      members_order: source
```

### Regenerate a Python source file

```text
Regenerate the attached Python file.

Requirements:
1. Preserve runtime behavior.
2. Preserve function and method signatures.
3. Add Google-style docstrings to every function, method, and class.
4. Use robust Purpose sections.
5. Use Args, Attributes, Returns, Raises, Notes, and Examples only where appropriate.
6. Do not document self or cls in Args.
7. Do not use Parameters or Return headings.
8. Do not put Returns in __init__ methods.
9. Implement the logging pattern in every exception handler.
10. Log recoverable handlers but preserve their fallback behavior.
11. Verify the final artifact before providing it.
```

 

## 🧭 Summary

Prompt engineering in Jimi is about reducing ambiguity. The most reliable prompts define the task,
context, constraints, and output format while explicitly preventing unsupported assumptions.

The practical rule is:

```text
Tell Jimi exactly what to do, what to use, what to avoid, and how to return the result.
```
