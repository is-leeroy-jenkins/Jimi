# 💬 Text Generation

Text generation is the core Jimi workflow for producing, transforming, summarizing, analyzing, and
structuring written output from user prompts. It is implemented through the Streamlit interface in
`app.py` and the Gemini provider wrapper in `gemini.py`.

Use text generation when the task starts with a prompt and returns written output such as Markdown,
summaries, explanations, tables, plans, documentation, analysis, or rewritten text.



## 🧭 Overview

A text-generation workflow begins when the user enters a prompt, selects a model, configures
optional generation settings, and submits the request. Jimi validates the input, builds the provider
request, sends it through the `Chat` wrapper, receives the model response, and renders the generated
text in the Streamlit interface.

```text
User prompt
    │
    ▼
app.py captures input and options
    │
    ▼
Chat.generate_text(...)
    │
    ▼
Gemini content generation request
    │
    ▼
Generated text response
    │
    ▼
app.py renders output
```

The application layer should remain responsible for the interface and workflow routing. The provider
wrapper should remain responsible for Gemini client creation, request configuration, tool selection,
response parsing, and exception logging.


## 🧱 Architecture

Text generation uses these Jimi components:

| Component     | Module      | Responsibility                                                               |
| ------------- | ----------- | ---------------------------------------------------------------------------- |
| Streamlit UI  | `app.py`    | Captures prompts, model selections, generation options, and displays output. |
| Chat wrapper  | `gemini.py` | Builds Gemini content requests and returns generated text.                   |
| Configuration | `config.py` | Provides API keys, model defaults, validation helpers, and runtime settings. |
| Logging       | `boogr.py`  | Captures exception metadata and writes diagnostics to SQLite.                |

The main provider-facing class is `Chat`. Its primary text-generation method is
`generate_text(...)`.

## 🔁 Basic Request Flow

A standard request follows this sequence:

1. User enters a prompt.
2. User selects a Gemini model.
3. User optionally configures generation settings.
4. `app.py` validates the prompt.
5. `app.py` passes the prompt and options to `Chat.generate_text(...)`.
6. `Chat` builds the content payload.
7. `Chat` builds `GenerateContentConfig`.
8. `Chat` resolves the API key and creates the GenAI client.
9. Gemini returns a response.
10. `Chat` extracts output text.
11. `app.py` renders the response.

Conceptual flow:

```text
Prompt + Options
        │
        ▼
Input validation
        │
        ▼
Content construction
        │
        ▼
Generation configuration
        │
        ▼
Gemini API request
        │
        ▼
Response extraction
        │
        ▼
Streamlit display
```


## 🧠 Chat Wrapper

The `Chat` class in `gemini.py` owns most of the text-generation behavior. It normalizes the user
prompt, optional context, content blocks, URLs, model parameters, tools, and response settings into
a provider request.

Typical responsibilities include:

* Resolving the active Gemini API key.
* Validating prompt input.
* Building conversation content.
* Adding optional context.
* Appending URL context when supplied.
* Building tool objects.
* Building generation configuration.
* Supporting streaming output.
* Extracting final text from the provider response.
* Logging provider or parsing failures.

Conceptual usage:

```python
from gemini import Chat

chat = Chat()
response = chat.generate_text(
	prompt="Summarize this architecture in five bullets.",
	model="gemini-2.5-flash-lite"
)
```


## ⚙️ Model Selection

Text generation depends on the selected Gemini model. The model determines capability, speed, cost,
context behavior, and tool support.

Common model-selection considerations:

| Selection Factor | Consideration                                                         |
| ---------------- | --------------------------------------------------------------------- |
| Speed            | Use lighter models for quick drafting, summaries, and iterative work. |
| Reasoning depth  | Use stronger models for complex analysis or planning.                 |
| Tool support     | Confirm the selected model supports requested tools.                  |
| Context size     | Use models with appropriate context capacity for long prompts.        |
| Cost             | Prefer smaller models for repetitive development tasks.               |
| Output type      | Choose models that support the desired response modality.             |

Jimi exposes model choices through the Streamlit interface and wrapper option lists.



## 🎛️ Generation Settings

Generation settings control how the model responds.

| Setting           | Purpose                                                                                  |
| ----------------- | ---------------------------------------------------------------------------------------- |
| `temperature`     | Controls randomness. Lower values are more deterministic; higher values are more varied. |
| `top_p`           | Controls nucleus sampling by limiting candidate tokens by probability mass.              |
| `top_k`           | Limits token selection to the top candidate tokens.                                      |
| `max_tokens`      | Limits maximum generated output length.                                                  |
| `frequency`       | Penalizes repeated tokens or phrases.                                                    |
| `presence`        | Encourages or discourages introducing new topics.                                        |
| `stops`           | Defines stop sequences that end generation.                                              |
| `instruct`        | Provides system-level instruction.                                                       |
| `response_format` | Requests a response MIME type where supported.                                           |

Practical defaults:

| Goal                 | Suggested Behavior                                |
| -------------------- | ------------------------------------------------- |
| Documentation        | Lower temperature, clear formatting instructions. |
| Brainstorming        | Moderate temperature, fewer constraints.          |
| Code-adjacent output | Lower temperature, explicit constraints.          |
| Summarization        | Low to moderate temperature, defined length.      |
| Creative drafting    | Moderate to higher temperature.                   |
| Structured tables    | Lower temperature and exact column instructions.  |



## 🧾 Prompt Structure

Text generation works best when the prompt defines the task, context, constraints, and output
format.

Recommended structure:

```text
Task:
Context:
Constraints:
Output format:
```

Example:

```text
Task:
Generate a Markdown section for the Jimi documentation.

Context:
The page explains text-generation workflows in a Streamlit AI assistant.

Constraints:
Do not invent files, classes, models, or unsupported features.
Use terminology consistent with app.py, gemini.py, config.py, and boogr.py.

Output format:
Use Markdown with headings, tables, examples, troubleshooting, and a summary.
```

This format gives Jimi enough instruction to produce output that is easier to validate and reuse.


## ✍️ Common Text-Generation Tasks

### Summarization

```text
Summarize the following content in 8 bullets.
Focus on purpose, workflow, dependencies, risks, and next steps.
```

### Markdown documentation

```text
Generate a Markdown user-guide page for this workflow.
Use icons only at heading level.
Include overview, architecture, usage, examples, troubleshooting, and summary.
```

### Rewriting

```text
Rewrite the following section to be clearer and more concise.
Preserve technical meaning and do not remove required details.
```

### Extraction

```text
Extract all requirements from the following text.
Return a Markdown table with Requirement, Owner, Deadline, Source, and Notes.
```

### Comparison

```text
Compare these two sections.
Return matches, differences, gaps, and recommended revisions.
```

### Planning

```text
Create an implementation plan.
Include phases, tasks, dependencies, risks, validation steps, and deliverables.
```


## 📊 Recommended Output Formats

Structured output is easier to review than unstructured prose.

| Format          | Best For                                     |
| --------------- | -------------------------------------------- |
| Bullets         | Summaries, findings, key points.             |
| Numbered steps  | Procedures, installation, deployment.        |
| Markdown tables | Requirements, comparisons, mappings, audits. |
| Checklists      | Validation, readiness, QA, release review.   |
| Code blocks     | Commands, config files, source snippets.     |
| Headed sections | Documentation pages and reports.             |

Example prompt:

```text
Return a Markdown table with these columns:
Step | Action | Owner | Validation | Notes
```


## 🧰 Tool-Grounded Generation

When supported by the selected model, Jimi can enable tools such as Google Search, URL context, code
execution, or maps-related grounding. Tool support depends on the active model and wrapper
implementation.

Tool-grounded generation is useful when:

* The answer depends on current information.
* The prompt references a URL or external source.
* The response should include grounded context.
* The workflow requires code execution support.
* The user needs location or maps-related context.

For source-code documentation workflows, avoid tool use unless current external information is
required. Project documentation should primarily rely on the actual source files and project
context.


## 🌐 URL Context

When URL context is available, Jimi can append URLs or URL-derived context to the request. This is
useful when the user wants the model to consider a specific web page or documentation page.

Recommended prompt:

```text
Use the provided URL context to answer the question.
Do not rely on unrelated sources.
If the URL content is insufficient, say what is missing.
```

URL-based workflows should clearly identify whether the source is being used as primary evidence or
supplemental context.


## 🔄 Streaming Output

Some text-generation workflows may support streaming. Streaming is useful for long responses because
it allows partial output to appear before the full response is complete.

Streaming is best for:

* Long summaries.
* Draft documents.
* Documentation generation.
* Extended analysis.
* Step-by-step output.

Streaming should still preserve error handling. If streaming fails, the exception should be logged
through `boogr.py`, and the UI should display a recoverable message when appropriate.


## 🧾 Conversation Context

Jimi can build structured conversation history so follow-up prompts can reuse prior exchanges. This
is useful when a workflow requires iterative refinement.

Example sequence:

```text
Prompt 1:
Generate architecture.md.

Prompt 2:
Make the logging architecture more detailed.

Prompt 3:
Add a validation checklist.
```

Conversation context is helpful, but long context can introduce ambiguity. When precision matters,
restate the required constraints in the follow-up prompt.


## 🧪 Validation Checklist

Before using text-generation output as final content, verify:

```text
The response follows the requested format.
The response does not invent files or features.
Project-specific terminology is correct.
Source constraints were followed.
Any generated commands are appropriate for the environment.
Any code snippets are syntactically plausible.
Any documentation paths match mkdocs.yml.
Sensitive information is not included.
The output can be copied into the intended file without structural edits.
```

For source or documentation workflows, run:

```powershell
python -m py_compile .\app.py
python -m py_compile .\gemini.py
python -m py_compile .\config.py
python -m py_compile .\boogr.py
python -m compileall .
mkdocs build
```


## 🧾 Exception Handling

Text-generation failures can occur because of missing prompts, missing API keys, unsupported models,
invalid tool configuration, provider errors, response parsing issues, or network failures.

Provider-level failures should log and raise:

```python
except Exception as e:
	ex = Error( e )
	ex.module = 'gemini'
	ex.cause = 'Chat'
	ex.method = 'generate_text( self, prompt, model )'
	Logger( ).write( ex )
	raise ex
```

UI-level recoverable failures should log and preserve fallback behavior:

```python
except Exception as e:
	ex = Error( e )
	ex.module = 'app'
	ex.cause = 'TextGeneration'
	ex.method = 'render_text_generation_panel( )'
	Logger( ).write( ex )
	st.warning( str( ex ) )
	return None
```

The rule is:

```text
One Error object.
One Logger write.
The same object is logged and raised or handled.
The original control flow is preserved.
```



## 🧰 Troubleshooting

### The output is too vague

Make the task more specific and define the output format.

Weak:

```text
Write something about this.
```

Better:

```text
Generate a Markdown section explaining the Chat wrapper request flow.
Include a table of settings and a troubleshooting section.
```

### The output invents details

Add stronger source boundaries.

```text
Use only the provided source files and project context.
Do not invent modules, files, classes, services, or screenshots.
```

### The output is too long

Set a length limit.

```text
Return no more than 8 bullets.
```

### The output is too short

Specify required sections.

```text
Include overview, workflow, examples, troubleshooting, validation, and summary.
```

### The model ignores formatting

Restate the desired format at the end of the prompt.

```text
Return only Markdown.
Do not include commentary before or after the Markdown.
```

### The provider call fails

Check:

* API key configuration.
* Selected model.
* Tool support.
* Network access.
* Provider quota.
* Exception logs.


## 📌 Best Practices

Use these practices for reliable text generation:

```text
Use a specific task.
Provide relevant context.
Set clear constraints.
Request a precise output format.
Tell Jimi what not to invent.
Use lower temperature for technical work.
Validate generated code or commands.
Review all documentation before committing it.
Use follow-up prompts for controlled refinement.
```

Avoid:

```text
Vague prompts.
Unbounded requests.
Mixing unrelated tasks in one prompt.
Asking for exact source-dependent output without providing source context.
Using generated text as final without review.
```

 

## 🧭 Summary

Text generation is Jimi’s most flexible workflow. It supports drafting, summarization, rewriting,
documentation, planning, analysis, and structured extraction. The most reliable results come from
prompts that define the task, context, constraints, and output format.

The operational rule is:

```text
Be specific about what Jimi should do, what it should use, what it should avoid, and how it should return the result.
```
