###### Jimi
![](https://github.com/is-leeroy-jenkins/Jimi/blob/main/resources/images/jimi_project.png)

A Python framework for building, deploying, and managing AI-powered assistants
tailored for federal data analysis, budget execution, and data science. Jimi integrates OpenAI's GPT, 
Google's Gemini API with multimodal support for text, image, audio, and file analysis 
and is designed with extensibility and federal applications in mind, it enables secure, scalable, and intelligent
automation of analytical tasks.

## 🛠️ Features

- Unified AI Framework: Integrates OpenAI APIs for text, image, audio, file analysis, transcription,
  and translation.
- Multimodal Capabilities: Supports text generation, image creation, image analysis, and document
  summarization.
- Vector Store Integration: Embedded vector store lookups for domain-specific knowledge retrieval.
- Web & File Search: Built-in support for semantic document and web search.

## Demo

![](https://github.com/is-leeroy-jenkins/Jimi/blob/main/resources/images/Jimi-functionality.gif)


## 🧭 Table of Contents

- 💬 **Text generation** with Gemini chat models
- 🖼️ **Image generation**
- 🔍 **Image analysis**
- ✏️ **Image editing**
- 🎧 **Audio transcription**
- 🌍 **Audio translation**
- 🔊 **Text-to-speech**
- 🔢 **Embedding generation**
- 📖 **Document-grounded Q&A**
- 📁 **File upload, listing, retrieval, summarization, and search**
- 🏛️ **Vector store and file-search-store operations**
- 📝 **Prompt engineering backed by SQLite**
- 🗃️ **Data export and management utilities**
- 🧠 **Local retrieval and prompt construction for GGUF inference paths**

## 🖥️ Local GGUF Support

The project is being expanded to support a local `Gemma-4-E4B-it.gguf` model for on-device use.

### What the local model is for

A local GGUF deployment is appropriate when you want:

[![HuggingFace](https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-sm.svg)](https://huggingface.co/leeroy-jankins/jimi)

* private inference on your own machine
* reduced dependency on remote APIs for text generation
* lower-latency prompt/response loops
* document-grounded local answering with retrieved context
* a hybrid workflow where Gemini handles hosted multimodal tasks and Gemma handles local text work


## 📦 Installation

#### 1. Clone the Repository

```
bash
git clone https://github.com/your-username/Jimi.git
cd Jimi
```

#### 2. Create and Activate a Virtual Environment

```
bash
python -m venv venv
venv\Scripts\activate
```

#### 3. Install Dependencies

```bash
    python -m pip install -r requirements.txt
```

#### 4. Install llama-cpp-python

```bash
    python -m pip install llama-cpp-python
```

## 🛠️ Install local LLM

- The local llm for jimi (based on Gemma-4) is on Huggingface and can be downloaded below
- Jimi expects the local llm to be at the location indicated by `MODEL_PATH` in [config.py](https://github.com/is-leeroy-jenkins/Jimi/blob/main/config.py#L55)

#### 5. Install Jimi

[![HuggingFace](https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-sm.svg)](https://huggingface.co/leeroy-jankins/jimi)

## 💻 Capabilities

| Capability        | Description                                                                 |
|-------------------|-----------------------------------------------------------------------------|
| Text Generation   | GPT-powered completions, instructions, and prompts                          |
| Image Generation  | DALL·E 3-based prompt-to-image generation                                   |
| Image Analysis    | Multimodal image+text inference using vision models                         |
| Document Summary  | File upload + prompt-driven summarization via OpenAI file API               |
| Web Search        | Integrated API call to perform web-based lookups                            |
| File Search       | Vector store lookup with prompt-based semantic matching                     |
| Model Registry    | Fine-tuned and base model tracking for GPT-4, GPT-4o, and others            |
| Assistant List    | Query and list named assistant objects from the OpenAI API                  |

## 🛠️ Requirements

- Python 3.10+
- OpenAI Python SDK
- Pydantic
- Numpy, Pandas
- Tiktoken
- Requests

## 📁 File Organization

- [app](https://github.com/is-leeroy-jenkins/Jimi/blob/main/app.py) – Main application framework
- [gemini](https://github.com/is-leeroy-jenkins/Jimi/blob/main/gemini.py) – Models used for structured output
- [agents](https://github.com/is-leeroy-jenkins/Jimi/blob/main/agents.py) – a prompt library w/ over 100 agents. 

## 🔐 Environment Variables

Set the following in your environment or `.env` file:

```bash

  GOOGLE_API_KEY=<your_api_key>

```

## 🚀 Streamlit UI

Jimi includes a **first-class, single-page Streamlit application** that exposes the framework’s
core capabilities through a unified graphical interface.

The Streamlit app is designed for:

* Interactive analysis
* Multimodal experimentation
* Demonstrations and internal tools
* Rapid prototyping on top of Jimi’s agents and models

The application runs entirely on top of Jimi’s core APIs and does **not** modify or duplicate
framework logic.

[![Streamlit App](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)](https://jimi-py.streamlit.app/)

![](https://github.com/is-leeroy-jenkins/Jimi/blob/main/resources/Jimi-streamlit.gif)

## Supported Capabilities

The Streamlit application supports the following workflows:

* 💬 **Chat-based Q&A**
* 📄 **Document-grounded question answering**
* 🖼️ **Image generation**
* ✏️ **Image editing**
* 🔍 **Image analysis (vision)**
* 🎙️ **Audio transcription**
* 🌍 **Audio translation**
* 🧠 **Dynamic model switching**
* 🧩 **Tool, reasoning, and source inspection (when available)**

All workflows are exposed from a **single application page** with explicit mode selection,
ensuring clear separation between text, document, image, and audio tasks.

---

### Running the Streamlit Application

From the project root:

```bash

python -m pip install streamlit streamlit-extras streamlit-pdf
python -m streamlit run app.py

```

Once running, the application will be available at:

```
http://localhost:8501
```


## 🧰 Overview

Jimi provides functionalty from the **Gemini Python SDK**y:

- **Gemini(base)** – holds the single `Gemini` client, env config, and shared helpers.
- **Chat / Assistant / Bro / Bubba** – opinionated text assistants using the **Responses API**.
- **Images* – image generation and vision analysis.
- **Embedding** – small/large/legacy embeddings with consistent return types.
- **TTS** – text-to-speech helpers (streaming to file).
- **Transcription / Translation** – Whisper-powered speech-to-text (+ translate).
- **Vector Stores** – list files, search via `file_search` tool, merge results.

### ✨ Features

- **Responses-first**: consistent `input=[{role, content:[{type:...}]}]` builders.
- **One client**: a single `OpenAI` instance per process for reliability and testability.
- **Typed containers**: Pydantic models for prompts/messages with pass-through `__init__`.
- **Vector search**: easy `file_search` tools + helpers to fetch file IDs from vector stores.
- **Audio**: TTS (stream-to-file), ASR (transcribe), and translate via Whisper.
- **Vision & Images**: multimodal analysis (4o/4o-mini) and image generation (DALL·E 3).
- **Guardrails**: tiny helpers that prevent recurring mistakes (e.g., `inputs` vs `input`,
  content `type` keys, size strings, binary file handling).
- **Uniform errors**: `GptError` + `ErrorDialog` with `module/cause/method` metadata.

## 🛠️ Requirements

Minimum practical requirements:

| Component             | Purpose                                                                                                                       |
| --------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| Python 3.10+          | Required runtime for the application and supporting libraries                                                                 |
| Streamlit             | Web application framework for the UI                                                                                          |
| Google GenAI SDK      | Hosted Gemini model access and multimodal API integration                                                                     |
| pandas                | Tabular data handling and export workflows                                                                                    |
| numpy                 | Numerical processing and array operations                                                                                     |
| plotly                | Interactive charts and visualizations                                                                                         |
| PyMuPDF               | PDF parsing and text extraction                                                                                               |
| reportlab             | PDF generation and reporting utilities                                                                                        |
| sentence-transformers | Local embedding generation for retrieval workflows                                                                            |
| sqlite-vec            | Vector search acceleration over SQLite-backed embeddings                                                                      |
| tiktoken              | Token counting and prompt sizing utilities                                                                                    |
| Pillow                | Image loading and manipulation support                                                                                        |
| requests              | HTTP requests for supporting service calls                                                                                    |
| `config.py`           | Project-local configuration module referenced by the app                                                                      |
| `boogr`               | Project-local support module referenced by the app and wrappers                                                               |
| Local GGUF runtime    | Required for local `Gemma-4-E4B-it.gguf` inference; typically a llama.cpp-compatible runtime or equivalent local serving path |

## 🔐 Configuration

The application reads configuration values from `config.py` and mirrors them into
`st.session_state` and environment variables at startup.

### Core environment variables

```bash
    GEMINI_API_KEY=<your_gemini_api_key>
    GOOGLE_API_KEY=<your_google_api_key>
    GOOGLE_CSE_ID=<your_custom_search_engine_id>
    GOOGLE_CLOUD_PROJECT_ID=<your_gcp_project_id>
    GOOGLE_CLOUD_LOCATION=<your_gcp_region>
    GOOGLEMAPS_API_KEY=<your_google_maps_key>
    GEOCODING_API_KEY=<your_geocoding_key>
```

### Notes

* `GEMINI_API_KEY` is used for Gemini-hosted model access.
* `GOOGLE_API_KEY` is used for Google services wired through the wrapper and UI.
* Cloud project and location values are needed for the Google Cloud and vector-store-related paths.
* Search and maps features depend on the corresponding Google credentials and service availability.

For local GGUF use, configure your chosen local runtime separately and ensure the
`Gemma-4-E4B-it.gguf` file is available to that runtime.


## ⚡ Quickstart

1) **Install**

```python

  python -m pip install google-genai google pydantic

```

2) **Configure**

```python

# Power your client via environment
export GOOGLE_API_KEY="sk-..."         # macOS/Linux
setx GOOGLE_API_KEY "sk-..."           # Windows

```

3) **Hello Jimi**

```python

  from jimi import Chat
  
  chat = Chat()
  print(chat.generate_text("Say hello in one short sentence."))

```

## 🔧 Configuration

- **Environment**
  - `GOOGLE_API_KEY` (required)
- **Models**
  - Text/Responses: e.g., `gemini-2.5-flash`, `gemini-3.0-flash`, 
  - Images: `gemini-2.5-flash-image`
  - Embeddings: `text-embedding-3-small`, `text-embedding-3-large`
  - TTS: `gemini-2.5-flash`,
  - ASR/Translate: `gemini-2.5-flash`
- **File Stores (optional)**
  - Configure your store IDs once; Jimi converts to lists when calling tools.

## 🧩 Design & Architecture

- **Single client**: `Client(api_key=...)` is created in `gemini.__init__()` and reused everywhere.
- **Schema helpers**: tiny, battle-tested builders ensure payloads are valid for the Responses API:
  - input text only
  - text + file
  - text + image

## 🔤 Text Generation

- Generate high-quality responses using OpenAI's GPT models.
- Supports parameter tuning (temperature, top_p, frequency penalties).
- Ideal for summarization, explanations, and knowledge retrieval.

```python

  from jimi import Chat
  
  chat = Chat( )
  response = jimi.generate_text( "Explain how random forests handle overfitting." )
  print( response )

```

## 🎨 Image Generation (NANO-BANANA)

- Convert natural language prompts into images using IMAGEN 3.
- Specify resolution and rendering quality options.
- Useful for creating visual illustrations and conceptual diagrams.

```python

image_url = images.generate_image("A conceptual illustration of quantum computing in federal AI")
print(f"Image URL: {image_url}")

```

### 🖼️ Image Analysis

- Analyze visual content by combining image and text prompts.

- Extract meaning, context, or structure from images.

- Leverages Imagen3/4's vision capabilities for advanced perception.

``` python
  
  url = "https://example.com/sample-image.png"
  response = jimi.analyze_image("Describe the primary elements in this image", url)
  print(response)

```

### 📄 Document Summarization

- Upload and process document files directly into the assistant.

- Use prompts to extract insights or summarize content.

- Supports PDFs, DOCX, and other file formats via Gemini's File Store API.

``` python
  
  file_path = "data/federal_strategy.pdf"
  summary = chat.summarize_document( prompt = "Summarize key national cybersecurity strategies.",
    path = file_path  )
    
  print( summary )

```

### 🔍 File Search with Vector Stores

- Embed and store documents in vector stores for semantic search.

- Retrieve contextually relevant content using natural language queries.

- Ideal for knowledge base querying and document Q&A systems.

``` python
  
  result = files.search_files( 'Legislation related to environmental impact funding' )
  print(result)

```

### 🔎 File & Web Search

- Performs semantic search over domain-specific document embeddings to retrieve relevant content.

- **File Search**: Query vector-embedded files using `vector_store_ids`.

- **Web Search**: Real-time information retrieval using Google Custom Search integration.

```python
  
  result = files.search_files( 'Legislation related to environmental impact funding' )
  print(result)

```

### 🌐 Web Search (Real-Time Querying)

- Perform web lookups in real time via OpenAI’s web-enabled models.

- Extract current events, news, and regulatory updates.

- No scraping required—returns model-interpreted summaries.

```python
  
  insights = text.search_web( 'Current status of the Federal AI Bill 2025' )
  print(insights)

```

### 🧾 Prompt & Message Structuring

- Build structured prompt schemas using Pydantic models.

- Define instructions, context, output goals, and data sources.

- Promotes reusable, interpretable prompt engineering.

```python
  
  from jimi import Chat
  p = Prompt(
      instruction="Create a budget summary",
      context="Federal Defense Budget FY25",
      output_indicator="summary",
      input_data="defense_budget_raw.csv"
  )
  
  print(p.model_dump())

```

## 🧪 Usage Examples

> The snippets below show idiomatic Jimi usage. They assume `chat = Chat()`, `img = Image()`,
> etc., and an `GEMINI_API_KEY` is present in your environment.

## 📝 Text generation

```python
    
    from jimi import Chat

    chat = Chat()
    out = chat.generate_text("Give me three bullet points on strict typing in Python.")
    print(out)
    
```

## 🌐 Web search (Responses)

```python

    from jimi import Chat

    chat = Chat()
    prompt = "Latest trends in Retrieval Augmented Generation. 3 bullets, 1 reference each."
    out = chat.search_web(prompt)  # internally uses web_search_options
    print(out)

```

## 📄 Summarize a document (file-grounded)

```python

    from jimi import Chat

    chat = Chat()
    out = chat.summarize_document(
        prompt="Summarize the document with a 5-bullet executive brief.",
        path="docs/paper.pdf"
    )
    print(out)
    
```

## 🗂️ File search (vector stores)

```python

    from jimi import Chat

    chat = Chat()
    # Assumes chat.vector_stores is configured with { "Appropriations": "...", "Guidance": "..." }
    out = chat.search_files("What are the major themes around FY2024 OCO funding?")
    print(out)

```

## 👀 Vision: analyze an image

```python

    from jimi import Image

    img = Image()
    out = img.analyze(
        text="Describe the chart and call out any anomalies in one paragraph.",
        path="https://example.com/plot.png"
    )
    print(out)
    
```

## 🖼️ Images: generate / edit

```python

    from jimi import Image

    img = Image()
    url = img.generate("A minimalist logo for 'Jimi' in monochrome, vector style")
    print(url)

    # If your SDK supports edits, ensure the correct API path (images.edit vs images.edits)
    # url = img.edit("Add subtle grid background", "logo.png", size="1024x1024")

```

## 🧬 Embeddings

```python

    from jimi import Embeddings

    emb = Embeddings()
    vec = emb.create_small_embedding("Vectorize this sentence.")
    print(len(vec), "dims")

```

## 🔊 Text-to-Speech (TTS)

```python

    from jimi import TTS

    tts = TTS()
    outfile = tts.save_audio("Hello from Jimi in a calm voice.", "out/hello.mp3")
    print("Saved:", outfile)

```

## 🎙️ Transcription / Translation (Whisper)

```python

    from jimi import Transcription, Translation

    asr = Transcription()
    text = asr.transcribe("audio/meeting.m4a")
    print(text)

    xlat = Translation()
    english = xlat.create("Translate this speech to English.", "audio/spanish.m4a")
    print(english)

```



## 📝 License

Jimi is published under the [MIT General Public License v3](https://github.com/is-leeroy-jenkins/Boo/blob/main/LICENSE).


