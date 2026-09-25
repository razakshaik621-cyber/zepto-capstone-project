# Zepto Support Assistant

A small GenAI-style customer support assistant for Zepto policy questions.

The service uses local document embeddings, ChromaDB retrieval, LangGraph routing, Pydantic structured output, and FastAPI.

## Features

* Loads 8 Zepto policy documents from `docs/`
* Splits documents into chunks
* Generates local embeddings using `sentence-transformers`
* Uses `all-MiniLM-L6-v2` for embeddings
* Stores and searches vectors using ChromaDB
* Retrieves the top 3 relevant policy chunks
* Uses LangGraph `StateGraph` for workflow orchestration
* Uses three named nodes:

  * `classify_intent`
  * `retrieve_and_answer`
  * `direct_answer`
* Supports deterministic offline mock mode using `MOCK_LLM=1`
* Returns Pydantic-validated structured responses
* Provides a FastAPI `POST /ask` endpoint
* Includes an optional Groq real-LLM path
* Includes retry logic for invalid structured LLM output
* Includes a Dockerfile for containerization

## Architecture

```text
User Query
    |
    v
FastAPI POST /ask
    |
    v
LangGraph StateGraph
    |
    v
classify_intent
    |
    +-----------------------------+
    |                             |
    | policy question             | general question
    v                             v
retrieve_and_answer          direct_answer
    |                             |
    v                             v
Query embedding              Canned response
    |
    v
ChromaDB
    |
    v
Top-3 relevant chunks
    |
    v
Structured response
    |
    v
Pydantic validation
    |
    v
JSON response
```

## Intent Classification

The `classify_intent` node checks the lowercased user query for policy-related keywords:

* `delivery`
* `return`
* `refund`
* `membership`
* `tracking`
* `cancel`
* `gift card`
* `support`

If any policy keyword is present, the query is routed to `retrieve_and_answer`.

Otherwise, it is routed to `direct_answer`.

## Retrieval Pipeline

For policy questions:

1. Load all 8 documents from `docs/`.
2. Split the documents into chunks.
3. Generate local embeddings using `all-MiniLM-L6-v2`.
4. Store the embeddings and metadata in ChromaDB.
5. Generate an embedding for the user's query.
6. Search ChromaDB using cosine similarity.
7. Retrieve the top 3 relevant chunks.
8. Generate a structured response containing the answer, source chunk IDs, and confidence.

## Structured Output

The API response is validated using Pydantic.

```json
{
  "answer": "string",
  "sources": ["source_chunk_id"],
  "confidence": 1.0
}
```

The `confidence` value must be between `0.0` and `1.0`.

## Mock Mode

The application uses deterministic offline mock mode by default.

`MOCK_LLM` defaults to `1`, so no external LLM API is required for the required submission tests.

PowerShell example:

```powershell
$env:MOCK_LLM="1"
uvicorn support_assistant.main:app --host 127.0.0.1 --port 7860
```

In mock mode:

* Policy questions use ChromaDB retrieval and return a deterministic answer based on the top retrieved chunk.
* General questions return the canned response:

```text
I can only answer questions about Zepto policies right now.
```

## Example Call Transcripts

The following examples were tested with the default `MOCK_LLM` setting.

### Example 1 — Policy Question

Request:

```json
{
  "query": "What is the return policy for damaged grocery items?"
}
```

Response:

```json
{
  "answer": "Based on the retrieved context: Grocery and perishable items may be reported for a return within 24 hours of delivery if damaged, spoiled, or incorrect; non-perishable packaged items may be returned within 7 days of delivery in an unop",
  "sources": [
    "doc_02_chunk_0",
    "doc_06_chunk_0",
    "doc_05_chunk_0"
  ],
  "confidence": 1.0
}
```

This demonstrates policy intent classification, ChromaDB top-3 retrieval, source tracking, and Pydantic structured output.

### Example 2 — General Question

Request:

```json
{
  "query": "What is the capital of India?"
}
```

Response:

```json
{
  "answer": "I can only answer questions about Zepto policies right now.",
  "sources": [],
  "confidence": 1.0
}
```

This demonstrates the general-question route and the deterministic canned response.

## FastAPI

Start the application:

```powershell
uvicorn support_assistant.main:app --host 127.0.0.1 --port 7860
```

The API endpoint is:

```text
POST /ask
```

Example request:

```json
{
  "query": "What is the return policy?"
}
```

Swagger documentation is available at:

```text
http://127.0.0.1:7860/docs
```

## Optional Real-LLM Mode

A real LLM path is included as an optional extension using Groq.

To enable it:

```powershell
$env:MOCK_LLM="0"
$env:GROQ_API_KEY="your_api_key"
```

The API key must be supplied through an environment variable and must never be hardcoded in source code or committed to GitHub.

The real-LLM response is validated using the Pydantic response model. Invalid structured output is retried up to three total attempts before an error is returned.

The required submission does not depend on the real-LLM path; the default mock mode is deterministic and offline.

## Docker

Build the image:

```powershell
docker build -t zepto-support-assistant ./support_assistant
```

Run the container:

```powershell
docker run -p 7860:7860 zepto-support-assistant
```

Then open:

```text
http://localhost:7860/docs
```

## Project Files

```text
support_assistant/
├── docs/
│   ├── doc_01.txt
│   ├── doc_02.txt
│   ├── doc_03.txt
│   ├── doc_04.txt
│   ├── doc_05.txt
│   ├── doc_06.txt
│   ├── doc_07.txt
│   └── doc_08.txt
├── main.py
├── requirements.txt
├── Dockerfile
└── README.md
```

## Design Notes

The assistant is intentionally policy-focused. It uses a local document corpus and deterministic mock mode for the required workflow, making the required demonstration reproducible without an external LLM API.
