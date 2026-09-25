# Zepto Support Assistant

A small GenAI-style customer support assistant for Zepto policy questions.

The service uses local document embeddings, ChromaDB retrieval, LangGraph routing, Pydantic structured output, and FastAPI.

## Features

- Loads 8 Zepto policy documents from `docs/`
- Splits documents into chunks
- Generates local embeddings using `sentence-transformers`
- Uses `all-MiniLM-L6-v2` for embeddings
- Stores and searches vectors using ChromaDB
- Retrieves the top 3 relevant policy chunks
- Uses LangGraph `StateGraph` for workflow orchestration
- Uses three named nodes:
  - `classify_intent`
  - `retrieve_and_answer`
  - `direct_answer`
- Supports deterministic offline mock mode using `MOCK_LLM=1`
- Returns Pydantic-validated structured responses
- Provides a FastAPI `POST /ask` endpoint
- Includes an optional Groq real-LLM path
- Includes retry logic for invalid structured LLM output
- Includes a Dockerfile for containerization

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
