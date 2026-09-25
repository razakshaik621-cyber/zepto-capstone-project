import json
import os
from pathlib import Path
from typing import TypedDict

import chromadb
from groq import Groq
from fastapi import FastAPI
from pydantic import BaseModel, Field
class AskRequest(BaseModel):
    query: str = Field(min_length=1)


class AskResponse(BaseModel):
    answer: str
    sources: list[str]
    confidence: float = Field(ge=0.0, le=1.0)
def parse_llm_response(raw_output: str) -> AskResponse:
    try:
        data = json.loads(raw_output)
        return AskResponse.model_validate(data)
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError("Invalid structured LLM response.") from exc
    def call_llm_with_retry(
    prompt: str,
    max_attempts: int = 3,
) -> AskResponse:
        last_error = None

    for attempt in range(max_attempts):
        try:
            raw_output = call_real_llm(prompt)
            return parse_llm_response(raw_output)
        except (ValueError, RuntimeError) as exc:
            last_error = exc

    raise RuntimeError(
        f"LLM failed structured-output validation after "
        f"{max_attempts} attempts."
    ) from last_error

from sentence_transformers import SentenceTransformer
from langgraph.graph import END, START, StateGraph


# ============================================================
# Configuration
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "docs"

CHROMA_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "zepto_support"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
PROMPT_TEMPLATE = """
Role:
You are a Zepto customer support policy assistant.

Context:
Use only the retrieved Zepto policy context provided below.
Retrieved context:
{context}

Task:
Answer the user's question using the retrieved policy context.

Format:
Return ONLY a valid JSON object with exactly these fields:
answer, sources, confidence.

The response must contain:
- answer: a concise factual answer
- sources: a list of retrieved source chunk IDs
- confidence: a number between 0.0 and 1.0

Do not invent policy details.
If the context does not contain the answer, say that the available policy context does not provide the answer and use an appropriate confidence value.
The "confidence" value must be a number between 0.0 and 1.0.
The "sources" field must contain only the retrieved chunk IDs that support the answer.
Do not invent policy details.
If the context does not contain the answer, say that the available policy context does not provide the answer and use an appropriate confidence value.

Negative constraint:
Do not use outside knowledge, do not make up policies, and do not claim information that is not supported by the retrieved context.

Few-shot example:
User question: What is the return period?
Context: Non-perishable packaged items may be returned within 7 days of delivery.
Answer: Non-perishable packaged items may be returned within 7 days of delivery.

Length:
Keep the answer concise and within 2-3 sentences.

User question:
{query}
"""

# MOCK_LLM is the graded/default mode.
# Unset, empty, or "1" means deterministic offline mock mode.
# "0" is reserved for the optional real-LLM extension.
MOCK_LLM = os.getenv("MOCK_LLM", "1") != "0"
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
def call_real_llm(prompt: str) -> str:
    if not GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY is required when MOCK_LLM=0."
        )

    client = Groq(api_key=GROQ_API_KEY)

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        temperature=0,
    )

    return response.choices[0].message.content.strip()


# ============================================================
# Embedding model and ChromaDB
# ============================================================

embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))

collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"},
)





def load_documents() -> list[dict[str, str]]:
    """Load the eight Zepto support documents from the docs folder."""
    documents = []

    for path in sorted(DOCS_DIR.glob("doc_*.txt")):
        text = path.read_text(encoding="utf-8").strip()

        if not text:
            continue

        documents.append(
            {
                "document_id": path.stem,
                "text": text,
            }
        )

    return documents


def chunk_text(text: str, chunk_size: int = 300) -> list[str]:
    """Split a document into small word-based chunks."""
    words = text.split()

    chunks = []

    for start in range(0, len(words), chunk_size):
        chunk = " ".join(words[start:start + chunk_size]).strip()

        if chunk:
            chunks.append(chunk)

    return chunks


def build_chunks() -> list[dict[str, str]]:
    """Create chunks while preserving the source document ID."""
    chunks = []

    for document in load_documents():
        document_id = document["document_id"]
        text = document["text"]

        for index, chunk in enumerate(chunk_text(text)):
            chunks.append(
                {
                    "chunk_id": f"{document_id}_chunk_{index}",
                    "document_id": document_id,
                    "text": chunk,
                }
            )

    return chunks
def ingest_documents() -> list[dict[str, str]]:
    """Embed all chunks and store them in the ChromaDB collection."""
    chunks = build_chunks()

    if not chunks:
        raise RuntimeError("No support documents were found.")

    ids = [chunk["chunk_id"] for chunk in chunks]
    texts = [chunk["text"] for chunk in chunks]
    metadatas = [
        {
            "document_id": chunk["document_id"],
            "chunk_id": chunk["chunk_id"],
        }
        for chunk in chunks
    ]

    embeddings = embedding_model.encode(
        texts,
        normalize_embeddings=True,
    ).tolist()

    collection.upsert(
        ids=ids,
        documents=texts,
        metadatas=metadatas,
        embeddings=embeddings,
    )

    return chunks
def retrieve_context(query: str, top_k: int = 3) -> list[dict[str, str]]:
    """Retrieve the top matching chunks using cosine similarity."""
    query_embedding = embedding_model.encode(
        [query],
        normalize_embeddings=True,
    ).tolist()[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
    )

    retrieved = []

    result_documents = results.get("documents", [[]])[0]
    result_ids = results.get("ids", [[]])[0]
    result_metadatas = results.get("metadatas", [[]])[0]

    for document, chunk_id, metadata in zip(
        result_documents,
        result_ids,
        result_metadatas,
    ):
        retrieved.append(
            {
                "chunk_id": chunk_id,
                "document_id": metadata["document_id"],
                "text": document,
            }
        )

    return retrieved
class AssistantState(TypedDict):
    query: str
    intent: str
    retrieved: list[dict[str, str]]
    response: AskResponse


def classify_intent(state: AssistantState) -> AssistantState:
    """Classify the user query into policy or general intent."""
    query = state["query"].lower()

    policy_keywords = [
        "delivery",
        "return",
        "refund",
        "membership",
        "tracking",
        "cancel",
        "gift card",
        "support",
    ]

    if any(keyword in query for keyword in policy_keywords):
        intent = "policy_question"
    else:
        intent = "general_question"

    state["intent"] = intent
    return state


def retrieve_and_answer(state: AssistantState) -> AssistantState:
    """Retrieve policy context and generate a deterministic mock answer."""
    retrieved = retrieve_context(state["query"], top_k=3)

    if not retrieved:
        state["response"] = AskResponse(
            answer="No relevant policy context was found.",
            sources=[],
            confidence=0.0,
        )
        return state

    top_chunk = retrieved[0]
    snippet = top_chunk["text"][:200]

    state["retrieved"] = retrieved
    state["response"] = AskResponse(
        answer=f"Based on the retrieved context: {snippet}",
        sources=[item["chunk_id"] for item in retrieved],
        confidence=1.0,
    )

    return state


def retrieve_and_answer(state: AssistantState) -> AssistantState:
    retrieved = retrieve_context(state["query"], top_k=3)

    if not retrieved:
        state["response"] = AskResponse(
            answer="No relevant policy context was found.",
            sources=[],
            confidence=0.0,
        )
        return state

    context = "\n\n".join(
        f"[{item['document_id']} / {item['chunk_id']}]\n{item['text']}"
        for item in retrieved
    )

    prompt = PROMPT_TEMPLATE.format(
        context=context,
        query=state["query"],
    )

    if MOCK_LLM:
        top_chunk = retrieved[0]
        snippet = top_chunk["text"][:200]

        state["response"] = AskResponse(
            answer=f"Based on the retrieved context: {snippet}",
            sources=[item["chunk_id"] for item in retrieved],
            confidence=1.0,
        )
        state["retrieved"] = retrieved
        return state

    response = call_llm_with_retry(prompt)

    state["response"] = response
    state["retrieved"] = retrieved
    return state
class AssistantState(TypedDict):
    query: str
    intent: str
    retrieved: list[dict[str, str]]
    response: AskResponse


def classify_intent(state: AssistantState) -> AssistantState:
    query = state["query"].lower()

    policy_keywords = [
        "delivery",
        "return",
        "refund",
        "membership",
        "tracking",
        "cancel",
        "gift card",
        "support",
    ]

    if any(keyword in query for keyword in policy_keywords):
        intent = "policy_question"
    else:
        intent = "general_question"

    state["intent"] = intent
    return state


def retrieve_and_answer(state: AssistantState) -> AssistantState:
    retrieved = retrieve_context(state["query"], top_k=3)

    if not retrieved:
        state["response"] = AskResponse(
            answer="No relevant policy context was found.",
            sources=[],
            confidence=0.0,
        )
        return state

    context = "\n\n".join(
        f"[{item['document_id']} / {item['chunk_id']}]\n{item['text']}"
        for item in retrieved
    )

    prompt = PROMPT_TEMPLATE.format(
        context=context,
        query=state["query"],
    )

    if MOCK_LLM:
        top_chunk = retrieved[0]
        snippet = top_chunk["text"][:200]

        state["response"] = AskResponse(
            answer=f"Based on the retrieved context: {snippet}",
            sources=[item["chunk_id"] for item in retrieved],
            confidence=1.0,
        )
        state["retrieved"] = retrieved
        return state

    response = call_llm_with_retry(prompt)

    state["response"] = response
    state["retrieved"] = retrieved
    return state


def direct_answer(state: AssistantState) -> AssistantState:
    state["retrieved"] = []

    state["response"] = AskResponse(
        answer="I can only answer questions about Zepto policies right now.",
        sources=[],
        confidence=1.0,
    )

    return state


def route_intent(state: AssistantState) -> str:
    if state["intent"] == "policy_question":
        return "retrieve_and_answer"

    return "direct_answer"


workflow = StateGraph(AssistantState)

workflow.add_node("classify_intent", classify_intent)
workflow.add_node("retrieve_and_answer", retrieve_and_answer)
workflow.add_node("direct_answer", direct_answer)

workflow.add_edge(START, "classify_intent")

workflow.add_conditional_edges(
    "classify_intent",
    route_intent,
    {
        "retrieve_and_answer": "retrieve_and_answer",
        "direct_answer": "direct_answer",
    },
)

workflow.add_edge("retrieve_and_answer", END)
workflow.add_edge("direct_answer", END)

graph = workflow.compile()


app = FastAPI(title="Zepto Support Assistant")


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    result = graph.invoke(
        {
            "query": request.query,
            "intent": "",
            "retrieved": [],
            "response": None,
        }
    )

    return result["response"]
