"""
RAG Agent.

Uses the FAISS index + chunks.json + documents.json produced by main.py to
answer questions about the organized documents:

  query -> embed (same model used for chunks) -> FAISS search -> pull
  matching chunk text -> feed as context to a Groq chat model -> answer,
  with the source file names it drew from.

Run directly for an interactive chat loop:

    python RAGAgent.py
"""

import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq

from config import GROQ_API_KEY

# Must match the model used to embed chunks in Chunk_Embed.py -- the query
# vector and the index vectors have to live in the same embedding space.
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

GROQ_MODEL = "openai/gpt-oss-120b"

client = Groq(api_key=GROQ_API_KEY)

_embedding_model = None


def _get_embedding_model():

    global _embedding_model

    if _embedding_model is None:
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    return _embedding_model


def load_vector_store(index_path="chunks.index", metadata_path="vector_metadata.json"):

    index = faiss.read_index(index_path)

    with open(metadata_path, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    return index, metadata


def load_chunk_lookup(chunks_path="chunks.json"):

    with open(chunks_path, "r", encoding="utf-8") as file:
        chunks = json.load(file)

    return {
        chunk["chunk_id"]: chunk
        for chunk in chunks
    }


def load_document_lookup(documents_path="documents.json"):

    with open(documents_path, "r", encoding="utf-8") as file:
        documents = json.load(file)

    return {
        document["id"]: document
        for document in documents
    }


def embed_query(query):

    model = _get_embedding_model()

    embedding = model.encode(
        query,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    return np.array([embedding], dtype=np.float32)


def retrieve(query, index, metadata, chunk_lookup, document_lookup, top_k=5):

    query_vector = embed_query(query)

    distances, indices = index.search(query_vector, top_k)

    results = []

    for distance, vector_id in zip(distances[0], indices[0]):

        if vector_id == -1:
            continue

        entry = metadata[vector_id]

        chunk = chunk_lookup.get(entry["chunk_id"])

        if chunk is None:
            continue

        document = document_lookup.get(entry["document_id"], {})

        results.append({
            "chunk_id": entry["chunk_id"],
            "document_id": entry["document_id"],
            "file_name": document.get("file_name", "Unknown"),
            "text": chunk["text"],
            "distance": float(distance)
        })

    return results


def classify_intent(query):
    """
    Decides whether a question needs METADATA aggregation (counts, lists,
    overviews of documents by category) or SEMANTIC content search
    (questions about what's actually written inside documents).

    Chunk-based retrieval is the wrong tool for "how many invoices do I
    have" -- no single chunk contains that answer, so the model would just
    guess from whatever 5 chunks happened to come back. Metadata questions
    get answered from documents.json directly, deterministically, instead.

    Returns {"type": "metadata" | "semantic", "topic": str}. `topic` is
    the short category keyword to match (e.g. "invoices"), or "" for a
    general overview ("how are my documents organized").
    """

    system_prompt = (
        "Classify the user's question about their personal document "
        "library. Respond with ONLY a JSON object, no other text.\n\n"
        'Use {"type": "metadata", "topic": "<short keyword>"} when the '
        "question asks for a COUNT, LIST, or OVERVIEW of documents by "
        "category or type -- e.g. \"how many invoices do I have\", "
        "\"list all my resumes\", \"what kinds of documents do I have\". "
        'Use topic: "" for general overview questions with no specific '
        "category.\n\n"
        'Use {"type": "semantic", "topic": ""} when the question asks '
        "about the CONTENT written inside one or more documents -- e.g. "
        "\"what was the total on the March invoice\", \"summarize my "
        "lease agreement\", \"who is the sender of the Acme contract\"."
    )

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            temperature=0
        )

        raw = response.choices[0].message.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()

        parsed = json.loads(raw)

        if parsed.get("type") not in {"metadata", "semantic"}:
            raise ValueError("unexpected type")

        return parsed

    except (json.JSONDecodeError, ValueError, KeyError, AttributeError):
        # Classification is a cheap best-effort step. If it fails for any
        # reason, fall back to the safer default: semantic chunk search.
        return {"type": "semantic", "topic": ""}


def _full_category_path(document):

    parts = [
        part for part in [document.get("category"), document.get("subcategory")]
        if part
    ]

    return "/".join(parts) if parts else "Uncategorized"


def _document_matches_topic(document, topic):

    if not topic:
        return False

    topic = topic.lower().strip()

    fields = [
        (document.get("category") or "").lower(),
        (document.get("subcategory") or "").lower()
    ]

    return any(
        topic in field or field in topic
        for field in fields if field
    )


def _category_overview(documents):

    counts = {}

    for document in documents:
        path = _full_category_path(document)
        counts[path] = counts.get(path, 0) + 1

    if not counts:
        return "You don't have any categorized documents yet.", []

    lines = ["Here's how your documents are currently organized:\n"]

    for category, count in sorted(counts.items()):
        lines.append(f"  - {category}: {count} document(s)")

    return "\n".join(lines), documents


def answer_metadata_query(topic, documents_path="documents.json"):
    """
    Deterministic count/list over document categories -- reads
    documents.json directly, no chunk search and no LLM guessing at
    numbers, so the count is always exactly right.
    """

    with open(documents_path, "r", encoding="utf-8") as file:
        documents = json.load(file)

    if not topic or not topic.strip():
        return _category_overview(documents)

    matches = [
        document for document in documents
        if _document_matches_topic(document, topic)
    ]

    if not matches:
        known_categories = sorted({
            _full_category_path(document)
            for document in documents
            if document.get("category")
        })
        categories_list = ", ".join(known_categories) if known_categories else "none yet"

        return (
            f'I couldn\'t find any documents matching "{topic}". '
            f"Categories currently on file: {categories_list}."
        ), []

    grouped = {}
    for document in matches:
        grouped.setdefault(_full_category_path(document), []).append(document["file_name"])

    lines = [f'You have {len(matches)} document(s) matching "{topic}":']

    for category, file_names in sorted(grouped.items()):
        lines.append(f"\n{category} ({len(file_names)}):")
        for file_name in sorted(file_names):
            lines.append(f"  - {file_name}")

    return "\n".join(lines), matches


def build_context(retrieved_chunks):

    parts = []

    for i, chunk in enumerate(retrieved_chunks, start=1):

        parts.append(
            f"[{i}] Source: {chunk['file_name']}\n{chunk['text']}"
        )

    return "\n\n".join(parts)


def generate_answer(query, retrieved_chunks, history=None):

    context = build_context(retrieved_chunks)

    system_prompt = (
        "You are a helpful assistant answering questions using ONLY the "
        "provided document excerpts. Cite the sources you used by their "
        "[number] tag. If the excerpts don't contain the answer, say you "
        "don't know instead of guessing."
    )

    user_prompt = f"Context:\n{context}\n\nQuestion: {query}"

    messages = [{"role": "system", "content": system_prompt}]

    if history:
        messages.extend(history)

    messages.append({"role": "user", "content": user_prompt})

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=0.2
    )

    return response.choices[0].message.content


def ask(query, top_k=5, history=None):
    """
    Full flow for a single question. Routes between two paths:

    - "metadata" (counts/lists/overviews, e.g. "how many invoices do I
      have") -> answered directly and deterministically from
      documents.json. No chunk search, no LLM-guessed numbers.
    - "semantic" (questions about document content) -> the original
      retrieve-then-generate flow over chunk embeddings.

    Returns (answer_text, list_of_matched_items).
    """

    intent = classify_intent(query)

    if intent["type"] == "metadata":

        answer, matches = answer_metadata_query(intent["topic"])

        # Normalize to the same shape retrieve() returns, so callers don't
        # need to know or care which path answered the question. There's
        # no chunk involved in a metadata answer, so chunk_id is None.
        sources = [
            {
                "chunk_id": None,
                "document_id": document.get("id"),
                "file_name": document.get("file_name", "Unknown"),
                "text": None,
                "distance": None
            }
            for document in matches
        ]

        return answer, sources

    index, metadata = load_vector_store()
    chunk_lookup = load_chunk_lookup()
    document_lookup = load_document_lookup()

    retrieved = retrieve(
        query, index, metadata, chunk_lookup, document_lookup, top_k
    )

    if not retrieved:
        return "I couldn't find anything relevant in the indexed documents.", []

    answer = generate_answer(query, retrieved, history)

    return answer, retrieved


if __name__ == "__main__":

    print("RAG Agent ready. Type 'exit' to quit.\n")

    conversation_history = []

    while True:

        query = input("You: ").strip()

        if query.lower() in {"exit", "quit"}:
            break

        if not query:
            continue

        try:
            answer, sources = ask(query, history=conversation_history)
        except Exception as error:
            print(f"\nSomething went wrong answering that: {error}\n")
            continue

        print(f"\nAssistant: {answer}\n")

        if sources:
            print("Sources:")
            for source in sources:
                file_name = source.get("file_name", "Unknown")
                chunk_id = source.get("chunk_id")
                if chunk_id is not None:
                    print(f"  - {file_name} (chunk {chunk_id})")
                else:
                    print(f"  - {file_name}")

        print()

        conversation_history.append({"role": "user", "content": query})
        conversation_history.append({"role": "assistant", "content": answer})