from sentence_transformers import SentenceTransformer
import numpy as np
from nltk.tokenize import sent_tokenize
import re
import nltk

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


def is_heading(line):

    if not line:
        return False

    # EDUCATION
    if (
        line.isupper()
        and len(line.split()) <= 8
    ):
        return True

    # Education:
    if line.endswith(":"):
        return True

    # 1. Introduction
    if re.match(
        r'^\d+(\.\d+)*[\.\)]?\s+',
        line
    ):
        return True

    # Introduction
    if (
        line.istitle()
        and len(line.split()) <= 8
        and len(line) < 80
    ):
        return True

    return False

def detect_sections(text):

    if text is None:
        return []

    if isinstance(text, list):
        text = "\n".join(
            str(item)
            for item in text
        )

    if not isinstance(text, str):
        text = str(text)

    if not text.strip():
        return []

    sections = []

    current_heading = "Document"
    current_content = []

    for line in text.splitlines():

        line = line.strip()

        if not line:
            continue

        if is_heading(line):

            if current_content:

                sections.append({
                    "heading": current_heading,
                    "content": "\n".join(current_content)
                })

            current_heading = line
            current_content = []

        else:

            current_content.append(line)

    if current_content:

        sections.append({
            "heading": current_heading,
            "content": "\n".join(current_content)
        })

    return sections

def detect_paragraphs(text):

    if text is None:
        return []

    if isinstance(text, list):
        text = "\n".join(
            str(item)
            for item in text
        )

    if not isinstance(text, str):
        text = str(text)

    if not text.strip():
        return []

    return [
        paragraph.strip()
        for paragraph in text.split("\n\n")
        if paragraph.strip()
    ]

def create_section_chunks(sections):

    chunks = []

    for section in sections:

        chunks.append(
            f"Section: {section['heading']}\n\n"
            f"{section['content']}"
        )

    return chunks

def create_paragraph_chunks(
    paragraphs,
    max_words=300
):

    chunks = []

    current_chunk = []

    current_size = 0

    for paragraph in paragraphs:

        paragraph_size = len(
            paragraph.split()
        )

        if (
            current_size + paragraph_size
            > max_words
            and current_chunk
        ):

            chunks.append(
                "\n\n".join(current_chunk)
            )

            current_chunk = []
            current_size = 0

        current_chunk.append(paragraph)

        current_size += paragraph_size

    if current_chunk:

        chunks.append(
            "\n\n".join(current_chunk)
        )

    return chunks

def create_sentence_chunks(
    text,
    max_sentences=10
):

    sentences = sent_tokenize(text)

    chunks = []

    for i in range(
        0,
        len(sentences),
        max_sentences
    ):

        chunks.append(
            " ".join(
                sentences[i:i + max_sentences]
            )
        )

    return chunks



def build_chunks(documents):

    all_chunks = []

    chunk_id = 1

    for document in documents:

        document_id = document["id"]

        text = document["content"]

        if isinstance(text, list):
            text = "\n".join(
                str(item)
                for item in text
            )

        if not isinstance(text, str):
            text = str(text)

        sections = detect_sections(text)

        if len(sections) > 1:

            chunks = create_section_chunks(
                sections
            )

        else:

            paragraphs = detect_paragraphs(
                text
            )

            if len(paragraphs) > 1:

                chunks = create_paragraph_chunks(
                    paragraphs
                )

            else:

                chunks = create_sentence_chunks(
                    text
                )

        for chunk_text in chunks:

            all_chunks.append({
                "chunk_id": chunk_id,
                "document_id": document_id,
                "text": chunk_text,
                "embedding": generate_embedding(chunk_text)
            })

            chunk_id += 1

    return all_chunks



def generate_embedding(text):

    embedding = model.encode(
        text,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    return embedding.tolist()



def embed_documents(documents, chunks):

    chunk_lookup = {
        chunk["chunk_id"]: chunk
        for chunk in chunks
    }

    document_embeddings = []

    for document in documents:

        document_id = document["id"]

        document_chunks = [
            chunk_lookup[chunk["chunk_id"]]
            for chunk in chunks
            if chunk["document_id"] == document_id
        ]

        if len(document_chunks) == 0:

            # No usable text was extracted for this document (empty/failed
            # extraction). Give it a zero vector instead of dropping it --
            # downstream code (relationship scoring) expects every document
            # to have a Document_Embedding, and a missing one crashes it.
            document_embeddings.append({
                "document_id": document_id,
                "Document_Embedding": np.zeros(
                    model.get_sentence_embedding_dimension()
                ).tolist()
            })

            continue

        document_embedding = np.zeros(
            len(document_chunks[0]["embedding"])
        )

        for chunk in document_chunks:

            embedding = np.array(
                chunk["embedding"]
            )

            if np.isnan(embedding).any():

                print(
                    "NaN chunk embedding",
                    chunk["chunk_id"]
                )

                continue

            document_embedding += embedding

        document_embedding = (
            document_embedding
            / len(document_chunks)
        )

        document_embeddings.append({
            "document_id": document_id,
            "Document_Embedding": document_embedding.tolist()
        })

    return document_embeddings
