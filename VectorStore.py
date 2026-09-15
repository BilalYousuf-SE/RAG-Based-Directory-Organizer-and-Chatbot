import faiss
import numpy as np
import json


def save_to_faiss(chunks, embeddings):

    embeddings = np.array(
        embeddings,
        dtype=np.float32
    )

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(
        dimension
    )

    index.add(embeddings)

    faiss.write_index(
        index,
        "chunks.index"
    )

    metadata = []

    for vector_id, chunk in enumerate(chunks):

        metadata.append({
            "vector_id": vector_id,
            "chunk_id": chunk["chunk_id"],
            "document_id": chunk["document_id"]
        })

    with open(
        "vector_metadata.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4
        )

    return index