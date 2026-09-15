from JsonManager import (
    save_json,
    load_json,
    create_json_file,
    update_all_documents_field,
    get_document_fields
)
from Extractor import discover_documents, extract_text, extract_keyberts
from Chunk_Embed import build_chunks, embed_documents
from Distributor import (
    create_category_folders,
    distribute_documents,
    discover_existing_folders,
    apply_version_label
)
from VectorStore import save_to_faiss
from Generating import (
    categorize_document,
    verify_document_placement,
    reset_token_usage,
    get_token_usage
)
from Clustering import (
    calculate_document_relationship,
    build_document_graph,
    find_communities,
    split_oversized_communities,
    find_duplicate_groups,
    find_version_pairs,
    group_version_pairs
)

from pathlib import Path
from datetime import datetime
import time


def convert_categories(categories_dict):

    return {
        "categories": [
            {
                "name": category_name,
                "document_ids": document_ids
            }
            for category_name, document_ids
            in categories_dict.items()
        ]
    }


def create_relationship_scores(
    documents,
    k=20,
    embedding_weight=0.4,
    keyword_weight=0.6,
    title_weight=0.0
):
    """
    Builds each document's top-k neighbor list using
    calculate_document_relationship() (embedding + keyword similarity)
    instead of raw embedding cosine similarity alone.

    Pure embedding similarity was pulling structurally-similar-but-
    unrelated documents together -- e.g. invoices and resumes are both
    short, form-like, boilerplate-heavy documents, so their averaged
    sentence embeddings can land close together in vector space (0.99+
    cosine similarity isn't unusual) even though their actual content has
    nothing to do with each other. Keyword weight is deliberately set
    HIGHER than embedding weight: with two totally disjoint keyword sets
    but near-identical embeddings, embedding_weight=0.6/keyword_weight=0.4
    still lands around 0.60 -- comfortably above the graph's default
    min_score=0.40 -- so the edge survives anyway. Flipping the ratio to
    0.4/0.6 pulls that same worst case down to ~0.40, right at the edge of
    getting filtered out, which is what actually lets community detection
    separate them instead of lumping them into one cluster.

    These weights (and main.py's build_document_graph min_score) are
    still dataset-dependent -- if you're still seeing mixed clusters after
    this change, try pushing keyword_weight higher still, or raising
    min_score a bit; the split_oversized_communities() safety net in
    Clustering.py exists precisely to catch whatever slips through.
    """

    relationship_json = []

    for document in documents:

        scored_neighbors = []

        for other in documents:

            if other["id"] == document["id"]:
                continue

            score = calculate_document_relationship(
                document,
                other,
                embedding_weight=embedding_weight,
                keyword_weight=keyword_weight,
                title_weight=title_weight
            )

            scored_neighbors.append({
                "document_id": other["id"],
                "score": score
            })

        scored_neighbors.sort(
            key=lambda neighbor: neighbor["score"],
            reverse=True
        )

        relationship_json.append({
            "document_id": document["id"],
            "Relationships": scored_neighbors[:k]
        })

    return relationship_json


def _format_duration(seconds):
    """Renders a duration as e.g. '2m 14.3s' or, under a minute, '4.87s'."""

    minutes = int(seconds // 60)
    remaining_seconds = seconds - (minutes * 60)

    if minutes:
        return f"{minutes}m {remaining_seconds:.1f}s"

    return f"{seconds:.2f}s"


def _build_folder_tree(categories):
    """
    Turns the flat {"Work Documents/Contracts": [ids], ...} mapping into
    a two-level tree: {"Work Documents": {"docs": [...], "children":
    {"Contracts": [...], "Research": [...]}}, ...}. Matches the "at most
    2 levels deep" rule categorize_document() is instructed to follow.
    """

    tree = {}

    for folder_name, document_ids in categories.items():

        parts = folder_name.split("/", 1)
        top_name = parts[0]

        node = tree.setdefault(top_name, {"docs": [], "children": {}})

        if len(parts) == 1:
            node["docs"].extend(document_ids)
        else:
            child_name = parts[1]
            node["children"].setdefault(child_name, []).extend(document_ids)

    return tree


def build_report(
    categories,
    all_documents_lookup,
    duplicate_report,
    labeled_version_groups,
    created_at_lookup,
    section_times,
    total_elapsed_seconds,
    token_usage
):
    """
    Builds the plain-text end-of-run report: which folders were created
    and what went into each, which documents were recognized as exact
    duplicates (and therefore left out of every folder), which documents
    look like different versions of each other -- grouped into chains and
    labeled v1/v2/... by creation date -- and how long each stage of the
    run took plus how many Groq tokens it used.
    """

    def file_name(document_id):
        return all_documents_lookup.get(
            document_id, {}
        ).get("file_name", f"Document ID {document_id}")

    lines = []

    lines.append("=" * 60)
    lines.append("DOCUMENT ORGANIZATION REPORT")
    lines.append("=" * 60)

    lines.append("\nFOLDERS CREATED")
    lines.append("-" * 60)

    if categories:

        tree = _build_folder_tree(categories)

        for top_name in sorted(tree.keys()):

            node = tree[top_name]
            total_documents = len(node["docs"]) + sum(
                len(child_ids) for child_ids in node["children"].values()
            )

            lines.append(f"\n{top_name} ({total_documents} document(s))")

            for document_id in sorted(node["docs"]):
                lines.append(f"   - {file_name(document_id)}")

            for child_name in sorted(node["children"].keys()):

                child_ids = node["children"][child_name]

                lines.append(f"   {child_name} ({len(child_ids)} document(s))")

                for document_id in sorted(child_ids):
                    lines.append(f"      - {file_name(document_id)}")

    else:
        lines.append("\nNone created.")

    lines.append("\n\nDUPLICATE DOCUMENTS (left out of every folder)")
    lines.append("-" * 60)

    if duplicate_report:

        for entry in duplicate_report:

            lines.append(f"\nKept: {file_name(entry['kept'])}")

            for duplicate_id in entry["duplicates"]:
                lines.append(f"   - Duplicate: {file_name(duplicate_id)}")

    else:
        lines.append("\nNone found.")

    lines.append("\n\nPOSSIBLE VERSIONS (grouped and labeled by creation date, still filed normally)")
    lines.append("-" * 60)

    if labeled_version_groups:

        for group in labeled_version_groups:

            lines.append("")

            for document_id, label in group:

                created_at = created_at_lookup.get(document_id)

                created_str = (
                    datetime.fromtimestamp(created_at).strftime("%Y-%m-%d %H:%M")
                    if created_at is not None else "unknown date"
                )

                lines.append(
                    f"   [{label}] {file_name(document_id)}  (created {created_str})"
                )

    else:
        lines.append("\nNone found.")

    lines.append("\n\nPERFORMANCE")
    lines.append("-" * 60)

    lines.append("\nTime per section:")

    for section_name, elapsed_seconds in section_times.items():
        lines.append(f"   - {section_name}: {_format_duration(elapsed_seconds)}")

    lines.append(f"\nTotal time taken: {_format_duration(total_elapsed_seconds)}")

    lines.append("\nLLM token usage:")

    if token_usage["calls"]:

        for label, stats in sorted(token_usage["by_label"].items()):
            lines.append(
                f"   - {label}: {stats['calls']} call(s), "
                f"{stats['total_tokens']} tokens "
                f"(prompt: {stats['prompt_tokens']}, completion: {stats['completion_tokens']})"
            )

        lines.append(
            f"\nTotal LLM calls: {token_usage['calls']}"
        )
        lines.append(
            f"Total tokens used: {token_usage['total_tokens']} "
            f"(prompt: {token_usage['prompt_tokens']}, "
            f"completion: {token_usage['completion_tokens']})"
        )

    else:
        lines.append("   No LLM calls were made.")

    return "\n".join(lines)


def build_documents(discovered_documents):

    processed_documents = []

    print("Extracting Keywords...")

    for index, document in enumerate(discovered_documents, start=1):

        text = extract_text(document)
        keywords = extract_keyberts(text, 10)

        processed_documents.append({
            "id": index,
            "file_name": document["name"],
            "path": document["path"],
            "extension": document["extension"],
            "created_at": document["created_at"],
            "text_length": len(text),
            "content": text,
            "keywords": keywords,
            "Document_Embedding": None,
            "Relationships": None,
            "category": None,
            "subcategory": None,
            "confidence": None,
            "version_label": None
        })

    return processed_documents


def run_pipeline(folder_path):

    reset_token_usage()

    section_times = {}
    pipeline_start = time.perf_counter()

    def mark(section_name, section_start):
        section_times[section_name] = time.perf_counter() - section_start

    print("Checking source path....")
    print("Collecting Documents from path....")

    discovery_start = time.perf_counter()

    # Folders this pipeline already created (this run or a previous one)
    # -- excluded from document discovery so already-organized files
    # don't get rediscovered as "new" documents, reclassified, and copied
    # back into the same folder they're already sitting in.
    existing_folders = discover_existing_folders(folder_path)
    top_level_existing_folders = {  
        folder.split("/")[0] for folder in existing_folders
    }

    discovered_documents = discover_documents(
        folder_path, exclude_dirs=top_level_existing_folders
    )

    mark("Document Discovery", discovery_start)

    extraction_start = time.perf_counter()
    processed_documents = build_documents(discovered_documents)
    mark("Text Extraction & Keywords", extraction_start)

    print("Making Chunks....")
    chunking_start = time.perf_counter()
    chunks = build_chunks(processed_documents)

    create_json_file("documents.json")
    create_json_file("chunks.json")
    save_json("documents.json", processed_documents)
    save_json("chunks.json", chunks)
    mark("Chunking", chunking_start)

    print("Embedding Chunks....")
    embedding_start = time.perf_counter()
    doc_embeds = embed_documents(processed_documents, chunks)
    update_all_documents_field("documents.json", "Document_Embedding", doc_embeds)
    mark("Embedding", embedding_start)

    print("Building vector store for RAG retrieval....")
    vector_store_start = time.perf_counter()
    # Reuse the per-chunk embeddings already computed in build_chunks
    # (all-MiniLM-L6-v2) instead of loading a second embedding model just
    # to re-encode everything for FAISS. One model, used consistently.
    chunk_embeddings = [chunk["embedding"] for chunk in chunks]
    save_to_faiss(chunks, chunk_embeddings)
    mark("Vector Store Build", vector_store_start)

    print("Checking for duplicate and version documents....")
    dedup_start = time.perf_counter()

    dedup_source = get_document_fields(
        "documents.json",
        ["id", "file_name", "Document_Embedding"]
    )

    # Exact duplicates: cosine similarity ~1 between document embeddings.
    # For each such group, one document is kept eligible for clustering/
    # folder placement and the rest are set aside entirely -- they are
    # reported at the end but never copied into a category folder.
    duplicate_groups = find_duplicate_groups(dedup_source)

    duplicate_ids = set()
    duplicate_report = []

    for group in duplicate_groups:

        kept_id = group[0]
        duplicate_of_kept = group[1:]

        # Exclude every document in the group -- including the "kept"
        # one -- from clustering/folder placement. "Kept" here is only
        # used to label the group in the report; it does NOT mean that
        # document still gets filed somewhere.
        duplicate_ids.update(duplicate_of_kept)

        duplicate_report.append({
            "kept": kept_id,
            "duplicates": duplicate_of_kept
        })

    # Likely different versions/drafts of the same document: similarity
    # 0.90-0.999. These are reported but still go through normal
    # clustering/placement -- unlike exact duplicates, we don't know which
    # version is "the" one to keep.
    version_pairs = find_version_pairs(dedup_source)

    # Label each version chain v1, v2, v3... by creation date (oldest
    # first) and bake that label into the file name, so it's obvious both
    # in the report and in the copy placed on disk which version is
    # which. Duplicate-group members are excluded from labeling -- they
    # never get filed at all, so there's nothing to usefully label.
    version_pairs_for_labeling = [
        pair for pair in version_pairs
        if pair[0] not in duplicate_ids and pair[1] not in duplicate_ids
    ]

    version_groups = group_version_pairs(version_pairs_for_labeling)

    created_at_lookup = {
        document["id"]: document["created_at"]
        for document in get_document_fields(
            "documents.json", ["id", "created_at"]
        )
    }

    version_labels = {}
    labeled_version_groups = []

    for group in version_groups:

        # Oldest file first -> v1, next oldest -> v2, and so on.
        ordered = sorted(
            group,
            key=lambda document_id: created_at_lookup[document_id]
        )

        labeled_group = []

        for position, document_id in enumerate(ordered, start=1):

            label = f"v{position}"
            version_labels[document_id] = label
            labeled_group.append((document_id, label))

        labeled_version_groups.append(labeled_group)

    if version_labels:

        current_file_names = {
            document["id"]: document["file_name"]
            for document in get_document_fields(
                "documents.json", ["id", "file_name"]
            )
        }

        update_all_documents_field(
            "documents.json",
            "file_name",
            [
                {
                    "document_id": document_id,
                    "file_name": apply_version_label(
                        current_file_names[document_id], label
                    )
                }
                for document_id, label in version_labels.items()
            ]
        )

        update_all_documents_field(
            "documents.json",
            "version_label",
            [
                {"document_id": document_id, "version_label": label}
                for document_id, label in version_labels.items()
            ]
        )

    mark("Duplicate & Version Detection", dedup_start)

    print("Building document relationships....")
    relationship_start = time.perf_counter()
    relationships = create_relationship_scores(
        [
            document for document in get_document_fields(
                "documents.json",
                ["id", "file_name", "keywords", "Document_Embedding"]
            )
            if document["id"] not in duplicate_ids
        ]
    )
    update_all_documents_field("documents.json", "Relationships", relationships)

    documents = [
        document for document in get_document_fields(
            "documents.json",
            [
                "id", "file_name", "path", "extension", "text_length",
                "content", "keywords", "Document_Embedding",
                "category", "subcategory", "confidence"
            ]
        )
        if document["id"] not in duplicate_ids
    ]

    relationships = [
        document for document in get_document_fields(
            "documents.json", ["id", "Relationships", "file_name"]
        )
        if document["id"] not in duplicate_ids
    ]
    mark("Relationship Scoring", relationship_start)

    print("Clustering documents....")
    clustering_start = time.perf_counter()
    graph = build_document_graph(relationships, 0.40)
    communities = find_communities(graph)

    # Safety net: if a community still comes out too large/mixed even
    # with keyword-aware relationship scoring, split it further on
    # embedding distance alone before it gets handed to the LLM as one
    # blob. A genuinely tight, single-topic community passes through
    # unchanged.
    communities = split_oversized_communities(communities, documents)
    mark("Clustering", clustering_start)

    document_lookup = {
        document["id"]: document
        for document in documents
    }

    # Full lookup (including duplicates) for the report generated at the
    # end of the run.
    all_documents_lookup = {
        document["id"]: document
        for document in get_document_fields(
            "documents.json", ["id", "file_name"]
        )
    }

    # Seed with folders that already exist on disk so the categorizer can
    # recognize and reuse them instead of creating near-duplicates.
    folders = list(existing_folders)
    categories = {}

    print("Naming folders....")
    naming_start = time.perf_counter()

    for cluster in communities:

        cluster_documents = [
            document_lookup[document_id]["file_name"]
            for document_id in cluster
        ]

        folder = categorize_document(
            cluster_documents,
            folders
        )

        if folder not in categories:
            categories[folder] = []

        categories[folder].extend(cluster)

        if folder not in folders:
            folders.append(folder)

    mark("Folder Naming (LLM)", naming_start)

    print("Verifying document placement....")
    verification_start = time.perf_counter()

    final_folder_names = list(categories.keys())

    # Single pass, per document: check it individually against the
    # finalized folder list using its own keywords (not just the cluster
    # it landed in). Catches the case where clustering pulled one
    # document into the wrong group.
    for folder_name in list(categories.keys()):

        for document_id in list(categories[folder_name]):

            document = document_lookup[document_id]

            suggested_folder = verify_document_placement(
                document, final_folder_names
            )

            if suggested_folder == folder_name:
                continue

            categories[folder_name].remove(document_id)

            if suggested_folder not in categories:
                categories[suggested_folder] = []
                final_folder_names.append(suggested_folder)

            categories[suggested_folder].append(document_id)

    # Drop any folder that ended up empty after reassignment.
    categories = {
        folder_name: document_ids
        for folder_name, document_ids in categories.items()
        if document_ids
    }
    mark("Placement Verification (LLM)", verification_start)

    print("Creating folders and distributing documents....")
    distribution_start = time.perf_counter()

    # Persist the folder each document ended up in as structured metadata
    # (category = top-level folder, subcategory = anything nested below
    # it). Without this, "how many invoices do I have" has no reliable
    # data to answer from and the RAG agent would have to guess from
    # retrieved chunk text -- which is exactly the kind of question chunk
    # search is bad at.
    category_values = []
    subcategory_values = []

    for folder_name, document_ids in categories.items():

        segments = folder_name.split("/")
        top_category = segments[0]
        subcategory = "/".join(segments[1:]) if len(segments) > 1 else None

        for document_id in document_ids:
            category_values.append({"document_id": document_id, "category": top_category})
            subcategory_values.append({"document_id": document_id, "subcategory": subcategory})

    update_all_documents_field("documents.json", "category", category_values)
    update_all_documents_field("documents.json", "subcategory", subcategory_values)

    # Snapshot the dict form (folder name -> document ids) before it's
    # converted to the list-of-dicts shape distribute_documents() expects
    # -- the report at the end of the run needs this shape.
    categories_for_report = categories

    categories = convert_categories(categories)

    create_category_folders(categories["categories"], folder_path)

    documents_for_distribution = get_document_fields(
        "documents.json", ["id", "file_name", "path"]
    )
    distribute_documents(categories, documents_for_distribution, folder_path)
    mark("Folder Creation & Distribution", distribution_start)

    total_elapsed = time.perf_counter() - pipeline_start

    print("Building report....")

    report_text = build_report(
        categories_for_report,
        all_documents_lookup,
        duplicate_report,
        labeled_version_groups,
        created_at_lookup,
        section_times,
        total_elapsed,
        get_token_usage()
    )

    report_path = str(Path(folder_path) / "Organization_Report.txt")

    with open(report_path, "w", encoding="utf-8") as report_file:
        report_file.write(report_text)

    print(f"\n{report_text}")

    print(f"\nDone. Documents organized and indexed for the RAG agent.")
    print(f"Report saved to: {report_path}")


if __name__ == "__main__":

    SOURCE_FOLDER = "C:\\Users\\dell\\Downloads\\Test"
    run_pipeline(SOURCE_FOLDER)