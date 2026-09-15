from pathlib import Path
from Readers import read_pdf, read_docx, read_pptx, read_txt
from keybert import KeyBERT

SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".pptx",
    ".txt"
}

# Load once at import time instead of once per document -- KeyBERT loads a
# sentence-transformers model under the hood, so re-creating it per call
# gets very slow once you're processing many documents.
_keybert_model = KeyBERT()

# Below this many non-whitespace characters, there isn't enough real
# content to meaningfully extract keywords from (e.g. extraction failed,
# or OCR still came back empty). Skip the model call and return [] instead
# of feeding it near-blank text.
MIN_CHARS_FOR_KEYWORDS = 20

# How many leading characters count as the document's "header" -- title,
# intro line, etc. On documents that are mostly a long repetitive table or
# list (addresses, sponsor names, line items), whole-document keyword
# extraction gets swamped by whatever repeats most, and the words that
# actually say what the document IS (usually near the top) never make it
# into the top keywords. Extracting the header separately and merging it
# in first fixes that without needing special-casing per document type.
HEADER_CHARS = 300


def discover_documents(folder_path, exclude_dirs=None):
    """
    exclude_dirs: top-level folder names to skip (e.g. folders this
    pipeline already created, like "Resumes" or "Finance"). Without this,
    re-running the pipeline on the same directory it distributes into
    rediscovers already-sorted files as "new" documents, reclassifies
    them, and tries to copy them back into the same folder they're
    already in -- which can hit a PermissionError if the file is still
    momentarily locked (OneDrive sync, an open preview, antivirus scan)
    from being placed there moments ago.
    """

    documents = []

    root = Path(folder_path)
    exclude_dirs = set(exclude_dirs or [])

    for file in root.rglob("*"):

        # Skip Office temp/lock files
        if file.name.startswith("~$"):
            continue

        relative_parts = file.relative_to(root).parts

        if relative_parts and relative_parts[0] in exclude_dirs:
            continue

        if (
            file.is_file()
            and file.suffix.lower() in SUPPORTED_EXTENSIONS
        ):

            documents.append({
                "path": str(file),
                "name": file.name,
                "extension": file.suffix.lower(),
                # File creation time. On Windows (this pipeline's target
                # platform) st_ctime IS the creation time; on Unix it's
                # metadata-change time instead (st_birthtime isn't
                # available on every platform), so this is used as the
                # best cross-platform proxy for "when was this created".
                "created_at": file.stat().st_ctime
            })

    return documents


def extract_text(document):

    extension = document["extension"]

    if extension == ".pdf":
        return read_pdf(document["path"])

    elif extension == ".docx":
        return read_docx(document["path"])

    elif extension == ".pptx":
        return read_pptx(document["path"])

    elif extension == ".txt":
        return read_txt(document["path"])

    return ""


def is_low_content(text, min_chars=MIN_CHARS_FOR_KEYWORDS):
    """True if there isn't enough real text to extract anything meaningful."""
    return len(text.strip()) < min_chars


def _merge_keyword_lists(priority_keywords, fallback_keywords, max_keywords):
    """Combine two keyword lists, keeping priority_keywords first and
    filling any remaining slots with fallback_keywords, de-duplicated."""

    merged = []
    seen = set()

    for keyword in priority_keywords + fallback_keywords:
        key = keyword.lower()
        if key not in seen:
            seen.add(key)
            merged.append(keyword)
        if len(merged) >= max_keywords:
            break

    return merged


def extract_keyberts(text, max_keywords=10):

    if is_low_content(text):
        return []

    header = text[:HEADER_CHARS]
    header_keywords = (
        [k for k, _ in _keybert_model.extract_keywords(header, top_n=max_keywords)]
        if not is_low_content(header)
        else []
    )

    full_keywords = [
        k for k, _ in _keybert_model.extract_keywords(text, top_n=max_keywords)
    ]

    return _merge_keyword_lists(header_keywords, full_keywords, max_keywords)