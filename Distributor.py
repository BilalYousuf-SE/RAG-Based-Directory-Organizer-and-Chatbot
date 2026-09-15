import shutil
import time
from pathlib import Path

# Characters that aren't valid in a folder name on most filesystems.
# "/" is deliberately excluded here -- it's used as the hierarchy
# separator and is handled by splitting the path into segments first.
INVALID_CHARS = '<>:"\\|?*'


def _sanitize_segment(segment):

    for char in INVALID_CHARS:
        segment = segment.replace(char, " - ")

    return segment.strip()


def apply_version_label(file_name, label):
    """
    Inserts a version label before the file extension, e.g.
    ("Marketing_Strategy_Plan.docx", "v2") ->
    "Marketing_Strategy_Plan_v2.docx". Returns file_name unchanged if
    label is falsy (document isn't part of a version chain).
    """

    if not label:
        return file_name

    stem = Path(file_name).stem
    suffix = Path(file_name).suffix

    # Already carries this exact label -- e.g. re-running labeling on a
    # document that was labeled on a previous pass -- don't double it up
    # into "..._v2_v2.docx".
    if stem.lower().endswith(f"_{label.lower()}"):
        return file_name

    return f"{stem}_{label}{suffix}"


def discover_existing_folders(destination_path, max_depth=3):
    """
    Walks the destination directory and returns existing folder paths
    (relative to destination_path, "/"-separated, e.g. "Finance/Invoices")
    up to max_depth levels deep.

    This is used to seed the `folders` list passed into categorize_document
    so the LLM can recognize and reuse folders that already exist on disk,
    instead of creating near-duplicate ones (e.g. "Invoices" vs "invoices ").
    """

    root = Path(destination_path)

    if not root.exists():
        return []

    existing_folders = []

    for path in root.rglob("*"):

        if not path.is_dir():
            continue

        relative_path = path.relative_to(root)

        if len(relative_path.parts) > max_depth:
            continue

        existing_folders.append(
            "/".join(relative_path.parts)
        )

    return sorted(existing_folders)


def create_category_folders(categories, destination_path):
    """
    Creates a folder for each category, including nested ones.
    category["name"] may be a hierarchical path like "Finance/Invoices".
    """

    root = Path(destination_path)

    root.mkdir(exist_ok=True)

    for category in categories:

        segments = [
            _sanitize_segment(segment)
            for segment in category["name"].split("/")
            if segment.strip()
        ]

        # Write the sanitized/normalized name back so distribute_documents
        # (which receives the same categories dict) builds the same path.
        category["name"] = "/".join(segments)

        folder = root.joinpath(*segments)

        folder.mkdir(
            parents=True,
            exist_ok=True
        )


def distribute_documents(
    categories,
    documents,
    destination_root
):

    document_lookup = {
        document["id"]: document
        for document in documents
    }

    for category in categories["categories"]:

        segments = category["name"].split("/")

        category_folder = Path(destination_root).joinpath(*segments)

        category_folder.mkdir(
            parents=True,
            exist_ok=True
        )

        for document_id in category["document_ids"]:

            document = document_lookup[
                document_id
            ]

            source_path = Path(document["path"])

            destination_path = (
                category_folder
                / document["file_name"]
            )

            # Already exactly where it belongs (e.g. re-running the
            # pipeline on an already-organized folder) -- nothing to copy.
            if source_path.resolve() == destination_path.resolve():
                continue

            _copy_with_retry(source_path, destination_path, document["file_name"])


def _copy_with_retry(source_path, destination_path, file_name, max_retries=3, retry_delay=1.0):
    """
    shutil.copy2 can fail with a transient PermissionError if the
    destination file is momentarily locked -- e.g. OneDrive still syncing
    the previous run's copy, an antivirus scan in progress, or Windows
    Explorer holding a preview handle. Retries briefly before giving up;
    if it's still locked after that, skips the file with a clear message
    instead of crashing the whole run.
    """

    for attempt in range(1, max_retries + 1):

        try:
            shutil.copy2(source_path, destination_path)
            return

        except PermissionError as error:

            if attempt == max_retries:
                print(
                    f"Skipped '{file_name}': still locked by another "
                    f"process after {max_retries} attempts ({error}). "
                    f"Close any program that has it open, or wait for "
                    f"OneDrive/antivirus to finish with it, then re-run."
                )
                return

            time.sleep(retry_delay)