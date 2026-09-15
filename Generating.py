import re
import time

from groq import Groq, RateLimitError
from config import GROQ_API_KEY

client = Groq(api_key=GROQ_API_KEY)

GROQ_MODEL = "openai/gpt-oss-120b"

# categorize_document/verify_document_placement each make one Groq call
# per cluster/document. On a big batch that adds up fast against Groq's
# tokens-per-minute limit (e.g. 8000 TPM on the on_demand tier), so a
# RateLimitError partway through a long run is expected, not a bug --
# this wraps every call so it waits out the limit and resumes instead of
# crashing the whole pipeline.
MAX_RATE_LIMIT_RETRIES = 8

# Running total of Groq token usage for the current pipeline run, broken
# down by which function made the call. main.py calls reset_token_usage()
# at the start of run_pipeline() and get_token_usage() at the end to pull
# the final numbers into the organization report.
_token_usage = {
    "calls": 0,
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0,
    "by_label": {}
}


def reset_token_usage():
    """Clears accumulated token usage -- call once before a pipeline run."""

    global _token_usage

    _token_usage = {
        "calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "by_label": {}
    }


def get_token_usage():
    """Returns the running token usage totals (see _token_usage shape)."""

    return _token_usage


def _record_token_usage(label, usage):

    if usage is None:
        return

    prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
    completion_tokens = getattr(usage, "completion_tokens", 0) or 0
    total_tokens = (
        getattr(usage, "total_tokens", 0)
        or (prompt_tokens + completion_tokens)
    )

    _token_usage["calls"] += 1
    _token_usage["prompt_tokens"] += prompt_tokens
    _token_usage["completion_tokens"] += completion_tokens
    _token_usage["total_tokens"] += total_tokens

    label_stats = _token_usage["by_label"].setdefault(
        label,
        {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    )

    label_stats["calls"] += 1
    label_stats["prompt_tokens"] += prompt_tokens
    label_stats["completion_tokens"] += completion_tokens
    label_stats["total_tokens"] += total_tokens


def _seconds_until_retry(error, default_seconds=2.0):
    """
    Groq's 429 message includes a hint like "Please try again in 577.5ms"
    or "in 2.3s" -- use it directly when present, since it's far more
    accurate than a blind guess. Falls back to default_seconds if the
    message doesn't match (format changed, different error variant, etc).
    """

    match = re.search(
        r"try again in ([\d.]+)\s*(ms|s)\b",
        str(error)
    )

    if not match:
        return default_seconds

    amount, unit = match.groups()
    amount = float(amount)

    return amount / 1000 if unit == "ms" else amount


def _create_completion(prompt, label="unlabeled"):
    """
    Thin wrapper around client.chat.completions.create() that retries on
    Groq 429 rate-limit errors with backoff, instead of letting them
    crash the pipeline mid-run. Every other error type is raised as-is.

    `label` identifies the calling function (e.g. "categorize_document")
    so token usage can be broken down by call site, not just totalled.
    """

    for attempt in range(1, MAX_RATE_LIMIT_RETRIES + 1):

        try:
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

            _record_token_usage(label, getattr(response, "usage", None))

            return response

        except RateLimitError as error:

            if attempt == MAX_RATE_LIMIT_RETRIES:
                raise

            # A little buffer on top of Groq's own estimate, and back off
            # further on repeated hits in case the limit is being refilled
            # slower than expected under sustained load.
            wait_seconds = _seconds_until_retry(error) + 0.5
            wait_seconds *= attempt

            print(
                f"Groq rate limit hit, waiting {wait_seconds:.1f}s "
                f"before retrying (attempt {attempt}/{MAX_RATE_LIMIT_RETRIES})..."
            )

            time.sleep(wait_seconds)


def _normalize_folder_path(raw_path):
    """
    Turn whatever the model returns into a clean, forward-slash separated
    hierarchical path, e.g. "Finance \\ Invoices" -> "Finance/Invoices".
    Each segment is title-cased and stripped independently.
    """

    raw_path = raw_path.replace("\\", "/")

    segments = [
        segment.strip().title()
        for segment in raw_path.split("/")
        if segment.strip()
    ]

    return "/".join(segments)


def categorize_document(cluster, folders):
    """
    Decides which folder a cluster of documents belongs to.

    `folders` is a list of existing folder paths -- both ones already on
    disk (seeded before the run) and ones created earlier in this run.
    Paths may be nested, shown as "Parent/Child". The model is asked to
    reuse an existing path exactly when it fits, or return a new path
    (optionally nested under an existing top-level folder) otherwise.
    """

    prompt = f"""
    You are organizing a user's documents into a folder structure.

    Existing folders (some may be nested, shown as Parent/Child):

    {folders}

    Document cluster:

    {cluster}

    Instructions:

    - Analyze the documents in the cluster.
    - Decide where this cluster belongs in the folder structure.
    - You MAY return a nested path using "/" as a separator, e.g.
      "Finance/Invoices", when the cluster is a specific topic within a
      broader existing or new category.
    - If the cluster clearly belongs to an existing folder (or existing
      nested path), return that exact path.
    - If it represents a new broad topic, return a new top-level folder name.
    - If it is a more specific topic within an existing top-level folder,
      return "ExistingFolder/NewSubfolder".
    - Keep hierarchy shallow: at most 2 levels deep (Parent/Child).
    - Prefer meaningful, human-friendly folder names.
    - Prefer broad categories over filenames or individual topics.
    - Avoid generic names such as:
    NewFolder
    Misc
    Other
    Category
    Documents
    Folder
    - Keep each segment concise (1-4 words).
    - Use title case.
    - Return only the folder path (e.g. "Finance/Invoices" or "Recipes").
      Do not explain your answer.
    """

    response = _create_completion(prompt, label="categorize_document")

    content = response.choices[0].message.content

    content = content.replace("```json", "")
    content = content.replace("```", "")
    content = content.strip()

    return _normalize_folder_path(content)


def verify_document_placement(document, final_folders):
    """
    Per-document sanity check, run AFTER all folders have been named.

    Clustering groups documents by embedding + keyword similarity, and
    occasionally one document ends up pulled into the wrong cluster --
    e.g. a user manual that happens to mention pricing, model numbers, or
    warranty terms drifting close enough to an invoice cluster to get
    swept in with it. categorize_document() only ever sees filenames per
    cluster, so it can't catch this.

    This checks ONE document at a time against the final folder list,
    using its actual extracted keywords -- a narrower, more reliable
    signal than the multi-document clustering decision. Returns the folder
    path this document should be filed under (may be the same one it
    already had).
    """

    prompt = f"""
    You are filing a single document into one of these existing folders.

    Folders:
    {final_folders}

    Document filename: {document.get('file_name')}
    Document keywords: {document.get('keywords', [])}

    Instructions:
    - Pick the single best-fitting folder from the list above.
    - Return the folder path EXACTLY as it appears in the list.
    - Only return something NOT in the list if none of them fit even loosely,
      in which case return a new folder name following the same rules used
      to create the folders above (concise, human-friendly, title case).
    - Return only the folder path. Do not explain your answer.
    """

    response = _create_completion(prompt, label="verify_document_placement")

    content = response.choices[0].message.content

    content = content.replace("```json", "")
    content = content.replace("```", "")
    content = content.strip()

    return _normalize_folder_path(content)