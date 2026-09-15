import json
from pathlib import Path


def create_json_file(file_path):

    Path(file_path).touch(
        exist_ok=True
    )


def delete_json_file(file_path):

    path = Path(file_path)

    if path.exists():
        path.unlink()


def clear_json_file(file_path):

    with open(
        file_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            [],
            file,
            indent=4
        )


def save_json(
    file_path,
    data
):

    with open(
        file_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            data,
            file,
            indent=4,
            ensure_ascii=False
        )


def load_json(file_path):

    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


def update_document_field(
    file_path,
    document_id,
    field_name,
    value
):

    documents = load_json(
        file_path
    )

    for document in documents:

        if document["id"] == document_id:

            document[field_name] = value

            break

    save_json(
        file_path,
        documents
    )


def update_all_documents_field(
    file_path,
    field_name,
    values
):

    documents = load_json(
        file_path
    )

    value_lookup = {
        item["document_id"]: item[field_name]
        for item in values
    }

    for document in documents:

        document_id = document["id"]

        if document_id in value_lookup:

            document[field_name] = (
                value_lookup[document_id]
            )

    save_json(
        file_path,
        documents
    )


def add_field_to_all_documents(
    file_path,
    field_name,
    default_value=None
):

    documents = load_json(
        file_path
    )

    for document in documents:

        document[field_name] = default_value

    save_json(
        file_path,
        documents
    )


def remove_field_from_all_documents(
    file_path,
    field_name
):

    documents = load_json(
        file_path
    )

    for document in documents:

        document.pop(
            field_name,
            None
        )

    save_json(
        file_path,
        documents
    )


def append_document(
    file_path,
    document
):

    documents = load_json(
        file_path
    )

    documents.append(
        document
    )

    save_json(
        file_path,
        documents
    )

def get_document_fields(
    file_path,
    fields
):

    documents = load_json(
        file_path
    )

    result = []

    for document in documents:

        filtered_document = {}

        for field in fields:

            filtered_document[field] = (
                document.get(field)
            )

        result.append(
            filtered_document
        )

    return result