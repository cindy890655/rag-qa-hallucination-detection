import csv
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DENSE_RESULTS_PATH = (
    PROJECT_ROOT
    / "data"
    / "wiki"
    / "rag_results_k3_phi3.jsonl"
)

BM25_RESULTS_PATH = (
    PROJECT_ROOT
    / "data"
    / "wiki"
    / "rag_results_bm25_k3_tokens64.jsonl"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "research"
    / "retriever_comparison.csv"
)


def load_jsonl(file_path: Path) -> list[dict[str, Any]]:
    """
    Load all non-empty JSON objects from a JSONL file.
    """
    if not file_path.exists():
        raise FileNotFoundError(
            f"Results file not found: {file_path}"
        )

    records: list[dict[str, Any]] = []

    with file_path.open(
        mode="r",
        encoding="utf-8"
    ) as input_file:
        for line_number, line in enumerate(
            input_file,
            start=1
        ):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON in {file_path.name}, "
                    f"line {line_number}"
                ) from error

            records.append(record)

    if not records:
        raise ValueError(
            f"No records found in {file_path}"
        )

    return records


def index_by_question_id(
    records: list[dict[str, Any]],
    source_name: str
) -> dict[str, dict[str, Any]]:
    """
    Index experiment records by question_id and reject duplicates.
    """
    indexed_records: dict[str, dict[str, Any]] = {}

    for record in records:
        question_id = record.get("question_id")

        if question_id is None:
            raise ValueError(
                f"Missing question_id in {source_name}"
            )

        question_id = str(question_id)

        if question_id in indexed_records:
            raise ValueError(
                f"Duplicate question_id '{question_id}' "
                f"in {source_name}"
            )

        indexed_records[question_id] = record

    return indexed_records


def get_retrieved_titles(
    record: dict[str, Any]
) -> str:
    """
    Combine retrieved document titles into one readable CSV cell.
    """
    documents = record.get(
        "retrieved_documents",
        []
    )

    return " | ".join(
        str(document.get("title", "")).strip()
        for document in documents
        if str(document.get("title", "")).strip()
    )


def get_top1_title(
    record: dict[str, Any]
) -> str:
    """
    Return the title of the highest-ranked retrieved document.
    """
    documents = record.get(
        "retrieved_documents",
        []
    )

    if not documents:
        return ""

    return str(
        documents[0].get("title", "")
    )


def get_top1_score(
    record: dict[str, Any]
) -> float | str:
    """
    Return the score of the highest-ranked retrieved document.
    """
    documents = record.get(
        "retrieved_documents",
        []
    )

    if not documents:
        return ""

    score = documents[0].get("score")

    if score is None:
        return ""

    return float(score)


def validate_matching_questions(
    dense_records: dict[str, dict[str, Any]],
    bm25_records: dict[str, dict[str, Any]]
) -> None:
    """
    Ensure both experiment files contain the same questions.
    """
    dense_ids = set(dense_records)
    bm25_ids = set(bm25_records)

    if dense_ids != bm25_ids:
        missing_from_bm25 = sorted(
            dense_ids - bm25_ids
        )

        missing_from_dense = sorted(
            bm25_ids - dense_ids
        )

        raise ValueError(
            "Question IDs do not match.\n"
            f"Missing from BM25: {missing_from_bm25}\n"
            f"Missing from Dense: {missing_from_dense}"
        )

    for question_id in dense_ids:
        dense_question = str(
            dense_records[question_id].get(
                "question",
                ""
            )
        ).strip()

        bm25_question = str(
            bm25_records[question_id].get(
                "question",
                ""
            )
        ).strip()

        if dense_question != bm25_question:
            raise ValueError(
                f"Question text mismatch for "
                f"{question_id}"
            )


def create_comparison_rows(
    dense_list: list[dict[str, Any]],
    dense_records: dict[str, dict[str, Any]],
    bm25_records: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    """
    Create aligned Dense-vs-BM25 comparison rows.

    Dense file order is preserved.
    """
    comparison_rows: list[dict[str, Any]] = []

    for dense_original in dense_list:
        question_id = str(
            dense_original["question_id"]
        )

        dense = dense_records[question_id]
        bm25 = bm25_records[question_id]

        comparison_rows.append({
            "question_id": question_id,
            "category": dense.get(
                "category",
                ""
            ),
            "question": dense.get(
                "question",
                ""
            ),

            "Dense Answer": dense.get(
                "answer",
                ""
            ),
            "Dense Runtime": dense.get(
                "runtime_seconds",
                ""
            ),
            "Dense Top-1 Title": get_top1_title(
                dense
            ),
            "Dense Top-1 Score": get_top1_score(
                dense
            ),
            "Dense Retrieved Titles":
                get_retrieved_titles(dense),
            "Dense Score": "",

            "BM25 Answer": bm25.get(
                "answer",
                ""
            ),
            "BM25 Runtime": bm25.get(
                "runtime_seconds",
                ""
            ),
            "BM25 Top-1 Title": get_top1_title(
                bm25
            ),
            "BM25 Top-1 Score": get_top1_score(
                bm25
            ),
            "BM25 Retrieved Titles":
                get_retrieved_titles(bm25),
            "BM25 Score": "",

            "Preferred Retriever": "",
            "Notes": "",
        })

    return comparison_rows


def save_csv(
    rows: list[dict[str, Any]],
    output_path: Path
) -> None:
    """
    Save comparison rows to CSV.
    """
    if not rows:
        raise ValueError(
            "No comparison rows to save"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    fieldnames = list(rows[0].keys())

    with output_path.open(
        mode="w",
        encoding="utf-8-sig",
        newline=""
    ) as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """
    Build the Dense-versus-BM25 manual evaluation CSV.
    """
    dense_list = load_jsonl(
        DENSE_RESULTS_PATH
    )

    bm25_list = load_jsonl(
        BM25_RESULTS_PATH
    )

    dense_records = index_by_question_id(
        dense_list,
        "Dense results"
    )

    bm25_records = index_by_question_id(
        bm25_list,
        "BM25 results"
    )

    validate_matching_questions(
        dense_records,
        bm25_records
    )

    comparison_rows = create_comparison_rows(
        dense_list=dense_list,
        dense_records=dense_records,
        bm25_records=bm25_records
    )

    save_csv(
        rows=comparison_rows,
        output_path=OUTPUT_PATH
    )

    print("=" * 70)
    print("RETRIEVER COMPARISON CSV CREATED")
    print("=" * 70)
    print(
        f"Dense records: {len(dense_list)}"
    )
    print(
        f"BM25 records: {len(bm25_list)}"
    )
    print(
        f"Comparison rows: "
        f"{len(comparison_rows)}"
    )
    print(
        f"Output saved to: {OUTPUT_PATH}"
    )
    print()
    print(
        "Manual score values: "
        "1 = correct, "
        "0.5 = partially correct, "
        "0 = incorrect"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()