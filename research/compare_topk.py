import csv
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent

WIKI_RESULTS_DIR = PROJECT_ROOT / "data" / "wiki"
EVALUATION_DIR = PROJECT_ROOT / "data" / "evaluation"

RESULT_FILES = {
    1: WIKI_RESULTS_DIR / "rag_results_k1_tokens64.jsonl",
    3: WIKI_RESULTS_DIR / "rag_results_k3_tokens64.jsonl",
    5: WIKI_RESULTS_DIR / "rag_results_k5_tokens64.jsonl",
}

OUTPUT_PATH = EVALUATION_DIR / "comparison_k1_k3_k5.csv"

VALID_LABELS = {"", "-1", "0", "1"}


def load_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    """
    Load a RAG JSONL result file and index records by question_id.
    """
    if not path.exists():
        raise FileNotFoundError(f"Result file not found: {path}")

    records: dict[str, dict[str, Any]] = {}

    with path.open(mode="r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON in {path} at line {line_number}"
                ) from error

            question_id = str(record.get("question_id", "")).strip()

            if not question_id:
                raise ValueError(
                    f"Missing question_id in {path} at line {line_number}"
                )

            if question_id in records:
                raise ValueError(
                    f"Duplicate question_id '{question_id}' in {path}"
                )

            records[question_id] = record

    if not records:
        raise ValueError(f"No records found in {path}")

    return records


def get_top_document(record: dict[str, Any]) -> tuple[str, str]:
    """
    Return the title and score of the top-ranked retrieved document.
    """
    documents = record.get("retrieved_documents", [])

    if not documents:
        return "", ""

    top_document = documents[0]
    title = str(top_document.get("title", "")).strip()
    score = top_document.get("score")

    if isinstance(score, (int, float)):
        score_text = f"{score:.6f}"
    else:
        score_text = ""

    return title, score_text


def get_retrieved_titles(record: dict[str, Any]) -> str:
    """
    Return all retrieved document titles in ranked order.
    """
    documents = record.get("retrieved_documents", [])
    titles = [
        str(document.get("title", "")).strip()
        for document in documents
        if str(document.get("title", "")).strip()
    ]
    return " | ".join(titles)


def load_existing_annotations(
    output_path: Path
) -> dict[str, dict[str, str]]:
    """
    Preserve manual labels and notes when regenerating the CSV.
    """
    if not output_path.exists():
        return {}

    annotations: dict[str, dict[str, str]] = {}

    with output_path.open(
        mode="r",
        encoding="utf-8-sig",
        newline=""
    ) as input_file:
        reader = csv.DictReader(input_file)

        for row in reader:
            question_id = str(row.get("question_id", "")).strip()

            if not question_id:
                continue

            annotations[question_id] = {
                "k1_label": str(row.get("k1_label", "")).strip(),
                "k3_label": str(row.get("k3_label", "")).strip(),
                "k5_label": str(row.get("k5_label", "")).strip(),
                "notes": str(row.get("notes", "")).strip(),
            }

    return annotations


def validate_label(label: str, column_name: str, question_id: str) -> str:
    """
    Keep only supported human-evaluation labels.
    """
    if label not in VALID_LABELS:
        raise ValueError(
            f"Invalid label '{label}' in {column_name} for {question_id}. "
            "Allowed values are -1, 0, 1, or blank."
        )

    return label


def check_question_sets(
    all_results: dict[int, dict[str, dict[str, Any]]]
) -> list[str]:
    """
    Ensure all experiment files contain the same question IDs.
    """
    question_sets = {
        top_k: set(records.keys())
        for top_k, records in all_results.items()
    }

    reference_k = min(question_sets)
    reference_ids = question_sets[reference_k]

    for top_k, question_ids in question_sets.items():
        if question_ids != reference_ids:
            missing = sorted(reference_ids - question_ids)
            extra = sorted(question_ids - reference_ids)

            raise ValueError(
                f"Question IDs do not match for K={top_k}. "
                f"Missing: {missing}; Extra: {extra}"
            )

    # Preserve the order from the K=1 file.
    return list(all_results[reference_k].keys())


def build_row(
    question_id: str,
    all_results: dict[int, dict[str, dict[str, Any]]],
    existing_annotations: dict[str, dict[str, str]]
) -> dict[str, Any]:
    """
    Build one combined comparison row.
    """
    records = {
        top_k: all_results[top_k][question_id]
        for top_k in RESULT_FILES
    }

    reference_record = records[1]
    annotation = existing_annotations.get(question_id, {})

    row: dict[str, Any] = {
        "question_id": question_id,
        "category": reference_record.get("category", ""),
        "question": reference_record.get("question", ""),
    }

    for top_k in (1, 3, 5):
        record = records[top_k]
        top_title, top_score = get_top_document(record)

        row[f"k{top_k}_answer"] = record.get("answer", "")
        row[f"k{top_k}_runtime_seconds"] = record.get(
            "runtime_seconds",
            ""
        )
        row[f"k{top_k}_top1_title"] = top_title
        row[f"k{top_k}_top1_score"] = top_score
        row[f"k{top_k}_retrieved_titles"] = get_retrieved_titles(
            record
        )

        saved_label = annotation.get(f"k{top_k}_label", "")
        row[f"k{top_k}_label"] = validate_label(
            saved_label,
            f"k{top_k}_label",
            question_id
        )

    row["notes"] = annotation.get("notes", "")

    return row


def main() -> None:
    """
    Combine K=1, K=3, and K=5 RAG outputs into one annotation CSV.
    """
    all_results = {
        top_k: load_jsonl(path)
        for top_k, path in RESULT_FILES.items()
    }

    question_ids = check_question_sets(all_results)
    existing_annotations = load_existing_annotations(OUTPUT_PATH)

    rows = [
        build_row(
            question_id=question_id,
            all_results=all_results,
            existing_annotations=existing_annotations
        )
        for question_id in question_ids
    ]

    fieldnames = [
        "question_id",
        "category",
        "question",
        "k1_answer",
        "k1_runtime_seconds",
        "k1_top1_title",
        "k1_top1_score",
        "k1_retrieved_titles",
        "k1_label",
        "k3_answer",
        "k3_runtime_seconds",
        "k3_top1_title",
        "k3_top1_score",
        "k3_retrieved_titles",
        "k3_label",
        "k5_answer",
        "k5_runtime_seconds",
        "k5_top1_title",
        "k5_top1_score",
        "k5_retrieved_titles",
        "k5_label",
        "notes",
    ]

    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open(
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

    print("=" * 70)
    print("COMPARISON CSV CREATED")
    print(f"Questions combined: {len(rows)}")
    print(f"Output: {OUTPUT_PATH}")
    print()
    print("Human labels:")
    print("  1  = correct")
    print("  0  = partially correct")
    print(" -1  = incorrect")
    print("=" * 70)


if __name__ == "__main__":
    main()
