import csv
import json
from pathlib import Path


# Project root:
# final/
# ├── data/
# │   └── wiki/
# └── research/
#     └── compare_generators.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "wiki"

MODEL_FILES = {
    "FLAN Base": DATA_DIR / "rag_results_k3_flanbase.jsonl",
    "FLAN Large": DATA_DIR / "rag_results_k3_flanlarge.jsonl",
    "Phi-3": DATA_DIR / "rag_results_k3_phi3.jsonl",
}

OUTPUT_FILE = DATA_DIR / "generator_comparison.csv"


def load_jsonl(file_path: Path) -> dict:
    """
    Load a JSONL result file and index each result by question_id.

    Returns:
        Dictionary mapping question_id to each result record.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Result file not found: {file_path}")

    results = {}

    with file_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON in {file_path.name}, line {line_number}: {error}"
                ) from error

            question_id = record.get("question_id")

            if not question_id:
                raise ValueError(
                    f"Missing question_id in {file_path.name}, line {line_number}"
                )

            results[question_id] = record

    return results


def compare_generators() -> None:
    """Combine generator outputs into one CSV file for manual evaluation."""

    all_model_results = {}

    for model_name, file_path in MODEL_FILES.items():
        model_results = load_jsonl(file_path)
        all_model_results[model_name] = model_results

        print(
            f"Loaded {len(model_results)} questions "
            f"for {model_name} from {file_path.name}"
        )

    first_model_name = next(iter(MODEL_FILES))
    first_model_results = all_model_results[first_model_name]

    question_ids = list(first_model_results.keys())

    # Confirm that every model contains the same question IDs.
    expected_ids = set(question_ids)

    for model_name, model_results in all_model_results.items():
        current_ids = set(model_results.keys())

        missing_ids = expected_ids - current_ids
        extra_ids = current_ids - expected_ids

        if missing_ids or extra_ids:
            raise ValueError(
                f"Question mismatch for {model_name}. "
                f"Missing: {sorted(missing_ids)}; "
                f"Extra: {sorted(extra_ids)}"
            )

    fieldnames = [
        "question_id",
        "category",
        "question",
        "FLAN Base Answer",
        "FLAN Base Runtime",
        "FLAN Base Score",
        "FLAN Large Answer",
        "FLAN Large Runtime",
        "FLAN Large Score",
        "Phi-3 Answer",
        "Phi-3 Runtime",
        "Phi-3 Score",
    ]

    with OUTPUT_FILE.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for question_id in question_ids:
            base_record = first_model_results[question_id]

            row = {
                "question_id": question_id,
                "category": base_record.get("category", ""),
                "question": base_record.get("question", ""),
            }

            for model_name in MODEL_FILES:
                record = all_model_results[model_name][question_id]

                row[f"{model_name} Answer"] = record.get("answer", "")
                row[f"{model_name} Runtime"] = record.get(
                    "runtime_seconds",
                    "",
                )

                # Leave score blank for manual grading.
                row[f"{model_name} Score"] = ""

            writer.writerow(row)

    print("\nGenerator comparison completed.")
    print(f"Number of questions: {len(question_ids)}")
    print(f"Output saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    compare_generators()