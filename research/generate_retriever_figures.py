from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# ---------------------------------------------------------------------
# File paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RETRIEVER_COMPARISON_CSV_PATH = (
    PROJECT_ROOT
    / "research"
    / "retriever_comparison.csv"
)

RETRIEVER_BAR_CHART_PATH = (
    PROJECT_ROOT
    / "research"
    / "retriever_bar.png"
)

RETRIEVER_TABLE_PATH = (
    PROJECT_ROOT
    / "research"
    / "retriever_table.png"
)


# ---------------------------------------------------------------------
# CSV column names
# ---------------------------------------------------------------------

DENSE_SCORE_COLUMN = "Dense Score"
BM25_SCORE_COLUMN = "BM25 Score"

DENSE_RUNTIME_COLUMN = "Dense Runtime"
BM25_RUNTIME_COLUMN = "BM25 Runtime"


# ---------------------------------------------------------------------
# Evaluation settings
# ---------------------------------------------------------------------

RETRIEVAL_METHODS = [
    "Dense",
    "BM25",
]

VALID_SCORES = {
    1.0,
    0.5,
    0.0,
}


def load_comparison_data(
    csv_path: Path
) -> pd.DataFrame:
    """
    Load the manually scored retriever comparison CSV.
    """
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Comparison CSV not found: {csv_path}"
        )

    dataframe = pd.read_csv(
        csv_path
    )

    required_columns = {
        DENSE_SCORE_COLUMN,
        BM25_SCORE_COLUMN,
        DENSE_RUNTIME_COLUMN,
        BM25_RUNTIME_COLUMN,
    }

    missing_columns = (
        required_columns
        - set(dataframe.columns)
    )

    if missing_columns:
        raise ValueError(
            "The comparison CSV is missing "
            f"required columns: "
            f"{sorted(missing_columns)}"
        )

    return dataframe


def clean_score_column(
    dataframe: pd.DataFrame,
    column_name: str
) -> pd.Series:
    """
    Convert a manual score column to numeric values and validate it.
    """
    scores = pd.to_numeric(
        dataframe[column_name],
        errors="coerce"
    )

    if scores.isna().any():
        invalid_rows = (
            scores[scores.isna()]
            .index
            .tolist()
        )

        raise ValueError(
            f"Column '{column_name}' contains "
            f"empty or invalid scores at rows: "
            f"{invalid_rows}"
        )

    invalid_scores = sorted(
        set(scores.unique())
        - VALID_SCORES
    )

    if invalid_scores:
        raise ValueError(
            f"Column '{column_name}' contains "
            f"invalid score values: "
            f"{invalid_scores}. "
            "Allowed values are 1, 0.5, and 0."
        )

    return scores.astype(float)


def clean_runtime_column(
    dataframe: pd.DataFrame,
    column_name: str
) -> pd.Series:
    """
    Convert a runtime column to numeric values.
    """
    runtimes = pd.to_numeric(
        dataframe[column_name],
        errors="coerce"
    )

    if runtimes.isna().any():
        invalid_rows = (
            runtimes[runtimes.isna()]
            .index
            .tolist()
        )

        raise ValueError(
            f"Column '{column_name}' contains "
            f"empty or invalid runtime values "
            f"at rows: {invalid_rows}"
        )

    if (runtimes < 0).any():
        raise ValueError(
            f"Column '{column_name}' contains "
            "negative runtime values"
        )

    return runtimes.astype(float)


def summarize_method(
    method_name: str,
    scores: pd.Series,
    runtimes: pd.Series
) -> dict:
    """
    Calculate evaluation statistics for one retrieval method.
    """
    total_questions = len(scores)

    if total_questions == 0:
        raise ValueError(
            f"No scores found for {method_name}"
        )

    correct_count = int(
        (scores == 1.0).sum()
    )

    partial_count = int(
        (scores == 0.5).sum()
    )

    incorrect_count = int(
        (scores == 0.0).sum()
    )

    mean_score = float(
        scores.mean()
    )

    average_runtime = float(
        runtimes.mean()
    )

    return {
        "Retrieval Method": method_name,
        "Correct": correct_count,
        "Partial": partial_count,
        "Incorrect": incorrect_count,
        "Mean Score": mean_score,
        "Avg. Runtime (s)": average_runtime,
    }


def build_summary_table(
    dataframe: pd.DataFrame
) -> pd.DataFrame:
    """
    Build the Dense-versus-BM25 summary table.
    """
    dense_scores = clean_score_column(
        dataframe,
        DENSE_SCORE_COLUMN
    )

    bm25_scores = clean_score_column(
        dataframe,
        BM25_SCORE_COLUMN
    )

    dense_runtimes = clean_runtime_column(
        dataframe,
        DENSE_RUNTIME_COLUMN
    )

    bm25_runtimes = clean_runtime_column(
        dataframe,
        BM25_RUNTIME_COLUMN
    )

    dense_summary = summarize_method(
        method_name="Dense",
        scores=dense_scores,
        runtimes=dense_runtimes
    )

    bm25_summary = summarize_method(
        method_name="BM25",
        scores=bm25_scores,
        runtimes=bm25_runtimes
    )

    summary_dataframe = pd.DataFrame([
        dense_summary,
        bm25_summary,
    ])

    return summary_dataframe


def add_segment_labels(
    axis: plt.Axes,
    bar_containers: list
) -> None:
    """
    Add count labels to non-empty stacked-bar segments.
    """
    for container in bar_containers:
        for bar in container:
            height = bar.get_height()

            if height <= 0:
                continue

            label_x = (
                bar.get_x()
                + bar.get_width() / 2
            )

            label_y = (
                bar.get_y()
                + height / 2
            )

            axis.text(
                label_x,
                label_y,
                f"{int(height)}",
                ha="center",
                va="center",
                fontsize=14
            )


def generate_bar_chart(
    summary_dataframe: pd.DataFrame,
    output_path: Path
) -> None:
    """
    Generate a stacked bar chart for retrieval-method evaluation.
    """
    method_names = (
        summary_dataframe[
            "Retrieval Method"
        ].tolist()
    )

    correct_counts = (
        summary_dataframe[
            "Correct"
        ].tolist()
    )

    partial_counts = (
        summary_dataframe[
            "Partial"
        ].tolist()
    )

    incorrect_counts = (
        summary_dataframe[
            "Incorrect"
        ].tolist()
    )

    figure, axis = plt.subplots(
        figsize=(12, 7)
    )

    correct_bars = axis.bar(
        method_names,
        correct_counts,
        label="Correct"
    )

    partial_bars = axis.bar(
        method_names,
        partial_counts,
        bottom=correct_counts,
        label="Partial"
    )

    partial_bottoms = [
        correct + partial
        for correct, partial in zip(
            correct_counts,
            partial_counts
        )
    ]

    incorrect_bars = axis.bar(
        method_names,
        incorrect_counts,
        bottom=partial_bottoms,
        label="Incorrect"
    )

    axis.set_title(
        "Human Evaluation by Retrieval Method",
        fontsize=22,
        fontweight="bold",
        pad=12
    )

    axis.set_xlabel(
        "Retrieval Method",
        fontsize=15
    )

    axis.set_ylabel(
        "Number of Questions",
        fontsize=15
    )

    total_questions = int(
        summary_dataframe.loc[
            0,
            [
                "Correct",
                "Partial",
                "Incorrect",
            ]
        ].sum()
    )

    axis.set_ylim(
        0,
        total_questions + 2
    )

    axis.tick_params(
        axis="both",
        labelsize=13
    )

    axis.grid(
        axis="y",
        alpha=0.3
    )

    axis.legend(
        fontsize=13
    )

    add_segment_labels(
        axis,
        [
            correct_bars,
            partial_bars,
            incorrect_bars,
        ]
    )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(figure)


def format_summary_for_table(
    summary_dataframe: pd.DataFrame
) -> list[list[str]]:
    """
    Convert summary values into display strings for the table image.
    """
    table_rows: list[list[str]] = []

    for _, row in summary_dataframe.iterrows():
        table_rows.append([
            str(
                row["Retrieval Method"]
            ),
            str(
                int(row["Correct"])
            ),
            str(
                int(row["Partial"])
            ),
            str(
                int(row["Incorrect"])
            ),
            f"{row['Mean Score']:.3f}",
            f"{row['Avg. Runtime (s)']:.2f}",
        ])

    return table_rows


def generate_table_image(
    summary_dataframe: pd.DataFrame,
    output_path: Path
) -> None:
    """
    Generate a PNG table containing retrieval evaluation statistics.
    """
    table_columns = [
        "Retrieval Method",
        "Correct",
        "Partial",
        "Incorrect",
        "Mean Score",
        "Avg. Runtime (s)",
    ]

    table_rows = format_summary_for_table(
        summary_dataframe
    )

    figure, axis = plt.subplots(
        figsize=(13, 4)
    )

    axis.axis("off")

    axis.set_title(
        "Retrieval Method Evaluation",
        fontsize=22,
        fontweight="bold",
        pad=24
    )

    table = axis.table(
        cellText=table_rows,
        colLabels=table_columns,
        cellLoc="center",
        colLoc="center",
        loc="center"
    )

    table.auto_set_font_size(False)

    table.set_fontsize(
        13
    )

    table.scale(
        1.0,
        2.0
    )

    for (
        row_index,
        column_index
    ), cell in table.get_celld().items():

        cell.set_linewidth(
            1.2
        )

        if row_index == 0:
            cell.set_text_props(
                fontweight="bold"
            )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(figure)


def print_summary(
    summary_dataframe: pd.DataFrame
) -> None:
    """
    Print the calculated retrieval statistics.
    """
    print("\n" + "=" * 70)
    print("RETRIEVER EVALUATION SUMMARY")
    print("=" * 70)

    for _, row in summary_dataframe.iterrows():
        print(
            f"\nRetrieval Method: "
            f"{row['Retrieval Method']}"
        )

        print(
            f"Correct: "
            f"{int(row['Correct'])}"
        )

        print(
            f"Partial: "
            f"{int(row['Partial'])}"
        )

        print(
            f"Incorrect: "
            f"{int(row['Incorrect'])}"
        )

        print(
            f"Mean Score: "
            f"{row['Mean Score']:.3f}"
        )

        print(
            f"Average Runtime: "
            f"{row['Avg. Runtime (s)']:.2f} seconds"
        )


def main() -> None:
    """
    Generate the retriever bar chart and summary table image.
    """
    comparison_dataframe = load_comparison_data(
        RETRIEVER_COMPARISON_CSV_PATH
    )

    summary_dataframe = build_summary_table(
        comparison_dataframe
    )

    generate_bar_chart(
        summary_dataframe=summary_dataframe,
        output_path=RETRIEVER_BAR_CHART_PATH
    )

    generate_table_image(
        summary_dataframe=summary_dataframe,
        output_path=RETRIEVER_TABLE_PATH
    )

    print_summary(
        summary_dataframe
    )

    print("\n" + "=" * 70)
    print("FIGURES CREATED")
    print("=" * 70)

    print(
        f"Bar chart saved to: "
        f"{RETRIEVER_BAR_CHART_PATH}"
    )

    print(
        f"Table image saved to: "
        f"{RETRIEVER_TABLE_PATH}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()