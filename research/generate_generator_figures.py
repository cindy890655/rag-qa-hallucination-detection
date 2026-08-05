from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

COMPARISON_CSV = (
    PROJECT_ROOT
    / "data"
    / "wiki"
    / "generator_comparison.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
)

SUMMARY_CSV = OUTPUT_DIR / "generator_summary.csv"
TABLE_PNG = OUTPUT_DIR / "generator_table.png"
BAR_PNG = OUTPUT_DIR / "generator_bar.png"
RUNTIME_PNG = OUTPUT_DIR / "generator_runtime.png"


MODEL_COLUMNS = {
    "FLAN Base": {
        "score": "FLAN Base Score",
        "runtime": "FLAN Base Runtime",
    },
    "FLAN Large": {
        "score": "FLAN Large Score",
        "runtime": "FLAN Large Runtime",
    },
    "Phi-3": {
        "score": "Phi-3 Score",
        "runtime": "Phi-3 Runtime",
    },
}


def load_comparison(path: Path) -> pd.DataFrame:
    """Load and validate the generator comparison CSV."""
    if not path.exists():
        raise FileNotFoundError(
            f"Generator comparison CSV not found: {path}"
        )

    df = pd.read_csv(path)

    required_columns = set()

    for columns in MODEL_COLUMNS.values():
        required_columns.add(columns["score"])
        required_columns.add(columns["runtime"])

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    return df


def create_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize score and runtime results for each generator."""
    summary_rows = []

    for model_name, columns in MODEL_COLUMNS.items():
        score_column = columns["score"]
        runtime_column = columns["runtime"]

        scores = pd.to_numeric(
            df[score_column],
            errors="coerce",
        ).dropna()

        runtimes = pd.to_numeric(
            df[runtime_column],
            errors="coerce",
        ).dropna()

        if scores.empty:
            raise ValueError(
                f"No valid scores found for {model_name}"
            )

        summary_rows.append(
            {
                "model": model_name,
                "correct": int((scores == 1.0).sum()),
                "partial": int((scores == 0.5).sum()),
                "incorrect": int((scores == 0.0).sum()),
                "mean_human_score": scores.mean(),
                "average_runtime": runtimes.mean(),
            }
        )

    return pd.DataFrame(summary_rows)


def generate_table_png(
    df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Generate a summary table image."""
    display_df = df[
        [
            "model",
            "correct",
            "partial",
            "incorrect",
            "mean_human_score",
            "average_runtime",
        ]
    ].copy()

    display_df.columns = [
        "Model",
        "Correct",
        "Partial",
        "Incorrect",
        "Mean Score",
        "Avg. Runtime (s)",
    ]

    display_df["Mean Score"] = display_df[
        "Mean Score"
    ].map(lambda value: f"{value:.3f}")

    display_df["Avg. Runtime (s)"] = display_df[
        "Avg. Runtime (s)"
    ].map(lambda value: f"{value:.2f}")

    fig, ax = plt.subplots(figsize=(10, 2.8))
    ax.axis("off")

    table = ax.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        cellLoc="center",
        colLoc="center",
        loc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.7)

    for column_index in range(len(display_df.columns)):
        table[(0, column_index)].set_text_props(
            weight="bold"
        )

    ax.set_title(
        "Generator Model Evaluation",
        fontsize=14,
        pad=16,
        weight="bold",
    )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def generate_stacked_bar_png(
    df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Generate a stacked bar chart of manual evaluation labels."""
    labels = df["model"]

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.bar(
        labels,
        df["correct"],
        label="Correct",
    )

    ax.bar(
        labels,
        df["partial"],
        bottom=df["correct"],
        label="Partial",
    )

    ax.bar(
        labels,
        df["incorrect"],
        bottom=df["correct"] + df["partial"],
        label="Incorrect",
    )

    for index, row in df.iterrows():
        ax.text(
            index,
            row["correct"] / 2,
            str(int(row["correct"])),
            ha="center",
            va="center",
            fontsize=10,
        )

        if row["partial"] > 0:
            ax.text(
                index,
                row["correct"] + row["partial"] / 2,
                str(int(row["partial"])),
                ha="center",
                va="center",
                fontsize=10,
            )

        if row["incorrect"] > 0:
            ax.text(
                index,
                row["correct"]
                + row["partial"]
                + row["incorrect"] / 2,
                str(int(row["incorrect"])),
                ha="center",
                va="center",
                fontsize=10,
            )

    total_questions = (
        df[["correct", "partial", "incorrect"]]
        .sum(axis=1)
        .max()
    )

    ax.set_title(
        "Human Evaluation by Generator Model",
        fontsize=14,
        weight="bold",
    )

    ax.set_xlabel("Generator Model")
    ax.set_ylabel("Number of Questions")
    ax.set_ylim(0, total_questions + 2)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def generate_runtime_png(
    df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Generate an average runtime bar chart."""
    fig, ax = plt.subplots(figsize=(8, 5))

    bars = ax.bar(
        df["model"],
        df["average_runtime"],
    )

    for bar, runtime in zip(
        bars,
        df["average_runtime"],
    ):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.15,
            f"{runtime:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_title(
        "Average Generation Runtime by Model",
        fontsize=14,
        weight="bold",
    )

    ax.set_xlabel("Generator Model")
    ax.set_ylabel("Average Runtime per Question (seconds)")
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def main() -> None:
    """Generate generator evaluation results and figures."""
    comparison_df = load_comparison(COMPARISON_CSV)
    summary_df = create_summary(comparison_df)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_df.to_csv(
        SUMMARY_CSV,
        index=False,
    )

    generate_table_png(
        summary_df,
        TABLE_PNG,
    )

    generate_stacked_bar_png(
        summary_df,
        BAR_PNG,
    )

    generate_runtime_png(
        summary_df,
        RUNTIME_PNG,
    )

    print("\nGenerator summary:")
    print(summary_df.to_string(index=False))

    print("\nGenerated:")
    print(f"- {SUMMARY_CSV}")
    print(f"- {TABLE_PNG}")
    print(f"- {BAR_PNG}")
    print(f"- {RUNTIME_PNG}")


if __name__ == "__main__":
    main()