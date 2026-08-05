
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent

SUMMARY_CSV = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "topk_summary.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
)

TABLE_PNG = OUTPUT_DIR / "model_table.png"
BAR_PNG = OUTPUT_DIR / "model_bar.png"


def load_summary(path: Path) -> pd.DataFrame:
    """Load and validate the Top-K summary CSV."""
    if not path.exists():
        raise FileNotFoundError(
            f"Summary CSV not found: {path}\n"
            "Run summarize_results.py first."
        )

    df = pd.read_csv(path)

    required_columns = {
        "top_k",
        "correct",
        "partial",
        "incorrect",
        "mean_human_score",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    return df.sort_values("top_k").reset_index(drop=True)


def generate_table_png(
    df: pd.DataFrame,
    output_path: Path
) -> None:
    """Generate a clean table image for direct use in Word."""
    display_df = df[
        [
            "top_k",
            "correct",
            "partial",
            "incorrect",
            "mean_human_score",
        ]
    ].copy()

    display_df.columns = [
        "Top-K",
        "Correct",
        "Partial",
        "Incorrect",
        "Mean Score",
    ]

    display_df["Mean Score"] = display_df[
        "Mean Score"
    ].map(lambda value: f"{value:.2f}")

    fig, ax = plt.subplots(figsize=(8.5, 2.8))
    ax.axis("off")

    table = ax.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        cellLoc="center",
        colLoc="center",
        loc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.7)

    for column_index in range(len(display_df.columns)):
        table[(0, column_index)].set_text_props(
            weight="bold"
        )

    ax.set_title(
        "Human Evaluation of Different Top-K Settings",
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
    output_path: Path
) -> None:
    """Generate a stacked bar chart of human labels."""
    labels = [f"K={value}" for value in df["top_k"]]

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

    ax.set_title(
        "Human Evaluation by Top-K Setting",
        fontsize=14,
        weight="bold",
    )
    ax.set_xlabel("Retrieval Top-K")
    ax.set_ylabel("Number of Questions")
    ax.set_ylim(0, df[
        ["correct", "partial", "incorrect"]
    ].sum(axis=1).max() + 2)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def main() -> None:
    """Generate Word-ready PNG results from the summary CSV."""
    df = load_summary(SUMMARY_CSV)

    generate_table_png(df, TABLE_PNG)
    generate_stacked_bar_png(df, BAR_PNG)

    print("Generated:")
    print(f"- {TABLE_PNG}")
    print(f"- {BAR_PNG}")


if __name__ == "__main__":
    main()
