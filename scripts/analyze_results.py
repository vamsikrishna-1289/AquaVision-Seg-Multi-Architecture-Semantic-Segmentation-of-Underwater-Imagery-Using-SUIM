from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS_DIR = PROJECT_ROOT / "results"

CSV_FILE = RESULTS_DIR / "model_comparison.csv"

FIGURES_DIR = PROJECT_ROOT / "outputs" / "analysis"

FIGURES_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ============================================================
# LOAD RESULTS
# ============================================================

def load_results():
    """Load model comparison results."""

    if not CSV_FILE.exists():
        raise FileNotFoundError(
            f"Results file not found:\n{CSV_FILE}"
        )

    data = pd.read_csv(CSV_FILE)

    print("\nLoaded results:")
    print(data.to_string(index=False))

    return data


# ============================================================
# MODEL COMPARISON
# ============================================================

def create_model_comparison(data):
    """Create comparison chart for all models."""

    models = data["model"]

    # --------------------------------------------------------
    # Pixel Accuracy
    # --------------------------------------------------------

    plt.figure(figsize=(9, 6))

    plt.bar(
        models,
        data["pixel_accuracy"]
    )

    plt.ylabel("Pixel Accuracy")
    plt.xlabel("Model")
    plt.title("Model Comparison - Pixel Accuracy")

    plt.ylim(0, 1)

    plt.tight_layout()

    output = FIGURES_DIR / "pixel_accuracy_comparison.png"

    plt.savefig(output, dpi=300)

    plt.close()

    print(f"\nSaved: {output}")

    # --------------------------------------------------------
    # Mean IoU
    # --------------------------------------------------------

    plt.figure(figsize=(9, 6))

    plt.bar(
        models,
        data["mean_iou"]
    )

    plt.ylabel("Mean IoU")
    plt.xlabel("Model")
    plt.title("Model Comparison - Mean IoU")

    plt.ylim(0, 1)

    plt.tight_layout()

    output = FIGURES_DIR / "mean_iou_comparison.png"

    plt.savefig(output, dpi=300)

    plt.close()

    print(f"Saved: {output}")

    # --------------------------------------------------------
    # Mean Dice
    # --------------------------------------------------------

    plt.figure(figsize=(9, 6))

    plt.bar(
        models,
        data["mean_dice"]
    )

    plt.ylabel("Mean Dice")
    plt.xlabel("Model")
    plt.title("Model Comparison - Mean Dice")

    plt.ylim(0, 1)

    plt.tight_layout()

    output = FIGURES_DIR / "mean_dice_comparison.png"

    plt.savefig(output, dpi=300)

    plt.close()

    print(f"Saved: {output}")

    # --------------------------------------------------------
    # Combined comparison
    # --------------------------------------------------------

    plt.figure(figsize=(10, 6))

    x = range(len(models))

    width = 0.25

    plt.bar(
        [i - width for i in x],
        data["pixel_accuracy"],
        width=width,
        label="Pixel Accuracy"
    )

    plt.bar(
        x,
        data["mean_iou"],
        width=width,
        label="Mean IoU"
    )

    plt.bar(
        [i + width for i in x],
        data["mean_dice"],
        width=width,
        label="Mean Dice"
    )

    plt.xticks(
        list(x),
        models
    )

    plt.ylabel("Score")
    plt.xlabel("Model")

    plt.title("Segmentation Model Comparison")

    plt.ylim(0, 1)

    plt.legend()

    plt.tight_layout()

    output = FIGURES_DIR / "model_comparison.png"

    plt.savefig(output, dpi=300)

    plt.close()

    print(f"Saved: {output}")


# ============================================================
# BEST MODEL
# ============================================================

def find_best_model(data):
    """Find the best model based on Mean IoU."""

    best_index = data["mean_iou"].idxmax()

    best_model = data.loc[
        best_index,
        "model"
    ]

    best_iou = data.loc[
        best_index,
        "mean_iou"
    ]

    best_dice = data.loc[
        best_index,
        "mean_dice"
    ]

    best_accuracy = data.loc[
        best_index,
        "pixel_accuracy"
    ]

    print("\n" + "=" * 60)
    print("BEST MODEL")
    print("=" * 60)

    print(f"Model          : {best_model}")
    print(f"Pixel Accuracy : {best_accuracy:.4f}")
    print(f"Mean IoU       : {best_iou:.4f}")
    print(f"Mean Dice      : {best_dice:.4f}")

    print("=" * 60)

    return best_model


# ============================================================
# PER-CLASS COMPARISON
# ============================================================

def create_per_class_comparison(data):
    """Create per-class IoU comparison."""

    class_columns = [
        "background_iou",
        "human_diver_iou",
        "aquatic_plants_iou",
        "wrecks_ruins_iou",
        "robots_iou",
        "reefs_invertebrates_iou",
        "fish_vertebrates_iou",
        "sea_floor_rocks_iou",
    ]

    class_names = [
        "Background",
        "Human Diver",
        "Aquatic Plants",
        "Wrecks/Ruins",
        "Robots",
        "Reefs/Invertebrates",
        "Fish/Vertebrates",
        "Sea Floor/Rocks",
    ]

    # --------------------------------------------------------
    # Create table
    # --------------------------------------------------------

    comparison = data[
        ["model"] + class_columns
    ].copy()

    comparison = comparison.set_index("model")

    comparison.columns = class_names

    print("\n" + "=" * 60)
    print("PER-CLASS IoU")
    print("=" * 60)

    print(
        comparison.to_string(
            float_format=lambda x: f"{x:.4f}"
        )
    )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    plt.figure(figsize=(14, 7))

    for model in comparison.index:

        plt.plot(
            class_names,
            comparison.loc[model],
            marker="o",
            label=model
        )

    plt.ylabel("IoU")

    plt.xlabel("Segmentation Class")

    plt.title("Per-Class IoU Comparison")

    plt.xticks(
        rotation=45,
        ha="right"
    )

    plt.ylim(0, 1)

    plt.legend()

    plt.tight_layout()

    output = FIGURES_DIR / "per_class_iou_comparison.png"

    plt.savefig(output, dpi=300)

    plt.close()

    print(f"\nSaved: {output}")


# ============================================================
# PERFORMANCE SUMMARY
# ============================================================

def print_summary(data):
    """Print a concise performance summary."""

    print("\n" + "=" * 60)
    print("PERFORMANCE SUMMARY")
    print("=" * 60)

    for _, row in data.iterrows():

        print(
            f"\n{row['model']}"
        )

        print(
            f"  Pixel Accuracy : "
            f"{row['pixel_accuracy']:.4f}"
        )

        print(
            f"  Mean IoU       : "
            f"{row['mean_iou']:.4f}"
        )

        print(
            f"  Mean Dice      : "
            f"{row['mean_dice']:.4f}"
        )

    print("\n" + "=" * 60)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("MODEL RESULT ANALYSIS")
    print("=" * 60)

    # --------------------------------------------------------
    # Load results
    # --------------------------------------------------------

    data = load_results()

    # --------------------------------------------------------
    # Create comparison charts
    # --------------------------------------------------------

    create_model_comparison(data)

    # --------------------------------------------------------
    # Find best model
    # --------------------------------------------------------

    find_best_model(data)

    # --------------------------------------------------------
    # Per-class comparison
    # --------------------------------------------------------

    create_per_class_comparison(data)

    # --------------------------------------------------------
    # Print summary
    # --------------------------------------------------------

    print_summary(data)

    print("\n" + "=" * 60)
    print("RESULT ANALYSIS COMPLETE")
    print("=" * 60)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()