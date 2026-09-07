import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
PROJECT_ROOT = Path(__file__).resolve().parent

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
)


def load_class_names(path: Path) -> list[str] | None:
    if not path.exists():
        return None

    return json.loads(path.read_text(encoding="utf-8"))


def plot_confusion_matrix(
    matrix: np.ndarray,
    class_names: list[str],
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=ax)

    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_yticklabels(class_names)

    threshold = matrix.max() / 2 if matrix.max() else 0
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            color = "white" if value > threshold else "black"
            ax.text(col, row, str(value), ha="center", va="center", color=color)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_history(history_path: Path, output_path: Path) -> None:
    history = json.loads(history_path.read_text(encoding="utf-8"))
    epochs = range(1, len(history.get("accuracy", [])) + 1)

    if not epochs:
        raise ValueError(f"No accuracy values found in {history_path}")

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(epochs, history["accuracy"], marker="o", label="Train Accuracy")
    if "val_accuracy" in history:
        ax.plot(epochs, history["val_accuracy"], marker="o", label="Val Accuracy")

    if "precision" in history:
        ax.plot(epochs, history["precision"], marker="s", label="Train Precision")
    if "val_precision" in history:
        ax.plot(epochs, history["val_precision"], marker="s", label="Val Precision")

    ax.set_title("Accuracy and Precision")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_metric_summary(accuracy: float, precision: float, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    names = ["Accuracy", "Weighted Precision"]
    values = [accuracy, precision]
    colors = ["#2f80ed", "#27ae60"]

    bars = ax.bar(names, values, color=colors, width=0.55)
    ax.set_title("Validation Metrics")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.25)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.02,
            f"{value * 100:.1f}%",
            ha="center",
            va="bottom",
            fontsize=11,
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create validation metrics and confusion matrix for the waste model."
    )
    parser.add_argument("--model", default=str(PROJECT_ROOT / "models" / "waste_classifier.keras"))
    parser.add_argument("--data-dir", default=str(PROJECT_ROOT / "dataset"))
    parser.add_argument("--split", default="val")
    parser.add_argument("--class-names", default=str(PROJECT_ROOT / "models" / "class_names.json"))
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--history", default=str(PROJECT_ROOT / "models" / "training_history.json"))
    parser.add_argument("--reports-dir", default=str(PROJECT_ROOT / "reports"))
    args = parser.parse_args()

    model_path = Path(args.model)
    split_dir = Path(args.data_dir) / args.split
    reports_dir = Path(args.reports_dir)
    class_names_path = Path(args.class_names)
    history_path = Path(args.history)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not split_dir.exists():
        raise FileNotFoundError(f"Dataset split not found: {split_dir}")

    reports_dir.mkdir(parents=True, exist_ok=True)

    dataset = tf.keras.utils.image_dataset_from_directory(
        split_dir,
        image_size=(args.image_size, args.image_size),
        batch_size=args.batch_size,
        label_mode="int",
        shuffle=False,
    )

    dataset_class_names = dataset.class_names
    saved_class_names = load_class_names(class_names_path)
    class_names = saved_class_names or dataset_class_names

    if class_names != dataset_class_names:
        raise ValueError(
            "Class names do not match the validation folders. "
            f"Model classes: {class_names}; Dataset classes: {dataset_class_names}"
        )

    model = tf.keras.models.load_model(model_path)

    y_true = np.concatenate([labels.numpy() for _images, labels in dataset])
    predictions = model.predict(dataset, verbose=1)
    y_pred = np.argmax(predictions, axis=1)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    matrix = confusion_matrix(y_true, y_pred)
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Weighted precision: {precision:.4f}")
    print(report)

    report_path = reports_dir / "classification_report.txt"
    report_path.write_text(
        f"Accuracy: {accuracy:.4f}\n"
        f"Weighted precision: {precision:.4f}\n\n"
        f"{report}",
        encoding="utf-8",
    )

    confusion_matrix_path = reports_dir / "confusion_matrix.png"
    plot_confusion_matrix(matrix, class_names, confusion_matrix_path)
    metrics_summary_path = reports_dir / "metrics_summary.png"
    plot_metric_summary(accuracy, precision, metrics_summary_path)
    print(f"Saved confusion matrix to: {confusion_matrix_path}")
    print(f"Saved metrics summary to: {metrics_summary_path}")
    print(f"Saved classification report to: {report_path}")

    if history_path.exists():
        history_plot_path = reports_dir / "accuracy_precision.png"
        plot_history(history_path, history_plot_path)
        print(f"Saved accuracy/precision plot to: {history_plot_path}")
    else:
        print(
            f"No history file found at {history_path}. "
            "Skipping accuracy/precision plot."
        )


if __name__ == "__main__":
    main()
