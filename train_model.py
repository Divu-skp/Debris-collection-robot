import argparse
import json
from pathlib import Path

import tensorflow as tf


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
PROJECT_ROOT = Path(__file__).resolve().parent


def count_images(class_dir: Path) -> int:
    return sum(
        1
        for path in class_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def build_class_weights(train_dir: Path, class_names: list[str]) -> dict[int, float]:
    counts = [count_images(train_dir / class_name) for class_name in class_names]
    total = sum(counts)

    if total == 0 or any(count == 0 for count in counts):
        return {}

    return {
        class_index: total / (len(class_names) * count)
        for class_index, count in enumerate(counts)
    }


def build_model(num_classes: int, image_size: int, learning_rate: float) -> tf.keras.Model:
    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.08),
            tf.keras.layers.RandomZoom(0.12),
            tf.keras.layers.RandomContrast(0.12),
        ],
        name="data_augmentation",
    )

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(image_size, image_size, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(image_size, image_size, 3))
    x = data_augmentation(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs, name="waste_classifier")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a biodegradable vs non-biodegradable waste classifier."
    )
    parser.add_argument("--data-dir", default=str(PROJECT_ROOT / "dataset"), help="Dataset root folder.")
    parser.add_argument("--output", default=str(PROJECT_ROOT / "models" / "waste_classifier.keras"))
    parser.add_argument("--class-names-output", default=str(PROJECT_ROOT / "models" / "class_names.json"))
    parser.add_argument("--history-output", default=str(PROJECT_ROOT / "models" / "training_history.json"))
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--fine-tune-epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--fine-tune-learning-rate", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    output_path = Path(args.output)
    class_names_path = Path(args.class_names_output)
    history_path = Path(args.history_output)

    if not train_dir.exists():
        raise FileNotFoundError(f"Training folder not found: {train_dir}")
    if not val_dir.exists():
        raise FileNotFoundError(f"Validation folder not found: {val_dir}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    class_names_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.parent.mkdir(parents=True, exist_ok=True)

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=(args.image_size, args.image_size),
        batch_size=args.batch_size,
        label_mode="int",
        shuffle=True,
        seed=args.seed,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        image_size=(args.image_size, args.image_size),
        batch_size=args.batch_size,
        label_mode="int",
        shuffle=False,
    )

    class_names = train_ds.class_names
    val_class_names = val_ds.class_names
    if class_names != val_class_names:
        raise ValueError(
            "Train and validation class folders must match exactly. "
            f"Train: {class_names}; Val: {val_class_names}"
        )

    class_names_path.write_text(json.dumps(class_names, indent=2), encoding="utf-8")
    print(f"Classes: {class_names}")
    print(f"Saved class names to: {class_names_path}")

    class_weights = build_class_weights(train_dir, class_names)
    if class_weights:
        print(f"Using class weights: {class_weights}")

    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    model = build_model(
        num_classes=len(class_names),
        image_size=args.image_size,
        learning_rate=args.learning_rate,
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            output_path,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=4,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=2,
            min_lr=1e-7,
            verbose=1,
        ),
    ]

    print("Starting feature-extraction training...")
    histories = []
    histories.append(model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        class_weight=class_weights or None,
        callbacks=callbacks,
    ))

    if args.fine_tune_epochs > 0:
        print("Starting fine-tuning...")
        base_model = next(
            layer
            for layer in model.layers
            if isinstance(layer, tf.keras.Model) and "mobilenet" in layer.name.lower()
        )
        base_model.trainable = True

        for layer in base_model.layers[:-30]:
            layer.trainable = False

        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=args.fine_tune_learning_rate
            ),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        histories.append(model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.epochs + args.fine_tune_epochs,
            initial_epoch=args.epochs,
            class_weight=class_weights or None,
            callbacks=callbacks,
        ))

    combined_history = {}
    for history in histories:
        for key, values in history.history.items():
            combined_history.setdefault(key, []).extend(float(value) for value in values)
    history_path.write_text(json.dumps(combined_history, indent=2), encoding="utf-8")
    print(f"Saved training history to: {history_path}")

    model.save(output_path)
    print(f"Saved trained model to: {output_path}")


if __name__ == "__main__":
    main()
