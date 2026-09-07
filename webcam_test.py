import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf


PROJECT_ROOT = Path(__file__).resolve().parent

def load_class_names(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(
            f"Class names file not found: {path}. Run train_model.py first."
        )

    return json.loads(path.read_text(encoding="utf-8"))


def frame_has_object(frame, threshold=0.006) -> tuple[bool, float]:
    h, w = frame.shape[:2]

    # Center box where you should hold/show the waste item.
    crop = frame[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(gray, 40, 120)
    edge_density = np.count_nonzero(edges) / edges.size

    return edge_density > threshold, edge_density

def predict_frame(
    model: tf.keras.Model,
    frame: np.ndarray,
    class_names: list[str],
    image_size: int,
) -> tuple[str, float]:
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb_frame, (image_size, image_size))
    batch = np.expand_dims(resized.astype(np.float32), axis=0)

    predictions = model.predict(batch, verbose=0)[0]
    class_index = int(np.argmax(predictions))
    confidence = float(predictions[class_index])

    return class_names[class_index], confidence


def label_color(label: str, locked: bool) -> tuple[int, int, int]:
    if locked:
        return (0, 220, 0)

    if "bio" in label.lower() and "non" not in label.lower():
        return (0, 180, 0)
    if "non" in label.lower():
        return(0,90,255)

    return (0, 90, 255)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run live webcam classification using a trained waste model."
    )
    parser.add_argument("--model", default=str(PROJECT_ROOT / "models" / "waste_classifier.keras"))
    parser.add_argument("--class-names", default=str(PROJECT_ROOT / "models" / "class_names.json"))
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--lock-threshold", type=float, default=0.90)
    parser.add_argument("--lock-frames", type=int, default=1)
    parser.add_argument("--predict-every", type=int, default=5)
    parser.add_argument("--object-threshold", type=float, default=0.006)
    args = parser.parse_args()
    if args.lock_threshold > 1:
        args.lock_threshold = args.lock_threshold / 100

    model_path = Path(args.model)
    class_names_path = Path(args.class_names)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}. Run train_model.py first.")

    class_names = load_class_names(class_names_path)
    model = tf.keras.models.load_model(model_path)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera index {args.camera}. Try --camera 1 if needed."
        )

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    print("Press q to quit.")
    print("Press r to reset locked prediction.")

    label = "Starting..."
    confidence = 0.0
    frame_count = 0

    locked_label = None
    locked_confidence = 0.0
    candidate_label = None
    candidate_count = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_count += 1

        if locked_label is None and frame_count % args.predict_every == 0:
            has_object, object_score = frame_has_object(
                frame,
                threshold=args.object_threshold,
            )

            if has_object:
                predicted_label, predicted_confidence = predict_frame(
                    model=model,
                    frame=frame,
                    class_names=class_names,
                    image_size=args.image_size,
                )
                label, confidence = predicted_label, predicted_confidence

                if confidence >= args.lock_threshold:
                    if candidate_label == label:
                        candidate_count += 1
                    else:
                        candidate_label = label
                        candidate_count = 1

                    if candidate_count >= args.lock_frames:
                        locked_label = label
                        locked_confidence = confidence
                else:
                    candidate_label = None
                    candidate_count = 0
            else:
                label = "No object"
                confidence = 0.0
                candidate_label = None
                candidate_count = 0

        shown_label = locked_label or label
        shown_confidence = locked_confidence if locked_label else confidence
        display_label = shown_label.replace("_", " ").title()

        if locked_label:
            text = f"LOCKED: {display_label} ({shown_confidence * 100:.1f}%)"
            status_text = "Press r to reset"
        else:
            text = f"{display_label}: {shown_confidence * 100:.1f}%"
            progress = min(candidate_count, args.lock_frames)
            status_text = (
                f"Locking at {args.lock_threshold * 100:.0f}%: "
                f"{progress}/{args.lock_frames}"
            )

        color = label_color(shown_label, locked_label is not None)

        cv2.rectangle(frame, (20, 20), (780, 115), (0, 0, 0), -1)
        cv2.putText(
            frame,
            text,
            (35, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.95,
            color,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            status_text,
            (35, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (220, 220, 220),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("Waste Classifier", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r"):
            locked_label = None
            locked_confidence = 0.0
            candidate_label = None
            candidate_count = 0
            label = "Starting..."
            confidence = 0.0

    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()