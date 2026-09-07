import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from arm_control import ServoArm


PROJECT_ROOT = Path(__file__).resolve().parent

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    import tensorflow as tf

    Interpreter = tf.lite.Interpreter


def load_class_names(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Class names file not found: {path}")

    return json.loads(path.read_text(encoding="utf-8"))


def frame_has_object(frame: np.ndarray, threshold: float) -> tuple[bool, float]:
    h, w = frame.shape[:2]
    crop = frame[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 40, 120)
    edge_density = np.count_nonzero(edges) / edges.size

    return edge_density > threshold, edge_density


def prepare_input(frame: np.ndarray, size: int, input_details: dict) -> np.ndarray:
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb_frame, (size, size))
    batch = np.expand_dims(resized, axis=0)

    input_dtype = input_details["dtype"]
    if input_dtype == np.float32:
        return batch.astype(np.float32)

    scale, zero_point = input_details["quantization"]
    if scale == 0:
        return batch.astype(input_dtype)

    quantized = batch.astype(np.float32) / scale + zero_point
    return np.clip(
        np.round(quantized),
        np.iinfo(input_dtype).min,
        np.iinfo(input_dtype).max,
    ).astype(input_dtype)


def dequantize_output(output: np.ndarray, output_details: dict) -> np.ndarray:
    if output_details["dtype"] == np.float32:
        return output.astype(np.float32)

    scale, zero_point = output_details["quantization"]
    return scale * (output.astype(np.float32) - zero_point)


def predict_frame(
    interpreter: Interpreter,
    frame: np.ndarray,
    class_names: list[str],
    image_size: int,
    input_details: dict,
    output_details: dict,
) -> tuple[str, float]:
    input_data = prepare_input(frame, image_size, input_details)

    interpreter.set_tensor(input_details["index"], input_data)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details["index"])[0]
    predictions = dequantize_output(output, output_details)
    class_index = int(np.argmax(predictions))
    confidence = float(predictions[class_index])

    return class_names[class_index], confidence


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Classify waste with webcam and sort it using the manipulator."
    )
    parser.add_argument("--model", default=str(PROJECT_ROOT / "models" / "waste_classifier.tflite"))
    parser.add_argument("--class-names", default=str(PROJECT_ROOT / "models" / "class_names.json"))
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--lock-threshold", type=float, default=0.95)
    parser.add_argument("--object-threshold", type=float, default=0.003)
    parser.add_argument("--predict-every", type=int, default=5)
    parser.add_argument("--cooldown", type=float, default=3.0)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print arm movements without moving GPIO servos.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Sort one detected object and exit.",
    )
    args = parser.parse_args()

    if args.lock_threshold > 1:
        args.lock_threshold = args.lock_threshold / 100

    class_names = load_class_names(Path(args.class_names))

    interpreter = Interpreter(model_path=args.model, num_threads=2)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    image_size = int(input_details["shape"][1])

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {args.camera}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    arm = ServoArm(dry_run=args.dry_run)

    frame_count = 0
    last_sort_time = 0.0

    print("Robot sorter started.")
    print("Press Ctrl+C to stop.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue

            frame_count += 1
            if frame_count % args.predict_every != 0:
                continue

            if time.time() - last_sort_time < args.cooldown:
                continue

            has_object, object_score = frame_has_object(
                frame,
                threshold=args.object_threshold,
            )

            if not has_object:
                print(f"No object | object score: {object_score:.4f}")
                continue

            label, confidence = predict_frame(
                interpreter=interpreter,
                frame=frame,
                class_names=class_names,
                image_size=image_size,
                input_details=input_details,
                output_details=output_details,
            )

            print(
                f"Detected {label} with {confidence * 100:.1f}% confidence "
                f"| object score: {object_score:.4f}"
            )

            if confidence < args.lock_threshold:
                continue

            print(f"LOCKED: {label}. Starting pick-and-place sequence.")
            arm.sort(label)
            last_sort_time = time.time()

            if args.once:
                break

    except KeyboardInterrupt:
        print("Stopping robot sorter.")
    finally:
        cap.release()
        arm.cleanup()


if __name__ == "__main__":
    main()
