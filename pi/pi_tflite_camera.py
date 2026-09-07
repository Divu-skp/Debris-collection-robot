import argparse
import json
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

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


def prepare_input(
    frame: np.ndarray,
    image_size: int,
    input_details: dict,
) -> np.ndarray:
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb_frame, (image_size, image_size))
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


def open_camera(source: str | int, width: int, height: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera source {source}. "
            "Try --camera 1, or use a USB webcam first."
        )

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


class PiCamera2Capture:
    def __init__(self, width: int, height: int) -> None:
        from picamera2 import Picamera2

        self.camera = Picamera2()
        config = self.camera.create_preview_configuration(
            main={"size": (width, height), "format": "RGB888"}
        )
        self.camera.configure(config)
        self.camera.start()

    def read(self) -> tuple[bool, np.ndarray]:
        rgb_frame = self.camera.capture_array()
        bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        return True, bgr_frame

    def release(self) -> None:
        self.camera.stop()


def label_color(label: str, locked: bool) -> tuple[int, int, int]:
    if locked:
        return (0, 220, 0)

    if "bio" in label.lower() and "non" not in label.lower():
        return (0, 180, 0)

    return (0, 90, 255)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the TensorFlow Lite waste classifier on Raspberry Pi."
    )
    parser.add_argument("--model", default=str(PROJECT_ROOT / "models" / "waste_classifier.tflite"))
    parser.add_argument("--class-names", default=str(PROJECT_ROOT / "models" / "class_names.json"))
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument(
        "--source",
        choices=["opencv", "picamera2"],
        default="opencv",
        help="Use opencv for USB/V4L2 cameras or picamera2 for the Pi Camera Module.",
    )
    parser.add_argument("--camera", default="0")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--lock-threshold", type=float, default=0.95)
    parser.add_argument("--object-threshold", type=float, default=0.003)
    parser.add_argument("--predict-every", type=int, default=5)
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Print predictions without opening a display window.",
    )
    args = parser.parse_args()

    if args.lock_threshold > 1:
        args.lock_threshold = args.lock_threshold / 100

    model_path = Path(args.model)
    class_names_path = Path(args.class_names)

    if not model_path.exists():
        raise FileNotFoundError(f"TFLite model not found: {model_path}")

    class_names = load_class_names(class_names_path)

    interpreter = Interpreter(model_path=str(model_path), num_threads=2)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    model_input_shape = input_details["shape"]
    model_image_size = int(model_input_shape[1])

    if args.image_size != model_image_size:
        print(
            f"Using model input size {model_image_size} instead of "
            f"--image-size {args.image_size}."
        )
        args.image_size = model_image_size

    if args.source == "picamera2":
        cap = PiCamera2Capture(args.width, args.height)
    else:
        camera_source: str | int = args.camera
        if str(args.camera).isdigit():
            camera_source = int(args.camera)

        cap = open_camera(camera_source, args.width, args.height)

    label = "No object"
    confidence = 0.0
    frame_count = 0
    locked_label = None
    locked_confidence = 0.0
    object_score = 0.0

    print("Press q to quit. Press r to reset locked prediction.")

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
                label, confidence = predict_frame(
                    interpreter=interpreter,
                    frame=frame,
                    class_names=class_names,
                    image_size=args.image_size,
                    input_details=input_details,
                    output_details=output_details,
                )

                if confidence >= args.lock_threshold:
                    locked_label = label
                    locked_confidence = confidence
                    print(f"LOCKED: {locked_label} ({locked_confidence * 100:.1f}%)")
            else:
                label = "No object"
                confidence = 0.0

        shown_label = locked_label or label
        shown_confidence = locked_confidence if locked_label else confidence
        display_label = shown_label.replace("_", " ").title()

        if args.headless:
            if frame_count % 30 == 0:
                print(
                    f"{display_label}: {shown_confidence * 100:.1f}% "
                    f"| object score: {object_score:.4f}"
                )
            continue

        if locked_label:
            text = f"LOCKED: {display_label} ({shown_confidence * 100:.1f}%)"
            status_text = "Press r to reset"
        else:
            text = f"{display_label}: {shown_confidence * 100:.1f}%"
            status_text = f"Object score: {object_score:.4f}"

        color = label_color(shown_label, locked_label is not None)
        cv2.rectangle(frame, (20, 20), (760, 115), (0, 0, 0), -1)
        cv2.putText(
            frame,
            text,
            (35, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            color,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            status_text,
            (35, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (220, 220, 220),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("Waste Classifier TFLite", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r"):
            locked_label = None
            locked_confidence = 0.0
            label = "No object"
            confidence = 0.0

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
