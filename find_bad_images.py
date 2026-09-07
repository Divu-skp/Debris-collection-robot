import argparse
import shutil
from pathlib import Path

import tensorflow as tf


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
PROJECT_ROOT = Path(__file__).resolve().parent


def destination_for(path: Path, root: Path, bad_dir: Path) -> Path:
    destination = bad_dir / path.relative_to(root)
    if not destination.exists():
        return destination

    stem = destination.stem
    suffix = destination.suffix
    parent = destination.parent

    counter = 1
    while True:
        candidate = parent / f"{stem}_{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def can_decode(path: Path) -> tuple[bool, str]:
    try:
        image_bytes = tf.io.read_file(str(path))
        image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
        _ = image.numpy()
        return True, ""
    except Exception as error:
        return False, str(error).splitlines()[0]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find corrupt images that can crash TensorFlow training."
    )
    parser.add_argument("--data-dir", default=str(PROJECT_ROOT / "dataset"))
    parser.add_argument("--bad-dir", default=str(PROJECT_ROOT / "bad_images"))
    parser.add_argument(
        "--move",
        action="store_true",
        help="Move corrupt images into --bad-dir, preserving folder structure.",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    bad_dir = Path(args.bad_dir)

    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset folder not found: {data_dir}")

    image_paths = [
        path
        for path in data_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]

    print(f"Checking {len(image_paths)} images in {data_dir}...")

    bad_images: list[tuple[Path, str]] = []

    for index, path in enumerate(image_paths, start=1):
        ok, reason = can_decode(path)
        if not ok:
            bad_images.append((path, reason))
            print(f"BAD: {path} | {reason}")

        if index % 500 == 0:
            print(f"Checked {index}/{len(image_paths)}...")

    if not bad_images:
        print("No corrupt images found.")
        return

    print(f"Found {len(bad_images)} corrupt image(s).")

    if not args.move:
        print("Run again with --move to move them out of the dataset.")
        return

    for path, _reason in bad_images:
        destination = destination_for(path, data_dir, bad_dir)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(path), str(destination))
        print(f"Moved: {path} -> {destination}")

    print(f"Moved {len(bad_images)} corrupt image(s) into {bad_dir}.")


if __name__ == "__main__":
    main()
