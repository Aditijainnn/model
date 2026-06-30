import argparse
import json
import time
from pathlib import Path

import torch
from PIL import Image
from torchvision import datasets

from predict import (
    GLOBAL_MODEL_PATH,
    NATIVE_WEIGHT,
    build_global_transform,
    build_transform,
    load_model,
)
from train import CLASS_ALIASES, canonical_dataset_root, count_dataset, stratified_indices


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def iter_labeled_images(dataset_dir: Path):
    for label_name, aliases in CLASS_ALIASES.items():
        label = 0 if label_name == "real" else 1
        source = next((dataset_dir / name for name in aliases if (dataset_dir / name).is_dir()), None)
        if source is None:
            continue
        for path in sorted(source.iterdir()):
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                yield path, label


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate held-out accuracy, precision, recall, F1, and latency.")
    parser.add_argument("--dataset", default="Dataset")
    parser.add_argument("--model", default="best_model.pth")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--all-data", action="store_true", help="Diagnostic only; includes training images.")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    model = load_model(Path(args.model))
    global_model = load_model(GLOBAL_MODEL_PATH)
    transform = build_transform()
    global_transform = build_global_transform()
    imagefolder = datasets.ImageFolder(canonical_dataset_root(dataset_dir))
    _, val_indices = stratified_indices(imagefolder.targets, val_fraction=0.2, seed=args.seed)
    held_out_keys = {
        (imagefolder.classes[imagefolder.samples[index][1]], Path(imagefolder.samples[index][0]).name)
        for index in val_indices
    }

    y_true = []
    y_pred = []
    latencies = []

    for image_path, label in iter_labeled_images(dataset_dir):
        label_name = "real" if label == 0 else "screen"
        if not args.all_data and (label_name, image_path.name) not in held_out_keys:
            continue
        image = Image.open(image_path).convert("RGB")
        tensor = transform(image)
        start = time.perf_counter()
        with torch.no_grad():
            native_probability = torch.softmax(model(tensor).mean(dim=0), dim=0)[1]
            global_probability = torch.softmax(
                global_model(global_transform(image)).mean(dim=0), dim=0
            )[1]
            probability = (
                NATIVE_WEIGHT * native_probability + (1 - NATIVE_WEIGHT) * global_probability
            ).item()
        latencies.append((time.perf_counter() - start) * 1000)
        y_true.append(label)
        y_pred.append(1 if probability >= 0.5 else 0)

    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)

    accuracy = (tp + tn) / max(len(y_true), 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)

    metrics = {
        "evaluation_split": "all data (diagnostic only)" if args.all_data else "held-out 20%",
        "evaluated_images": len(y_true),
        "dataset": count_dataset(dataset_dir),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "latency_ms_per_image": sum(latencies) / max(len(latencies), 1),
        "cost_per_image_usd": 0.0,
    }

    print(json.dumps(metrics, indent=2))
    Path("evaluation.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
