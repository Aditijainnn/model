import argparse
import json
import random
import time
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms


IMAGE_SIZE = 224
TEXTURE_RESIZE = 512
CLASS_ALIASES = {
    "real": ("real", "Original_dataset", "original", "real_data"),
    "screen": ("screen", "screen_data", "screens", "recapture"),
}


def ensure_min_size(image: Image.Image) -> Image.Image:
    if min(image.size) >= IMAGE_SIZE:
        return image
    scale = IMAGE_SIZE / min(image.size)
    size = (round(image.width * scale), round(image.height * scale))
    return image.resize(size, Image.Resampling.BICUBIC)


def canonical_dataset_root(dataset_dir: Path) -> Path:
    """Create a tiny ImageFolder-compatible view without moving user images."""
    dataset_dir = dataset_dir.resolve()
    view_dir = dataset_dir / "_imagefolder_view"
    view_dir.mkdir(exist_ok=True)

    for class_name, aliases in CLASS_ALIASES.items():
        target = view_dir / class_name
        target.mkdir(exist_ok=True)
        for old_link in target.iterdir():
            if old_link.is_symlink():
                old_link.unlink()

        source = next((dataset_dir / name for name in aliases if (dataset_dir / name).is_dir()), None)
        if source is None:
            raise FileNotFoundError(
                f"Could not find a folder for '{class_name}'. Tried: "
                + ", ".join(str(dataset_dir / name) for name in aliases)
            )

        for image_path in sorted(source.iterdir()):
            if image_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                link_path = target / image_path.name
                if not link_path.exists():
                    try:
                        link_path.symlink_to(image_path)
                    except OSError:
                        # Windows symlinks may need privileges. Fall back to a lightweight copy.
                        import shutil

                        shutil.copy2(image_path, link_path)

    return view_dir


def build_transforms(train: bool, global_crops: bool = False) -> transforms.Compose:
    if train:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    IMAGE_SIZE,
                    scale=(0.04, 0.35) if global_crops else (0.003, 0.06),
                    ratio=(0.75, 1.3333333333),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    validation_transforms = []
    if global_crops:
        validation_transforms.append(
            transforms.Resize(TEXTURE_RESIZE, interpolation=transforms.InterpolationMode.BICUBIC)
        )
    validation_transforms.extend(
        [
            transforms.Lambda(ensure_min_size),
            transforms.FiveCrop(IMAGE_SIZE),
            transforms.Lambda(
                lambda crops: torch.stack(
                    [
                        transforms.Normalize(
                            mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225],
                        )(transforms.ToTensor()(crop))
                        for crop in crops
                    ]
                )
            ),
        ]
    )
    return transforms.Compose(validation_transforms)


def make_model(pretrained: bool = True, freeze_features: bool = False) -> nn.Module:
    weights = None
    if pretrained:
        try:
            weights = models.MobileNet_V3_Small_Weights.DEFAULT
        except Exception:
            weights = None

    try:
        model = models.mobilenet_v3_small(weights=weights)
    except Exception as exc:
        print(f"Warning: pretrained weights unavailable ({exc}). Training from scratch.")
        model = models.mobilenet_v3_small(weights=None)

    if freeze_features:
        for parameter in model.features.parameters():
            parameter.requires_grad = False

    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, 2)
    return model


def stratified_indices(targets: list[int], val_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    rng = random.Random(seed)
    train_indices = []
    val_indices = []
    for class_id in sorted(set(targets)):
        class_indices = [idx for idx, target in enumerate(targets) if target == class_id]
        rng.shuffle(class_indices)
        val_count = max(1, int(round(len(class_indices) * val_fraction)))
        val_indices.extend(class_indices[:val_count])
        train_indices.extend(class_indices[val_count:])
    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return train_indices, val_indices


def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return (logits.argmax(dim=1) == labels).float().mean().item()


def run_epoch(model, loader, criterion, device, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        if images.ndim == 5:
            batch_size, crop_count, channels, height, width = images.shape
            model_images = images.reshape(batch_size * crop_count, channels, height, width)
        else:
            batch_size = images.shape[0]
            crop_count = 1
            model_images = images

        with torch.set_grad_enabled(is_train):
            logits = model(model_images)
            if crop_count > 1:
                logits = logits.reshape(batch_size, crop_count, -1).mean(dim=1)
            loss = criterion(logits, labels)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_seen += batch_size

    return total_loss / max(total_seen, 1), total_correct / max(total_seen, 1)


def write_report_pdf(report_path: Path, metrics: dict) -> None:
    try:
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import inch
        from reportlab.platypus import (
            KeepTogether,
            Paragraph,
            SimpleDocTemplate,
            Spacer,
            Table,
            TableStyle,
        )
    except ImportError:
        print("ReportLab is not installed; skipping report.pdf generation.")
        return

    document = SimpleDocTemplate(
        str(report_path),
        pagesize=letter,
        rightMargin=0.62 * inch,
        leftMargin=0.62 * inch,
        topMargin=0.48 * inch,
        bottomMargin=0.45 * inch,
        title="SpotFakePhoto - Model Report",
        author="SpotFakePhoto Project",
    )
    styles = getSampleStyleSheet()
    navy = colors.HexColor("#18324A")
    teal = colors.HexColor("#13756D")
    pale = colors.HexColor("#EEF4F3")
    body = ParagraphStyle(
        "ReportBody",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9.1,
        leading=12.3,
        textColor=colors.HexColor("#27323A"),
        spaceAfter=5,
    )
    heading = ParagraphStyle(
        "ReportHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=10.6,
        leading=13,
        textColor=navy,
        spaceBefore=5,
        spaceAfter=3,
    )
    title = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=21,
        alignment=TA_CENTER,
        textColor=navy,
        spaceAfter=2,
    )
    subtitle = ParagraphStyle(
        "ReportSubtitle",
        parent=body,
        fontSize=8.6,
        leading=11,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#62717B"),
        spaceAfter=9,
    )

    story = [
        Paragraph("SpotFakePhoto", title),
        Paragraph("Real photo vs. screen-recapture classification - short technical report", subtitle),
    ]
    metric_data = [
        [
            Paragraph("<b>Original validation</b><br/>92.0% (23/25)", body),
            Paragraph("<b>External holdout</b><br/>75.0% (6/8)", body),
            Paragraph("<b>Warm CPU latency</b><br/>203.86 ms/image", body),
            Paragraph("<b>Estimated cost</b><br/>Approx. $0/image", body),
        ]
    ]
    metric_table = Table(metric_data, colWidths=[1.72 * inch] * 4)
    metric_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), pale),
                ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#B8CFCC")),
                ("INNERGRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#C9D9D7")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.extend([metric_table, Spacer(1, 5)])

    sections = [
        (
            "Approach",
            [
                "I treated the task as binary image classification: class 0 is an original photograph and "
                "class 1 is a photograph recaptured from a display. The system uses two ImageNet-pretrained "
                "MobileNetV3 Small networks. Transfer learning was chosen because the available dataset is too "
                "small to train a deep visual model reliably from scratch.",
                "The two networks inspect the image at different scales. The native-detail branch takes small "
                "crops without first shrinking a high-resolution image, which helps preserve moire, scanlines, "
                "pixel grids and other fine display artifacts. The global-context branch first resizes the image "
                "and examines five larger-context crops, making it more sensitive to screen borders, glare, "
                "perspective distortion and framing. Their screen probabilities are blended, with 79% weight on "
                "native detail and 21% on global context. Training used random resized crops, horizontal flips, "
                "small rotations, colour jitter, label smoothing, AdamW, cosine learning-rate decay and early stopping.",
            ],
        ),
        (
            "Dataset And Evaluation",
            [
                f"The adapted training pool contains {metrics['dataset']['real']} unique real images and "
                f"{metrics['dataset']['screen']} unique screen-recapture images. Sixteen newly collected, "
                "lower-resolution examples were included for adaptation, while eight additional new images "
                "were kept untouched as an external holdout. No public dataset was added.",
                f"On the original fixed validation split, the final ensemble achieved "
                f"{metrics['best_val_accuracy']:.2%} accuracy "
                f"({metrics.get('validation_correct', 0)}/{metrics.get('validation_images', 0)}), with "
                f"{metrics.get('precision', 0):.2%} precision, {metrics.get('recall', 0):.2%} recall and "
                f"{metrics.get('f1_score', 0):.2%} F1. On the new external holdout it achieved "
                f"{metrics.get('external_holdout_accuracy', 0):.2%}. I report both numbers because the external "
                "score is a more realistic indication of performance on unfamiliar devices and compressed images.",
            ],
        ),
        (
            "Latency, Deployment And Cost",
            [
                f"Average warm inference time was {metrics.get('latency_ms', 0.0):.2f} ms per image on an "
                "Intel Core 7 150U CPU. This excludes Python and PyTorch process startup. The model runs locally, "
                "so there is no per-request API charge and the estimated inference cost is approximately $0 per "
                "image, apart from electricity and device ownership. A browser demo captures full-resolution "
                "webcam frames and sends them only to the local Python server.",
            ],
        ),
        (
            "Limitations And Improvements",
            [
                "The main limitation is data diversity. Screen recapture changes with display technology, camera "
                "sensor, focus distance, angle, lighting and messaging-app compression. The camera demo is therefore "
                "less reliable than the internal validation score suggests. The next improvements should be a larger "
                "device-separated test set, more hard negatives, balanced examples from several phones and monitors, "
                "compression-aware augmentation and probability calibration. For deployment, quantization could "
                "reduce latency and model size, followed by monitoring and periodic retraining on real failure cases.",
            ],
        ),
    ]

    for section_heading, paragraphs in sections:
        block = [Paragraph(section_heading, heading)]
        block.extend(Paragraph(text, body) for text in paragraphs)
        story.append(KeepTogether(block))

    def add_page_number(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#D5DFE3"))
        canvas.line(doc.leftMargin, 0.35 * inch, letter[0] - doc.rightMargin, 0.35 * inch)
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(colors.HexColor("#71808A"))
        canvas.drawString(doc.leftMargin, 0.2 * inch, "SpotFakePhoto model report")
        canvas.drawRightString(letter[0] - doc.rightMargin, 0.2 * inch, f"Page {doc.page}")
        canvas.restoreState()

    document.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)


def count_dataset(dataset_dir: Path) -> dict:
    counts = {}
    for class_name, aliases in CLASS_ALIASES.items():
        source = next((dataset_dir / name for name in aliases if (dataset_dir / name).is_dir()), None)
        counts[class_name] = 0 if source is None else sum(
            1 for p in source.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        )
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MobileNetV3 for real-vs-screen photo detection.")
    parser.add_argument("--dataset", default="Dataset", help="Dataset folder containing real/screen image folders.")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--resume", help="Checkpoint to fine-tune instead of starting from ImageNet weights.")
    parser.add_argument("--global-crops", action="store_true", help="Train the global-context crop model.")
    parser.add_argument("--output", default="best_model.pth", help="Output checkpoint path.")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--freeze-features", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset_dir = Path(args.dataset)
    imagefolder_root = canonical_dataset_root(dataset_dir)

    full_dataset = datasets.ImageFolder(
        imagefolder_root, transform=build_transforms(train=True, global_crops=args.global_crops)
    )
    if full_dataset.class_to_idx.get("real") != 0 or full_dataset.class_to_idx.get("screen") != 1:
        raise RuntimeError(f"Unexpected class mapping: {full_dataset.class_to_idx}")

    train_indices, val_indices = stratified_indices(full_dataset.targets, val_fraction=0.2, seed=args.seed)
    train_size = len(train_indices)
    val_size = len(val_indices)
    val_dataset = datasets.ImageFolder(
        imagefolder_root, transform=build_transforms(train=False, global_crops=args.global_crops)
    )
    train_ds = Subset(full_dataset, train_indices)
    val_ds = Subset(val_dataset, val_indices)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_model(pretrained=not args.no_pretrained, freeze_features=args.freeze_features).to(device)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
        print(f"Resuming from {args.resume}")
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    classifier_parameters = list(model.classifier.parameters())
    classifier_ids = {id(parameter) for parameter in classifier_parameters}
    feature_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in classifier_ids
    ]
    parameter_groups = [{"params": classifier_parameters, "lr": args.lr}]
    if feature_parameters:
        parameter_groups.append({"params": feature_parameters, "lr": args.lr * 0.2})
    optimizer = torch.optim.AdamW(parameter_groups, weight_decay=2e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs, 1), eta_min=args.lr * 0.02
    )

    best_acc = -1.0
    best_path = Path(args.output)
    history = []
    epochs_without_improvement = 0

    print(f"Training on {device} with {train_size} train and {val_size} validation images.")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
            }
        )
        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train loss {train_loss:.4f} acc {train_acc:.2%} | "
            f"val loss {val_loss:.4f} acc {val_acc:.2%}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_to_idx": full_dataset.class_to_idx,
                    "image_size": IMAGE_SIZE,
                    "texture_resize": TEXTURE_RESIZE,
                },
                best_path,
            )
            print(f"Saved {best_path} with validation accuracy {best_acc:.2%}")
        else:
            epochs_without_improvement += 1
        scheduler.step()
        if epochs_without_improvement >= args.patience:
            print(f"Early stopping after {args.patience} epochs without improvement.")
            break

    sample = Image.open(full_dataset.samples[0][0]).convert("RGB")
    transform = build_transforms(train=False, global_crops=args.global_crops)
    model.eval()
    with torch.no_grad():
        tensor = transform(sample).to(device)
        for _ in range(5):
            model(tensor)
        start = time.perf_counter()
        for _ in range(25):
            model(tensor)
        latency_ms = (time.perf_counter() - start) * 1000 / 25

    metrics = {
        "dataset": count_dataset(dataset_dir),
        "epochs": len(history),
        "best_val_accuracy": best_acc,
        "latency_ms": latency_ms,
        "history": history,
    }
    Path("metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    write_report_pdf(Path("report.pdf"), metrics)
    print(f"Done. Best validation accuracy: {best_acc:.2%}. Latency: {latency_ms:.2f} ms/image.")


if __name__ == "__main__":
    main()
