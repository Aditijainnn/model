"""Predict the probability that an image is a photo taken of a screen.

Usage:
    python predict.py some_image.jpg

Prints one number from 0 to 1:
    0 = real photo, 1 = photo of a screen / recapture
"""

from pathlib import Path
import sys

import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms


IMAGE_SIZE = 224
TEXTURE_RESIZE = 512
MODEL_PATH = Path(__file__).with_name("best_model.pth")
GLOBAL_MODEL_PATH = Path(__file__).with_name("global_model.pth")
NATIVE_WEIGHT = 0.79


def ensure_min_size(image: Image.Image) -> Image.Image:
    if min(image.size) >= IMAGE_SIZE:
        return image
    scale = IMAGE_SIZE / min(image.size)
    size = (round(image.width * scale), round(image.height * scale))
    return image.resize(size, Image.Resampling.BICUBIC)


def build_transform() -> transforms.Compose:
    return transforms.Compose(
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


def build_global_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(TEXTURE_RESIZE, interpolation=transforms.InterpolationMode.BICUBIC),
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


def make_model() -> nn.Module:
    model = models.mobilenet_v3_small(weights=None)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, 2)
    return model


def load_model(model_path: Path = MODEL_PATH) -> nn.Module:
    if not model_path.exists():
        raise FileNotFoundError(
            f"{model_path.name} was not found. Train the model first with: python train.py"
        )

    checkpoint = torch.load(model_path, map_location="cpu")
    model = make_model()
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def predict_image(
    image: Image.Image,
    native_model: nn.Module,
    global_model: nn.Module,
) -> float:
    image = image.convert("RGB")
    with torch.no_grad():
        native_logits = native_model(build_transform()(image)).mean(dim=0)
        global_logits = global_model(build_global_transform()(image)).mean(dim=0)
        native_probability = torch.softmax(native_logits, dim=0)[1]
        global_probability = torch.softmax(global_logits, dim=0)[1]
    return float((NATIVE_WEIGHT * native_probability + (1 - NATIVE_WEIGHT) * global_probability).item())


def predict(image_path: str) -> float:
    native_model = load_model()
    global_model = load_model(GLOBAL_MODEL_PATH)
    return predict_image(Image.open(image_path), native_model, global_model)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python predict.py path/to/image.jpg")
    print(f"{predict(sys.argv[1]):.4f}")
