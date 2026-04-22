from __future__ import annotations

import csv
import io
import sys
from pathlib import Path

import numpy as np
import streamlit as st
import torch
from PIL import Image


APP_DIR = Path(__file__).resolve().parent
Q2_DIR = APP_DIR.parent
if str(Q2_DIR) not in sys.path:
    sys.path.insert(0, str(Q2_DIR))

from dataloader import build_sample_pairs, train_test_split_pairs
from train_unet import NUM_CLASSES, UNetSmall


ARTIFACT_DIR = APP_DIR
MODEL_PATH = ARTIFACT_DIR / "best_unet_model.pt"
METRICS_CSV_PATH = ARTIFACT_DIR / "metrics_history.csv"
TEST_METRICS_PATH = ARTIFACT_DIR / "test_metrics.json"
LOSS_CURVE_PATH = ARTIFACT_DIR / "train_loss_curve.svg"
MIOU_CURVE_PATH = ARTIFACT_DIR / "train_miou_curve.svg"
MDICE_CURVE_PATH = ARTIFACT_DIR / "train_mdice_curve.svg"
IMAGE_SIZE = (160, 120)
DATA_SEED = 42


def build_palette(num_classes: int) -> np.ndarray:
    palette = []
    for class_index in range(num_classes):
        palette.append(
            [
                (37 * class_index) % 256,
                (67 * class_index + 53) % 256,
                (97 * class_index + 101) % 256,
            ]
        )
    return np.asarray(palette, dtype=np.uint8)


PALETTE = build_palette(NUM_CLASSES)


@st.cache_data(show_spinner=False)
def load_history() -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with METRICS_CSV_PATH.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            rows.append(
                {
                    "epoch": int(row["epoch"]),
                    "train_loss": float(row["train_loss"]),
                    "train_miou": float(row["train_miou"]),
                    "train_mdice": float(row["train_mdice"]),
                    "test_loss": float(row["test_loss"]),
                    "test_miou": float(row["test_miou"]),
                    "test_mdice": float(row["test_mdice"]),
                }
            )
    return rows


@st.cache_data(show_spinner=False)
def load_test_summary() -> dict[str, float | int | list[int]]:
    import json

    return json.loads(TEST_METRICS_PATH.read_text(encoding="utf-8"))


@st.cache_resource(show_spinner=False)
def load_model() -> UNetSmall:
    model = UNetSmall(num_classes=NUM_CLASSES)
    try:
        state_dict = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    except TypeError:
        state_dict = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


@st.cache_data(show_spinner=False)
def get_test_sample_lookup() -> dict[str, dict[str, Path]]:
    samples = build_sample_pairs(
        rgb_dir=Q2_DIR / "data" / "CameraRGB",
        mask_dir=Q2_DIR / "data" / "CameraMask",
    )
    _, test_samples = train_test_split_pairs(samples, seed=DATA_SEED)

    lookup: dict[str, dict[str, Path]] = {}
    for sample in test_samples:
        lookup[sample.image_path.name] = {
            "image_path": sample.image_path,
            "mask_path": sample.mask_path,
        }
    return lookup


def preprocess_uploaded_image(image_bytes: bytes) -> tuple[np.ndarray, torch.Tensor]:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    resized = image.resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
    image_np = np.asarray(resized, dtype=np.float32) / 255.0
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0).contiguous()
    return np.asarray(image), image_tensor


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    return PALETTE[mask]


def predict_mask(image_tensor: torch.Tensor) -> np.ndarray:
    model = load_model()
    with torch.no_grad():
        logits = model(image_tensor)
    prediction = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
    return prediction


def render_metrics_page() -> None:
    st.title("Question 2 Segmentation Dashboard")
    st.caption("Training curves from the final run and evaluation scores on the held-out test set.")

    history = load_history()
    summary = load_test_summary()

    metric_cols = st.columns(4)
    metric_cols[0].metric("Final Test mIoU", f"{summary['final_test_miou']:.4f}")
    metric_cols[1].metric("Final Test mDice", f"{summary['final_test_mdice']:.4f}")
    metric_cols[2].metric("Best Test mIoU", f"{summary['best_test_miou']:.4f}")
    metric_cols[3].metric("Best Test mDice", f"{summary['best_test_mdice']:.4f}")

    st.subheader("Training Curves")
    curve_cols = st.columns(3)
    curve_cols[0].image(str(LOSS_CURVE_PATH), caption="Training Loss", use_container_width=True)
    curve_cols[1].image(str(MIOU_CURVE_PATH), caption="Training mIoU", use_container_width=True)
    curve_cols[2].image(str(MDICE_CURVE_PATH), caption="Training mDice", use_container_width=True)

    st.subheader("Run Summary")
    info_cols = st.columns(4)
    info_cols[0].metric("Epochs", int(summary["epochs"]))
    info_cols[1].metric("Batch Size", int(summary["batch_size"]))
    info_cols[2].metric("Train Samples", int(summary["train_samples"]))
    info_cols[3].metric("Test Samples", int(summary["test_samples"]))

    st.dataframe(history, use_container_width=True, hide_index=True)


def render_prediction_page() -> None:
    st.title("Test-Set Mask Comparison")
    st.caption(
        "Upload exactly 4 RGB images from `Q2/data/CameraRGB` that belong to the seeded test split. "
        "Their original filenames must be preserved so the app can find the correct ground-truth masks."
    )

    sample_lookup = get_test_sample_lookup()
    uploaded_files = st.file_uploader(
        "Upload 4 test images",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
    )

    with st.expander("Show a few valid test filenames"):
        valid_names = sorted(sample_lookup.keys())[:12]
        st.code(", ".join(valid_names), language=None)

    if not uploaded_files:
        st.info("Upload 4 images to view the ground-truth and predicted segmentation masks.")
        return

    if len(uploaded_files) != 4:
        st.warning(f"Please upload exactly 4 images. You uploaded {len(uploaded_files)}.")
        return

    for uploaded_file in uploaded_files:
        st.markdown(f"### {uploaded_file.name}")

        if uploaded_file.name not in sample_lookup:
            st.error(
                "This file is not part of the expected test split. "
                "Upload an image from `Q2/data/CameraRGB` using its original filename."
            )
            continue

        file_bytes = uploaded_file.getvalue()
        original_image, image_tensor = preprocess_uploaded_image(file_bytes)

        prediction_mask = predict_mask(image_tensor)
        sample_info = sample_lookup[uploaded_file.name]
        ground_truth_mask = Image.open(sample_info["mask_path"]).split()[0]
        ground_truth_mask = ground_truth_mask.resize(IMAGE_SIZE, Image.Resampling.NEAREST)
        ground_truth_np = np.asarray(ground_truth_mask, dtype=np.uint8)

        preview_cols = st.columns(3)
        preview_cols[0].image(original_image, caption="Uploaded Image", use_container_width=True)
        preview_cols[1].image(colorize_mask(ground_truth_np), caption="Ground-Truth Mask", use_container_width=True)
        preview_cols[2].image(colorize_mask(prediction_mask), caption="Predicted Mask", use_container_width=True)


def main() -> None:
    st.set_page_config(
        page_title="Q2 Segmentation App",
        page_icon=":bar_chart:",
        layout="wide",
    )

    page = st.sidebar.radio(
        "Navigate",
        ["Training Metrics", "Prediction Viewer"],
    )

    st.sidebar.markdown(
        "\n".join(
            [
                "**Artifacts**",
                f"- Model: `{MODEL_PATH.name}`",
                f"- Metrics: `{METRICS_CSV_PATH.name}`",
                f"- Test summary: `{TEST_METRICS_PATH.name}`",
            ]
        )
    )

    if page == "Training Metrics":
        render_metrics_page()
    else:
        render_prediction_page()


if __name__ == "__main__":
    main()
