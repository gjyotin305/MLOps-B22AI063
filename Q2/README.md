# Q2 Segmentation Pipeline

This folder contains a custom dataloader and a UNet-based semantic segmentation training script for the dataset in `Q2/data`.

## Files

- `dataloader.py`: builds paired RGB-mask samples, applies an 80/20 train-test split with seed `42`, and returns PyTorch dataloaders.
- `train_unet.py`: trains a lightweight UNet for `23` classes, computes training and test metrics, and saves plots and artifacts in `Question2/`.

## Run

Use the existing virtual environment from `Q1/.venv`:

```bash
python Q2/train_unet.py --epochs 15
```

The command writes the following outputs to the repo-level `Question2/` folder:

- `train_loss_curve.svg`
- `train_miou_curve.svg`
- `train_mdice_curve.svg`
- `metrics_history.csv`
- `test_metrics.json`
- `best_unet_model.pt`

## App

The deployed frontend for Question 2 lives at `Q2/Question2/app.py`. It provides:

- a training-metrics page with the saved loss, mIoU, and mDice plots plus test-set scores
- a prediction page where you upload 4 test images and compare ground-truth vs predicted masks

Install the UI dependency in the existing venv, then launch the app:

```bash
python -m pip install -r Q2/Question2/requirements_app.txt
streamlit run Q2/Question2/app.py
```
