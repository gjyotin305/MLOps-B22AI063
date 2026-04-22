# Q2 Segmentation Pipeline

This folder contains a custom dataloader and a UNet-based semantic segmentation training script for the dataset in `Q2/data`.

## Files

- `dataloader.py`: builds paired RGB-mask samples, applies an 80/20 train-test split with seed `42`, and returns PyTorch dataloaders.
- `train_unet.py`: trains a lightweight UNet for `23` classes, computes training and test metrics, and saves plots and artifacts in `Question2/`.

## Run

Use the existing virtual environment from `Q1/.venv`:

```bash
Q1/.venv/bin/python Q2/train_unet.py --epochs 15
```

The command writes the following outputs to the repo-level `Question2/` folder:

- `train_loss_curve.svg`
- `train_miou_curve.svg`
- `train_mdice_curve.svg`
- `metrics_history.csv`
- `test_metrics.json`
- `best_unet_model.pt`
