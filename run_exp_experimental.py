import os
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from torch.cuda.amp import autocast, GradScaler
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from thop import profile
from tqdm import tqdm

# ============================================================
# CONFIG
# ============================================================
DATASETS = ["MNIST", "FashionMNIST"]

MODELS = {
    "resnet18": models.resnet18,
    "resnet34": models.resnet34,
    "resnet50": models.resnet50
}

BATCH_SIZES = [16, 32]
OPTIMIZERS = ["sgd", "adam"]
LRS = [1e-3]
EPOCHS_LIST = [5]
PIN_MEMORY_LIST = [True]

USE_AMP = True
NUM_CLASSES = 10

DEVICE_LIST = ["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"]

DL_RESULTS_PATH = "results/dl_ablation_results.jsonl"
SVM_RESULTS_PATH = "results/svm_results.jsonl"

os.makedirs("results", exist_ok=True)
os.makedirs("results/plots", exist_ok=True)

torch.backends.cudnn.benchmark = True

# ============================================================
# STREAMING JSON WRITER
# ============================================================
def stream_write_json(result, filepath):
    with open(filepath, "a") as f:
        f.write(json.dumps(result) + "\n")
        f.flush()

# ============================================================
# DATA LOADERS
# ============================================================
def get_dataloaders(dataset_name, batch_size, pin_memory):
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    if dataset_name == "MNIST":
        dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
        testset = datasets.MNIST("./data", train=False, transform=transform)
    else:
        dataset = datasets.FashionMNIST("./data", train=True, download=True, transform=transform)
        testset = datasets.FashionMNIST("./data", train=False, transform=transform)

    train_len = int(0.7 * len(dataset))
    val_len = int(0.1 * len(dataset))
    test_len = len(dataset) - train_len - val_len

    train_set, val_set, _ = random_split(dataset, [train_len, val_len, test_len])

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=pin_memory
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=pin_memory
    )

    test_loader = DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=pin_memory
    )

    return train_loader, val_loader, test_loader

# ============================================================
# MODEL BUILDER
# ============================================================
def build_model(model_name):
    model = MODELS[model_name](weights=None)
    model.conv1 = nn.Conv2d(
        1, 64, kernel_size=7, stride=2, padding=3, bias=False
    )
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model

# ============================================================
# EVALUATION
# ============================================================
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    return 100.0 * correct / total

# ============================================================
# TRAINING (WITH PLOTTING)
# ============================================================
def train_model(model, train_loader, val_loader, optimizer, criterion, device, epochs, save_path):
    # model = torch.compile(model)
    model.train()
    scaler = GradScaler(enabled=USE_AMP)

    train_losses = []
    val_accuracies = []

    start = time.time()

    epoch_bar = tqdm(range(epochs), desc="Epochs", position=0)

    for epoch in epoch_bar:
        running_loss = 0

        batch_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{epochs}",
            position=1,
            leave=False
        )

        # ------------------------
        # TRAIN LOOP
        # ------------------------
        for x, y in batch_bar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            with autocast(enabled=USE_AMP):
                out = model(x)
                loss = criterion(out, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            batch_bar.set_postfix(loss=loss.item())

        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)

        # ------------------------
        # VALIDATION LOOP
        # ------------------------
        val_acc = evaluate(model, val_loader, device)
        val_accuracies.append(val_acc)

        print(f"Epoch {epoch+1}: Train Loss={avg_loss:.4f}, Val Acc={val_acc:.2f}%")

    # ------------------------
    # SAVE TRAIN–VAL GRAPH
    # ------------------------
    epochs_range = range(1, epochs + 1)

    plt.figure()
    plt.plot(epochs_range, train_losses, label="Train Loss")
    plt.plot(epochs_range, val_accuracies, label="Validation Accuracy")

    plt.xlabel("Epochs")
    plt.ylabel("Metric Value")
    plt.title("Training Loss vs Validation Accuracy")
    plt.legend()

    plt.savefig(save_path)
    plt.close()

    return (time.time() - start) * 1000

# ============================================================
# FLOPs
# ============================================================
def compute_flops(model, input_shape, device=None):
    if device is None:
        device = next(model.parameters()).device

    model = model.to(device)
    model.eval()

    dummy = torch.randn(*input_shape, device=device)

    with torch.no_grad():
        flops, params = profile(
            model,
            inputs=(dummy,),
            verbose=False
        )

    return {
        "GFLOPs": flops / 1e9,
        "MParams": params / 1e6
    }

# ============================================================
# DEEP LEARNING EXPERIMENTS (Q1a + Q2)
# ============================================================
def run_dl_experiments():
    for dataset in DATASETS:
        for model_name in MODELS:
            for device in DEVICE_LIST:
                for bs in BATCH_SIZES:
                    for opt_name in OPTIMIZERS:
                        for lr in LRS:
                            for epochs in EPOCHS_LIST:
                                for pin in PIN_MEMORY_LIST:

                                    train_loader, val_loader, test_loader = get_dataloaders(
                                        dataset, bs, pin
                                    )

                                    model = build_model(model_name).to(device)

                                    optimizer = (
                                        optim.SGD(model.parameters(), lr=lr, momentum=0.9)
                                        if opt_name == "sgd"
                                        else optim.Adam(model.parameters(), lr=lr)
                                    )

                                    criterion = nn.CrossEntropyLoss()

                                    plot_path = f"results/plots/{dataset}_{model_name}_{opt_name}_bs{bs}.png"

                                    train_time = train_model(
                                        model,
                                        train_loader,
                                        val_loader,
                                        optimizer,
                                        criterion,
                                        device,
                                        epochs,
                                        plot_path
                                    )

                                    test_acc = evaluate(model, test_loader, device)
                                    flops = compute_flops(model, input_shape=(1, 1, 28, 28))

                                    result = {
                                        "dataset": dataset,
                                        "model": model_name,
                                        "device": device,
                                        "batch_size": bs,
                                        "optimizer": opt_name,
                                        "learning_rate": lr,
                                        "epochs": epochs,
                                        "pin_memory": pin,
                                        "use_amp": USE_AMP,
                                        "test_accuracy": test_acc,
                                        "train_time_ms": train_time,
                                        "flops": flops,
                                        "plot_saved": plot_path
                                    }

                                    print(result)
                                    stream_write_json(result, DL_RESULTS_PATH)

# ============================================================
# SVM EXPERIMENTS (Q1b)
# ============================================================
def run_svm_experiments():
    transform = transforms.ToTensor()

    for dataset_name in DATASETS:
        dataset = (
            datasets.MNIST("./data", train=True, transform=transform)
            if dataset_name == "MNIST"
            else datasets.FashionMNIST("./data", train=True, transform=transform)
        )

        testset = (
            datasets.MNIST("./data", train=False, transform=transform)
            if dataset_name == "MNIST"
            else datasets.FashionMNIST("./data", train=False, transform=transform)
        )

        X_train = dataset.data.view(len(dataset), -1).numpy()
        y_train = dataset.targets.numpy()
        X_test = testset.data.view(len(testset), -1).numpy()
        y_test = testset.targets.numpy()

        for kernel in ["rbf", "poly"]:
            clf = svm.SVC(kernel=kernel, C=1.0)

            start = time.time()
            clf.fit(X_train[:10000], y_train[:10000])
            train_time = (time.time() - start) * 1000

            acc = accuracy_score(y_test, clf.predict(X_test)) * 100

            result = {
                "dataset": dataset_name,
                "model": f"svm-{kernel}",
                "kernel": kernel,
                "train_samples": 10000,
                "test_accuracy": acc,
                "train_time_ms": train_time
            }

            stream_write_json(result, SVM_RESULTS_PATH)

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("Running Deep Learning Experiments...")
    run_dl_experiments()

    print("Running SVM Experiments...")
    run_svm_experiments()

    print("All experiments completed.")