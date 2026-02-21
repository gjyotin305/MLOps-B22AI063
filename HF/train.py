from datasets import load_dataset
import torch
from huggingface_hub import HfApi
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from torchvision import transforms, models
from tqdm.auto import tqdm
import os
from sklearn.metrics import confusion_matrix, classification_report

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

dataset_hf = load_dataset(
    'Chiranjeev007/CIFAR-10_Subset'
)

# ==========================
# Configuration
# ==========================
BATCH_SIZE = 256
NUM_CLASSES = 10
EPOCHS = 3
LR = 1e-3
DEVICE="mps"
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WANDB_PROJECT = os.getenv("WANDB_PROJECT", "cifar10-resnet18")
WANDB_RUN_NAME = os.getenv("WANDB_RUN_NAME", "hf-train")

# ==========================
# Transforms
# ==========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ResNet input size
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ==========================
# HF Dataset -> Custom DataLoaders
# ==========================
def resolve_columns(hf_split):
    image_col = None
    for candidate in ("img", "image"):
        if candidate in hf_split.column_names:
            image_col = candidate
            break
    if image_col is None:
        raise ValueError(f"Could not find image column in {hf_split.column_names}")

    label_col = "label" if "label" in hf_split.column_names else None
    if label_col is None:
        raise ValueError(f"Could not find label column in {hf_split.column_names}")

    return image_col, label_col

train_split_name = "train" if "train" in dataset_hf else list(dataset_hf.keys())[0]
eval_split_name = "test" if "test" in dataset_hf else (
    "validation" if "validation" in dataset_hf else train_split_name
)

img_col, label_col = resolve_columns(dataset_hf[train_split_name])

label_names = None
label_feature = dataset_hf[train_split_name].features.get(label_col)
if label_feature is not None and hasattr(label_feature, "names"):
    label_names = label_feature.names

class HFDatasetWrapper(Dataset):
    def __init__(self, hf_split, image_col, label_col, transform):
        self.hf_split = hf_split
        self.image_col = image_col
        self.label_col = label_col
        self.transform = transform

    def __len__(self):
        return len(self.hf_split)

    def __getitem__(self, idx):
        row = self.hf_split[idx]
        image = row[self.image_col]
        label = row[self.label_col]
        image = self.transform(image)
        return image, int(label)

train_dataset = HFDatasetWrapper(dataset_hf[train_split_name], img_col, label_col, transform)
eval_dataset = HFDatasetWrapper(dataset_hf[eval_split_name], img_col, label_col, transform)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
eval_dataloader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Train split: {train_split_name} ({len(train_dataset)} samples)")
print(f"Eval split: {eval_split_name} ({len(eval_dataset)} samples)")

# ==========================
# Load ResNet-18
# ==========================
model = models.resnet18(pretrained=True)
# Replace final layer
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

model = model.to(DEVICE)

# ==========================
# Loss & Optimizer
# ==========================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

if WANDB_AVAILABLE:
    wandb.init(
        project=WANDB_PROJECT,
        name=WANDB_RUN_NAME,
        config={
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "lr": LR,
            "num_classes": NUM_CLASSES,
            "model": "resnet18",
            "dataset": "Chiranjeev007/CIFAR-10_Subset",
            "device": str(DEVICE),
        },
    )
    wandb.watch(model, log="all", log_freq=50)
else:
    print("wandb is not installed. Skipping wandb logging.")

def pbar_log(pbar, **metrics):
    """Log metrics on tqdm progress bar."""
    formatted = {}
    for k, v in metrics.items():
        if isinstance(v, float):
            formatted[k] = f"{v:.4f}"
        else:
            formatted[k] = v
    pbar.set_postfix(formatted)

def log_wandb_prediction_table(model, eval_dataset, max_samples=16):
    if not WANDB_AVAILABLE:
        return

    table = wandb.Table(columns=["index", "image", "true_label", "pred_label", "confidence"])
    n_samples = min(max_samples, len(eval_dataset))

    model.eval()
    with torch.no_grad():
        for idx in range(n_samples):
            row = eval_dataset.hf_split[idx]
            pil_img = row[eval_dataset.image_col]
            true_id = int(row[eval_dataset.label_col])

            inp = eval_dataset.transform(pil_img).unsqueeze(0).to(DEVICE)
            logits = model(inp)
            probs = torch.softmax(logits, dim=1)
            conf, pred = torch.max(probs, dim=1)

            pred_id = int(pred.item())
            confidence = float(conf.item())

            true_label = label_names[true_id] if label_names else str(true_id)
            pred_label = label_names[pred_id] if label_names else str(pred_id)

            table.add_data(
                idx,
                wandb.Image(pil_img),
                true_label,
                pred_label,
                confidence,
            )

    wandb.log({"eval/prediction_samples": table})

def log_wandb_eval_reports(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    class_labels = label_names if label_names else [str(i) for i in range(NUM_CLASSES)]

    # Class-wise accuracy from confusion matrix diagonal
    classwise_acc = []
    for idx in range(NUM_CLASSES):
        total_for_class = cm[idx].sum()
        acc = (cm[idx, idx] / total_for_class) if total_for_class > 0 else 0.0
        classwise_acc.append(acc)

    report_dict = classification_report(
        y_true,
        y_pred,
        labels=list(range(NUM_CLASSES)),
        target_names=class_labels,
        output_dict=True,
        zero_division=0,
    )
    report_text = classification_report(
        y_true,
        y_pred,
        labels=list(range(NUM_CLASSES)),
        target_names=class_labels,
        zero_division=0,
    )
    print("\nTest Classification Report:")
    print(report_text)

    if not WANDB_AVAILABLE:
        return

    wandb.log(
        {
            "eval/confusion_matrix": wandb.plot.confusion_matrix(
                y_true=y_true,
                preds=y_pred,
                class_names=class_labels,
            )
        }
    )

    classwise_table = wandb.Table(columns=["class", "accuracy"])
    for cls_name, acc in zip(class_labels, classwise_acc):
        classwise_table.add_data(cls_name, acc)
    wandb.log(
        {
            "eval/classwise_accuracy_bar": wandb.plot.bar(
                classwise_table,
                "class",
                "accuracy",
                title="Class-wise Accuracy on Test Set",
            )
        }
    )

    report_table = wandb.Table(columns=["label", "precision", "recall", "f1-score", "support"])
    for label, metrics in report_dict.items():
        if isinstance(metrics, dict):
            report_table.add_data(
                label,
                float(metrics.get("precision", 0.0)),
                float(metrics.get("recall", 0.0)),
                float(metrics.get("f1-score", 0.0)),
                float(metrics.get("support", 0.0)),
            )
    wandb.log({"eval/classification_report": report_table})

# ==========================
# Training Loop
# ==========================
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(
        train_dataloader,
        desc=f"Epoch {epoch + 1}/{EPOCHS}",
        unit="batch"
    )
    global_step = epoch * len(train_dataloader)

    for batch_idx, (images, labels) in enumerate(pbar, start=1):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        
        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        avg_loss = running_loss / batch_idx
        avg_acc = 100 * correct / total
        pbar_log(pbar, loss=avg_loss, acc=avg_acc)
        if WANDB_AVAILABLE:
            wandb.log(
                {
                    "train/loss": avg_loss,
                    "train/acc": avg_acc,
                    "epoch": epoch + 1,
                },
                step=global_step + batch_idx,
            )

    epoch_loss = running_loss / len(train_dataloader)
    epoch_acc = 100 * correct / total

    pbar.write(
        f"Epoch [{epoch+1}/{EPOCHS}] "
        f"Loss: {epoch_loss:.4f} "
        f"Accuracy: {epoch_acc:.2f}%"
    )

    model.eval()
    eval_correct = 0
    eval_total = 0
    eval_running_loss = 0.0
    all_eval_preds = []
    all_eval_labels = []

    with torch.no_grad():
        eval_pbar = tqdm(
            eval_dataloader,
            desc=f"Eval {epoch + 1}/{EPOCHS}",
            unit="batch"
        )
        for batch_idx, (images, labels) in enumerate(eval_pbar, start=1):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)
            eval_running_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            eval_total += labels.size(0)
            eval_correct += (preds == labels).sum().item()
            all_eval_preds.extend(preds.cpu().tolist())
            all_eval_labels.extend(labels.cpu().tolist())

            eval_avg_loss = eval_running_loss / batch_idx
            eval_avg_acc = 100 * eval_correct / eval_total
            pbar_log(eval_pbar, eval_loss=eval_avg_loss, eval_acc=eval_avg_acc)

    final_eval_loss = eval_running_loss / len(eval_dataloader)
    final_eval_acc = 100 * eval_correct / eval_total
    print(f"Eval [{epoch+1}/{EPOCHS}] Loss: {final_eval_loss:.4f} Accuracy: {final_eval_acc:.2f}%")
    if WANDB_AVAILABLE:
        wandb.log(
            {
                "epoch/train_loss": epoch_loss,
                "epoch/train_acc": epoch_acc,
                "eval/loss": final_eval_loss,
                "eval/acc": final_eval_acc,
                "epoch": epoch + 1,
            },
            step=(epoch + 1) * len(train_dataloader),
        )
        log_wandb_prediction_table(model, eval_dataset, max_samples=16)
        log_wandb_eval_reports(all_eval_labels, all_eval_preds)

print("Training Complete!")

# ==========================
# Save Model
# ==========================
torch.save(model.state_dict(), "setB.pth")
print("Model saved!")

api = HfApi()
api.upload_file(
    path_or_fileobj="setB.pth",
    path_in_repo="setB.pth",
    repo_id="gjyotin305/minor_resnet18",
    repo_type="model",
)
print("Model saved to Hugging Face!")


if WANDB_AVAILABLE:
    wandb.save("setB.pth")
    wandb.finish()
