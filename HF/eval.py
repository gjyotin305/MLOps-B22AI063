from datasets import load_dataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from tqdm.auto import tqdm
from sklearn.metrics import confusion_matrix, classification_report, f1_score


MODEL_URL = "https://huggingface.co/gjyotin305/minor_resnet18/resolve/main/setB.pth"
BATCH_SIZE = 256
NUM_CLASSES = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


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


def pbar_log(pbar, **metrics):
    formatted = {}
    for k, v in metrics.items():
        if isinstance(v, float):
            formatted[k] = f"{v:.4f}"
        else:
            formatted[k] = v
    pbar.set_postfix(formatted)


def main():
    dataset_hf = load_dataset("Chiranjeev007/CIFAR-10_Subset")
    train_split_name = "train" if "train" in dataset_hf else list(dataset_hf.keys())[0]
    eval_split_name = "test" if "test" in dataset_hf else (
        "validation" if "validation" in dataset_hf else train_split_name
    )

    image_col, label_col = resolve_columns(dataset_hf[train_split_name])
    label_names = None
    label_feature = dataset_hf[train_split_name].features.get(label_col)
    if label_feature is not None and hasattr(label_feature, "names"):
        label_names = label_feature.names
    if label_names is None:
        label_names = [str(i) for i in range(NUM_CLASSES)]

    eval_dataset = HFDatasetWrapper(dataset_hf[eval_split_name], image_col, label_col, transform)
    eval_dataloader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(f"Eval split: {eval_split_name} ({len(eval_dataset)} samples)")

    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

    state_dict = torch.hub.load_state_dict_from_url(
        MODEL_URL, map_location=DEVICE, progress=True
    )
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    model.load_state_dict(state_dict)

    model = model.to(DEVICE)
    model.eval()
    print("Model loaded from HF URL.")

    criterion = nn.CrossEntropyLoss()
    eval_running_loss = 0.0
    eval_correct = 0
    eval_total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        pbar = tqdm(eval_dataloader, desc="Evaluating", unit="batch")
        for batch_idx, (images, labels) in enumerate(pbar, start=1):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)
            eval_running_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            eval_total += labels.size(0)
            eval_correct += (preds == labels).sum().item()

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

            avg_loss = eval_running_loss / batch_idx
            avg_acc = 100 * eval_correct / eval_total
            pbar_log(pbar, loss=avg_loss, acc=avg_acc)

    final_loss = eval_running_loss / len(eval_dataloader)
    final_acc = 100 * eval_correct / eval_total
    macro_f1 = f1_score(all_labels, all_preds, average="macro")

    cm = confusion_matrix(all_labels, all_preds, labels=list(range(NUM_CLASSES)))
    classwise_acc = []
    for idx in range(NUM_CLASSES):
        class_total = cm[idx].sum()
        acc = (cm[idx, idx] / class_total) if class_total > 0 else 0.0
        classwise_acc.append(acc)

    report = classification_report(
        all_labels,
        all_preds,
        labels=list(range(NUM_CLASSES)),
        target_names=label_names,
        zero_division=0,
    )

    print(f"\nFinal Eval Loss: {final_loss:.4f}")
    print(f"Final Eval Accuracy: {final_acc:.2f}%")
    print(f"Final Eval Macro F1: {macro_f1:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClass-wise Accuracy:")
    for cls_name, acc in zip(label_names, classwise_acc):
        print(f"{cls_name}: {acc * 100:.2f}%")

    print("\nClassification Report:")
    print(report)


if __name__ == "__main__":
    main()
