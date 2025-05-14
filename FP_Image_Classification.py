import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
from tqdm import tqdm

# -------- Step 1: Visualization (no transform) --------
raw_dataset = datasets.ImageFolder('Rice_Image_Dataset')

def show_images_per_class(dataset, samples_per_class=4):
    class_to_idx = {v: k for k, v in dataset.class_to_idx.items()}
    images_by_class = {i: [] for i in range(len(dataset.classes))}

    for img, label in dataset:
        if len(images_by_class[label]) < samples_per_class:
            images_by_class[label].append((img, label))
        if all(len(v) == samples_per_class for v in images_by_class.values()):
            break

    fig, axes = plt.subplots(len(dataset.classes), samples_per_class, figsize=(12, 6))
    for class_idx, img_list in images_by_class.items():
        for i, (img, label) in enumerate(img_list):
            ax = axes[class_idx][i]
            ax.imshow(img)
            ax.set_title(dataset.classes[label])
            ax.axis("off")
    plt.tight_layout()
    plt.show()

show_images_per_class(raw_dataset)

# -------- Step 2: Transform to RGB & normalize --------
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.1259, 0.1272, 0.1301], std=[0.2949, 0.2980, 0.3062])
])

full_dataset = datasets.ImageFolder('Rice_Image_Dataset', transform=transform)

# -------- Step 3: Train/Val/Test Split --------
train_size = int(0.7 * len(full_dataset))
val_size = int(0.15 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size

train_set, val_set, test_set = random_split(full_dataset, [train_size, val_size, test_size],
                                            generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader = DataLoader(val_set, batch_size=32)
test_loader = DataLoader(test_set, batch_size=32)

# -------- Step 4: CNN --------
class RiceClassify(nn.Module):
    def __init__(self):
        super(RiceClassify, self).__init__()
        self.in_to_h1 = nn.Conv2d(3, 16, kernel_size=5, padding=2)
        self.h1_to_h2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.h2_to_h3 = nn.Linear(32 * 16 * 16, 32)
        self.h3_to_out = nn.Linear(32, 4)

    def forward(self, x):
        x = F.relu(self.in_to_h1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.h1_to_h2(x))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.h2_to_h3(x))
        return self.h3_to_out(x)

# -------- Step 5: Accuracy & Confusion Matrix --------
def evaluate(model, loader, device, label="Validation", show_matrix=True):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            preds = torch.argmax(outputs, dim=1)

            correct += (preds == y).sum().item()
            total += y.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    acc = correct / total
    print(f"{label} Accuracy: {acc:.4f}")

    if show_matrix:
        cm = confusion_matrix(all_labels, all_preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=full_dataset.classes)
        disp.plot(cmap='Blues')
        plt.title(f"{label} Confusion Matrix")
        plt.show()

    return acc

# -------- Step 6: Classification Metrics --------
def compute_classification_metrics(model, loader, class_names, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    cm = confusion_matrix(y_true, y_pred)

    metrics = {
        "Class": [],
        "Sensitivity (Recall)": [],
        "Specificity": [],
        "Precision": [],
        "F1 Score": [],
        "Accuracy": [],
        "False Positive Rate": [],
        "False Negative Rate": []
    }

    for i in range(len(cm)):
        TP = cm[i, i]
        FN = np.sum(cm[i, :]) - TP
        FP = np.sum(cm[:, i]) - TP
        TN = np.sum(cm) - (TP + FP + FN)

        sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
        accuracy = (TP + TN) / (TP + TN + FP + FN)
        fpr = FP / (FP + TN) if (FP + TN) > 0 else 0
        fnr = FN / (TP + FN) if (TP + FN) > 0 else 0

        metrics["Class"].append(class_names[i])
        metrics["Sensitivity (Recall)"].append(round(sensitivity, 4))
        metrics["Specificity"].append(round(specificity, 4))
        metrics["Precision"].append(round(precision, 4))
        metrics["F1 Score"].append(round(f1, 4))
        metrics["Accuracy"].append(round(accuracy, 4))
        metrics["False Positive Rate"].append(round(fpr, 4))
        metrics["False Negative Rate"].append(round(fnr, 4))

    df = pd.DataFrame(metrics)
    print("\nClassification Report Per Class:")
    print(df.to_string(index=False))
    return df

# -------- Step 7: Train Model --------
def train_model(epochs=5, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RiceClassify().to(device)

    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    test_accuracies = []
    training_losses = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        print(f"\nEpoch {epoch+1}/{epochs}")
        for x, y in tqdm(train_loader):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            outputs = model(x)
            loss = loss_fn(outputs, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Training Loss: {running_loss:.4f}")
        acc = evaluate(model, test_loader, device, label=f"Test (Epoch {epoch+1})", show_matrix=False)
        test_accuracies.append(acc)
        training_losses.append(running_loss)

    # Plot training loss and test accuracy
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), training_losses, marker='o', color='red')
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Training Loss Over Epochs")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), test_accuracies, marker='o', color='blue')
    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy")
    plt.title("Test Accuracy Over Epochs")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Final Validation Set Evaluation (with confusion matrix and metrics)
    print("\nFinal Validation Set Evaluation")
    evaluate(model, val_loader, device, label="Validation", show_matrix=True)
    compute_classification_metrics(model, val_loader, full_dataset.classes, device)

# -------- Step 8: Run Training --------
train_model(epochs=5)