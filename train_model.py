
import json
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split

# ------------------------------------------------
# 1. Dataset and model paths
# ------------------------------------------------

DATASET_PATH = Path(
    r"C:\Users\Naveen\Downloads\'plant_disease_model.keras'\PlantVillage"
)

MODEL_PATH = "plant_disease_model.pth"
CLASS_PATH = "class_names.json"

device = torch.device("cpu")
torch.set_num_threads(4)

# ------------------------------------------------
# 2. Image preprocessing
# ------------------------------------------------

transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ------------------------------------------------
# 3. Load only actual disease categories
# ------------------------------------------------

print("Loading dataset...")

if not DATASET_PATH.exists():
    raise FileNotFoundError(
        f"Dataset folder not found: {DATASET_PATH}"
    )

full_dataset = datasets.ImageFolder(
    root=str(DATASET_PATH),
    transform=transform
)

# Exclude the nested PlantVillage folder
valid_classes = [
    name for name in full_dataset.classes
    if name != "PlantVillage"
]

class_names = []
filtered_samples = []

for class_name in valid_classes:
    old_index = full_dataset.class_to_idx[class_name]

    class_samples = [
        (path, old_index)
        for path, label in full_dataset.samples
        if label == old_index
    ]

    # Ignore empty folders
    if class_samples:
        new_index = len(class_names)
        class_names.append(class_name)

        for path, _ in class_samples:
            filtered_samples.append((path, new_index))

if len(class_names) != 15:
    raise ValueError(
        f"Expected 15 categories, found {len(class_names)}. "
        "Please check the dataset folder."
    )

# Create a clean dataset with only 15 categories
full_dataset.samples = filtered_samples
full_dataset.imgs = filtered_samples
full_dataset.targets = [
    label for _, label in filtered_samples
]
full_dataset.classes = class_names
full_dataset.class_to_idx = {
    name: index for index, name in enumerate(class_names)
}

print("Categories:", len(class_names))
print("Images:", len(full_dataset))

if len(full_dataset) != 20638:
    raise ValueError(
        f"Expected 20638 images, found {len(full_dataset)}. "
        "Please check the dataset folder structure."
    )

# Save category names
with open(CLASS_PATH, "w") as file:
    json.dump(class_names, file, indent=4)

print("Dataset verified successfully!")

# ------------------------------------------------
# 4. Split training and validation data
# ------------------------------------------------

train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size

generator = torch.Generator().manual_seed(42)

train_dataset, val_dataset = random_split(
    full_dataset,
    [train_size, val_size],
    generator=generator
)

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=0
)

print("Training images:", train_size)
print("Validation images:", val_size)

# ------------------------------------------------
# 5. Load MobileNetV2
# ------------------------------------------------

print("Loading MobileNetV2...")

weights = models.MobileNet_V2_Weights.DEFAULT

model = models.mobilenet_v2(weights=weights)

# Freeze feature extraction layers
for parameter in model.features.parameters():
    parameter.requires_grad = False

# Replace output layer for 15 categories
model.classifier[1] = nn.Linear(
    model.last_channel,
    len(class_names)
)

model = model.to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.classifier.parameters(),
    lr=0.001
)

# ------------------------------------------------
# 6. Train model
# ------------------------------------------------

EPOCHS = 2

for epoch in range(EPOCHS):

    model.train()
    total_loss = 0.0

    print(f"\nStarting Epoch {epoch + 1}/{EPOCHS}")

    for batch_number, (images, labels) in enumerate(train_loader):

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if (batch_number + 1) % 100 == 0:
            print(
                f"Epoch {epoch + 1}: "
                f"Batch {batch_number + 1}/{len(train_loader)}"
            )

    # ------------------------------------------------
    # 7. Validate model
    # ------------------------------------------------

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)

            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            correct += (
                predictions.cpu() == labels
            ).sum().item()

            total += labels.size(0)

    accuracy = 100 * correct / total

    average_loss = total_loss / len(train_loader)

    print(f"Epoch: {epoch + 1}/{EPOCHS}")
    print(f"Training Loss: {average_loss:.4f}")
    print(f"Validation Accuracy: {accuracy:.2f}%")

# ------------------------------------------------
# 8. Save trained model
# ------------------------------------------------

torch.save(model.state_dict(), MODEL_PATH)

print("\n================================")
print("TRAINING COMPLETED SUCCESSFULLY!")
print("Model saved:", MODEL_PATH)
print("Class names saved:", CLASS_PATH)
print("Total categories:", len(class_names))
print("================================")
