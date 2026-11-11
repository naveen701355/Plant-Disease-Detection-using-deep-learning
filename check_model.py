
import json
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split

DATASET_PATH = Path(
    r"C:\Users\Naveen\Downloads\'plant_disease_model.keras'\PlantVillage"
)

device = torch.device("cpu")
torch.set_num_threads(4)

transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Load dataset
dataset = datasets.ImageFolder(
    str(DATASET_PATH),
    transform=transform
)

# Remove nested PlantVillage folder
class_names = [
    name for name in dataset.classes
    if name != "PlantVillage"
]

samples = []
for name in class_names:
    old_index = dataset.class_to_idx[name]
    new_index = len([
        item for item in class_names[:class_names.index(name)]
    ])

    for path, label in dataset.samples:
        if label == old_index:
            samples.append((path, new_index))

dataset.samples = samples
dataset.imgs = samples
dataset.targets = [label for _, label in samples]
dataset.classes = class_names
dataset.class_to_idx = {
    name: i for i, name in enumerate(class_names)
}

# Reproduce the original validation split
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

generator = torch.Generator().manual_seed(42)

_, val_dataset = random_split(
    dataset,
    [train_size, val_size],
    generator=generator
)

val_loader = DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=0
)

# Load saved class names
with open("class_names.json", "r") as file:
    saved_classes = json.load(file)

# Load trained model
model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(
    model.last_channel,
    len(saved_classes)
)

model.load_state_dict(
    torch.load(
        "plant_disease_model.pth",
        map_location="cpu",
        weights_only=True
    )
)

model.eval()

# Evaluate each category
correct = [0] * len(saved_classes)
total = [0] * len(saved_classes)

with torch.no_grad():
    for images, labels in val_loader:
        outputs = model(images)
        predictions = outputs.argmax(dim=1)

        for actual, predicted in zip(labels, predictions):
            total[actual.item()] += 1

            if actual.item() == predicted.item():
                correct[actual.item()] += 1

print("\nCLASS-WISE VALIDATION RESULTS")
print("--------------------------------")

for i, name in enumerate(saved_classes):
    accuracy = (
        100 * correct[i] / total[i]
        if total[i] > 0 else 0
    )

    print(
        f"{name}: {accuracy:.2f}% "
        f"({correct[i]}/{total[i]})"
    )

healthy_index = saved_classes.index("Tomato_healthy")

print("\nTomato Healthy Accuracy:")
print(
    f"{100 * correct[healthy_index] / total[healthy_index]:.2f}%"
    if total[healthy_index] > 0 else "No validation images found"
)
