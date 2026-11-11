
from pathlib import Path

dataset_path = Path(
    r"C:\Users\Naveen\Downloads\'plant_disease_model.keras'\PlantVillage"
)

if not dataset_path.exists():
    print("Dataset folder not found!")
else:
    folders = [p for p in dataset_path.iterdir() if p.is_dir()]

    print("Dataset found successfully!")
    print("Total plant categories:", len(folders))

    total_images = 0

    for folder in folders:
        count = sum(
            1 for p in folder.iterdir()
            if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
        )
        total_images += count
        print(folder.name, ":", count, "images")

    print("Total images:", total_images)
