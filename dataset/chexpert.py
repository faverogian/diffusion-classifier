import os
import polars as pl
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class CheXpertDataset(Dataset):
    def __init__(self, data_path, split="train"):
        """
        Args:
            data_path (str): Path to the CheXpert dataset.
            split (str): Dataset split to use. One of "train", "valid", "test".

        Note:
            Everything is derived from the train split.
        """
        self.data_path = data_path
        self.split = split if split != "test" else "valid"

        # Get images path
        self.img_dir = os.path.join(self.data_path, "train")

        # Get CSV path
        self.csv_path = os.path.join(self.data_path, f"train.csv")

        # Load and filter CSV for Study1 frontal images
        self.data = self._filter_study1_frontal()

        # Check the dataset split and apply filtering accordingly
        if split == "train": # Take first 80% of the data
            self.data = self.data.head(int(self.data.height * 0.8))
        elif split == "valid": # Take first half of last 20%) of the data
            self.data = self.data.tail(int(self.data.height * 0.2))
            self.data = self.data.head(int(self.data.height * 0.5))
        elif split == "test": # Take second half of last 20% of the data
            self.data = self.data.tail(int(self.data.height * 0.2))
            self.data = self.data.tail(int(self.data.height * 0.5))

        # Print length of dataset
        print(f"Dataset length: {len(self.data)}")

        # Print label column name
        print(f"Label column name: {self.data.columns[1]}")
        
        # Get unique label and their counts from the polars dataframe
        print(self.data.group_by("Lung Opacity").count().sort("Lung Opacity"))

    def transforms(self):
        return transforms.Compose([
            transforms.Resize((64,64)),
            transforms.ToTensor(),
    ])

    def _filter_study1_frontal(self):
        # Load CSV
        df = pl.read_csv(self.csv_path)

        # Filter for Study1 frontal images
        study1_frontal = df.filter(pl.col("Path").str.contains("study1/view1_frontal.jpg"))

        # Keep only relevant columns (Path and Lung Opacity label)
        study1_frontal = study1_frontal.select(["Path", "Lung Opacity"])

        # Replace NaN labels with 0
        study1_frontal = study1_frontal.with_columns(
            pl.col("Lung Opacity").fill_null(0)
        )

        # Drop rows where label is -1
        study1_frontal = study1_frontal.filter(pl.col("Lung Opacity") != -1)

        return study1_frontal

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get row data
        row = self.data[idx]
        rel_path = row["Path"].item().split("/")[1:]
        rel_path = os.path.join(*rel_path)
        img_path = os.path.join(self.data_path, rel_path)
        label = int(row["Lung Opacity"].item())

        # Load image
        image = Image.open(img_path).convert("RGB")

        # Convert image to tensor
        image = self.transforms()(image)

        return image, label
    
class CheXpertDataLoader:
    def __init__(self, data_path, batch_size=64, num_workers=4):
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Initialize datasets
        self.train_dataset = CheXpertDataset(data_path=self.data_path, split="train")
        self.val_dataset = CheXpertDataset(data_path=self.data_path, split="valid")
        self.test_dataset = CheXpertDataset(data_path=self.data_path, split="test")

        # Initialize DataLoaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )

        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )

    def collate_fn(self, batch):
        # Extract the images and labels from the batch
        images, labels = zip(*batch)

        return {
            "images": torch.stack(images),
            "prompt": torch.tensor(labels)
        }

    def get_train_loader(self):
        return self.train_loader

    def get_val_loader(self):
        return self.val_loader

    def get_test_loader(self):
        return self.test_loader