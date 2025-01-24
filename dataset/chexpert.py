import os
import polars as pl
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from utils.wavelet import wavelet_dec_2

class CheXpertDataset(Dataset):
    def __init__(self, data_path, split="train", wavelet_transform=False):
        """
        Args:
            data_path (str): Path to the CheXpert dataset.
            split (str): Dataset split to use. One of "train", "valid", "test".

        Note:
            Everything is derived from the train split.
        """
        self.wavelet_transform = wavelet_transform
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
        print(self.data.group_by("Pleural Effusion").count().sort("Pleural Effusion"))

    def transforms(self):
        return transforms.Compose([
            transforms.Resize((256,256)),
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5)
    ])

    def _filter_study1_frontal(self):
        # Load CSV
        df = pl.read_csv(self.csv_path)

        # Filter for Study1 frontal images
        study1_frontal = df.filter(pl.col("Path").str.contains("study1/view1_frontal.jpg"))

        # Keep only relevant columns (Path and Pleural Effusion label)
        study1_frontal = study1_frontal.select(["Path", "Pleural Effusion", "No Finding"])

        # Replace NaN labels with 0
        study1_frontal = study1_frontal.with_columns(
            pl.col("Pleural Effusion").fill_null(0),
            pl.col("No Finding").fill_null(0)
        )

        # Drop rows where label is -1
        study1_frontal = study1_frontal.filter(
            pl.col("Pleural Effusion") != -1,
            pl.col("No Finding") != -1    
        )

        # Create new column that is XOR of Pleural Effusion and No Finding
        study1_frontal = study1_frontal.with_columns(
            ((pl.col("Pleural Effusion")>0) ^ (pl.col("No Finding")>0)).alias("healthy_or_sick")
        )

        # Drop rows where healthy_or_sick is 0
        study1_frontal = study1_frontal.filter(pl.col("healthy_or_sick") == 1)

        # Separate the active and inactive labels
        active_df = study1_frontal.filter(pl.col("Pleural Effusion") == 1)
        inactive_df = study1_frontal.filter(pl.col("Pleural Effusion") == 0)

        # Take the minimum count of the two labels
        min_count = min(active_df.height, inactive_df.height)

        # Sample the data to have equal number of active and inactive labels
        active_df = active_df.sample(n=min_count, with_replacement=False, seed=42)
        inactive_df = inactive_df.sample(n=min_count, with_replacement=False, seed=42)

        # Concatenate the two dataframes
        study1_frontal = pl.concat([active_df, inactive_df])

        # Shuffle the dataframe
        study1_frontal = study1_frontal.sample(n=len(study1_frontal), shuffle=True, seed=42)

        '''# Load resnet50 mistakes
        resnet50_mistakes = []
        with open("/home/mila/g/gian.favero/diffusion-classifier/mistakes/mistakes-ddpm-chexpert.txt", "r") as f:
            for line in f:
                resnet50_mistakes.append(line.strip())

        # Add resnet50_mistake column
        study1_frontal = study1_frontal.with_columns(
            pl.lit(0).alias("resnet50_mistake")
        )

        for i in range(len(study1_frontal)):
            rel_path = study1_frontal[i]['Path'].item().split("/")[1:]
            rel_path = ("/").join(rel_path)
            if rel_path in resnet50_mistakes:
                study1_frontal[i, "resnet50_mistake"] = 1
            else:
                study1_frontal[i, "resnet50_mistake"] = 0

        # Drop rows where resnet50_mistake is 0
        study1_frontal = study1_frontal.filter(pl.col("resnet50_mistake") != 0)'''

        return study1_frontal

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get row data
        row = self.data[idx]
        rel_path = row["Path"].item().split("/")[1:]
        rel_path = os.path.join(*rel_path)
        img_path = os.path.join(self.data_path, rel_path)
        label = int(row["Pleural Effusion"].item())

        # Load image
        image = Image.open(img_path).convert("RGB")

        # Convert image to tensor
        image = self.transforms()(image)

        if self.wavelet_transform:
            image = wavelet_dec_2(image) / 2

        return image, label #, rel_path
    
class CheXpertDataLoader:
    def __init__(self, wavelet_transform, data_path, cf_label=None, batch_size=64, num_workers=4):
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.cf_label = cf_label

        # Initialize datasets
        self.train_dataset = CheXpertDataset(data_path=self.data_path, split="train", wavelet_transform=wavelet_transform)
        self.val_dataset = CheXpertDataset(data_path=self.data_path, split="valid", wavelet_transform=wavelet_transform)
        self.test_dataset = CheXpertDataset(data_path=self.data_path, split="test", wavelet_transform=wavelet_transform)

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

        if self.cf_label is not None:
            # Make all labels the cf_label
            labels = [self.cf_label for _ in labels]

        return {
            "images": torch.stack(images),
            "prompt": torch.tensor(labels),
            #"rel_path": rel_path
        }

    def get_train_loader(self):
        return self.train_loader

    def get_val_loader(self):
        return self.val_loader

    def get_test_loader(self):
        return self.test_loader