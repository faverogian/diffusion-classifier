import os
import polars as pl
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from utils.wavelet import wavelet_dec_2

class mimicOodDataset(Dataset):
    def __init__(self, data_path, wavelet_transform=False):
        """
        Args:
            data_path (str): Path to the MIMIC-CXR dataset.
            wavelet_transform (bool): Whether to apply wavelet transform to the images.

        Note:
            Everything is derived from the train split.
        """
        self.wavelet_transform = wavelet_transform
        self.data_path = data_path

        # Get CSV path
        self.csv_path = os.path.join(self.data_path, f"mimic_pa_metadata.csv")

        # Load and filter CSV for the dataset
        self.data = self._filter_data()

        # Print length of dataset
        print(f"Dataset length: {len(self.data)}")

        # Print label column name
        print(f"Label column name: {self.data.columns[1]}")
        
        # Get unique label and their counts from the polars dataframe
        print(self.data.group_by("Pleural Effusion").count().sort("Pleural Effusion"))

    def transforms(self):
        return transforms.Compose([
            transforms.Resize((64,64)),
            transforms.ToTensor(),
            #transforms.Normalize(0.5, 0.5)
    ])

    def _filter_data(self):
        # Load CSV
        df = pl.read_csv(self.csv_path)

        # Keep only relevant columns (Path and Pleural Effusion label)
        relevant_df = df.select(["image_path", "Pleural Effusion"])

        # Replace NaN labels with 0
        relevant_df = relevant_df.with_columns(
            pl.col("Pleural Effusion").fill_null(0)
        )

        # Drop rows where label is -1
        relevant_df = relevant_df.filter(pl.col("Pleural Effusion") != -1)

        # Balance the counts of 0 and 1 in the "Pleural Effusion" column
        count_0 = relevant_df.filter(pl.col("Pleural Effusion") == 0).height
        count_1 = relevant_df.filter(pl.col("Pleural Effusion") == 1).height
        min_count = min(count_0, count_1)

        # Keep only `min_count` rows for each label to balance the dataset
        relevant_df = relevant_df.filter(pl.col("Pleural Effusion") == 0).limit(min_count).vstack(
            relevant_df.filter(pl.col("Pleural Effusion") == 1).limit(min_count)
        )

        return relevant_df

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get row data
        row = self.data[idx]
        rel_path = row["image_path"].item().split("/")[2:]
        img_path = os.path.join(self.data_path, *rel_path)
        label = int(row["Pleural Effusion"].item())

        # Load image
        image = Image.open(img_path).convert("RGB")

        # Convert image to tensor
        image = self.transforms()(image)

        if self.wavelet_transform:
            image = wavelet_dec_2(image) / 2 # Keep in range [-1, 1])

        return image, label

class mimicOodDataLoader:
    def __init__(self, wavelet_transform, data_path, batch_size=64, num_workers=4):
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Initialize dataset
        self.dataset = mimicOodDataset(data_path=self.data_path, wavelet_transform=wavelet_transform)

        # Initialize DataLoader
        self.data_loader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            shuffle=True,
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

    def get_data_loader(self):
        return self.data_loader