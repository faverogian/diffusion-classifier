import os
import polars as pl
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from utils.wavelet import wavelet_dec_2
import numpy as np

class PadChestDataset(Dataset):
    def __init__(self, data_path, wavelet_transform=False):
        """
        Args:
            data_path (str): Path to the PadChest dataset.
            wavelet_transform (bool): Whether to apply wavelet transform to the images.

        Note:
            Everything is derived from the train split.
        """
        self.wavelet_transform = wavelet_transform
        self.data_path = data_path

        # Get CSV path
        self.csv_path = os.path.join(self.data_path, f"padchest-v1.csv")

        # Load and filter CSV for the dataset
        self.data = self._filter_data()

        # Print length of dataset
        print(f"Dataset length: {len(self.data)}")

        # Print label column name
        print(f"Label column name: {self.data.columns[1]}")
        
        # Get unique label and their counts from the polars dataframe
        print(self.data.group_by("PleuralEffusion").count().sort("PleuralEffusion"))

    def transforms(self):
        return transforms.Compose([
            transforms.Resize((64,64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    def _filter_data(self):
        # Load CSV
        df = pl.read_csv(self.csv_path)

        # Keep only relevant columns (Path and Pleural Effusion label)
        relevant_df = df.select(["ImageID", "PleuralEffusion"])

        # Replace NaN labels with 0
        relevant_df = relevant_df.with_columns(
            pl.col("PleuralEffusion").fill_null(0)
        )

        # Drop rows where label is -1
        relevant_df = relevant_df.filter(pl.col("PleuralEffusion") != -1)

        return relevant_df

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get row data
        row = self.data[idx]
        rel_path = row["ImageID"].item()
        img_path = os.path.join(self.data_path, rel_path)
        label = int(row["PleuralEffusion"].item())

        image = Image.open(img_path)

        # Resize to 64x64
        image = image.resize((64, 64))

        # Convert to numpy array
        image = np.array(image).astype(np.float32)

        # Convert to tensor
        image = torch.from_numpy(image).unsqueeze(0)

        # Divide by 255*255 to keep in range [0, 1]
        image /= 255*255

        # Normalize to [-1, 1]
        image = (image - 0.5) / 0.5

        # Convert to RGB (3 channels)
        image = image.repeat(3, 1, 1)

        if self.wavelet_transform:
            image = wavelet_dec_2(image) / 2 # Keep in range [-1, 1])

        return image, label
    
class PadChestDataLoader:
    def __init__(self, wavelet_transform, data_path, batch_size=64, num_workers=4):
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Initialize dataset
        self.dataset = PadChestDataset(data_path=self.data_path, wavelet_transform=wavelet_transform)

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