from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class MandalaClassificationDataset(Dataset):
    """
    Dataset for loading:
      - image
      - completion time
      - class label
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        image_col: str,
        image_path_col: str,
        target_col: str,
        time_col: str,
        transform: Optional[object] = None,
        use_time_feature: bool = True,
    ):
        self.df = dataframe.reset_index(drop=True).copy()
        self.image_col = image_col
        self.image_path_col = image_path_col
        self.target_col = target_col
        self.time_col = time_col
        self.transform = transform
        self.use_time_feature = use_time_feature

        required_columns = [self.image_col, self.image_path_col, self.target_col]
        if self.use_time_feature:
            required_columns.append(self.time_col)

        missing = [col for col in required_columns if col not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required columns in metadata CSV: {missing}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        image_path = Path(row[self.image_path_col])
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        target = torch.tensor(int(row[self.target_col]), dtype=torch.long)

        if self.use_time_feature:
            completion_time = torch.tensor(float(row[self.time_col]), dtype=torch.float32)
        else:
            completion_time = torch.tensor(0.0, dtype=torch.float32)

        return image, completion_time, target, str(image_path.name)
