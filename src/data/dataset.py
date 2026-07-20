from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class ManifestImageDataset(Dataset):
    """
    DVC가 고정한 CSV manifest를 읽는 공통 Dataset.

    공통으로 고정되는 것:
      - 어떤 sample이 train/valid/test에 속하는지
      - sample의 label

    모델별로 바꿀 수 있는 것:
      - transform
      - model forward 내부 reshape / patch / sequence 처리
    """

    def __init__(
        self,
        manifest_path: str | Path,
        project_root: str | Path,
        transform: Callable | None = None,
        class_to_idx: dict[str, int] | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.project_root = Path(project_root)
        self.transform = transform

        self.dataframe = pd.read_csv(self.manifest_path)
        required_columns = {"path", "label"}
        missing = required_columns - set(self.dataframe.columns)
        if missing:
            raise ValueError(
                f"Manifest missing required columns: {sorted(missing)}"
            )

        labels = sorted(self.dataframe["label"].unique().tolist())

        if class_to_idx is None:
            self.class_to_idx = {
                label: index for index, label in enumerate(labels)
            }
        else:
            self.class_to_idx = class_to_idx

        self.classes = [
            label
            for label, _ in sorted(
                self.class_to_idx.items(),
                key=lambda item: item[1],
            )
        ]

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int):
        row = self.dataframe.iloc[index]

        image_path = self.project_root / row["path"]
        image = Image.open(image_path).convert("RGB")
        label = self.class_to_idx[row["label"]]

        if self.transform is not None:
            image = self.transform(image)

        return image, label
