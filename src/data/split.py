from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PARAMS_PATH = PROJECT_ROOT / "params.yaml"
RAW_DIR = PROJECT_ROOT / "data" / "raw"
SPLIT_DIR = PROJECT_ROOT / "data" / "splits"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_params() -> dict:
    with PARAMS_PATH.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)["data"]


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scan_images() -> pd.DataFrame:
    records: list[dict[str, str]] = []

    for image_path in sorted(RAW_DIR.rglob("*")):
        if not image_path.is_file():
            continue
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        relative_path = image_path.relative_to(PROJECT_ROOT).as_posix()

        # 현재 Kaggle 구조:
        # data/raw/train/apple/*.jpg
        # data/raw/validation/apple/*.jpg
        # data/raw/test/apple/*.jpg
        # 따라서 이미지의 직계 부모 폴더명을 class label로 사용한다.
        label = image_path.parent.name

        records.append(
            {
                "path": relative_path,
                "label": label,
                "sha256": file_sha256(image_path),
            }
        )

    if not records:
        raise RuntimeError(f"No image files found under {RAW_DIR}")

    dataframe = pd.DataFrame(records)

    duplicate_count = int(dataframe.duplicated(subset=["sha256"]).sum())
    if duplicate_count:
        print(f"[SPLIT] Removing exact duplicate images: {duplicate_count}")
        dataframe = dataframe.drop_duplicates(subset=["sha256"]).reset_index(drop=True)

    return dataframe


def validate_ratios(train_ratio: float, valid_ratio: float, test_ratio: float) -> None:
    total = train_ratio + valid_ratio + test_ratio
    if abs(total - 1.0) > 1e-9:
        raise ValueError(
            f"Split ratios must sum to 1.0. Current total: {total}"
        )


def stratified_split(
    dataframe: pd.DataFrame,
    seed: int,
    train_ratio: float,
    valid_ratio: float,
    test_ratio: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    validate_ratios(train_ratio, valid_ratio, test_ratio)

    train_df, remain_df = train_test_split(
        dataframe,
        test_size=(1.0 - train_ratio),
        random_state=seed,
        stratify=dataframe["label"],
    )

    remain_total = valid_ratio + test_ratio
    test_share_of_remain = test_ratio / remain_total

    valid_df, test_df = train_test_split(
        remain_df,
        test_size=test_share_of_remain,
        random_state=seed,
        stratify=remain_df["label"],
    )

    return (
        train_df.sort_values("path").reset_index(drop=True),
        valid_df.sort_values("path").reset_index(drop=True),
        test_df.sort_values("path").reset_index(drop=True),
    )


def save_split(name: str, dataframe: pd.DataFrame) -> None:
    output_path = SPLIT_DIR / f"{name}.csv"
    dataframe.to_csv(output_path, index=False)
    print(f"[SPLIT] {name:5s}: {len(dataframe):5d} -> {output_path}")


def main() -> None:
    params = load_params()

    seed = int(params["seed"])
    train_ratio = float(params["train_ratio"])
    valid_ratio = float(params["valid_ratio"])
    test_ratio = float(params["test_ratio"])

    dataframe = scan_images()

    train_df, valid_df, test_df = stratified_split(
        dataframe=dataframe,
        seed=seed,
        train_ratio=train_ratio,
        valid_ratio=valid_ratio,
        test_ratio=test_ratio,
    )

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)

    save_split("train", train_df)
    save_split("valid", valid_df)
    save_split("test", test_df)

    class_counts = (
        dataframe.groupby("label")
        .size()
        .rename("total")
        .sort_index()
    )
    class_counts.to_csv(SPLIT_DIR / "class_counts.csv")

    print(f"[SPLIT] Classes: {dataframe['label'].nunique()}")
    print(f"[SPLIT] Total images: {len(dataframe)}")
    print("[SPLIT] Fixed dataset manifests created successfully.")


if __name__ == "__main__":
    main()
