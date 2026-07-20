from pathlib import Path
import shutil

import kagglehub
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PARAMS_PATH = PROJECT_ROOT / "params.yaml"
RAW_DIR = PROJECT_ROOT / "data" / "raw"


def load_dataset_name() -> str:
    with PARAMS_PATH.open("r", encoding="utf-8") as file:
        params = yaml.safe_load(file)
    return params["data"]["kaggle_dataset"]


def copy_dataset(source: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)

    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination)


def main() -> None:
    dataset_name = load_dataset_name()

    print(f"[DOWNLOAD] Kaggle dataset: {dataset_name}")
    cache_path = Path(kagglehub.dataset_download(dataset_name))

    print(f"[DOWNLOAD] Kaggle cache: {cache_path}")
    copy_dataset(cache_path, RAW_DIR)

    print(f"[DOWNLOAD] Dataset copied to: {RAW_DIR}")


if __name__ == "__main__":
    main()
