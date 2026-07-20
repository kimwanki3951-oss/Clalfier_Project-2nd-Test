from __future__ import annotations

import csv
import hashlib
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent

LOCAL_DATA_DIR = PROJECT_ROOT / "data"
LOCAL_RAW_DIR = LOCAL_DATA_DIR / "raw"
LOCAL_SPLITS_DIR = LOCAL_DATA_DIR / "splits"

DRIVE_ROOT = Path("/content/drive/MyDrive")
DRIVE_PROJECT_DIR = DRIVE_ROOT / "Clalfier_Project-2nd-Test"
DRIVE_DATASETS_DIR = DRIVE_PROJECT_DIR / "datasets"

DVC_LOCK_PATH = PROJECT_ROOT / "dvc.lock"
PARAMS_PATH = PROJECT_ROOT / "params.yaml"

DATASET_NAME = "fruit-and-vegetable-image-recognition"
DATASET_SOURCE = (
    "kaggle://kritikseth/fruit-and-vegetable-image-recognition"
)


def run(command: list[str]) -> None:
    """외부 명령어를 프로젝트 루트에서 실행한다."""

    print("\n" + "=" * 72)
    print("[RUN]", " ".join(command))
    print("=" * 72)

    subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=True,
    )


def ensure_drive_mounted() -> None:
    """Google Drive가 Colab에 마운트됐는지 확인한다."""

    if not DRIVE_ROOT.exists():
        raise RuntimeError(
            "Google Drive가 마운트되지 않았습니다.\n\n"
            "Colab에서 아래 코드를 먼저 실행하세요.\n\n"
            "from google.colab import drive\n"
            "drive.mount('/content/drive')"
        )

    print(f"[DRIVE] Google Drive detected: {DRIVE_ROOT}")


def ensure_dvc_repository() -> None:
    """DVC 저장소가 없으면 초기화한다."""

    dvc_dir = PROJECT_ROOT / ".dvc"

    if dvc_dir.exists():
        print("[DVC] Existing DVC repository detected.")
        return

    print("[DVC] Initializing DVC repository.")
    run(["dvc", "init"])


def load_yaml(path: Path) -> dict[str, Any]:
    """YAML 파일을 딕셔너리로 읽는다."""

    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file)

    return loaded if isinstance(loaded, dict) else {}


def calculate_file_sha256(path: Path) -> str:
    """파일의 SHA256 해시를 계산한다."""

    sha256 = hashlib.sha256()

    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            sha256.update(chunk)

    return sha256.hexdigest()


def calculate_pipeline_fingerprint() -> str:
    """
    dvc.lock과 params.yaml을 기준으로 데이터셋 버전 지문을 만든다.

    데이터 또는 파이프라인 파라미터가 바뀌면
    dvc.lock 또는 params.yaml이 바뀌므로 지문도 달라진다.
    """

    sha256 = hashlib.sha256()

    required_files = [
        DVC_LOCK_PATH,
        PARAMS_PATH,
    ]

    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(
                f"버전 계산에 필요한 파일이 없습니다: {path}"
            )

        sha256.update(path.name.encode("utf-8"))
        sha256.update(path.read_bytes())

    return sha256.hexdigest()


def count_csv_rows(csv_path: Path) -> int:
    """CSV 헤더를 제외한 데이터 행 수를 계산한다."""

    if not csv_path.exists():
        return 0

    with csv_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        reader = csv.reader(file)
        row_count = sum(1 for _ in reader)

    return max(row_count - 1, 0)


def count_raw_images() -> int:
    """raw 폴더에 있는 이미지 파일 수를 계산한다."""

    image_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".gif",
        ".webp",
        ".tif",
        ".tiff",
    }

    if not LOCAL_RAW_DIR.exists():
        return 0

    return sum(
        1
        for path in LOCAL_RAW_DIR.rglob("*")
        if path.is_file()
        and path.suffix.lower() in image_extensions
    )


def get_split_statistics() -> dict[str, int]:
    """Train, Validation, Test CSV의 행 수를 가져온다."""

    train_count = count_csv_rows(
        LOCAL_SPLITS_DIR / "train.csv"
    )
    valid_count = count_csv_rows(
        LOCAL_SPLITS_DIR / "valid.csv"
    )
    test_count = count_csv_rows(
        LOCAL_SPLITS_DIR / "test.csv"
    )

    return {
        "train": train_count,
        "valid": valid_count,
        "test": test_count,
        "total_unique_images": (
            train_count + valid_count + test_count
        ),
    }


def get_class_count() -> int:
    """train.csv의 label 또는 class 열에서 클래스 수를 계산한다."""

    train_csv = LOCAL_SPLITS_DIR / "train.csv"

    if not train_csv.exists():
        return 0

    with train_csv.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        fieldnames = reader.fieldnames or []

        possible_columns = [
            "label",
            "class",
            "class_name",
            "category",
            "target",
        ]

        class_column = next(
            (
                column
                for column in possible_columns
                if column in fieldnames
            ),
            None,
        )

        if class_column is None:
            return 0

        classes = {
            row[class_column]
            for row in reader
            if row.get(class_column)
        }

    return len(classes)


def find_existing_version(
    fingerprint: str,
) -> Path | None:
    """
    동일한 fingerprint를 가진 기존 데이터셋 버전을 찾는다.
    """

    if not DRIVE_DATASETS_DIR.exists():
        return None

    for version_dir in DRIVE_DATASETS_DIR.iterdir():
        if not version_dir.is_dir():
            continue

        metadata_path = version_dir / "metadata.yaml"

        if not metadata_path.exists():
            continue

        metadata = load_yaml(metadata_path)

        if metadata.get("fingerprint") == fingerprint:
            return version_dir

    return None


def build_version_name(fingerprint: str) -> str:
    """시간과 fingerprint 앞 8자리로 버전명을 만든다."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_hash = fingerprint[:8]

    return f"v{timestamp}_{short_hash}"


def copy_directory(
    source: Path,
    destination: Path,
) -> None:
    """폴더를 목적지로 복사한다."""

    if not source.exists():
        raise FileNotFoundError(
            f"복사할 폴더가 없습니다: {source}"
        )

    if destination.exists():
        shutil.rmtree(destination)

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(f"[COPY] {source}")
    print(f"    -> {destination}")

    shutil.copytree(
        source,
        destination,
    )

    print(f"[COPY] Completed: {destination}")


def copy_file(
    source: Path,
    destination: Path,
) -> None:
    """파일을 목적지로 복사한다."""

    if not source.exists():
        raise FileNotFoundError(
            f"복사할 파일이 없습니다: {source}"
        )

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    shutil.copy2(
        source,
        destination,
    )

    print(f"[COPY] {source}")
    print(f"    -> {destination}")


def create_metadata(
    version_name: str,
    fingerprint: str,
) -> dict[str, Any]:
    """현재 데이터셋 버전의 metadata를 생성한다."""

    split_statistics = get_split_statistics()
    raw_image_count = count_raw_images()

    metadata: dict[str, Any] = {
        "version": version_name,
        "created_at": datetime.now().isoformat(
            timespec="seconds"
        ),
        "dataset": {
            "name": DATASET_NAME,
            "source": DATASET_SOURCE,
            "raw_image_count": raw_image_count,
            "unique_image_count": split_statistics[
                "total_unique_images"
            ],
            "duplicate_count": max(
                raw_image_count
                - split_statistics["total_unique_images"],
                0,
            ),
            "class_count": get_class_count(),
        },
        "splits": {
            "train": split_statistics["train"],
            "valid": split_statistics["valid"],
            "test": split_statistics["test"],
        },
        "version_control": {
            "fingerprint": fingerprint,
            "dvc_lock_sha256": calculate_file_sha256(
                DVC_LOCK_PATH
            ),
            "params_sha256": calculate_file_sha256(
                PARAMS_PATH
            ),
        },
        "paths": {
            "raw": "raw",
            "splits": "splits",
            "dvc_lock": "dvc.lock",
            "params": "params.yaml",
        },
    }

    return metadata


def save_yaml(
    data: dict[str, Any],
    destination: Path,
) -> None:
    """딕셔너리를 YAML 파일로 저장한다."""

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with destination.open(
        "w",
        encoding="utf-8",
    ) as file:
        yaml.safe_dump(
            data,
            file,
            allow_unicode=True,
            sort_keys=False,
        )

    print(f"[WRITE] {destination}")


def update_latest_file(
    version_name: str,
    fingerprint: str,
) -> None:
    """가장 최근 데이터셋 버전을 latest.yaml에 기록한다."""

    latest_data = {
        "latest_version": version_name,
        "fingerprint": fingerprint,
        "updated_at": datetime.now().isoformat(
            timespec="seconds"
        ),
        "path": f"datasets/{version_name}",
    }

    save_yaml(
        latest_data,
        DRIVE_DATASETS_DIR / "latest.yaml",
    )


def create_dataset_version() -> Path:
    """Google Drive에 데이터셋 버전을 생성한다."""

    print("\n" + "=" * 72)
    print("[VERSION] Creating dataset version")
    print("=" * 72)

    fingerprint = calculate_pipeline_fingerprint()

    print(f"[VERSION] Fingerprint: {fingerprint}")

    existing_version = find_existing_version(
        fingerprint
    )

    if existing_version is not None:
        print(
            "[VERSION] Identical dataset version already exists."
        )
        print(f"[VERSION] Existing path: {existing_version}")

        update_latest_file(
            existing_version.name,
            fingerprint,
        )

        return existing_version

    version_name = build_version_name(fingerprint)
    version_dir = DRIVE_DATASETS_DIR / version_name

    print(f"[VERSION] New version: {version_name}")

    copy_directory(
        LOCAL_RAW_DIR,
        version_dir / "raw",
    )

    copy_directory(
        LOCAL_SPLITS_DIR,
        version_dir / "splits",
    )

    copy_file(
        DVC_LOCK_PATH,
        version_dir / "dvc.lock",
    )

    copy_file(
        PARAMS_PATH,
        version_dir / "params.yaml",
    )

    metadata = create_metadata(
        version_name=version_name,
        fingerprint=fingerprint,
    )

    save_yaml(
        metadata,
        version_dir / "metadata.yaml",
    )

    update_latest_file(
        version_name=version_name,
        fingerprint=fingerprint,
    )

    return version_dir


def main() -> None:
    ensure_drive_mounted()
    ensure_dvc_repository()

    # Kaggle 다운로드 및 데이터 분할
    run(["dvc", "repro"])

    # Google Drive에 버전 단위로 저장
    version_dir = create_dataset_version()

    print("\n" + "=" * 72)
    print("[DONE] Dataset pipeline completed.")
    print(f"[DONE] Dataset version: {version_dir.name}")
    print(f"[DONE] Google Drive path: {version_dir}")
    print("[DONE] latest.yaml has been updated.")
    print("=" * 72)


if __name__ == "__main__":
    main()
