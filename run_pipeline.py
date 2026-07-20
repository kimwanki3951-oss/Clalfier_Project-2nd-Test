from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
LOCAL_DATA_DIR = PROJECT_ROOT / "data"

# Colab에서 drive.mount("/content/drive") 실행 후 접근 가능한 경로
DRIVE_ROOT = Path("/content/drive/MyDrive")

# Google Drive 안에 저장할 폴더
DRIVE_PROJECT_DIR = DRIVE_ROOT / "Clalfier_Project-2nd-Test"
DRIVE_DATA_DIR = DRIVE_PROJECT_DIR / "data"


def run(command: list[str]) -> None:
    print("\n" + "=" * 72)
    print("[RUN]", " ".join(command))
    print("=" * 72)

    subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=True,
    )


def ensure_dvc_repository() -> None:
    if (PROJECT_ROOT / ".dvc").exists():
        print("[DVC] Existing DVC repository detected.")
        return

    print("[DVC] Initializing DVC repository.")
    run(["dvc", "init"])


def ensure_drive_mounted() -> None:
    if not DRIVE_ROOT.exists():
        raise RuntimeError(
            "Google Drive가 마운트되지 않았습니다.\n"
            "Colab에서 아래 코드를 먼저 실행하세요:\n\n"
            "from google.colab import drive\n"
            "drive.mount('/content/drive')"
        )

    print(f"[DRIVE] Google Drive detected: {DRIVE_ROOT}")


def copy_directory(source: Path, destination: Path) -> None:
    if not source.exists():
        print(f"[SKIP] Source directory not found: {source}")
        return

    if destination.exists():
        print(f"[REMOVE] Existing Drive directory: {destination}")
        shutil.rmtree(destination)

    destination.parent.mkdir(parents=True, exist_ok=True)

    print(f"[COPY] {source}")
    print(f"    -> {destination}")

    shutil.copytree(source, destination)

    print(f"[COPY] Completed: {destination}")


def copy_file(source: Path, destination: Path) -> None:
    if not source.exists():
        print(f"[SKIP] Source file not found: {source}")
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)

    print(f"[COPY] {source}")
    print(f"    -> {destination}")


def copy_pipeline_outputs_to_drive() -> None:
    print("\n" + "=" * 72)
    print("[DRIVE] Copying pipeline outputs to Google Drive")
    print("=" * 72)

    DRIVE_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # 원본 Kaggle 데이터
    copy_directory(
        LOCAL_DATA_DIR / "raw",
        DRIVE_DATA_DIR / "raw",
    )

    # Train / Validation / Test manifest
    copy_directory(
        LOCAL_DATA_DIR / "splits",
        DRIVE_DATA_DIR / "splits",
)
    # 재현성에 필요한 DVC 기록 파일
    copy_file(
        PROJECT_ROOT / "dvc.lock",
        DRIVE_PROJECT_DIR / "dvc.lock",
    )

    copy_file(
        PROJECT_ROOT / "params.yaml",
        DRIVE_PROJECT_DIR / "params.yaml",
    )


def main() -> None:
    ensure_drive_mounted()
    ensure_dvc_repository()

    # Kaggle 다운로드 및 데이터 분할
    run(["dvc", "repro"])

    # DVC push 대신 마운트된 Drive로 직접 복사
    copy_pipeline_outputs_to_drive()

    print("\n" + "=" * 72)
    print("[DONE] Dataset pipeline completed.")
    print(f"[DONE] Google Drive location: {DRIVE_PROJECT_DIR}")
    print("[DONE] Commit dvc.lock and updated source files to GitHub.")
    print("=" * 72)


if __name__ == "__main__":
    main()
