from __future__ import annotations

import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
REMOTE_NAME = "gdrive_storage"
GDRIVE_FOLDER_ID = "1nVaOpihiS5mKL0eJwxj-D8-1WKA17PAJ"
GDRIVE_URL = f"gdrive://{GDRIVE_FOLDER_ID}"


def run(command: list[str]) -> None:
    print("\n" + "=" * 72)
    print("[RUN]", " ".join(command))
    print("=" * 72)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def ensure_dvc_repository() -> None:
    if (PROJECT_ROOT / ".dvc").exists():
        print("[DVC] Existing DVC repository detected.")
        return
    print("[DVC] Initializing DVC repository.")
    run(["dvc", "init"])


def configure_remote() -> None:
    run(["dvc", "remote", "add", "--default", "--force", REMOTE_NAME, GDRIVE_URL])
    print(f"[DVC] Default remote: {REMOTE_NAME}")
    print(f"[DVC] Google Drive folder ID: {GDRIVE_FOLDER_ID}")


def main() -> None:
    ensure_dvc_repository()
    configure_remote()
    run(["dvc", "repro"])
    print("\n[DVC] Starting Google Drive push.")
    print("[DVC] First use may request Google OAuth authentication.")
    print("[DVC] Sign in with an account that has write access to the folder.")
    run(["dvc", "push"])
    print("\n[DONE] Dataset pipeline reproduced and pushed to Google Drive.")
    print("[DONE] Commit dvc.lock and project config files to GitHub.")


if __name__ == "__main__":
    main()
