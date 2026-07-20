# Common Dataset DVC Pipeline — Google Drive Push Ready

## Fixed DVC Remote

- Remote name: `gdrive_storage`
- Google Drive folder ID: `1nVaOpihiS5mKL0eJwxj-D8-1WKA17PAJ`
- DVC URL: `gdrive://1nVaOpihiS5mKL0eJwxj-D8-1WKA17PAJ`

## Colab execution

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd <YOUR_REPOSITORY>
pip install -r requirements.txt
python run_pipeline.py
```

`run_pipeline.py` executes:

```text
DVC repository check
        ↓
Google Drive remote configuration
        ↓
dvc repro
        ↓
Kaggle download
        ↓
Stratified 70 / 15 / 15 split
        ↓
dvc push
        ↓
Google Drive DVC remote
```

The first Google Drive use may request Google OAuth authentication.
Authenticate with a Google account that has write access to the target folder.
A public link alone does not guarantee DVC write access.

## GitHub commit after execution

```bash
git add .dvc/config dvc.yaml dvc.lock params.yaml requirements.txt run_pipeline.py src Common_Dataset_DVC_Pipeline.ipynb
git commit -m "Add common DVC dataset pipeline"
git push
```

Do not commit Google OAuth credentials or token files.

## Dataset rule

All models use the same manifests:

```text
data/splits/train.csv
data/splits/valid.csv
data/splits/test.csv
```

CNN / RNN / Transformer can have different transforms, but sample membership stays fixed.
