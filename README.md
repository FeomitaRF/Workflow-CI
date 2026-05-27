# Workflow-CI - MLflow Project

Repository ini dibuat untuk Kriteria 3: membuat workflow CI menggunakan MLflow Project.

## Struktur

```text
Workflow-CI
├── .github/workflows/mlflow-ci.yml
├── .workflow/README.md
├── MLProject
│   ├── modelling.py
│   ├── conda.yaml
│   ├── MLproject
│   ├── breast_cancer_preprocessing/breast_cancer_preprocessed.csv
│   ├── docker_hub.txt
│   └── requirements.txt
├── requirements.txt
├── README.md
└── .gitignore
```

## Menjalankan MLflow Project secara lokal

```bash
pip install -r requirements.txt
mlflow run MLProject --env-manager=local
```

Hasil training akan tersimpan di:

```text
MLProject/mlruns
MLProject/artifacts
MLProject/latest_run_id.txt
```

## Secrets GitHub Actions

Tambahkan secrets berikut di GitHub repository:

```text
DOCKERHUB_USERNAME
DOCKERHUB_TOKEN
```

`DOCKERHUB_TOKEN` sebaiknya memakai Access Token dari Docker Hub, bukan password utama.

## Docker Image

Workflow akan membuat image dengan format:

```text
<DOCKERHUB_USERNAME>/breast-cancer-mlflow-model:latest
<DOCKERHUB_USERNAME>/breast-cancer-mlflow-model:<GITHUB_SHA>
```

Tautan Docker Hub ditulis di `MLProject/docker_hub.txt`.
