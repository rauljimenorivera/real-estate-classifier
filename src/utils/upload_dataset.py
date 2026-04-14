import wandb
from pathlib import Path

ENTITY = "rauljr"
PROJECT = "real-estate-classifier"
DATASET_DIR = "data/raw/dataset"

def upload_dataset():
    run = wandb.init(
        entity=ENTITY,
        project=PROJECT,
        job_type="dataset-upload"
    )

    artifact = wandb.Artifact(
        name="real-estate-dataset",
        type="dataset",
        description="MIT Scenes dataset with 15 classes for real-estate image classification",
        metadata={
            "classes": 15,
            "splits": ["training", "validation"],
            "expected_processed_splits": ["train", "val", "test"],
            "source": "MIT Indoor/Outdoor Scenes",
        }
    )

    artifact.add_dir(DATASET_DIR)
    run.log_artifact(artifact)
    run.finish()
    print("Dataset subido correctamente a W&B")


if __name__ == "__main__":
    upload_dataset()