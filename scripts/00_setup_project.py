from pathlib import Path


def init_project_structure():
    """
    Creates all necessary local directories for the MLOps pipeline.
    Uses exist_ok=True so it safely ignores folders that already exist.
    """
    # This dynamically finds the root 'miniLAVAD' folder
    project_root = Path(__file__).resolve().parents[1]

    # List of directories that Git ignores but the pipeline needs
    directories = [
        "data/raw",
        "data/processed",
        "models/saved_lora",
        "mlruns",
        "notebooks"
    ]

    print("Initializing project structure...")
    for dir_name in directories:
        dir_path = project_root / dir_name
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"  [OK] {dir_path}")


if __name__ == "__main__":
    init_project_structure()
    print("Project structure is ready for experiments!")