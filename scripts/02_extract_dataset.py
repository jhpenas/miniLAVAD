import os
import zipfile
from pathlib import Path
from tqdm import tqdm

def extract_zip_files(raw_dir: Path):
    """
    Finds all .zip files in the raw directory and extracts them into their own subfolders.
    Skips extraction if the target folder already exists and has files.
    """
    zip_files = list(raw_dir.glob("*.zip"))

    if not zip_files:
        print("No .zip files found in the raw directory.")
        return

    print(f"Found {len(zip_files)} zip archives. Checking extraction status...")

    for zip_path in zip_files:
        # Create a target folder named after the zip file
        extract_target = raw_dir / zip_path.stem

        # Skip if already extracted
        if extract_target.exists() and any(extract_target.iterdir()):
            print(f"  [SKIPPED] {zip_path.name} is already extracted.")
            continue

        print(f"  [EXTRACTING] {zip_path.name}...")
        extract_target.mkdir(parents=True, exist_ok=True)

        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                members = zip_ref.namelist()
                for member in tqdm(members, desc=f"Extracting {zip_path.stem}", leave=False):
                    zip_ref.extract(member, path=extract_target)
            print(f"  [SUCCESS] {zip_path.name} extracted to {extract_target.name}/")
        except zipfile.BadZipFile:
            print(f"  [ERROR] {zip_path.name} is corrupted or still downloading. Skipping.")

if __name__ == "__main__":
    # --- CONFIGURATION ---
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

    # Extract all downloaded zip files safely
    extract_zip_files(RAW_DATA_DIR)