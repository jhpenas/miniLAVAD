import os
from pathlib import Path
import time
import dropbox
from tqdm import tqdm
from dotenv import load_dotenv
import requests

load_dotenv()

def download_file_with_retry(dbx, shared_link, file_name, local_file_path, max_retries=5):
    """
    Downloads a file with chunked writing, progress bar, and automatic retry on connection drops.
    """
    for attempt in range(1, max_retries + 1):
        try:
            print(f"  [DOWNLOADING] {file_name} (Attempt {attempt}/{max_retries})...")

            # Request file from Dropbox
            _, res = dbx.sharing_get_shared_link_file(url=shared_link, path=f"/{file_name}")

            # Get content size for progress bar
            total_size = int(res.headers.get('content-length', 0))

            # Write in chunks with a progress bar
            with open(local_file_path, "wb") as f, tqdm(
                    desc=file_name,
                    total=total_size,
                    unit='iB',
                    unit_scale=True,
                    unit_divisor=1024,
            ) as bar:
                for chunk in res.iter_content(chunk_size=1024 * 1024):  # 1MB chunks
                    if chunk:
                        f.write(chunk)
                        bar.update(len(chunk))

            print(f"  [SUCCESS] {file_name} downloaded successfully.")
            return True

        except Exception as e:
            print(f"  [WARNING] Download interrupted on {file_name}: {e}")
            if local_file_path.exists():
                local_file_path.unlink()  # Delete partial corrupted file

            if attempt < max_retries:
                sleep_time = 5 * attempt
                print(f"  Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)
            else:
                print(f"  [ERROR] Failed to download {file_name} after {max_retries} attempts.")
                return False


def download_dropbox_folder(access_token: str, shared_link: str, output_dir: Path):
    dbx = dropbox.Dropbox(access_token)

    print("Resolving shared link...")
    try:
        result = dbx.sharing_get_shared_link_metadata(url=shared_link)
    except Exception as e:
        print(f"Error resolving shared link: {e}")
        return

    print("Scanning remote folder contents...")
    try:
        entries = []
        res = dbx.files_list_folder(path="", shared_link=dropbox.files.SharedLink(url=shared_link))
        entries.extend(res.entries)
        while res.has_more:
            res = dbx.files_list_folder_continue(cursor=res.cursor)
            entries.extend(res.entries)

        file_entries = [e for e in entries if isinstance(e, dropbox.files.FileMetadata)]
        print(f"Found {len(file_entries)} files to download.")

        output_dir.mkdir(parents=True, exist_ok=True)

        for entry in file_entries:
            file_name = entry.name
            local_file_path = output_dir / file_name

            if local_file_path.exists():
                print(f"  [SKIPPED] {file_name} already exists locally.")
                continue

            success = download_file_with_retry(dbx, shared_link, file_name, local_file_path)
            if not success:
                print("Stopping pipeline due to download failure.")
                return

        print("\nAll dataset files downloaded successfully into data/raw/!")

    except Exception as e:
        print(f"Error during folder download: {e}")


def download_github_folder(repo_owner: str, repo_name: str, folder_path: str, local_target_dir: Path):
    """Recursively downloads files and subfolders from a GitHub repository via the API."""
    api_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/contents/{folder_path}"

    response = requests.get(api_url)
    if response.status_code != 200:
        print(f"  [ERROR] Failed to fetch GitHub directory: {folder_path}. HTTP {response.status_code}")
        return

    items = response.json()
    local_target_dir.mkdir(parents=True, exist_ok=True)

    for item in items:
        if item['type'] == 'file':
            file_name = item['name']
            download_url = item['download_url']
            local_file_path = local_target_dir / file_name

            if local_file_path.exists():
                print(f"  [SKIPPED] {file_name} already exists locally.")
                continue

            print(f"  [DOWNLOADING] GitHub file: {folder_path}/{file_name}...")
            file_res = requests.get(download_url)

            with open(local_file_path, "wb") as f:
                f.write(file_res.content)

        elif item['type'] == 'dir':
            print(f"  [SCANNING FOLDER] {item['path']}...")
            new_folder_path = item['path']  # GitHub API provides the full relative path here
            new_local_target = local_target_dir / item['name']

            # Recursive call to dive into the subfolder
            download_github_folder(repo_owner, repo_name, new_folder_path, new_local_target)


if __name__ == "__main__":
    # --- CONFIGURATION ---
    # Dropbox settings
    DROPBOX_SHARED_LINK = "https://www.dropbox.com/scl/fo/2aczdnx37hxvcfdo4rq4q/AOjRokSTaiKxXmgUyqdcI6k?rlkey=5bg7mxxbq46t7aujfch46dlvz&e=1&dl=0"
    DROPBOX_ACCESS_TOKEN = os.getenv("DROPBOX_ACCESS_TOKEN")

    # GitHub settings
    GITHUB_OWNER = "Xuange923"
    GITHUB_REPO = "Surveillance-Video-Understanding"
    GITHUB_FOLDER = "UCF Annotation"

    # Project paths
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
    GITHUB_OUTPUT_DIR = RAW_DATA_DIR / "UCF_Annotation"

    # 1. Download video files from Dropbox
    if not DROPBOX_ACCESS_TOKEN:
        print("ERROR: Please set your DROPBOX_ACCESS_TOKEN in the .env file.")
    else:
        download_dropbox_folder(DROPBOX_ACCESS_TOKEN, DROPBOX_SHARED_LINK, RAW_DATA_DIR)

    # 2. Download the annotations from GitHub recursively
    print("\n--- Starting GitHub Annotation Download ---")
    download_github_folder(GITHUB_OWNER, GITHUB_REPO, GITHUB_FOLDER, GITHUB_OUTPUT_DIR)
    print("GitHub annotations downloaded successfully!")