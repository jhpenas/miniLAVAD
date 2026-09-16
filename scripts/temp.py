import os
from pathlib import Path
import dropbox
from dotenv import load_dotenv

load_dotenv()

if __name__ == "__main__":
    dbx = dropbox.Dropbox(os.getenv("DROPBOX_ACCESS_TOKEN"))
    shared_link = "https://www.dropbox.com/scl/fo/2aczdnx37hxvcfdo4rq4q/AOjRokSTaiKxXmgUyqdcI6k?rlkey=5bg7mxxbq46t7aujfch46dlvz&e=1&dl=0"

    try:
        # Let's inspect the contents of the shared link root
        res = dbx.files_list_folder(path="", shared_link=dropbox.files.SharedLink(url=shared_link))
        print(f"Total entries found at root: {len(res.entries)}")
        for entry in res.entries:
            print(f" - [{type(entry).__name__}] {entry.name} (Path: {getattr(entry, 'path_lower', 'N/A')})")
    except Exception as e:
        print(f"Inspection error: {e}")