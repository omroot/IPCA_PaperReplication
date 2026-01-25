"""
Module to check for required CSV data files and download them from Dropbox if missing.
"""

import os
import zipfile
import tempfile
import requests

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

REQUIRED_CSV_FILES = [
    "crsp_monthly_returns.csv",
    "datashare.csv",
]

DROPBOX_FOLDER_URL = (
    "https://www.dropbox.com/scl/fo/qytgfmm1zse9mwqam36mt/"
    "AL5ejDfWyx93pUAI5kkjnME?rlkey=nm3n0o1pdymsw4zkkm247c13y&st=msvrls3l&dl=1"
)


def get_missing_files():
    """Return list of required CSV files not present in the data directory."""
    missing = []
    for fname in REQUIRED_CSV_FILES:
        fpath = os.path.join(DATA_DIR, fname)
        if not os.path.isfile(fpath):
            missing.append(fname)
    return missing


def download_from_dropbox():
    """Download the Dropbox folder as a zip and extract CSV files to data/."""
    os.makedirs(DATA_DIR, exist_ok=True)

    print(f"Downloading data from Dropbox...")
    response = requests.get(DROPBOX_FOLDER_URL, stream=True)
    response.raise_for_status()

    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        tmp_path = tmp.name
        for chunk in response.iter_content(chunk_size=8192):
            tmp.write(chunk)

    try:
        with zipfile.ZipFile(tmp_path, "r") as zf:
            for member in zf.namelist():
                if member.endswith(".csv"):
                    filename = os.path.basename(member)
                    if not filename:
                        continue
                    target_path = os.path.join(DATA_DIR, filename)
                    print(f"  Extracting {filename}...")
                    with zf.open(member) as src, open(target_path, "wb") as dst:
                        dst.write(src.read())
        print("Download complete.")
    finally:
        os.unlink(tmp_path)


def ensure_data():
    """Check for missing CSV files and download them if needed. Returns the data directory path."""
    missing = get_missing_files()
    if missing:
        print(f"Missing data files: {', '.join(missing)}")
        download_from_dropbox()
        still_missing = get_missing_files()
        if still_missing:
            raise FileNotFoundError(
                f"After download, still missing: {', '.join(still_missing)}"
            )
    return DATA_DIR


if __name__ == "__main__":
    ensure_data()
