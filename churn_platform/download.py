"""Download the raw IBM Telco Customer Churn CSV.

Usage (from the project root):
    python -m churn_platform.download            # skip if file already exists
    python -m churn_platform.download --force    # re-download

The file is stored in data/raw/ and is NOT committed to Git. Instead, anyone
can reproduce it with this command. The SHA-256 hash is printed so you can
check that you and a reviewer are working with byte-identical data.
"""

import argparse
import hashlib
import sys
import urllib.request

from churn_platform.config import DATASET_URL, RAW_DATA_FILE


def sha256_of(path) -> str:
    """Return the SHA-256 hex digest of a file, read in chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(force: bool = False) -> None:
    if RAW_DATA_FILE.exists() and not force:
        print(f"Already present: {RAW_DATA_FILE}")
    else:
        RAW_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
        # Download to a temporary name first so a failed download never
        # leaves a half-written file that looks valid.
        tmp_path = RAW_DATA_FILE.with_suffix(".part")
        print(f"Downloading {DATASET_URL}")
        try:
            urllib.request.urlretrieve(DATASET_URL, tmp_path)
        except OSError as exc:
            tmp_path.unlink(missing_ok=True)
            sys.exit(f"Download failed: {exc}")
        tmp_path.replace(RAW_DATA_FILE)
        print(f"Saved to {RAW_DATA_FILE}")

    print(f"Size:    {RAW_DATA_FILE.stat().st_size} bytes")
    print(f"SHA-256: {sha256_of(RAW_DATA_FILE)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="re-download even if the file exists")
    download(force=parser.parse_args().force)
