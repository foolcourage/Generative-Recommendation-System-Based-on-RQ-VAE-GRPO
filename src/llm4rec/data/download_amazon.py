"""Download Amazon 2018 category reviews and metadata with resume support."""

from __future__ import annotations

import argparse
import gzip
import shutil
from pathlib import Path

import requests
from tqdm import tqdm

from llm4rec.utils import project_root, resolve_project_path

BASE_URL = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2"


def _download(url: str, destination: Path, force: bool = False) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0 and not force:
        print(f"[skip] {destination} already exists")
        return

    partial = destination.with_suffix(destination.suffix + ".part")
    start = partial.stat().st_size if partial.exists() else 0
    headers = {"Range": f"bytes={start}-"} if start else {}
    response = requests.get(url, headers=headers, stream=True, timeout=(30, 120))
    response.raise_for_status()
    if start and response.status_code != 206:
        start = 0
        partial.unlink(missing_ok=True)

    total = int(response.headers.get("content-length", 0)) + start
    mode = "ab" if start else "wb"
    with partial.open(mode) as handle, tqdm(
        total=total, initial=start, unit="B", unit_scale=True, desc=destination.name
    ) as progress:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                handle.write(chunk)
                progress.update(len(chunk))
    shutil.move(str(partial), str(destination))

    # Fail early on proxy/login HTML saved under a gzip filename.
    try:
        with gzip.open(destination, "rb") as handle:
            handle.read(1)
    except OSError as exc:
        destination.unlink(missing_ok=True)
        raise RuntimeError(f"Downloaded file is not valid gzip: {url}") from exc


def download_category(category: str, output_dir: Path, force: bool = False) -> tuple[Path, Path]:
    review_name = f"{category}_5.json.gz"
    metadata_name = f"meta_{category}.json.gz"
    review_path = output_dir / review_name
    metadata_path = output_dir / metadata_name
    _download(f"{BASE_URL}/categoryFilesSmall/{review_name}", review_path, force=force)
    _download(f"{BASE_URL}/metaFiles2/{metadata_name}", metadata_path, force=force)
    return review_path, metadata_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", default="All_Beauty")
    parser.add_argument("--output-dir", default="data/raw")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output_dir = resolve_project_path(args.output_dir, project_root())
    review_path, metadata_path = download_category(args.category, output_dir, force=args.force)
    print(f"Reviews: {review_path}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()

