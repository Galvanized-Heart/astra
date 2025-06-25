import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

from astra.constants import PROJECT_ROOT


API_ROOT = "https://sabiork.h-its.org/sabioRestWebServices"
OUT_DIR = Path.joinpath(PROJECT_ROOT, "data", "raw", "sabiork")
BATCH_SIZE = 500
SLEEP = 1.0 # seconds
TIMEOUT = (15, 120) # (connect, read)


def request_entry_ids(query: str) -> list[str]:
    """Return every EntryID matching the provided query"""
    url = f"{API_ROOT}/searchKineticLaws/entryIDs"
    params = {"q": query, "format": "txt"}
    try:
        r = requests.get(url, params=params, timeout=TIMEOUT)
        r.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Error during request: {e}")
        raise

    return [x.strip() for x in r.text.split() if x.strip().isdigit()]


def chunks(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i+size]


def get_all_entry_ids() -> list[str]:
    """Queries SABIO-RK API for entire list of entires"""
    try:
        ids = request_entry_ids("*:*")
        if ids:
            print(f"Retreived {len(ids):,} IDs")
            return ids
    except ValueError:
        pass
    except requests.exceptions.RequestException:
        pass
    sys.exit("Could not retrieve EntryIDs.")


def download_batch(id_batch, idx, normalized=True):
    """Downloads a batch of kinetic laws."""
    url = f"{API_ROOT}/kineticLaws"
    params = {"kinlawids": ",".join(id_batch)}
    if not normalized:
        params["normalized"] = "false"
    dest = OUT_DIR / f"sabio_batch_{idx:05d}.xml"

    # Check if the file already exists. Skip if it does.
    if dest.exists():
        print(f"Skipping existing file: {dest}")
        return

    start_time = time.time()
    try:
        with requests.get(url, params=params, stream=True, timeout=TIMEOUT) as r:
            r.raise_for_status()
            with open(dest, "wb") as fh:
                for blk in r.iter_content(8192):
                    fh.write(blk)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Saved {dest}  (+{len(id_batch)}) in {elapsed_time:.2f} seconds")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading batch {idx}: {e}")
        return

    time.sleep(SLEEP)


def determine_downloaded_indices(output_directory: Path) -> set[int]:
    """Determines the indices of already downloaded batches based on existing files."""
    downloaded_indices = set()
    for file in output_directory.glob("sabio_batch_*.xml"):
        try:
            index = int(file.stem.split("_")[-1])  # Extract index from filename
            downloaded_indices.add(index)
        except ValueError:
            print(f"Warning: Could not parse index from filename: {file.name}")
    return downloaded_indices


def prepare_batches_for_download(all_ids: list[str], batch_size: int, downloaded_indices: set[int]) -> list[tuple[int, list[str]]]:
    """Creates batches from all IDs and filters out already downloaded ones."""
    batches = list(chunks(all_ids, batch_size))
    batches_to_download = [(i + 1, batch) for i, batch in enumerate(batches) if (i + 1) not in downloaded_indices]
    return batches_to_download


def download_batches_concurrently(batches_to_download: list[tuple[int, list[str]]], num_threads: int, normalized: bool):
    """Downloads batches concurrently using a thread pool."""
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(download_batch, batch, i, normalized=normalized) for i, batch in batches_to_download]
        for future in futures:
            future.result()  # Raise any exceptions that occurred in the threads.


def cleanup_incomplete_files(output_dir: Path, downloaded_indices: set[int], batches_to_download: list[tuple[int, list[str]]], num_threads: int):
    """Cleans up potentially incomplete files after an interruption."""
    print("\nInterrupted! Cleaning up potentially incomplete files...")
    # Calculate the range of files to potentially delete based on the current batch number
    # Assuming file names are sabio_batch_00001.xml, etc.
    end_index = len(downloaded_indices) + len(batches_to_download)  # Index of the last batch attempted

    for i in range(max(1, end_index - num_threads + 1), end_index + 1):  # prevent index to be less than 1
        file_path = output_dir / f"sabio_batch_{i:05d}.xml"
        if file_path.exists():
            try:
                os.remove(file_path)
                print(f"Deleted potentially incomplete file: {file_path}")
            except OSError as e:
                print(f"Error deleting file {file_path}: {e}")
    print("Cleanup complete.")


def main():
    # Parse CLI arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=BATCH_SIZE,
                        help="IDs per request (default 500)")
    parser.add_argument("--raw-units", action="store_true",
                        help="use original units (normalized=false)")
    parser.add_argument("--threads", type=int, default=4,
                        help="Number of threads to use (default 4)")
    args = parser.parse_args()

    OUT_DIR.mkdir(exist_ok=True)

    # Get all EntryIDs from SABIO-RK API
    ids = get_all_entry_ids()

    # Determine already downloaded batches
    downloaded_indices = determine_downloaded_indices(OUT_DIR)

    # Prepare batches for download
    batches_to_download = prepare_batches_for_download(ids, args.batch, downloaded_indices)

    print(f"Found {len(downloaded_indices)} existing batches.")
    print(f"Downloading {len(batches_to_download)} batches.")

    try:
        # Download batches concurrently
        download_batches_concurrently(batches_to_download, args.threads, normalized=not args.raw_units)

    except KeyboardInterrupt:
        cleanup_incomplete_files(OUT_DIR, downloaded_indices, batches_to_download, args.threads)
        raise  # Re-raise KeyboardInterrupt to terminate the script

    finally:
        print(f"\nFinished: {len(ids):,} entries (attempted) to be written to {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()