import argparse
import sys
import time
from pathlib import Path

import requests

from astra.constants import PROJECT_ROOT

API_ROOT = "https://sabiork.h-its.org/sabioRestWebServices"
OUT_DIR = Path.joinpath(PROJECT_ROOT, "data", "raw", "sabiork")
BATCH_SIZE = 500
SLEEP = 1.0
TIMEOUT = (15, 120) # (connect, read)

def request_entry_ids(query: str) -> list[str]:
    """Return every EntryID matching the provided query"""
    url = f"{API_ROOT}/searchKineticLaws/entryIDs"
    params = {"q": query, "format": "txt"}
    r = requests.get(url, params=params, timeout=TIMEOUT)
    if r.status_code == 400:
        raise ValueError(f"Bad query syntax: {query!r}")
    r.raise_for_status()
    return [x.strip() for x in r.text.split() if x.strip().isdigit()]

def get_all_entry_ids() -> list[str]:
    """Queries SABIO-RK API for entire list of entires"""
    try:
        ids = request_entry_ids("*:*")
        if ids:
            print(f"Retreived {len(ids):,} IDs")
            return ids
    except ValueError:
        pass
    sys.exit("Could not retrieve EntryIDs.")

def chunks(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i+size]

def download_batch(id_batch, idx, normalized=True):
    url = f"{API_ROOT}/kineticLaws"
    params = {"kinlawids": ",".join(id_batch)}
    if not normalized:
        params["normalized"] = "false"
    dest = OUT_DIR / f"sabio_batch_{idx:05d}.xml"
    with requests.get(url, params=params, stream=True, timeout=TIMEOUT) as r:
        r.raise_for_status()
        with open(dest, "wb") as fh:
            for blk in r.iter_content(8192):
                fh.write(blk)
    print(f"Saved {dest}  (+{len(id_batch)})")
    time.sleep(SLEEP)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=BATCH_SIZE,
                        help="IDs per request (default 500)")
    parser.add_argument("--raw-units", action="store_true",
                        help="use original units (normalized=false)")
    args = parser.parse_args()

    OUT_DIR.mkdir(exist_ok=True)
    ids = get_all_entry_ids()

    for i, batch in enumerate(chunks(ids, args.batch), 1):
        download_batch(batch, i, normalized=not args.raw_units)

    print(f"\nFinished: {len(ids):,} entries written to {OUT_DIR.resolve()}")

if __name__ == "__main__":
    main()
