from pathlib import Path
from astra.constants import PROJECT_ROOT

OUTPUT_DIR = PROJECT_ROOT.joinpath("data", "raw", "sabiork")

print(OUTPUT_DIR)