from pathlib import Path

# Define root path of project for file manipulations
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Define path to data
DATA_PATH = PROJECT_ROOT/"data"

# Define path to embeddings on Balam scratch (internal use only)
EMB_PATH = PROJECT_ROOT.parent.parent/"maxkirby"/"astra-embeddings"