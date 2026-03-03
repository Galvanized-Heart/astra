# Set cache directories
export SCRATCH_CACHE_DIR=".cache"
export HF_HOME="${SCRATCH_CACHE_DIR}/huggingface"
mkdir -p "$HF_HOME"

# Run cacher
uv run -c "from transformers import EsmModel, AutoTokenizer; model_name='facebook/esm2_t33_650M_UR50D'; AutoTokenizer.from_pretrained(model_name); EsmModel.from_pretrained(model_name)"