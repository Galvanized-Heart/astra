# Astra
Astra is a multi-task model for predicting Michaelis-Menten kinetic parameters from elementary kinetic rate decompositions.

## Getting Started
Astra uses `uv` for development. To install `uv`, you can follow the instructions <a href=https://docs.astral.sh/uv/getting-started/installation>here</a> for updated instructions. At the time of writing, you can follow the instructions below to install `uv` and create the correct environment:
```
# Download and install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create .venv
uv venv

# Sync uv .venv with uv.lock
uv sync --locked

# Sync addtional dev dependencies
uv sync --dev
```

## Brief Tutorial for uv Usage
To run a script using 

