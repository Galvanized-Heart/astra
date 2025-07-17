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
Run scripts
```
uv run hello_world.py
```
<br>

Add dependencies to `pyproject.toml`
```
uv add pandas
```
<br>

Remove dependencies from `pyproject.toml` and `uv.lock`
```
uv remove pandas
```
<br>

Sync environment with dependencies from `pyproject.toml`
```
uv sync
```
- `--locked` flag is used to sync environment from `uv.lock`. 
<br>

Update lockfile
```
uv lock
```
<br>
