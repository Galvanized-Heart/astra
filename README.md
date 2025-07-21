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
Run scripts (e.g. Python or shell commands)
```
uv run <COMMAND>
```
<br>

Add dependencies to `pyproject.toml`
```
uv add <PACKAGE>
```
<br>

Remove dependencies from `pyproject.toml` and `uv.lock`
```
uv remove <PACKAGE>>
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

## NeurIPs 2025 Worshop Planning:
NeurIPs 2025 Workshop Dates:
- Suggested Paper Submission - Aug 22nd, 2025
- Mandatory Accept/Reject - Sep 22, 2025
- Conference - Dec 2, 2025 to Dec 6, 2025

NeurIPs 2025 Potential Workshops:
- <a href=https://icml.cc/virtual/2025/workshop/39959>2nd Workshop on Multi-modal Foundation Models and Large Language Models for Life Sciences</a> (temporary link)
- <a href=https://ai4sciencecommunity.github.io/neurips25>AI for Science: The Reach and Limits of AI for Scientific Discovery</a>

Writing TODOs:
- Write abstract
- Write paper outline
- Write first draft
- Get first draft feedback
- Implement feedback

Software TODOs:
- Build training logic
	- Needs to accept train_path, valid_path
- Create loss logic
- Create splitting logic (mmseqs2, tanimoto, random)
	- Splitting happens outside of DataModule
- Create model architectures
- Compare ablations (individual, naive combined, basic recomp, advanced recomp)
	- XGBoost can't do multiple regression, it just uses more models.

Optional TODOs:
- Create featurizations (optional)
