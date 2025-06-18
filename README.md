# astra

## Command Line Interface (CLI)
- If you are installing as a user, `pip install -e .` then activate the virtual environment using `source .venv/bin/activate`. You can then run the CLI through `astra hello`.

- If you are installing as a developer with uv, use `uv sync` for synchronizing your environment (uv.lock file is provided in the repository). You can run the CLI through `uv run astra hello`.
    - Note: you can still activate the virtual environment once uv is synchronized and activate and run the same as the user.