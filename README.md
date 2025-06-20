# astra

## Command Line Interface (CLI)
- If you are installing as a user, `pip install -e .` then activate the virtual environment using `source .venv/bin/activate`. You can then run the CLI through `astra hello`.

- If you are installing as a developer with uv, use `uv sync` for synchronizing your environment (uv.lock file is provided in the repository). You can run the CLI through `uv run astra hello`.
    - Note: you can still activate the virtual environment once uv is synchronized and activate and run the same as the user.

## Testing
- Tests are kept in in the `tests` directory. To run these tests, nagivate to the root directory (i.e. `astra`, not `astra/src/astra`) and run `uv run pytest` in the command line.
- To run a specific test, run `uv run pytest -q <TEST SCRIPT>` (e.g. `uv run pytest -q test_constants.py`)

