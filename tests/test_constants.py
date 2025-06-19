from pathlib import Path

from astra.constants import PROJECT_ROOT

def test_project_root_is_correct():
    """
    Tests that the PROJECT_ROOT constant is a Path object, is absolute,
    and points to the correct directory by checking for key files.
    """
    # Test if pathlib.Path object
    assert isinstance(PROJECT_ROOT, Path), "PROJECT_ROOT should be a pathlib.Path object"

    # Test if path is absolute
    assert PROJECT_ROOT.is_absolute(), "PROJECT_ROOT should be an absolute path"

    # Test if path is a directory
    assert PROJECT_ROOT.is_dir(), "PROJECT_ROOT should point to a valid directory"

    # Test for key files/folders at the root
    assert (PROJECT_ROOT / "pyproject.toml").is_file(), \
        "The file 'pyproject.toml' was not found at the project root."

    assert (PROJECT_ROOT / "src").is_dir(), \
        "The 'src' directory was not found at the project root."