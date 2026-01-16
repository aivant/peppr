import types
from pathlib import Path
import peppr

DOCS_DIR = Path(__file__).parents[1] / "docs"


def test_public_api_reference():
    """
    Check if every function/class that is part of the public API is documented
    in the API reference, i.e. in the ``api.rst`` file.
    """
    with open(DOCS_DIR / "api.rst") as api_reference_file:
        api_reference = api_reference_file.read()

    for attr_name in dir(peppr):
        if attr_name.startswith("_"):
            continue
        attribute = getattr(peppr, attr_name)
        # Check classes and functions
        if isinstance(attribute, (type, types.FunctionType)):
            assert attr_name in api_reference, f"Missing entry for '{attr_name}'"
