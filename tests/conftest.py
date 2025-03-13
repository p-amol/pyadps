# This file contains pyfixtures of pyadps
import pytest

# Creating a temporary single ensemble binary file
@pytest.fixture
def binfile(tmp_path):
    # Reading a file containing a single ensemble
    ensemble = "tests/test_data/ensemble.000"
    with open(ensemble, "rb") as f:
        ensemble = f.read()

    # Creating a temporary file for manipulation
    binfile = tmp_path/"binfile.000"
    with open(binfile, "wb") as f:
        f.write(ensemble)
    return binfile