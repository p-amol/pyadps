import numpy as np
import pytest
from pyadps.utils.pyreadrdi import fileheader

def test_fileheader_with_sample_file():
    result = fileheader('./tests/test_data/test.000')
    datatype, byte, byteskip, address_offset, dataid, ensemble, error_code = result
    assert error_code == 0
    assert isinstance(datatype, np.ndarray)
    assert isinstance(byte, np.ndarray)
    assert isinstance(byteskip, np.ndarray)
    assert isinstance(address_offset, np.ndarray)
    assert isinstance(dataid, np.ndarray)
    assert isinstance(ensemble, int)

    # Check specific values (if known, or check structure)
    assert len(datatype) > 0  # At least one datatype should be found
    assert ensemble > 0  # There should be at least one ensemble


def test_fileheader_file_not_found():
    result = fileheader('non_existent_file.bin')
    _, _, _, _, _, _, error_code = result
    assert error_code != 0 # Error code should indicate failure

def test_fileheader_wrong_format():
    with open('./tests/test_data/invalid_format.bin', 'wb') as file:
        file.write(b'\x00\x00\x00\x00')
    result = fileheader('invalid_format.bin')
    _, _, _, _, _, _, error_code = result
    # Assertions
    assert error_code != 0  # Error code should indicate wrong format


if __name__ == '__main__':
    pytest.main()
