import pytest
import random
import numpy as np
from unittest import mock
from pyadps.utils.pyreadrdi import fileheader, ErrorCode


################ FileHeader ####################
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

def test_fileheader_wrong_format(tmp_path):
    temp_file = tmp_path/"test_file.bin"
    binary_data = (b'\x00\x00\x00\x00\x00\x00')
    temp_file.write_bytes(binary_data)
    result = fileheader(temp_file)
    _, _, _, _, _, _, error_code = result
    # Assertions
    assert error_code == 5 # Error code should indicate wrong format

def test_fileheader_permission_denied():
    # Mock the open function to raise a PermissionError when attempting to open the file
    with mock.patch("builtins.open", side_effect=PermissionError("Permission Denied")):
        result = fileheader("somefile.000")
    # Assert that the function handles the PermissionError correctly
    assert result[6] == 2

def test_fileheader_IO_Error():
    # Mock the open function to raise a I/O Error when attempting to open the file
    with mock.patch("builtins.open", side_effect=OSError("I/O Error")):
        result = fileheader("somefile.000")
    # Assert that the function handles the I/O Error correctly
    assert result[6] == 3

def test_fileheader_Memory_Error():
    # Mock the open function to raise a Memory Error when attempting to open the file
    with mock.patch("builtins.open", side_effect=MemoryError("Not enough memory")):
        result = fileheader("somefile.000")
    # Assert that the function handles the Memory Error correctly
    assert result[6] == 4

def test_fileheader_unknown_error():
    # Mock the open function to raise an unknown error when attempting to open the file
    with mock.patch("builtins.open", side_effect=Exception("Unknown error")):
        result = fileheader("somefile.000")
    # Assert that the function handles the unknown Error correctly
    assert result[6] == 99

def test_fileheader_unexpected_end_of_file(tmp_path):
    temp_file = tmp_path/"test_file.bin"
    binary_data = (b'\x7f\x7f\xf0\x02\x00\x06\x12\x00\x4d\x00\x8e\x00\x80\x01\xfa\x01\x74')
    temp_file.write_bytes(binary_data)
    result = fileheader(temp_file)
    error_code = result[6]
    # Assertions
    assert error_code == 8

def test_fileheader_ID_not_found(tmp_path):
    temp_file = tmp_path/"test_file.bin"
    binary_data = ( b'\x7f\x7f\x18\x00\x00\x06\x12\x00\x4d\x00\x8e\x00\x80'\
                    b'\x01\xfa\x01\x74\x02\x00\x00\x00\x00\x64\x00\x00\x00'\
                    b'\x7c\x7c\x18\x00\x00\x06\x12\x00\x4d\x00\x8e\x00\x80'\
                    b'\x01\xfa\x01\x74\x02\x00')
    temp_file.write_bytes(binary_data)
    result = fileheader(temp_file)
    error_code = result[6]
    # Assertions
    assert error_code == 6

def test_fileheader_datatype_mismatch(tmp_path):
    temp_file = tmp_path/"test_file.bin"
    binary_data = ( b'\x7f\x7f\x18\x00\x00\x06\x12\x00\x4d\x00\x8e\x00\x80'\
                    b'\x01\xfa\x01\x74\x02\x00\x00\x00\x00\x64\x00\x00\x00'\
                    b'\x7f\x7f\x18\x00\x00\x05\x12\x00\x4d\x00\x00\x80'\
                    b'\x01\xfa\x01\x74\x00')
    temp_file.write_bytes(binary_data)
    result = fileheader(temp_file)
    error_code = result[6]
    # Assertions
    assert error_code == 7



################ unit -> ErrorCode ####################
@pytest.fixture
def ErrorCode_data_collector():
    defined_error_codes =  {"SUCCESS": (0, "Success"),
                            "FILE_NOT_FOUND": (1, "Error: File not found."),
                            "PERMISSION_DENIED": (2, "Error: Permission denied."),
                            "IO_ERROR": (3, "IO Error: Unable to open file."),
                            "OUT_OF_MEMORY": (4, "Error: Out of memory."),
                            "WRONG_RDIFILE_TYPE": (5, "Error: Wrong RDI File Type."),
                            "ID_NOT_FOUND": (6, "Error: Data type ID not found."),
                            "DATATYPE_MISMATCH": (7, "Warning: Data type mismatch."),
                            "FILE_CORRUPTED": (8, "Warning: File Corrupted."),
                            "VALUE_ERROR": (9, "Value Error for incorrect argument."),
                            "UNKNOWN_ERROR": (99, "Unknown error.")}
    collected_error_codes = {}
    for i in dir(ErrorCode):
        if (i.startswith("__") or i.endswith("__")):
            pass
        else:
            temp_error_code = getattr(ErrorCode, i)
            collected_error_codes[i] = (temp_error_code.value[0], temp_error_code.value[1])
            
    return defined_error_codes, collected_error_codes
    
def test_ErrorCode(ErrorCode_data_collector):
    defined_error_codes, collected_error_codes = ErrorCode_data_collector
    # Checks wheter the number of attributes changed or not.
    assert len(defined_error_codes) == len(collected_error_codes)
    
    # Checks each attributes with last updated attribute values.
    for i in defined_error_codes:
        assert defined_error_codes[i] == collected_error_codes[i]
    
    # Checks wheter the function get_message is working correctly.
    random_error_code = random.randint(0, 10)
    check = ErrorCode.get_message(random_error_code)
    assert  any(check == value[1] for value in defined_error_codes.values())
    
    assert ErrorCode.get_message(15) == 'Error: Invalid error code.'

if __name__ == '__main__':
    pytest.main()
