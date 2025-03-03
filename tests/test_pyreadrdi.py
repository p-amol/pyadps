import pytest
import random, _io
import numpy as np
from unittest import mock
from pyadps.utils.pyreadrdi import  fileheader, ErrorCode, safe_open, \
                                    safe_read, fixedleader, datatype, \
                                    variableleader

############# pyfixtures #############
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

@pytest.fixture
def binfile(tmp_path):
    ensemble = "tests/test_data/ensemble.000"
    with open(ensemble, "rb") as f:
        ensemble = f.read()

    binfile = tmp_path/"binfile.000"
    with open(binfile, "wb") as f:
        f.write(ensemble)
        print(binfile)
    return binfile


############# ErrorCode #############
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

############# safe_open #############
def test_safe_open_without_parameter(binfile):
    with pytest.raises(TypeError):
        safe_open()

def test_safe_open_with_wrong_parameter():
    result = safe_open(123)
    assert len(result) == 2
    assert result[1] == ErrorCode.UNKNOWN_ERROR

def test_safe_open_with_ghost_file():
    result = safe_open("ghost_file.000")
    assert len(result) == 2
    assert result[1] == ErrorCode.FILE_NOT_FOUND

def test_safe_open_with_pure_file(binfile):
    result = safe_open(binfile)
    assert result[1] == ErrorCode.SUCCESS
    assert len(result) == 2
    assert isinstance(result[0], _io.BufferedReader)

def test_safe_open_without_file_permission(mocker):
    mock_open = mocker.patch("builtins.open")
    mock_open.side_effect=PermissionError("Permission deined")
    result = safe_open("file.000")
    assert len(result) == 2
    assert result[1] == ErrorCode.PERMISSION_DENIED

def test_safe_open_io_error(mocker):
    mock_open = mocker.patch("builtins.open")
    mock_open.side_effect=IOError("IOError")
    result = safe_open("file.000")
    assert len(result) == 2
    assert result[1] == ErrorCode.IO_ERROR

def test_safe_open_memory_error(mocker):
    mock_open = mocker.patch("builtins.open")
    mock_open.side_effect=MemoryError("Out Of Memory")
    result = safe_open("file.000")
    assert len(result) == 2
    assert result[1] == ErrorCode.OUT_OF_MEMORY

############# safe_read #############
def test_safe_read_without_parameter():
    with pytest.raises(TypeError):
        safe_read()

def test_safe_read_without_num_bytes(binfile):
    with open (binfile, 'rb') as f:
        with pytest.raises(TypeError):
            safe_read(f, "abc")

def test_safe_read_with_pure_file(binfile):
    with open (binfile, 'rb') as f:
        result = safe_read(f, 10)
        f.seek(0)
        assert result[0] == f.read(10)
    assert len(result) == 2
    assert result[1] == ErrorCode.SUCCESS

def test_safe_read_less_bytes_to_read(binfile):
    with open (binfile, 'rb') as f:
        result = safe_read(f, 800)
    assert result[0] == None
    assert len(result) == 2
    assert result[1] == ErrorCode.FILE_CORRUPTED

############# fileheader #############
def test_fileheader_without_rdi_file():
    with pytest.raises(TypeError):
        result = fileheader()

def test_fileheader_with_wrong_parameter():
    result = fileheader(123)
    assert len(result) == 7
    assert result[6] == 99

def test_fileheader_with_ghost_file():
    result = fileheader('ghost_file.000')
    _, _, _, _, _, _, error_code = result
    assert len(result) == 7
    assert error_code == 1 # Error code should indicate failure

def test_fileheader_with_pure_file():
    result = fileheader('./tests/test_data/test.000')
    datatype, byte, byteskip, address_offset, dataid, ensemble, error_code = result
    assert error_code == 0
    assert len(result) == 7
    assert isinstance(datatype, np.ndarray)
    assert isinstance(byte, np.ndarray)
    assert isinstance(byteskip, np.ndarray)
    assert isinstance(address_offset, np.ndarray)
    assert isinstance(dataid, np.ndarray)
    assert isinstance(ensemble, int)
    # Check specific values (if known, or check structure)
    assert len(datatype) > 0  # At least one datatype should be found
    assert ensemble > 0  # There should be at least one ensemble

def test_fileheader_without_file_access():
    # Mock the open function to raise a PermissionError when attempting to open the file
    with mock.patch("builtins.open", side_effect=PermissionError("Permission Denied")):
        result = fileheader("somefile.000")
    # Assert that the function handles the PermissionError correctly
    assert len(result) == 7
    assert result[6] == 2

def test_fileheader_wrong_format(tmp_path):
    temp_file = tmp_path/"test_file.bin"
    binary_data = (b'\x00\x00\x00\x00\x00\x00')
    temp_file.write_bytes(binary_data)
    result = fileheader(temp_file)
    _, _, _, _, _, _, error_code = result
    # Assertions
    assert len(result) == 7
    assert error_code == 5 # Error code should indicate wrong format

def test_fileheader_IO_error():
    # Mock the open function to raise a I/O Error when attempting to open the file
    with mock.patch("builtins.open", side_effect=OSError("I/O Error")):
        result = fileheader("somefile.000")
    # Assert that the function handles the I/O Error correctly
    assert len(result) == 7
    assert result[6] == 3

def test_fileheader_Memory_error():
    # Mock the open function to raise a Memory Error when attempting to open the file
    with mock.patch("builtins.open", side_effect=MemoryError("Not enough memory")):
        result = fileheader("somefile.000")
    # Assert that the function handles the Memory Error correctly
    assert len(result) == 7
    assert result[6] == 4

def test_fileheader_unknown_error():
    # Mock the open function to raise an unknown error when attempting to open the file
    with mock.patch("builtins.open", side_effect=Exception("Unknown error")):
        result = fileheader("somefile.000")
    # Assert that the function handles the unknown Error correctly
    assert len(result) == 7
    assert result[6] == 99

def test_fileheader_unexpected_end_of_file(tmp_path):
    temp_file = tmp_path/"test_file.bin"
    binary_data = (b'\x7f\x7f\xf0\x02\x00\x06\x12\x00\x4d\x00\x8e\x00\x80\x01\xfa\x01\x74')
    temp_file.write_bytes(binary_data)
    result = fileheader(temp_file)
    error_code = result[6]
    # Assertions
    assert len(result) == 7
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
    assert len(result) == 7
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
    assert len(result) == 7
    assert error_code == 7

#############   fixedleader  #############
def test_fixleader_without_rdi_file():
    with pytest.raises(TypeError):
        result = fixedleader()

def test_fixleader_with_wrong_parameter():
    result = fixedleader(123)
    assert len(result) == 3
    assert result[2] == 99

def test_fixleader_with_ghost_file():
    result = fixedleader('ghost_file.000')
    assert len(result) == 3
    assert result[2] == 1

def test_fixleader_with_pure_file(binfile):
    result = fixedleader(binfile)
    assert len(result) == 3
    assert result[2] == 0
    assert result[1] >= 0

def test_fixleader_without_file_access(mocker):
    mock_open = mocker.patch("builtins.open")
    mock_open.side_effect=PermissionError("Permission deined")
    result = fixedleader("nofile.000")
    assert len(result) == 3
    assert result[2] == 2

def test_fixleader_corrupted_file(binfile):
    with open(binfile, "r+b") as f:
        f.write(f.read())
        f.seek(0)
        f.truncate(839)
        f.seek(0)
        result = fixedleader(binfile)
        assert len(result) == 3
        assert result[2] == 8

def test_fixleader_corrupted_fix_lead_data(binfile):
    with open(binfile, "r+b") as f:
        f.seek(0)
        f.truncate(60)
        result = fixedleader(binfile)
        assert len(result) == 3
        assert result[2] == 8

def test_fixleader_corrupted_fix_lead_id(binfile):
    with open(binfile, "r+b") as f:
        f.seek(18)
        f.write(b'\x70\x7f')
        f.seek(0)
        result = fixedleader(binfile)
        assert len(result) == 3
        assert result[2] == 6

def test_fixleader_corrupted_fourth_fix_lead_id(binfile):
    with open(binfile, "r+b") as f:
        for i in range(2):
            f.write(f.read())
            f.seek(i*754)
        f.seek(2280)
        f.write(b'\x70\x7f')
        f.seek(0)
        result = fixedleader(binfile)
        assert len(result) == 3
        assert result[2] == 6

############# Variableleader #############
def test_varleader_without_rdi_file():
    with pytest.raises(TypeError):
        result = variableleader()

def test_varleader_with_wrong_parameter():
    result = variableleader(123)
    assert len(result) == 3
    assert result[2] == 99

def test_varleader_with_ghost_file():
    result = variableleader('ghost_file.000')
    assert len(result) == 3
    assert result[2] == 1

def test_varleader_with_pure_file(binfile):
    result = variableleader(binfile)
    assert len(result) == 3
    assert result[2] == 0
    assert result[1] >= 0

def test_varleader_without_file_access(mocker):
    mock_open = mocker.patch("builtins.open")
    mock_open.side_effect=PermissionError("Permission deined")
    result = variableleader("nofile.000")
    assert len(result) == 3
    assert result[2] == 2

def test_varleader_corrupted_file(binfile):
    with open(binfile, "r+b") as f:
        f.write(f.read())
        f.seek(0)
        f.truncate(894)
        result = variableleader(binfile)
        assert len(result) == 3
        assert result[2] == 8

def test_varleader_corrupted_var_lead_data(binfile):
    with open(binfile, "r+b") as f:
        f.seek(0)
        f.truncate(138)
        result = variableleader(binfile)
        assert len(result) == 3
        assert result[2] == 8

def test_varleader_corrupted_var_lead_id(binfile):
    with open(binfile, "r+b") as f:
        f.seek(77)
        f.write(b'\x70\x7f')
        f.seek(0)
        result = variableleader(binfile)
        assert len(result) == 3
        assert result[2] == 6

def test_varleader_corrupted_fourth_var_lead_id(binfile):
    with open(binfile, "r+b") as f:
        for i in range(2):
            f.write(f.read())
            f.seek(i*754)
        f.seek(2339)
        f.write(b'\x70\x7f')
        f.seek(0)
        result = variableleader(binfile)
        assert len(result) == 3
        assert result[2] == 6


if __name__ == '__main__':
    pytest.main()
