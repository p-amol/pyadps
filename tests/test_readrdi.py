import pytest
import numpy as np
from pyadps import ReadFile, error_code, check_equal, \
                   FileHeader, FixedLeader, VariableLeader, \
                   Velocity, Correlation, Echo, PercentGood, \
                    Status

############# FileHeader #############
def test_FileHeader_without_parameter():
    with pytest.raises(TypeError):
        result = FileHeader()

def test_FileHeader_with_pure_RDI_file(binfile):
    result = FileHeader(binfile)
    # Checking the quality of result itself
    assert type(result) == FileHeader
    # assert len(dir(result)) == 39
    
    # Check the values of attributes
    assert result.byteskip[0] == 754
    assert len(result.dataid[0]) == 6
    assert result.ensembles == 1
    assert result.error == 0
    assert result.warning == "Success"

    # Testing the check_file method 
    assert len(result.check_file()) == 6
    assert result.check_file()['File Size Match'] == True
    
    # Testing the data_types method
    assert len(result.data_types(0)) == 6
    assert result.data_types(0)[4] == 'Echo'

############# FixedLeader #############
def test_FixLeader_whithout_parameter():
    with pytest.raises(TypeError):
        result = FixedLeader()

def test_FixedLeaser_with_pure_file(binfile):
    result = FixedLeader(binfile)
    # Checking the quality of result itself
    assert type(result) == FixedLeader
    # assert len(dir(result)) == 74

    # Checking the value of the attributes
    assert result.beams.data[0] == 4
    assert result.cells.data[0] == 30
    assert result.depth_cell_len.data[0] == 1600
    assert result.warning == "Success"
    assert result.pings.data[0] == 14

    # Checking the field method
    assert len(result.field()) == 35
    assert result.field()['Lag Length'] == 53
    assert result.field()['Pings'] == 14
    assert result.field()['Beam Angle'] == 20
    assert result.field()['Xmit Pulse Len'] == 1734

    # Checking the system_configuration method
    assert len(result.system_configuration()) == 7
    assert result.system_configuration()['Beam Pattern'] == 'Convex'
    assert result.system_configuration()['Beam Direction'] == 'Up'
    assert result.system_configuration()['Beam Angle'] == '20'

    # Checking the ex_coordinate_trans metod
    assert len(result.ex_coord_trans()) == 4
    assert result.ex_coord_trans()['Tilt Correction'] == np.True_
    assert result.ex_coord_trans()['Coordinates'] == 'Earth Coordinates'

    # Checking the ez_sensor method
    assert len(result.ez_sensor()) == 7
    assert result.ez_sensor()['Conductivity Sensor'] == False
    assert result.ez_sensor()['Temperature Sensor'] == True

def test_FixedLeader_with_nonuniform_data(binfile):
    with open(binfile, "r+b") as f:
        f.write(f.read())
        f.seek(754+26)
        f.write(b'\x06\x28')
        f.seek(0)
        result = FixedLeader(binfile)
        # Checking the functionality of the is_uniform method
        assert result.is_uniform()['Beams'] == np.False_
        assert result.is_uniform()['Cells'] == np.False_
        assert result.is_uniform()['Pings'] == np.True_
        assert result.is_uniform()['Depth Cell Len'] == np.True_


############# VariableLeader #############
def test_VariableLeader_without_parameter():
    with pytest.raises(TypeError):
        result = VariableLeader()

def test_VariableLeader_with_pure_RDI_file(binfile):
    result = VariableLeader(binfile)
    assert type(result) == VariableLeader
    # assert len(dir(result)) == 84

    # Checking the value of the attributes
    assert result.depth_of_transducer.data[0] == 4456
    assert result.ensembles == 1
    assert result.pressure.data[0] == 447551
    assert result.roll.data[0] == 208
    assert result.speed_of_sound.data[0] == 1498
    assert result.warning == 'Success'

    # Checking the functionality of adc_channel method
    assert len(result.adc_channel()) == 3
    assert result.adc_channel()['Xmit Voltage'][0] == 257.40443700000003
    assert result.adc_channel()['Xmit Current'] == np.array([3.50704])
    assert result.adc_channel()['Ambient Temperature'] == 97.45030766051309

    # Checking the functionality of bitresult method
    assert len(result.bitresult()) == 8
    assert result.bitresult()['Reserved #1'] == 0
    assert result.bitresult()['DEMOD 1 Error'] == 0
    assert result.bitresult()['DEMOD 0 Error'] == 0
    assert result.bitresult()['Timing Card Error'] == 0

############# HelperFunctions #############
def attribute_assertion(result, test):
    if test == "velocity":
        assert type(result) == Velocity
        assert result.missing_value == "-32768"
        assert result.valid_max == 32768
    else:
        assert isinstance(result, (Correlation, Echo, PercentGood, Status))
        assert result.long_name in ("Correlation Magnitude", "Echo Intensity",
                                    "Percent Good", "Status Data Format")
        assert result.valid_max == 255

    # assert len(dir(result)) == 39
    assert result.beams[0] == 4
    assert len(result.data) == result.beams
    assert len(result.data[0]) == result.cells
    assert result.error == 0
    assert result.warning == 'Success'

############# Velocity #############
def test_velocity_without_parameter():
    with pytest.raises(TypeError):
        result = Velocity()

def test_velocity_with_pure_RDI_file(binfile):
    result = Velocity(binfile)
    attribute_assertion(result, "velocity")

############# Correlation #############
def test_correlation_without_parameter():
    with pytest.raises(TypeError):
        result = Correlation()

def test_correlation_with_pure_RDI_file(binfile):
    result = Correlation(binfile)
    attribute_assertion(result, "Correlation")

############# Echo #############
def test_Echo_without_parameter():
    with pytest.raises(TypeError):
        result = Echo()

def test_Echo_with_pure_RDI_file(binfile):
    result = Echo(binfile)
    attribute_assertion(result, "Correlation")

############# ReadFile #############
def test_ReadFile_without_parameter():
    with pytest.raises(TypeError):
        result = ReadFile()

def test_ReadFile_with_pure_RDI_file(binfile):
    result = ReadFile(binfile)
    assert type(result) == ReadFile
    # assert len(dir(result)) == 136

    assert result.variableleader.ensembles == 1
    assert result.variableleader.depth_of_transducer.data[0] == 4456
    assert result.variableleader.pressure.data[0] == 447551
    assert result.variableleader.temperature.data[0] == 1024

    assert result.fixedleader.beams.data[0] == 4
    assert result.fixedleader.cells.data[0] == 30
    assert result.fixedleader.cpu_version.data[0] == 50
    assert result.fixedleader.pings.data[0] == 14

    assert len(result.velocity.data) == 4
    assert result.correlation.data[0][1][0] == 122
    assert result.echo.data[0][1][0] == 161
    assert result.percentgood.data[0][1][0] == 0

@pytest.mark.parametrize("code, value",
                            [(0, "Data type is healthy"),
                            (1, "End of file"),
                            (2, "File Corrupted (ID not recognized)"),
                            (3, "Wrong file type"),
                            (4, "Data type mismatch"),
                            (5, "Unknown error")])
def test_error_code(code, value):
    assert error_code(code) == value

def test_check_equal():
    array_1 = np.full ((1, 10), 6)
    assert check_equal(array_1[0])
    array_1[0][9] = 7
    assert not check_equal(array_1[0])

