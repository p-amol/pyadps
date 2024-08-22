import os
import sys
import unittest
import numpy as np

# Assuming fortreadrdi and readrdi are in the same directory
sys.path.append(os.path.dirname(__file__))

from pyadps.utils.readrdi import FileHeader

class TestFileHeader(unittest.TestCase):

    def setUp(self):
        # Path to your binary RDI ADCP file
        self.rdi_file = 'EQ213000.000'
        self.header_fortran = FileHeader(self.rdi_file, run="fortran")

    def test_filename(self):
        self.assertEqual(self.header_fortran.filename, self.rdi_file)

    def test_total_ensembles(self):
        self.assertGreater(self.header_fortran.ensembles, 0)

    def test_header_information(self):
        self.assertIn('Header ID', self.header_fortran.header)
        self.assertIn('Source ID', self.header_fortran.header)
        self.assertIn('Bytes', self.header_fortran.header)
        self.assertIn('Spare', self.header_fortran.header)
        self.assertIn('Data Types', self.header_fortran.header)
        self.assertIn('Address Offset', self.header_fortran.header)

    def test_check_file(self):
        file_check = self.header_fortran.check_file()
        self.assertIn('System File Size (B)', file_check)
        self.assertIn('Calculated File Size (B)', file_check)
        self.assertIn('File Size Match', file_check)
        self.assertIn('Byte Uniformity', file_check)
        self.assertIn('Data Type Uniformity', file_check)

    def test_data_types(self):
        data_types = self.header_fortran.data_types(ens=0)
        expected_data_types = [
            "Fixed Leader", "Variable Leader", "Velocity", "Correlation",
            "Echo", "Percent Good", "Status", "Bottom Track", "ID not Found"
        ]
        for dtype in data_types:
            self.assertIn(dtype, expected_data_types)

    def test_check_file_print(self):
        # This test checks if the method runs without errors
        self.header_fortran.check_file_print()

if __name__ == "__main__":
    unittest.main()
