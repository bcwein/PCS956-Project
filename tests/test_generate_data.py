# tests/test_generate_data.py

import pytest
from src.generate_data import generate_synthetic_vineyard_data

def test_generate_synthetic_vineyard_data():
    """
    Test that synthetic data generation works and produces the correct structure.
    """
    data = generate_synthetic_vineyard_data()
    
    # Check that the output is a DataFrame
    assert isinstance(data, pd.DataFrame)
    
    # Check that the DataFrame contains the expected columns
    expected_columns = ['DTM', 'CHM', 'NDVI', 'LAI', 'Botrytis_Risk']
    assert all(col in data.columns for col in expected_columns)

    # Check that the DataFrame has non-zero entries
    assert len(data) > 0

# If pytest is installed, you can run tests from the command line:
# pytest tests/test_generate_data.py
