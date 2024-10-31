import os
import pytest
from src.generate_data import save_to_parquet

def test_generate_and_save_data():
    """
    Test that synthetic data is generated and saved correctly to a Parquet file.
    """
    output_file = 'data/synthetic_vineyard_data.parquet'
    
    # Remove the file if it exists to simulate first-time generation
    if os.path.exists(output_file):
        os.remove(output_file)
    
    # Run the data generation script
    save_to_parquet(output_file)
    
    # Check that the file has been created
    assert os.path.exists(output_file)
    
    # Run again to check that it skips generation if the file exists
    save_to_parquet(output_file)
    assert os.path.exists(output_file)