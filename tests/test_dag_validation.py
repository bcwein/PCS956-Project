import pytest
import pandas as pd
from src.preprocessing.dag_validation import run_dag_validation_tests

def test_dag_validation():
    """
    Test the DAG validation logic with synthetic data.
    """
    # Load synthetic data
    data = pd.read_parquet('data/synthetic_vineyard_data.parquet')
    
    # Run DAG validation tests
    results = run_dag_validation_tests(data)
    
    # Assert each result has correlation and p_value
    for key, result in results.items():
        assert 'correlation_coefficient' in result, f"{key} missing correlation_coefficient"
        assert 'p_value' in result, f"{key} missing p_value"
        assert isinstance(result['correlation_coefficient'], float), "Correlation should be a float"
        assert isinstance(result['p_value'], float), "p_value should be a float"
