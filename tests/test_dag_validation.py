import pytest
import pandas as pd
from src.preprocessing.dag_validation import run_dag_validation_tests, evaluate_residual_independence

@pytest.fixture
def load_data():
    """Load the synthetic vineyard data and drop rows with NaNs."""
    data = pd.read_parquet('data/synthetic_vineyard_data.parquet')
    return data.dropna()


def test_residual_independence(load_data):
    """Test independence of residuals from a conditioning variable."""
    X = load_data[['CHM', 'LAI']]
    y = load_data['Botrytis_Risk']
    conditioning_var = load_data['DTM']
    
    # Run independence test
    result = evaluate_residual_independence(X, y, conditioning_var)
    
    # Check result keys and values
    assert 'correlation_coefficient' in result
    assert 'p_value' in result
    assert isinstance(result['correlation_coefficient'], float)
    assert isinstance(result['p_value'], float)
    assert 0 <= result['p_value'] <= 1


def test_dag_validation(load_data):
    """Test the DAG validation on synthetic vineyard data."""
    results = run_dag_validation_tests(load_data)

    # Ensure each test result contains correlation coefficient and p-value
    for key, result in results.items():
        assert 'correlation_coefficient' in result, f"{key} missing correlation_coefficient."
        assert 'p_value' in result, f"{key} missing p_value."
        assert isinstance(result['correlation_coefficient'], float), "Correlation should be a float."
        assert isinstance(result['p_value'], float), "p_value should be a float."
        assert 0 <= result['p_value'] <= 1, f"{key} p_value should be between 0 and 1."
