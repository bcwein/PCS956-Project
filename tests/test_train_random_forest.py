import pytest
import pandas as pd
import os
from src.models.train_random_forest import prepare_data, train_random_forest
from sklearn.ensemble import RandomForestClassifier

# Path to synthetic data file
data_path = 'data/synthetic_vineyard_data.parquet'

@pytest.fixture
def load_data():
    """
    Fixture to load the synthetic vineyard data.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"{data_path} not found. Please generate the data first.")
    return pd.read_parquet(data_path)

def test_prepare_data(load_data):
    """
    Test if data preparation works as expected, including handling NaNs and extracting features and targets.
    """
    X, y, nan_mask, full_data = prepare_data(load_data)
    
    # Check filtered features and target output
    assert isinstance(X, pd.DataFrame), "X should be a DataFrame"
    assert isinstance(y, pd.Series), "y should be a Series"
    assert set(X.columns) == {'DTM', 'CHM', 'NDVI', 'LAI'}, "Unexpected columns in features"
    assert y.name == 'Botrytis_Risk_Class', "Target should be Botrytis_Risk_Class"
    
    # Verify nan_mask and full_data
    assert isinstance(nan_mask, pd.Series), "nan_mask should be a Series"
    assert nan_mask.shape[0] == full_data.shape[0], "nan_mask should match the full data length"
    assert nan_mask.isna().sum() == 0, "nan_mask should contain no NaNs"

    # Ensure that `full_data` contains `NaN` values in the same indices as `nan_mask`
    assert full_data.loc[nan_mask].isna().any(axis=1).all(), "full_data should retain NaNs at the correct indices"

def test_train_random_forest(load_data):
    """
    Test if Random Forest model trains and returns expected outputs, including model accuracy and metrics.
    """
    X, y, _, _ = prepare_data(load_data)  # Use filtered data for training
    metrics, feature_importance, model = train_random_forest(X, y, max_depth=5)  # Test with max_depth set to 5
    
    # Check metrics output
    assert 'accuracy' in metrics, "Metrics should include accuracy"
    assert isinstance(metrics['accuracy'], float), "Accuracy should be a float"
    assert metrics['accuracy'] > 0.5, "Baseline accuracy should be reasonable"

    # Check classification report
    assert 'classification_report' in metrics, "Metrics should include classification report"
    assert isinstance(metrics['classification_report'], dict), "Classification report should be a dictionary"
    
    # Check feature importance output
    assert isinstance(feature_importance, pd.DataFrame), "Feature importance should be a DataFrame"
    assert not feature_importance.empty, "Feature importance should not be empty"
    assert set(feature_importance.columns) == {'feature', 'importance'}, "Feature importance should have correct columns"
    
    # Check model type
    assert isinstance(model, RandomForestClassifier), "Model should be an instance of RandomForestClassifier"

def test_train_random_forest_no_max_depth(load_data):
    """
    Test if Random Forest model trains correctly without specifying max_depth.
    """
    X, y, _, _ = prepare_data(load_data)  # Use filtered X and y for training
    metrics, feature_importance, model = train_random_forest(X, y)  # Default max_depth=None
    
    # Check metrics output
    assert 'accuracy' in metrics, "Metrics should include accuracy"
    assert isinstance(metrics['accuracy'], float), "Accuracy should be a float"
    assert metrics['accuracy'] > 0.5, "Baseline accuracy should be reasonable without max_depth constraint"

    # Check classification report
    assert 'classification_report' in metrics, "Metrics should include classification report"
    assert isinstance(metrics['classification_report'], dict), "Classification report should be a dictionary"
    
    # Check feature importance output
    assert isinstance(feature_importance, pd.DataFrame), "Feature importance should be a DataFrame"
    assert not feature_importance.empty, "Feature importance should not be empty"
    assert set(feature_importance.columns) == {'feature', 'importance'}, "Feature importance should have correct columns"
    
    # Check model type
    assert isinstance(model, RandomForestClassifier), "Model should be an instance of RandomForestClassifier"
