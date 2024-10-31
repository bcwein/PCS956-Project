import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

def prepare_data(data, risk_threshold=0.5):
    """
    Prepares data by separating NaN values, creating a binary target variable, and
    returning both the full dataset (with NaNs) and filtered data for model training.
    """
    # Track indices with NaN values to preserve grid structure
    nan_mask = data[['DTM', 'CHM', 'NDVI', 'LAI', 'Botrytis_Risk']].isna().any(axis=1)

    # Create binary target based on risk threshold
    data['Botrytis_Risk_Class'] = (data['Botrytis_Risk'] > risk_threshold).astype(int)
    
    # Filter out NaN values for training data
    filtered_data = data.dropna(subset=['DTM', 'CHM', 'NDVI', 'LAI', 'Botrytis_Risk'])
    
    # Define features and target variable for filtered data
    X = filtered_data[['DTM', 'CHM', 'NDVI', 'LAI']]
    y = filtered_data['Botrytis_Risk_Class']
    
    return X, y, nan_mask, data  # Return full data for reshaping later


def train_random_forest(X, y, max_depth=None, test_size=0.2, random_state=42):
    """
    Trains a Random Forest classifier, returning evaluation metrics, feature importances, 
    and the trained model.
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Initialize and train model with max_depth parameter
    rf_clf = RandomForestClassifier(random_state=random_state, max_depth=max_depth)
    rf_clf.fit(X_train, y_train)

    # Predictions and evaluation metrics
    y_pred = rf_clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    metrics = {
        'accuracy': accuracy,
        'classification_report': classification_report(y_test, y_pred, output_dict=True)
    }

    # Feature importances
    feature_importances = rf_clf.feature_importances_
    feature_importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': feature_importances
    }).sort_values(by='importance', ascending=False)

    return metrics, feature_importance_df, rf_clf