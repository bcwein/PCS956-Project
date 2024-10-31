import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

def calculate_residuals(X, y):
    """
    Fits a linear regressor between X and y and returns the residuals.
    """
    reg = LinearRegression()
    reg.fit(X, y)
    y_pred = reg.predict(X)
    residuals = y - y_pred
    return residuals

def test_residual_independence(X, y, conditioning_var):
    """
    Tests the independence of residuals from a conditioning variable by:
    - Calculating residuals from the linear regression
    - Checking correlation of residuals with the conditioning variable
    - Visualizing the residuals
    
    Parameters:
        X (pd.DataFrame): Features used for regression.
        y (pd.Series): Target variable.
        conditioning_var (pd.Series): Variable we are testing residuals against.
    
    Returns:
        result (dict): Correlation coefficient and p-value of the independence test.
    """
    # Calculate residuals
    residuals = calculate_residuals(X, y)
    
    # Test correlation of residuals with the conditioning variable
    corr_coef, p_value = pearsonr(conditioning_var, residuals)
    
    # Plot residuals vs conditioning variable
    plt.figure(figsize=(8, 6))
    plt.scatter(conditioning_var, residuals, alpha=0.5)
    plt.title(f'Residuals vs {conditioning_var.name} (r={corr_coef:.2f}, p={p_value:.3f})')
    plt.xlabel(conditioning_var.name)
    plt.ylabel('Residuals')
    plt.axhline(0, color='red', linestyle='--')
    plt.show()
    
    # Return correlation coefficient and p-value
    result = {'correlation_coefficient': corr_coef, 'p_value': p_value}
    return result

def run_dag_validation_tests(data):
    """
    Runs conditional independence tests on the data according to DAG assumptions.
    Returns a dictionary with test results for each condition.
    """
    results = {}

    # 1. Test DTM and Botrytis Risk conditional on CHM and LAI
    X = data[['CHM', 'LAI']]
    y = data['Botrytis_Risk']
    conditioning_var = data['DTM']
    results['DTM_Botrytis_ConditionedOn_CHM_LAI'] = test_residual_independence(X, y, conditioning_var)

    # 2. Test CHM and NDVI dependence
    X = data[['CHM']]
    y = data['NDVI']
    conditioning_var = data['CHM']
    results['CHM_NDVI_DirectDependence'] = test_residual_independence(X, y, conditioning_var)

    # Additional tests as per the DAG structure
    return results
