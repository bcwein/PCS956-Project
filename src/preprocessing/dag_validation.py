import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from scipy.stats import spearmanr, chi2_contingency
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def calculate_residuals(X, y, is_binary=False):
    """
    Fits a linear or logistic regressor between X and y and returns the residuals.
    """
    if is_binary:
        reg = LogisticRegression(max_iter=1000)
        reg.fit(X, y)
        y_pred_proba = reg.predict_proba(X)[:, 1]  # Probability for positive class
        y_pred = (y_pred_proba > 0.5).astype(int)
    else:
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
    """
    residuals = calculate_residuals(X, y)
    corr_coef, p_value = spearmanr(conditioning_var, residuals)

    plt.figure(figsize=(8, 6))
    plt.scatter(conditioning_var, residuals, alpha=0.5)
    plt.title(f'Residuals vs {conditioning_var.name} (r={corr_coef:.2f}, p={p_value:.3f})')
    plt.xlabel(conditioning_var.name)
    plt.ylabel('Residuals')
    plt.axhline(0, color='red', linestyle='--')
    plt.show()
    
    result = {'correlation_coefficient': corr_coef, 'p_value': p_value}
    return result

def find_all_conditional_independences(dag, max_combinations=300):
    """
    Identifies potential sets of conditional independences based on DAG structure using d-separation logic.
    Caps the number of combinations generated for testing.
    """
    conditional_independences = []
    child_to_parents = {}
    parent_to_children = {}

    for cause, effect in dag:
        if effect not in child_to_parents:
            child_to_parents[effect] = []
        child_to_parents[effect].append(cause)

        if cause not in parent_to_children:
            parent_to_children[cause] = []
        parent_to_children[cause].append(effect)

    count = 0
    for effect, parents in child_to_parents.items():
        if len(parents) > 1:
            for i, parent1 in enumerate(parents):
                for parent2 in parents[i+1:]:
                    conditional_independences.append((parent1, parent2, effect))
                    conditional_independences.append((parent2, parent1, effect))
                    count += 2
                    if count >= max_combinations:
                        return conditional_independences

        for parent in parents:
            if parent in parent_to_children:
                for child in parent_to_children[parent]:
                    if child != effect:
                        conditional_independences.append((parent, child, effect))
                        count += 1
                        if count >= max_combinations:
                            return conditional_independences

    return conditional_independences


def run_conditional_independence_tests(data, y, conditional_independences):
    """
    Runs conditional independence tests on the data based on identified sets of conditional independences.
    """
    results = {}

    for X, Y, Z in conditional_independences:
        is_X_binary = data[X].nunique() == 2 if X != 'Diabetes_binary' else True
        is_Y_binary = data[Y].nunique() == 2 if Y != 'Diabetes_binary' else True

        if X == 'Diabetes_binary':
            X_data = y.dropna().to_frame()
        else:
            X_data = data[[X]].dropna()
        
        if Y == 'Diabetes_binary':
            Y_data = y.dropna().to_frame()
        else:
            Y_data = data[[Y]].dropna()
        
        Z_data = data[[Z]].dropna()

        common_index = X_data.index.intersection(Y_data.index).intersection(Z_data.index)
        X_data = X_data.loc[common_index]
        Y_data = Y_data.loc[common_index]
        Z_data = Z_data.loc[common_index]

        residuals_X = calculate_residuals(Z_data, X_data[X], is_binary=is_X_binary)
        residuals_Y = calculate_residuals(Z_data, Y_data[Y], is_binary=is_Y_binary)

        if is_X_binary or is_Y_binary:
            contingency_table = pd.crosstab(residuals_X, residuals_Y)
            chi2, p_value, _, _ = chi2_contingency(contingency_table)
            test_stat = chi2
        else:
            corr_coef, p_value = spearmanr(residuals_X, residuals_Y)
            test_stat = corr_coef
        
        results[(X, Y, Z)] = {
            'type': 'binary-numeric' if is_X_binary or is_Y_binary else 'numeric-numeric',
            'test_statistic': test_stat,
            'p_value': p_value
        }

    return results

def run_dag_validation_tests(data, y, dag):
    """
    Runs direct and conditional independence tests on the data according to DAG assumptions.
    Returns a dictionary with test results for each condition.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("The 'data' argument must be a pandas DataFrame")

    results = {}

    for cause, effect in tqdm(dag, desc="Testing direct dependencies"):
        if not isinstance(cause, str) or not isinstance(effect, str):
            raise ValueError(f"Cause and effect must be strings. Got {type(cause)} and {type(effect)}")

        if cause not in data.columns and cause != 'Diabetes_binary':
            raise KeyError(f"Column '{cause}' not found in data")
        if effect not in data.columns and effect != 'Diabetes_binary':
            raise KeyError(f"Column '{effect}' not found in data")

        if cause == 'Diabetes_binary':
            cause_data = y.dropna()
        else:
            cause_data = data[cause].dropna()

        if effect == 'Diabetes_binary':
            effect_data = y.dropna()
        else:
            effect_data = data[effect].dropna()

        # Ensure data is 1-dimensional
        cause_data = cause_data.squeeze()
        effect_data = effect_data.squeeze()

        common_index = cause_data.index.intersection(effect_data.index)
        cause_data = cause_data.loc[common_index]
        effect_data = effect_data.loc[common_index]

        # Check if either variable is binary
        is_cause_binary = pd.api.types.is_bool_dtype(cause_data) or pd.api.types.is_categorical_dtype(cause_data) or len(cause_data.unique()) <= 2
        is_effect_binary = pd.api.types.is_bool_dtype(effect_data) or pd.api.types.is_categorical_dtype(effect_data) or len(effect_data.unique()) <= 2

        if is_cause_binary or is_effect_binary:
            # Use chi-squared test for binary data
            contingency_table = pd.crosstab(cause_data, effect_data)
            chi2, p_value, _, _ = chi2_contingency(contingency_table)
            test_stat = chi2
        else:
            # Use Spearman's correlation for continuous data
            corr_coef, p_value = spearmanr(cause_data, effect_data)
            test_stat = corr_coef

        results[(cause, effect)] = {
            'type': 'binary' if is_cause_binary or is_effect_binary else 'numeric-numeric',
            'test_stat': test_stat,  # Ensure the test statistic is recorded
            'p_value': p_value
        }

    conditional_independences = find_all_conditional_independences(dag)
    conditional_results = run_conditional_independence_tests(data, y, conditional_independences)

    return results, conditional_results