# analysis.py
# This module contains functions for performing data analysis.
# Updated to use scikit-learn for regression to avoid environment issues.

import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
from sklearn.linear_model import LinearRegression # Using scikit-learn for regression

def perform_summary_statistics(df):
    """
    Calculates summary statistics for the numerical columns in the DataFrame.
    """
    numerical_cols = df.select_dtypes(include=['number']).columns
    if not numerical_cols.empty:
        stats = df[numerical_cols].describe()
        stats = stats.reset_index()
        stats = stats.rename(columns={'index': 'Statistic'})
        return stats
    return pd.DataFrame()

def create_frequency_table(df, column_name):
    """
    Calculates a detailed frequency distribution table for a specified column.
    """
    if column_name not in df.columns or not pd.api.types.is_categorical_dtype(df[column_name]) and not pd.api.types.is_object_dtype(df[column_name]):
        return None

    frequency_counts = df[column_name].value_counts()
    frequency_table = frequency_counts.reset_index()
    frequency_table.columns = [column_name, 'Frequency']
    total_count = len(df)
    frequency_table['Percentage (%)'] = (frequency_table['Frequency'] / total_count * 100).round(2)
    return frequency_table

def calculate_grouped_frequency(df, group_by_col, agg_col, agg_type='mean'):
    """
    Calculates an aggregation (mean, sum, count) of a column, grouped by another.
    """
    if group_by_col in df.columns and agg_col in df.columns:
        if pd.api.types.is_numeric_dtype(df[agg_col]) or agg_type == 'count':
            try:
                result = df.groupby(group_by_col)[agg_col].agg(agg_type)
                return result.reset_index()
            except Exception:
                return None
    return None

def calculate_histogram(df, column_name, bins=10):
    """
    Calculates the data needed for a histogram for a numerical column.
    """
    if column_name not in df.columns or not pd.api.types.is_numeric_dtype(df[column_name]):
        return None
    
    counts, bin_edges = np.histogram(df[column_name].dropna(), bins=bins)
    
    hist_df = pd.DataFrame({
        'Bin': [f'{edge:.2f}-{bin_edges[i+1]:.2f}' for i, edge in enumerate(bin_edges[:-1])],
        'Frequency': counts
    })
    return hist_df

def perform_correlation_analysis(df):
    """
    Calculates the correlation matrix for the numerical columns.
    """
    numerical_cols = df.select_dtypes(include=['number'])
    if len(numerical_cols.columns) > 1:
        corr_matrix = numerical_cols.corr()
        corr_matrix = corr_matrix.reset_index()
        return corr_matrix.rename(columns={'index': 'Variable'})
    return pd.DataFrame()

def perform_chi_square_test(df, col1, col2):
    """
    Performs a Chi-Square test of independence between two categorical columns.
    """
    if col1 not in df.columns or col2 not in df.columns:
        return None
        
    contingency_table = pd.crosstab(df[col1], df[col2])
    
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    
    results = {
        'Chi-Square Statistic': [chi2],
        'P-value': [p],
        'Degrees of Freedom': [dof]
    }
    return pd.DataFrame(results)

def perform_reliability_analysis(df, columns):
    """
    Calculates Cronbach's alpha for a set of columns to test reliability.
    """
    if not all(col in df.columns for col in columns):
        return None
    
    sub_df = df[columns].dropna()
    
    if sub_df.shape[0] < 2 or sub_df.shape[1] < 2:
        return None

    k = sub_df.shape[1]
    sum_item_variances = sub_df.var(axis=0, ddof=1).sum()
    total_score_variance = sub_df.sum(axis=1).var(ddof=1)
    
    if total_score_variance == 0:
        return pd.DataFrame({
            'Cronbach\'s Alpha': [1.0 if sum_item_variances == 0 else 0.0],
            'Number of Items': [k],
            'Number of Samples': [sub_df.shape[0]]
        })

    alpha = (k / (k - 1)) * (1 - (sum_item_variances / total_score_variance))
    
    results = {
        'Cronbach\'s Alpha': [alpha],
        'Number of Items': [k],
        'Number of Samples': [sub_df.shape[0]]
    }
    return pd.DataFrame(results)

# --- Rewritten Regression Function using scikit-learn ---
def perform_regression_analysis(df, dependent_var, independent_vars):
    """
    Performs a multiple linear regression using scikit-learn
    and returns a table of coefficients.
    """
    if dependent_var not in df.columns or not all(col in df.columns for col in independent_vars):
        return None
    
    # Prepare the data, dropping rows with missing values for the model
    model_df = df[[dependent_var] + independent_vars].dropna()
    
    if model_df.empty:
        return None

    X = model_df[independent_vars]
    y = model_df[dependent_var]
    
    # Create and fit the model
    model = LinearRegression()
    model.fit(X, y)
    
    # Create the results table
    coefficients = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
    intercept = pd.DataFrame({'Coefficient': model.intercept_}, index=['Intercept'])
    
    # Combine intercept and coefficients into a single DataFrame
    results_df = pd.concat([intercept, coefficients]).reset_index()
    results_df = results_df.rename(columns={'index': 'Variable'})
    
    return results_df
