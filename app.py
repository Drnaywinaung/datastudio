# app.py
# This script runs a Flask web server to provide a UI for the data analysis.
# Updated to be compatible with the scikit-learn based regression in analysis.py.

from flask import Flask, render_template, jsonify, request
import pandas as pd
import io

from analysis import (
    perform_summary_statistics, 
    create_frequency_table, 
    calculate_grouped_frequency,
    calculate_histogram, 
    perform_correlation_analysis,
    perform_chi_square_test,
    perform_reliability_analysis,
    perform_regression_analysis # Using the scikit-learn version
)

app = Flask(__name__)

def read_file_to_dataframe(file):
    """Reads an uploaded file (CSV or Excel) into a pandas DataFrame."""
    if file.filename.endswith('.csv'):
        return pd.read_csv(io.StringIO(file.stream.read().decode("UTF8")))
    elif file.filename.endswith(('.xls', '.xlsx')):
        return pd.read_excel(file.stream)
    return None

@app.route('/')
def index():
    """Serves the main HTML page."""
    return render_template('index.html')

@app.route('/api/get-columns', methods=['POST'])
def get_columns():
    """Reads an uploaded file and returns its column names."""
    try:
        file = request.files.get('file')
        if not file: return jsonify({'status': 'error', 'message': 'No file uploaded.'}), 400
        df = read_file_to_dataframe(file)
        if df is None: return jsonify({'status': 'error', 'message': 'Unsupported file type. Please upload a CSV or Excel file.'}), 400

        numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        return jsonify({'status': 'success', 'numerical': numerical_cols, 'categorical': categorical_cols})
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'An error occurred: {str(e)}'}), 500

@app.route('/api/analyze', methods=['POST'])
def analyze_data():
    """Runs the selected analysis and returns the results."""
    try:
        file = request.files.get('file')
        analysis_type = request.form.get('analysis_type')
        chart_type = request.form.get('chart_type', 'table')

        if not (file and analysis_type):
            return jsonify({'status': 'error', 'message': 'Missing file or analysis type.'}), 400

        df = read_file_to_dataframe(file)
        if df is None: return jsonify({'status': 'error', 'message': 'Unsupported file type.'}), 400
        
        analysis_result = None
        visualization_type = chart_type
        chart_data = None
        title = "Analysis Result"

        # --- Analysis Logic ---
        if analysis_type == 'descriptive_statistics':
            analysis_result = perform_summary_statistics(df)
            title, visualization_type = 'Descriptive Statistics', 'table'

        elif analysis_type == 'frequency_distribution':
            column = request.form.get('column')
            if not column: return jsonify({'status': 'error', 'message': 'Column not specified.'}), 400
            analysis_result = create_frequency_table(df, column)
            title = f'Frequency Distribution for {column}'
            chart_data = analysis_result[[column, 'Frequency']].to_json(orient='values')

        elif analysis_type == 'grouped_aggregation':
            group_col, agg_col = request.form.get('group_by_col'), request.form.get('agg_col')
            agg_type = request.form.get('agg_type', 'mean')
            if not (group_col and agg_col): return jsonify({'status': 'error', 'message': 'Columns not specified.'}), 400
            analysis_result = calculate_grouped_frequency(df, group_col, agg_col, agg_type)
            title = f'{agg_type.capitalize()} of {agg_col} by {group_col}'
            chart_data = analysis_result.to_json(orient='values')
        
        elif analysis_type == 'histogram':
            column = request.form.get('column')
            bins = int(request.form.get('bins', 10))
            if not column: return jsonify({'status': 'error', 'message': 'Column not specified.'}), 400
            analysis_result = calculate_histogram(df, column, bins)
            title, visualization_type = f'Histogram for {column}', 'bar'
            chart_data = analysis_result.to_json(orient='values')

        elif analysis_type == 'correlation_matrix':
            analysis_result = perform_correlation_analysis(df)
            title, visualization_type = 'Correlation Matrix', 'table'
            
        elif analysis_type == 'chi_square':
            col1, col2 = request.form.get('col1'), request.form.get('col2')
            if not (col1 and col2): return jsonify({'status': 'error', 'message': 'Two columns must be selected.'}), 400
            analysis_result = perform_chi_square_test(df, col1, col2)
            title, visualization_type = f'Chi-Square Test: {col1} vs {col2}', 'table'

        elif analysis_type == 'reliability':
            cols = request.form.getlist('columns[]')
            if not cols or len(cols) < 2: return jsonify({'status': 'error', 'message': 'At least two columns must be selected.'}), 400
            analysis_result = perform_reliability_analysis(df, cols)
            title, visualization_type = 'Reliability Analysis (Cronbach\'s Alpha)', 'table'
            
        elif analysis_type == 'regression':
            dep_var = request.form.get('dependent_var')
            ind_vars = request.form.getlist('independent_vars[]') 
            if not (dep_var and ind_vars): return jsonify({'status': 'error', 'message': 'Dependent/Independent variables not specified.'}), 400
            analysis_result = perform_regression_analysis(df, dep_var, ind_vars)
            title, visualization_type = f'Regression Coefficients: {dep_var} ~ {", ".join(ind_vars)}', 'table'

        else:
            return jsonify({'status': 'error', 'message': 'Invalid analysis type.'}), 400

        if analysis_result is None or analysis_result.empty:
            return jsonify({'status': 'error', 'message': 'Analysis could not be completed. Please check your column selections.'}), 500

        response = {
            'status': 'success', 'title': title, 'visualization': visualization_type,
            'data': analysis_result.to_json(orient='split')
        }
        if chart_data:
            response['chart_data'] = chart_data

        return jsonify(response)

    except Exception as e:
        return jsonify({'status': 'error', 'message': f'An unexpected error occurred: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)
