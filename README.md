# Export Analysis and Fleet Estimation Project

## Overview
This project is a Streamlit-based web application designed to analyze export data, predict future demands, and estimate the optimal fleet size for a logistics operation. It uses machine learning techniques to forecast export volumes and provides tools for cost optimization in fleet management.

## Features
- Data loading and preprocessing from Excel or CSV files
- Time series analysis and forecasting using NeuralProphet
- Interactive visualizations of data trends and model predictions
- Fleet size estimation based on demand forecasts and cost parameters
- Cost optimization for different reverse logistics scenarios
- Export of results to Excel for further analysis

## Installation
1. Clone this repository
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```
3. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

## Usage
1. Upload your export data (Excel or CSV format)
2. Configure model parameters and cost options
3. Run the analysis to get predictions and fleet estimations
4. Explore the interactive visualizations and results
5. Export the results to Excel for further analysis

## File Structure
- `app.py`: Main Streamlit application
- `excel_based_solution.py`: Handles Excel-based data processing and analysis
- `export_analyzer.py`: Contains the ExportAnalyzer class for time series analysis
- `fleet_estimator.py`: Implements the FleetEstimator class for fleet size optimization
- `utils.py`: Utility functions for simulations and plotting
- `documentation.py`: for the documentation

## Contact
paul.longourpl@gmail.com