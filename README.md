# 📊 ML Dashboards

Interactive dashboard visualizations for machine learning algorithms.  
Dashboard visualizations for www.ispm-edu.com

## Features

This Streamlit application provides interactive visualizations for understanding:

- **Linear Regression**: Explore how linear regression fits data with real-time parameter adjustments
- **Logistic Regression**: Understand classification boundaries and model performance metrics

## Installation

1. Clone the repository:
```bash
git clone https://github.com/AndryRAB/ml-dashboards.git
cd ml-dashboards
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the Streamlit application:

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

## Features Detail

### Linear Regression
- Generate synthetic data or upload your own CSV
- Adjust parameters like noise level, slope, and intercept
- View regression line fit with interactive plots
- Analyze residuals to check model assumptions
- Export results as CSV

**Metrics:**
- R² Score (Coefficient of Determination)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)

### Logistic Regression
- Generate binary classification data or upload CSV
- Visualize decision boundaries
- Interactive parameter tuning (regularization strength)
- Confusion matrix for train and test sets
- ROC curve with AUC score
- Export predictions as CSV

**Metrics:**
- Accuracy
- Precision, Recall, F1-Score
- Confusion Matrix
- ROC-AUC Score

## Requirements

- Python 3.8+
- streamlit
- numpy
- pandas
- scikit-learn
- matplotlib
- plotly
- seaborn

See `requirements.txt` for specific versions.

## Project Structure

```
ml-dashboards/
├── app.py                      # Main Streamlit application
├── pages/
│   ├── __init__.py
│   ├── linear_regression.py    # Linear regression visualization
│   └── logistic_regression.py  # Logistic regression visualization
├── requirements.txt            # Python dependencies
├── .gitignore
└── README.md
```

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
