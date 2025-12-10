"""Linear Regression Visualization Page"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import seaborn as sns

def show():
    st.header("📈 Linear Regression Visualization")
    st.markdown("""
    Linear regression is a supervised learning algorithm used to predict a continuous target variable 
    based on one or more input features by fitting a linear equation to the observed data.
    """)
    
    # Sidebar controls
    st.sidebar.header("Configuration")
    
    # Data source selection
    data_source = st.sidebar.radio(
        "Data Source:",
        ["Generate Synthetic Data", "Upload CSV"]
    )
    
    if data_source == "Generate Synthetic Data":
        # Parameters for synthetic data
        n_samples = st.sidebar.slider("Number of samples", 20, 500, 100, 10)
        noise_level = st.sidebar.slider("Noise level", 0.0, 50.0, 10.0, 1.0)
        slope = st.sidebar.slider("True slope", -5.0, 5.0, 2.0, 0.1)
        intercept = st.sidebar.slider("True intercept", -50.0, 50.0, 5.0, 1.0)
        
        # Generate synthetic data
        np.random.seed(42)
        X = np.linspace(0, 10, n_samples)
        y = slope * X + intercept + np.random.normal(0, noise_level, n_samples)
        X = X.reshape(-1, 1)
        
        st.sidebar.markdown(f"""
        **True Equation:**  
        y = {slope:.2f}x + {intercept:.2f}
        """)
        
    else:
        uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.sidebar.write("Preview:", df.head())
            
            if len(df.columns) >= 2:
                x_col = st.sidebar.selectbox("Select X column:", df.columns)
                y_col = st.sidebar.selectbox("Select Y column:", df.columns)
                
                X = df[x_col].values.reshape(-1, 1)
                y = df[y_col].values
            else:
                st.error("CSV must have at least 2 columns")
                return
        else:
            st.info("Please upload a CSV file to continue")
            return
    
    # Fit linear regression model
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    
    # Calculate metrics
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y, y_pred)
    
    # Display model equation
    st.subheader("Fitted Model")
    st.markdown(f"""
    **Regression Equation:**  
    y = {model.coef_[0]:.4f}x + {model.intercept_:.4f}
    """)
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("R² Score", f"{r2:.4f}")
    with col2:
        st.metric("MSE", f"{mse:.4f}")
    with col3:
        st.metric("RMSE", f"{rmse:.4f}")
    with col4:
        st.metric("MAE", f"{mae:.4f}")
    
    # Visualization tabs
    tab1, tab2, tab3 = st.tabs(["Scatter Plot with Regression Line", "Residual Plot", "Data Table"])
    
    with tab1:
        st.subheader("Regression Line Fit")
        
        # Create interactive plot with Plotly
        fig = go.Figure()
        
        # Add scatter plot
        fig.add_trace(go.Scatter(
            x=X.flatten(),
            y=y,
            mode='markers',
            name='Actual Data',
            marker=dict(size=8, color='blue', opacity=0.6)
        ))
        
        # Add regression line
        fig.add_trace(go.Scatter(
            x=X.flatten(),
            y=y_pred,
            mode='lines',
            name='Regression Line',
            line=dict(color='red', width=2)
        ))
        
        fig.update_layout(
            xaxis_title='X',
            yaxis_title='Y',
            hovermode='closest',
            width=800,
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Residual Analysis")
        
        # Calculate residuals
        residuals = y - y_pred
        
        # Create residual plot
        fig_residual = go.Figure()
        
        fig_residual.add_trace(go.Scatter(
            x=y_pred,
            y=residuals,
            mode='markers',
            name='Residuals',
            marker=dict(size=8, color='green', opacity=0.6)
        ))
        
        # Add horizontal line at y=0
        fig_residual.add_hline(y=0, line_dash="dash", line_color="red")
        
        fig_residual.update_layout(
            xaxis_title='Predicted Values',
            yaxis_title='Residuals',
            hovermode='closest',
            width=800,
            height=500
        )
        
        st.plotly_chart(fig_residual, use_container_width=True)
        
        st.markdown("""
        **Residual Plot Interpretation:**
        - Points should be randomly scattered around the horizontal line at 0
        - Patterns in residuals indicate the model may not be capturing all relationships
        - Increasing/decreasing spread suggests heteroscedasticity
        """)
    
    with tab3:
        st.subheader("Data and Predictions")
        
        # Create DataFrame
        results_df = pd.DataFrame({
            'X': X.flatten(),
            'Actual Y': y,
            'Predicted Y': y_pred,
            'Residual': residuals
        })
        
        st.dataframe(results_df, use_container_width=True)
        
        # Download button
        csv = results_df.to_csv(index=False)
        st.download_button(
            label="Download Data as CSV",
            data=csv,
            file_name="linear_regression_results.csv",
            mime="text/csv"
        )
    
    # Information section
    with st.expander("ℹ️ About Linear Regression"):
        st.markdown("""
        ### What is Linear Regression?
        
        Linear regression models the relationship between a dependent variable and one or more 
        independent variables by fitting a linear equation to observed data.
        
        ### Key Metrics:
        
        - **R² (Coefficient of Determination)**: Proportion of variance in the dependent variable 
          that is predictable from the independent variable(s). Ranges from 0 to 1, where 1 indicates 
          perfect prediction.
        
        - **MSE (Mean Squared Error)**: Average of the squared differences between predicted and 
          actual values. Lower is better.
        
        - **RMSE (Root Mean Squared Error)**: Square root of MSE, in the same units as the target 
          variable. Lower is better.
        
        - **MAE (Mean Absolute Error)**: Average of absolute differences between predicted and 
          actual values. More robust to outliers than MSE.
        
        ### Assumptions:
        
        1. **Linearity**: The relationship between X and Y is linear
        2. **Independence**: Observations are independent of each other
        3. **Homoscedasticity**: Constant variance of residuals
        4. **Normality**: Residuals are normally distributed
        """)
