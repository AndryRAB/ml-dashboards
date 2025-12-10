"""
ML Dashboards - Interactive visualizations for Linear and Logistic Regression
Dashboard visualizations for www.ispm-edu.com
"""

import streamlit as st

# Page configuration
st.set_page_config(
    page_title="ML Dashboards",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main page
st.title("📊 Machine Learning Dashboards")
st.markdown("""
This interactive dashboard provides visualizations for understanding machine learning regression algorithms.

**Available Visualizations:**
- **Linear Regression**: Visualize how linear regression fits data and explore the impact of different parameters
- **Logistic Regression**: Understand classification boundaries and model performance

Select a visualization from the sidebar to get started!
""")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select a Visualization:",
    ["Home", "Linear Regression", "Logistic Regression"]
)

if page == "Home":
    st.header("Welcome to ML Dashboards")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Linear Regression")
        st.markdown("""
        Explore linear regression with:
        - Interactive data generation
        - Real-time parameter adjustment
        - Visual regression line fitting
        - Performance metrics (R², MSE, RMSE)
        """)
    
    with col2:
        st.subheader("📉 Logistic Regression")
        st.markdown("""
        Understand logistic regression through:
        - Binary classification visualization
        - Decision boundary display
        - Interactive parameter tuning
        - Confusion matrix and accuracy metrics
        """)
    
    st.info("👈 Use the sidebar to navigate to a specific visualization!")

elif page == "Linear Regression":
    from pages import linear_regression
    linear_regression.show()

elif page == "Logistic Regression":
    from pages import logistic_regression
    logistic_regression.show()
