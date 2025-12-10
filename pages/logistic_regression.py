"""Logistic Regression Visualization Page"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.model_selection import train_test_split
import seaborn as sns

def show():
    st.header("📉 Logistic Regression Visualization")
    st.markdown("""
    Logistic regression is a supervised learning algorithm used for binary classification. 
    It models the probability that an instance belongs to a particular class.
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
        n_samples = st.sidebar.slider("Number of samples", 50, 1000, 300, 50)
        n_features = 2  # Fixed at 2 for 2D visualization
        class_separation = st.sidebar.slider("Class separation", 0.5, 3.0, 1.5, 0.1)
        
        # Generate synthetic data
        np.random.seed(42)
        
        # Generate two classes with different means
        n_per_class = n_samples // 2
        
        # Class 0
        X_class0 = np.random.randn(n_per_class, n_features) - class_separation
        y_class0 = np.zeros(n_per_class)
        
        # Class 1
        X_class1 = np.random.randn(n_per_class, n_features) + class_separation
        y_class1 = np.ones(n_per_class)
        
        # Combine
        X = np.vstack([X_class0, X_class1])
        y = np.concatenate([y_class0, y_class1])
        
        # Shuffle
        indices = np.random.permutation(len(y))
        X = X[indices]
        y = y[indices]
        
    else:
        uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.sidebar.write("Preview:", df.head())
            
            if len(df.columns) >= 3:
                x1_col = st.sidebar.selectbox("Select X1 column:", df.columns)
                x2_col = st.sidebar.selectbox("Select X2 column:", df.columns)
                y_col = st.sidebar.selectbox("Select Y (target) column:", df.columns)
                
                X = df[[x1_col, x2_col]].values
                y = df[y_col].values
            else:
                st.error("CSV must have at least 3 columns (2 features + 1 target)")
                return
        else:
            st.info("Please upload a CSV file to continue")
            return
    
    # Model parameters
    st.sidebar.subheader("Model Parameters")
    C = st.sidebar.slider("Regularization (C)", 0.01, 10.0, 1.0, 0.1,
                          help="Inverse of regularization strength. Smaller values specify stronger regularization.")
    max_iter = st.sidebar.slider("Max iterations", 100, 1000, 200, 100)
    
    # Split data
    test_size = st.sidebar.slider("Test set size", 0.1, 0.5, 0.2, 0.05)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    # Fit logistic regression model
    model = LogisticRegression(C=C, max_iter=max_iter, random_state=42)
    model.fit(X_train, y_train)
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    # Display model coefficients
    st.subheader("Fitted Model")
    st.markdown(f"""
    **Decision Boundary Equation:**  
    {model.coef_[0][0]:.4f} × X₁ + {model.coef_[0][1]:.4f} × X₂ + {model.intercept_[0]:.4f} = 0
    """)
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Training Accuracy", f"{train_accuracy:.4f}")
    with col2:
        st.metric("Test Accuracy", f"{test_accuracy:.4f}")
    with col3:
        st.metric("Training Samples", f"{len(X_train)}")
    
    # Visualization tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Decision Boundary", "Confusion Matrix", "ROC Curve", "Data Table"])
    
    with tab1:
        st.subheader("Decision Boundary Visualization")
        
        # Create mesh for decision boundary
        h = 0.02  # step size in the mesh
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        
        # Predict on mesh
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        # Create Plotly figure
        fig = go.Figure()
        
        # Add decision boundary
        fig.add_trace(go.Contour(
            x=xx[0],
            y=yy[:, 0],
            z=Z,
            colorscale=[[0, 'lightblue'], [1, 'lightcoral']],
            opacity=0.3,
            showscale=False,
            hoverinfo='skip'
        ))
        
        # Add training data
        for class_val in [0, 1]:
            mask = y_train == class_val
            fig.add_trace(go.Scatter(
                x=X_train[mask, 0],
                y=X_train[mask, 1],
                mode='markers',
                name=f'Train Class {int(class_val)}',
                marker=dict(
                    size=8,
                    color='blue' if class_val == 0 else 'red',
                    symbol='circle',
                    opacity=0.6
                )
            ))
        
        # Add test data
        for class_val in [0, 1]:
            mask = y_test == class_val
            fig.add_trace(go.Scatter(
                x=X_test[mask, 0],
                y=X_test[mask, 1],
                mode='markers',
                name=f'Test Class {int(class_val)}',
                marker=dict(
                    size=10,
                    color='darkblue' if class_val == 0 else 'darkred',
                    symbol='x',
                    opacity=0.8
                )
            ))
        
        fig.update_layout(
            xaxis_title='Feature 1 (X₁)',
            yaxis_title='Feature 2 (X₂)',
            hovermode='closest',
            width=800,
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        **Visualization Guide:**
        - **Circles**: Training data
        - **X markers**: Test data
        - **Shaded regions**: Predicted class regions
        - The boundary line separates the two classes
        """)
    
    with tab2:
        st.subheader("Confusion Matrix")
        
        # Calculate confusion matrices
        cm_train = confusion_matrix(y_train, y_train_pred)
        cm_test = confusion_matrix(y_test, y_test_pred)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Training Set**")
            fig_cm_train, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(cm_train, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=['Class 0', 'Class 1'],
                       yticklabels=['Class 0', 'Class 1'])
            ax.set_ylabel('Actual')
            ax.set_xlabel('Predicted')
            ax.set_title('Training Set Confusion Matrix')
            st.pyplot(fig_cm_train)
            plt.close()
        
        with col2:
            st.markdown("**Test Set**")
            fig_cm_test, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(cm_test, annot=True, fmt='d', cmap='Greens', ax=ax,
                       xticklabels=['Class 0', 'Class 1'],
                       yticklabels=['Class 0', 'Class 1'])
            ax.set_ylabel('Actual')
            ax.set_xlabel('Predicted')
            ax.set_title('Test Set Confusion Matrix')
            st.pyplot(fig_cm_test)
            plt.close(fig_cm_test)
        
        # Classification report
        st.subheader("Classification Report (Test Set)")
        report = classification_report(y_test, y_test_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.style.format("{:.4f}"), use_container_width=True)
    
    with tab3:
        st.subheader("ROC Curve")
        
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(y_test, y_test_proba)
        roc_auc = auc(fpr, tpr)
        
        # Create ROC curve plot
        fig_roc = go.Figure()
        
        fig_roc.add_trace(go.Scatter(
            x=fpr,
            y=tpr,
            mode='lines',
            name=f'ROC Curve (AUC = {roc_auc:.4f})',
            line=dict(color='darkorange', width=2)
        ))
        
        fig_roc.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(color='navy', width=2, dash='dash')
        ))
        
        fig_roc.update_layout(
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            title='Receiver Operating Characteristic (ROC) Curve',
            hovermode='closest',
            width=700,
            height=600
        )
        
        st.plotly_chart(fig_roc, use_container_width=True)
        
        st.metric("ROC AUC Score", f"{roc_auc:.4f}")
        
        st.markdown("""
        **ROC Curve Interpretation:**
        - **AUC = 1.0**: Perfect classifier
        - **AUC = 0.5**: Random classifier (no better than chance)
        - **AUC > 0.8**: Generally considered good performance
        - The closer the curve follows the top-left corner, the better the model
        """)
    
    with tab4:
        st.subheader("Predictions Data")
        
        # Create DataFrame for test set
        results_df = pd.DataFrame({
            'Feature_1': X_test[:, 0],
            'Feature_2': X_test[:, 1],
            'Actual_Class': y_test.astype(int),
            'Predicted_Class': y_test_pred.astype(int),
            'Probability_Class_1': y_test_proba,
            'Correct': (y_test == y_test_pred)
        })
        
        st.dataframe(results_df, use_container_width=True)
        
        # Download button
        csv = results_df.to_csv(index=False)
        st.download_button(
            label="Download Predictions as CSV",
            data=csv,
            file_name="logistic_regression_predictions.csv",
            mime="text/csv"
        )
    
    # Information section
    with st.expander("ℹ️ About Logistic Regression"):
        st.markdown("""
        ### What is Logistic Regression?
        
        Despite its name, logistic regression is a classification algorithm. It models the probability 
        that an instance belongs to a particular class using the logistic (sigmoid) function.
        
        ### Key Concepts:
        
        - **Decision Boundary**: The line (or hyperplane) that separates different classes
        - **Probability**: The model outputs probabilities between 0 and 1
        - **Threshold**: Typically 0.5 is used to convert probabilities to class predictions
        
        ### Key Metrics:
        
        - **Accuracy**: Proportion of correct predictions
        - **Precision**: Of predicted positives, how many are actually positive
        - **Recall (Sensitivity)**: Of actual positives, how many are correctly predicted
        - **F1-Score**: Harmonic mean of precision and recall
        - **AUC-ROC**: Area under the ROC curve, measures model's ability to distinguish classes
        
        ### Confusion Matrix:
        
        - **True Positives (TP)**: Correctly predicted positive cases
        - **True Negatives (TN)**: Correctly predicted negative cases
        - **False Positives (FP)**: Incorrectly predicted as positive (Type I error)
        - **False Negatives (FN)**: Incorrectly predicted as negative (Type II error)
        
        ### Parameters:
        
        - **C (Regularization)**: Controls the trade-off between fitting the training data 
          and keeping the model simple. Smaller values mean stronger regularization.
        """)
