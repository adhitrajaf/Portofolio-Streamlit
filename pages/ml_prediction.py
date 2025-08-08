import streamlit as st
import pandas as pd
import numpy as np
import io
from models.house_price_predictor import HousePricePredictor
from models.churn_predictor import ChurnPredictor

def show():
    """Display the ML prediction page"""
    st.markdown('<h1 class="main-header">Machine Learning Prediction Interface</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    Upload your data and get predictions from my trained machine learning models! 
    This interface supports both **House Price Prediction** and **Customer Churn Analysis**.
    """)
    
    # Model selection for prediction
    prediction_model = st.selectbox(
        "Choose a prediction model:",
        ["🏠 House Price Prediction", "📞 Customer Churn Prediction"]
    )
    
    if prediction_model == "🏠 House Price Prediction":
        house_predictor = HousePricePredictor()
        house_predictor.show_prediction_interface()
    
    elif prediction_model == "📞 Customer Churn Prediction":
        churn_predictor = ChurnPredictor()
        churn_predictor.show_prediction_interface()