import streamlit as st
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from models.house_price_viz import HousePriceVisualization
from models.churn_viz import ChurnVisualization

def show():
    """Display the model visualization page"""
    st.markdown('<h1 class="main-header">Model Visualization & Performance</h1>', unsafe_allow_html=True)
    
    # Model selection
    model_choice = st.selectbox(
        "Select a model to visualize:",
        ["House Price Prediction", "Customer Churn Prediction"]
    )
    
    if model_choice == "House Price Prediction":
        house_viz = HousePriceVisualization()
        house_viz.show_visualizations()
    
    elif model_choice == "Customer Churn Prediction":
        churn_viz = ChurnVisualization()
        churn_viz.show_visualizations()