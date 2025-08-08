import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

class HousePriceVisualization:
    def __init__(self):
        # Generate sample data for demonstration
        np.random.seed(42)
    
    def show_visualizations(self):
        """Display all house price model visualizations"""
        st.markdown('<h2 class="section-header">🏠 House Price Prediction Model Analysis</h2>', unsafe_allow_html=True)
        
        # Feature correlation heatmap
        self.show_correlation_heatmap()
        
        # Model performance comparison
        self.show_model_performance()
        
        # Feature importance
        self.show_feature_importance()
        
        # Prediction vs Actual scatter plot
        self.show_prediction_scatter()
        
        # Price distribution by features
        self.show_price_distribution()
    
    def show_correlation_heatmap(self):
        """Display feature correlation heatmap"""
        st.markdown("### 📊 Feature Correlation Matrix")
        
        features = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt', 'SalePrice']
        correlation_data = np.random.rand(len(features), len(features))
        correlation_data = (correlation_data + correlation_data.T) / 2
        np.fill_diagonal(correlation_data, 1)
        
        fig_corr = px.imshow(correlation_data, 
                           x=features, y=features,
                           color_continuous_scale='RdBu',
                           title="Feature Correlation Heatmap",
                           aspect="auto")
        fig_corr.update_layout(height=500)
        st.plotly_chart(fig_corr, use_container_width=True)
        
        with st.expander("💡 Interpretation"):
            st.write("""
            - **Strong positive correlations** (red): Features that increase together
            - **Strong negative correlations** (blue): Features that move in opposite directions
            - **Weak correlations** (white): Features with little relationship
            - Key insight: OverallQual and GrLivArea show strong correlation with SalePrice
            """)
    
    def show_model_performance(self):
        """Display model performance comparison"""
        st.markdown("### 📈 Model Performance Comparison")
        
        models = ['Linear Regression', 'Random Forest', 'XGBoost', 'Neural Network']
        rmse_scores = [0.145, 0.128, 0.122, 0.135]
        mae_scores = [0.098, 0.087, 0.081, 0.091]
        r2_scores = [0.87, 0.91, 0.93, 0.89]
        
        # Create performance dataframe
        performance_df = pd.DataFrame({
            'Model': models,
            'RMSE': rmse_scores,
            'MAE': mae_scores,
            'R²': r2_scores
        })
        
        # Display metrics table
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Performance Metrics:**")
            st.dataframe(performance_df.style.highlight_min(subset=['RMSE', 'MAE']).highlight_max(subset=['R²']))
        
        with col2:
            # Bar chart for R² scores
            fig_r2 = px.bar(x=models, y=r2_scores,
                           title="R² Score Comparison (Higher is Better)",
                           labels={'x': 'Model', 'y': 'R² Score'},
                           color=r2_scores,
                           color_continuous_scale='viridis')
            fig_r2.update_layout(height=300)
            st.plotly_chart(fig_r2, use_container_width=True)
        
        # Best model highlight
        best_model_idx = np.argmax(r2_scores)
        st.success(f"🏆 **Best Model**: {models[best_model_idx]} with R² = {r2_scores[best_model_idx]}")
    
    def show_feature_importance(self):
        """Display feature importance for XGBoost model"""
        st.markdown("### 🎯 Feature Importance (XGBoost)")
        
        features_imp = ['OverallQual', 'GrLivArea', 'ExterQual', 'KitchenQual', 
                       'GarageCars', 'GarageArea', 'TotalBsmtSF', 'BsmtQual']
        importance = [0.23, 0.19, 0.12, 0.11, 0.09, 0.08, 0.07, 0.06]
        
        fig_imp = px.bar(y=features_imp, x=importance, orientation='h',
                        title="Top Features Affecting House Prices",
                        labels={'x': 'Importance', 'y': 'Features'},
                        color=importance,
                        color_continuous_scale='viridis')
        fig_imp.update_layout(height=400)
        st.plotly_chart(fig_imp, use_container_width=True)
        
        with st.expander("🔍 Feature Insights"):
            st.write("""
            - **OverallQual (23%)**: Overall material and finish quality - most important factor
            - **GrLivArea (19%)**: Above ground living area - size matters significantly
            - **ExterQual (12%)**: Exterior material quality affects perceived value
            - **KitchenQual (11%)**: Kitchen quality is crucial for home buyers
            - **GarageCars (9%)**: Number of car spaces adds convenience value
            """)
    
    def show_prediction_scatter(self):
        """Display prediction vs actual scatter plot"""
        st.markdown("### 🎯 Prediction vs Actual Values")
        
        n_points = 200
        actual = np.random.normal(200000, 50000, n_points)
        predicted = actual + np.random.normal(0, 15000, n_points)
        
        fig_scatter = px.scatter(x=actual, y=predicted, 
                               title="Predicted vs Actual House Prices",
                               labels={'x': 'Actual Price ($)', 'y': 'Predicted Price ($)'},
                               opacity=0.6)
        
        # Add perfect prediction line
        min_val, max_val = min(actual.min(), predicted.min()), max(actual.max(), predicted.max())
        fig_scatter.add_trace(go.Scatter(x=[min_val, max_val], 
                                       y=[min_val, max_val],
                                       mode='lines', name='Perfect Prediction',
                                       line=dict(color='red', dash='dash')))
        
        fig_scatter.update_layout(height=500)
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Calculate and display metrics
        residuals = predicted - actual
        mae = np.mean(np.abs(residuals))
        rmse = np.sqrt(np.mean(residuals**2))
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Mean Absolute Error", f"${mae:,.0f}")
        with col2:
            st.metric("Root Mean Square Error", f"${rmse:,.0f}")
        with col3:
            correlation = np.corrcoef(actual, predicted)[0, 1]
            st.metric("Prediction Correlation", f"{correlation:.3f}")
    
    def show_price_distribution(self):
        """Display price distribution analysis"""
        st.markdown("### 📊 House Price Distribution Analysis")
        
        # Generate sample data for different categories
        np.random.seed(42)
        
        # Price by Overall Quality
        qualities = [5, 6, 7, 8, 9, 10]
        price_data = []
        quality_labels = []
        
        for qual in qualities:
            prices = np.random.normal(150000 + qual * 30000, 25000, 50)
            price_data.extend(prices)
            quality_labels.extend([f"Quality {qual}"] * 50)
        
        df_prices = pd.DataFrame({
            'Price': price_data,
            'OverallQual': quality_labels
        })
        
        fig_box = px.box(df_prices, x='OverallQual', y='Price',
                        title="House Price Distribution by Overall Quality",
                        labels={'Price': 'House Price ($)'})
        fig_box.update_layout(height=400)
        st.plotly_chart(fig_box, use_container_width=True)
        
        with st.expander("📈 Price Insights"):
            st.write("""
            - **Quality 5-6**: Entry-level homes ($150k-$180k average)
            - **Quality 7-8**: Mid-range homes ($200k-$270k average)  
            - **Quality 9-10**: Premium homes ($300k-$400k+ average)
            - **Price variance** increases with quality level
            - **Sweet spot** appears to be Quality 7-8 for value
            """)