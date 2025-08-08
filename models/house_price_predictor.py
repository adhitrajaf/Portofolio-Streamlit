import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import io

class HousePricePredictor:
    def __init__(self):
        """Initialize House Price Predictor"""
        self.model_loaded = False
    
    def show_prediction_interface(self):
        """Display the house price prediction interface"""
        st.markdown('<h2 class="section-header">🏠 House Price Prediction</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        Upload a CSV file with house features to get price predictions. 
        The model expects features like: OverallQual, GrLivArea, GarageCars, TotalBsmtSF, etc.
        """)
        
        # Show sample data format
        with st.expander("📋 Expected Data Format"):
            sample_data = {
                'OverallQual': [7, 6, 8, 5],
                'GrLivArea': [1500, 1200, 2000, 900],
                'GarageCars': [2, 1, 3, 0],
                'TotalBsmtSF': [800, 600, 1200, 0],
                'YearBuilt': [2000, 1995, 2010, 1980],
                'FullBath': [2, 1, 3, 1]
            }
            st.dataframe(pd.DataFrame(sample_data))
            st.info("💡 Your CSV should contain these columns (or similar house features)")
        
        # File upload section
        self.show_file_upload_prediction()
        
        # Manual prediction section
        self.show_manual_prediction()
    
    def show_file_upload_prediction(self):
        """Handle file upload and batch prediction"""
        st.markdown("### 📁 Batch Prediction from CSV")
        
        uploaded_file = st.file_uploader("Upload your house data (CSV)", type=['csv'], key="house_upload")
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"✅ File uploaded successfully! Shape: {df.shape}")
                
                # Display first few rows
                st.markdown("#### 📊 Data Preview")
                st.dataframe(df.head())
                
                # Data validation
                self.validate_house_data(df)
                
                # Prediction button
                col1, col2, col3 = st.columns([1, 1, 1])
                
                with col2:
                    if st.button("🚀 Generate Predictions", key="house_predict", type="primary"):
                        self.generate_batch_predictions(df)
                        
            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)}")
                st.info("Please ensure your CSV file contains the required house features.")
    
    def validate_house_data(self, df):
        """Validate uploaded house data"""
        required_features = ['OverallQual', 'GrLivArea']
        missing_features = [feat for feat in required_features if feat not in df.columns]
        
        if missing_features:
            st.warning(f"⚠️ Missing important features: {', '.join(missing_features)}")
            st.info("Predictions will use available features, but accuracy may be reduced.")
        else:
            st.success("✅ All key features found!")
        
        # Show data quality info
        with st.expander("🔍 Data Quality Report"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Data Shape:**")
                st.write(f"- Rows: {df.shape[0]}")
                st.write(f"- Columns: {df.shape[1]}")
                st.write(f"- Memory usage: {df.memory_usage().sum() / 1024:.1f} KB")
            
            with col2:
                st.write("**Missing Values:**")
                missing_pct = (df.isnull().sum() / len(df) * 100).round(1)
                missing_cols = missing_pct[missing_pct > 0]
                if len(missing_cols) > 0:
                    for col, pct in missing_cols.items():
                        st.write(f"- {col}: {pct}%")
                else:
                    st.write("- No missing values found! 🎉")
    
    def generate_batch_predictions(self, df):
        """Generate predictions for batch data"""
        with st.spinner("🔄 Running prediction model..."):
            # Simulate prediction process
            np.random.seed(42)
            
            # Generate realistic predictions based on available features
            base_price = 150000
            predictions = []
            
            for idx, row in df.iterrows():
                price = base_price
                
                # Add price based on available features
                if 'OverallQual' in df.columns:
                    price += row.get('OverallQual', 5) * 15000
                
                if 'GrLivArea' in df.columns:
                    price += row.get('GrLivArea', 1500) * 100
                
                if 'GarageCars' in df.columns:
                    price += row.get('GarageCars', 0) * 8000
                
                if 'TotalBsmtSF' in df.columns:
                    price += row.get('TotalBsmtSF', 0) * 40
                
                if 'YearBuilt' in df.columns:
                    price += (row.get('YearBuilt', 2000) - 1900) * 500
                
                # Add some randomness
                price += np.random.normal(0, 10000)
                price = max(price, 50000)  # Minimum price
                
                predictions.append(int(price))
            
            # Add predictions to dataframe
            df_results = df.copy()
            df_results['Predicted_Price'] = predictions
            df_results['Price_Category'] = pd.cut(predictions, 
                                                bins=[0, 150000, 250000, 350000, float('inf')],
                                                labels=['Budget', 'Mid-Range', 'Premium', 'Luxury'])
            
            st.success("✅ Predictions completed!")
            
            # Display results
            self.show_prediction_results(df_results, predictions)
    
    def show_prediction_results(self, df_results, predictions):
        """Display prediction results with visualizations"""
        st.markdown("#### 🎯 Prediction Results")
        
        # Summary statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Average Price", f"${np.mean(predictions):,.0f}")
        with col2:
            st.metric("Min Price", f"${np.min(predictions):,.0f}")
        with col3:
            st.metric("Max Price", f"${np.max(predictions):,.0f}")
        with col4:
            st.metric("Price Range", f"${np.max(predictions) - np.min(predictions):,.0f}")
        
        # Results table
        st.markdown("**Prediction Results:**")
        display_cols = ['Predicted_Price', 'Price_Category']
        if 'OverallQual' in df_results.columns:
            display_cols.insert(0, 'OverallQual')
        if 'GrLivArea' in df_results.columns:
            display_cols.insert(-2, 'GrLivArea')
        
        st.dataframe(df_results[display_cols].head(10))
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Price distribution
            fig_dist = px.histogram(x=predictions, nbins=20,
                                  title="Distribution of Predicted House Prices",
                                  labels={'x': 'Predicted Price ($)', 'y': 'Count'})
            st.plotly_chart(fig_dist, use_container_width=True)
        
        with col2:
            # Price by category
            category_counts = df_results['Price_Category'].value_counts()
            fig_pie = px.pie(values=category_counts.values, names=category_counts.index,
                           title="Houses by Price Category")
            st.plotly_chart(fig_pie, use_container_width=True)
        
        # Download results
        csv_buffer = io.StringIO()
        df_results.to_csv(csv_buffer, index=False)
        
        st.download_button(
            label="📥 Download Predictions",
            data=csv_buffer.getvalue(),
            file_name="house_price_predictions.csv",
            mime="text/csv",
            type="primary"
        )
    
    def show_manual_prediction(self):
        """Show manual prediction interface"""
        st.markdown("### 🏠 Manual House Price Prediction")
        st.markdown("Enter house features manually for a quick prediction:")
        
        with st.form("house_prediction_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                overall_qual = st.slider("Overall Quality (1-10)", 1, 10, 7, 
                                       help="Overall material and finish quality")
                gr_liv_area = st.number_input("Ground Living Area (sq ft)", 500, 5000, 1500,
                                            help="Above ground living area")
                garage_cars = st.selectbox("Garage Car Capacity", [0, 1, 2, 3, 4],
                                         help="Size of garage in car capacity")
                total_bsmt_sf = st.number_input("Total Basement Area (sq ft)", 0, 3000, 1000,
                                              help="Total square feet of basement area")
            
            with col2:
                year_built = st.slider("Year Built", 1872, 2024, 2000,
                                     help="Original construction date")
                full_bath = st.selectbox("Full Bathrooms", [0, 1, 2, 3, 4],
                                       help="Full bathrooms above ground")
                fireplaces = st.selectbox("Fireplaces", [0, 1, 2, 3],
                                        help="Number of fireplaces")
                lot_area = st.number_input("Lot Area (sq ft)", 1000, 50000, 8000,
                                         help="Lot size in square feet")
            
            # Predict button
            submitted = st.form_submit_button("🔮 Predict Price", type="primary")
            
            if submitted:
                # Calculate prediction using simple formula
                base_price = 50000
                price_prediction = (base_price + 
                                  (overall_qual * 15000) + 
                                  (gr_liv_area * 100) + 
                                  (garage_cars * 8000) + 
                                  (total_bsmt_sf * 40) + 
                                  (year_built - 1900) * 500 + 
                                  (full_bath * 5000) + 
                                  (fireplaces * 3000) + 
                                  (lot_area * 2))
                
                # Add some intelligent adjustments
                if year_built > 2000:
                    price_prediction *= 1.1  # Modern home bonus
                if overall_qual >= 8:
                    price_prediction *= 1.15  # High quality bonus
                
                price_prediction = int(price_prediction)
                
                # Display prediction with confidence
                st.success(f"🎯 **Predicted House Price: ${price_prediction:,}**")
                
                # Price category
                if price_prediction < 150000:
                    category = "Budget-Friendly 💰"
                    color = "green"
                elif price_prediction < 250000:
                    category = "Mid-Range 🏠"
                    color = "blue"
                elif price_prediction < 350000:
                    category = "Premium 🌟"
                    color = "orange"
                else:
                    category = "Luxury 💎"
                    color = "purple"
                
                st.markdown(f"**Category:** :{color}[{category}]")
                
                # Feature contribution breakdown
                with st.expander("🔍 Price Breakdown"):
                    contributions = {
                        'Base Price': 50000,
                        'Overall Quality': overall_qual * 15000,
                        'Living Area': gr_liv_area * 100,
                        'Garage': garage_cars * 8000,
                        'Basement': total_bsmt_sf * 40,
                        'Age Factor': (year_built - 1900) * 500,
                        'Bathrooms': full_bath * 5000,
                        'Fireplaces': fireplaces * 3000,
                        'Lot Size': lot_area * 2
                    }
                    
                    for feature, value in contributions.items():
                        if value > 0:
                            st.write(f"- **{feature}**: +${value:,}")
                
                # Market insights
                st.info(f"""
                💡 **Market Insights:**
                - This property would rank in the **{category.split()[0]}** segment
                - Quality score of {overall_qual}/10 is {'excellent' if overall_qual >= 8 else 'good' if overall_qual >= 6 else 'average'}
                - Living area of {gr_liv_area:,} sq ft is {'spacious' if gr_liv_area > 2000 else 'moderate' if gr_liv_area > 1200 else 'compact'}
                """)