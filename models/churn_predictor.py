import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io

class ChurnPredictor:
    def __init__(self):
        """Initialize Customer Churn Predictor"""
        self.model_loaded = False
    
    def show_prediction_interface(self):
        """Display the customer churn prediction interface"""
        st.markdown('<h2 class="section-header">📞 Customer Churn Prediction</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        Upload customer data to predict churn probability. 
        The model analyzes factors like tenure, monthly charges, contract type, and service usage.
        """)
        
        # Show sample data format
        with st.expander("📋 Expected Data Format"):
            sample_data = {
                'tenure': [12, 24, 6, 48],
                'MonthlyCharges': [65.5, 89.3, 45.2, 120.0],
                'TotalCharges': [786.0, 2143.2, 271.2, 5760.0],
                'Contract': ['Month-to-month', 'One year', 'Month-to-month', 'Two year'],
                'InternetService': ['DSL', 'Fiber optic', 'DSL', 'Fiber optic'],
                'PaymentMethod': ['Electronic check', 'Credit card', 'Bank transfer', 'Electronic check']
            }
            st.dataframe(pd.DataFrame(sample_data))
            st.info("💡 Your CSV should contain these columns (or similar customer features)")
        
        # File upload section
        self.show_file_upload_prediction()
        
        # Manual prediction section
        self.show_manual_prediction()
    
    def show_file_upload_prediction(self):
        """Handle file upload and batch churn prediction"""
        st.markdown("### 📁 Batch Churn Analysis from CSV")
        
        uploaded_file = st.file_uploader("Upload customer data (CSV)", type=['csv'], key="churn_upload")
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"✅ File uploaded successfully! Shape: {df.shape}")
                
                # Display first few rows
                st.markdown("#### 📊 Data Preview")
                st.dataframe(df.head())
                
                # Data validation
                self.validate_churn_data(df)
                
                # Prediction button
                col1, col2, col3 = st.columns([1, 1, 1])
                
                with col2:
                    if st.button("🚀 Analyze Customer Churn", key="churn_predict", type="primary"):
                        self.generate_batch_churn_analysis(df)
                        
            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)}")
                st.info("Please ensure your CSV file contains customer data with relevant features.")
    
    def validate_churn_data(self, df):
        """Validate uploaded customer data"""
        important_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
        missing_features = [feat for feat in important_features if feat not in df.columns]
        
        if missing_features:
            st.warning(f"⚠️ Missing important features: {', '.join(missing_features)}")
            st.info("Analysis will use available features, but accuracy may be reduced.")
        else:
            st.success("✅ All key features found!")
        
        # Show data quality info
        with st.expander("🔍 Data Quality Report"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Dataset Overview:**")
                st.write(f"- Total customers: {df.shape[0]:,}")
                st.write(f"- Features available: {df.shape[1]}")
                st.write(f"- Data types: {df.dtypes.value_counts().to_dict()}")
            
            with col2:
                st.write("**Data Quality:**")
                missing_pct = (df.isnull().sum() / len(df) * 100).round(1)
                missing_cols = missing_pct[missing_pct > 0]
                if len(missing_cols) > 0:
                    for col, pct in missing_cols.head(5).items():
                        st.write(f"- {col}: {pct}% missing")
                else:
                    st.write("- No missing values detected! 🎉")
                
                # Data range validation
                if 'MonthlyCharges' in df.columns:
                    charges_range = df['MonthlyCharges'].agg(['min', 'max'])
                    st.write(f"- Monthly charges: ${charges_range['min']:.1f} - ${charges_range['max']:.1f}")
    
    def generate_batch_churn_analysis(self, df):
        """Generate churn analysis for batch data"""
        with st.spinner("🔄 Analyzing customer churn patterns..."):
            # Generate realistic churn predictions
            np.random.seed(42)
            n_customers = len(df)
            
            # Calculate churn probabilities based on available features
            churn_probabilities = []
            
            for idx, row in df.iterrows():
                base_risk = 0.2  # Base 20% churn risk
                
                # Tenure factor (newer customers more likely to churn)
                tenure = row.get('tenure', 12)
                if tenure < 6:
                    base_risk += 0.35
                elif tenure < 12:
                    base_risk += 0.25
                elif tenure < 24:
                    base_risk += 0.15
                elif tenure > 48:
                    base_risk -= 0.15
                
                # Monthly charges factor
                monthly_charges = row.get('MonthlyCharges', 65)
                if monthly_charges > 100:
                    base_risk += 0.25
                elif monthly_charges > 80:
                    base_risk += 0.15
                elif monthly_charges < 30:
                    base_risk -= 0.1
                
                # Contract factor
                contract = str(row.get('Contract', 'Month-to-month')).lower()
                if 'month' in contract:
                    base_risk += 0.3
                elif 'one' in contract:
                    base_risk += 0.05
                elif 'two' in contract:
                    base_risk -= 0.2
                
                # Internet service factor
                internet = str(row.get('InternetService', 'DSL')).lower()
                if 'fiber' in internet:
                    base_risk += 0.15
                elif 'no' in internet:
                    base_risk -= 0.1
                
                # Payment method factor
                payment = str(row.get('PaymentMethod', 'Credit card')).lower()
                if 'electronic check' in payment:
                    base_risk += 0.2
                
                # Add randomness and ensure valid probability
                base_risk += np.random.normal(0, 0.1)
                churn_prob = np.clip(base_risk, 0.01, 0.99)
                churn_probabilities.append(churn_prob)
            
            # Add predictions to dataframe
            df_results = df.copy()
            df_results['Churn_Probability'] = np.array(churn_probabilities) * 100
            df_results['Churn_Prediction'] = ['High Risk' if p > 0.6 else 'Medium Risk' if p > 0.3 else 'Low Risk' 
                                            for p in churn_probabilities]
            df_results['Risk_Score'] = pd.cut(churn_probabilities, 
                                            bins=[0, 0.3, 0.6, 1.0], 
                                            labels=['Low', 'Medium', 'High'])
            df_results['Retention_Priority'] = df_results['Churn_Probability'].apply(
                lambda x: 'Urgent' if x > 70 else 'High' if x > 50 else 'Medium' if x > 30 else 'Low'
            )
            
            st.success("✅ Churn analysis completed!")
            
            # Display comprehensive results
            self.show_churn_analysis_results(df_results, churn_probabilities)
    
    def show_churn_analysis_results(self, df_results, churn_probabilities):
        """Display comprehensive churn analysis results"""
        st.markdown("#### 🎯 Churn Analysis Results")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        high_risk = (np.array(churn_probabilities) > 0.7).sum()
        medium_risk = ((np.array(churn_probabilities) > 0.3) & (np.array(churn_probabilities) <= 0.7)).sum()
        low_risk = (np.array(churn_probabilities) <= 0.3).sum()
        avg_churn_prob = np.mean(churn_probabilities) * 100
        
        with col1:
            st.metric("🔴 High Risk", high_risk, f"{high_risk/len(df_results)*100:.1f}%")
        with col2:
            st.metric("🟡 Medium Risk", medium_risk, f"{medium_risk/len(df_results)*100:.1f}%")
        with col3:
            st.metric("🟢 Low Risk", low_risk, f"{low_risk/len(df_results)*100:.1f}%")
        with col4:
            st.metric("📊 Avg Churn Risk", f"{avg_churn_prob:.1f}%")
        
        # Results table with key columns
        st.markdown("**Customer Risk Assessment:**")
        display_cols = ['Churn_Probability', 'Churn_Prediction', 'Risk_Score', 'Retention_Priority']
        
        # Add relevant input features if available
        if 'tenure' in df_results.columns:
            display_cols.insert(0, 'tenure')
        if 'MonthlyCharges' in df_results.columns:
            display_cols.insert(-3, 'MonthlyCharges')
        
        # Color-code the results
        styled_df = df_results[display_cols].head(15).style.apply(
            lambda x: ['background-color: #ffebee' if 'High' in str(val) 
                      else 'background-color: #fff3e0' if 'Medium' in str(val)
                      else 'background-color: #e8f5e8' if 'Low' in str(val)
                      else '' for val in x], axis=0)
        
        st.dataframe(styled_df)
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Churn probability distribution
            fig_dist = px.histogram(x=np.array(churn_probabilities)*100, nbins=20,
                                  title="Churn Probability Distribution",
                                  labels={'x': 'Churn Probability (%)', 'y': 'Number of Customers'},
                                  color_discrete_sequence=['#1f77b4'])
            
            # Add risk threshold lines
            fig_dist.add_vline(x=30, line_dash="dash", line_color="orange", 
                             annotation_text="Medium Risk (30%)")
            fig_dist.add_vline(x=70, line_dash="dash", line_color="red", 
                             annotation_text="High Risk (70%)")
            
            st.plotly_chart(fig_dist, use_container_width=True)
        
        with col2:
            # Risk level distribution
            risk_counts = df_results['Risk_Score'].value_counts()
            colors = ['#d62728' if risk == 'High' else '#ff7f0e' if risk == 'Medium' else '#2ca02c' 
                     for risk in risk_counts.index]
            
            fig_pie = px.pie(values=risk_counts.values, names=risk_counts.index,
                           title="Customer Risk Distribution",
                           color_discrete_sequence=colors)
            st.plotly_chart(fig_pie, use_container_width=True)
        
        # Advanced analytics
        self.show_advanced_churn_analytics(df_results)
        
        # Download results
        csv_buffer = io.StringIO()
        df_results.to_csv(csv_buffer, index=False)
        
        st.download_button(
            label="📥 Download Complete Analysis",
            data=csv_buffer.getvalue(),
            file_name="customer_churn_analysis.csv",
            mime="text/csv",
            type="primary"
        )
    
    def show_advanced_churn_analytics(self, df_results):
        """Show advanced churn analytics"""
        st.markdown("#### 📈 Advanced Churn Analytics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Churn by tenure segments
            if 'tenure' in df_results.columns:
                df_results['Tenure_Segment'] = pd.cut(df_results['tenure'], 
                                                    bins=[0, 6, 12, 24, 48, 100],
                                                    labels=['0-6m', '6-12m', '12-24m', '24-48m', '48m+'])
                
                tenure_churn = df_results.groupby('Tenure_Segment')['Churn_Probability'].mean()
                
                fig_tenure = px.bar(x=tenure_churn.index, y=tenure_churn.values,
                                  title="Average Churn Risk by Tenure",
                                  labels={'x': 'Tenure Segment', 'y': 'Avg Churn Probability (%)'},
                                  color=tenure_churn.values,
                                  color_continuous_scale='Reds')
                st.plotly_chart(fig_tenure, use_container_width=True)
        
        with col2:
            # Revenue at risk analysis
            if 'MonthlyCharges' in df_results.columns:
                df_results['Revenue_at_Risk'] = (df_results['MonthlyCharges'] * 
                                               df_results['Churn_Probability'] / 100)
                
                total_revenue = df_results['MonthlyCharges'].sum()
                revenue_at_risk = df_results['Revenue_at_Risk'].sum()
                
                st.metric("💰 Monthly Revenue at Risk", 
                         f"${revenue_at_risk:,.0f}",
                         f"{revenue_at_risk/total_revenue*100:.1f}% of total")
                
                # Top revenue at risk customers
                top_risk_customers = df_results.nlargest(10, 'Revenue_at_Risk')[
                    ['MonthlyCharges', 'Churn_Probability', 'Revenue_at_Risk']
                ].round(2)
                
                st.markdown("**Top Revenue at Risk:**")
                st.dataframe(top_risk_customers)
        
        # Business recommendations
        self.show_business_recommendations(df_results)
    
    def show_business_recommendations(self, df_results):
        """Show actionable business recommendations"""
        st.markdown("#### 💼 Business Recommendations")
        
        high_risk_count = len(df_results[df_results['Risk_Score'] == 'High'])
        medium_risk_count = len(df_results[df_results['Risk_Score'] == 'Medium'])
        
        if high_risk_count > 0:
            st.error(f"""
            🚨 **Immediate Action Required**: {high_risk_count} customers at high risk
            
            **Recommended Actions:**
            - Launch immediate retention campaign for high-risk customers
            - Offer personalized discounts or service upgrades
            - Assign dedicated account managers for top revenue customers
            - Implement proactive customer service outreach
            """)
        
        if medium_risk_count > 0:
            st.warning(f"""
            ⚠️ **Monitor Closely**: {medium_risk_count} customers at medium risk
            
            **Recommended Actions:**
            - Implement customer satisfaction surveys
            - Offer loyalty program enrollment
            - Provide additional support and training
            - Monitor usage patterns for early warning signs
            """)
        
        st.info("""
        💡 **General Recommendations:**
        - Focus on customer onboarding for new customers (0-6 months)
        - Review pricing strategy for high monthly charge customers
        - Promote longer-term contracts with incentives
        - Improve fiber optic service quality and support
        - Encourage automatic payment methods vs. electronic checks
        """)
    
    def show_manual_prediction(self):
        """Show manual churn prediction interface"""
        st.markdown("### 📞 Manual Churn Risk Assessment")
        st.markdown("Enter customer details for individual churn risk assessment:")
        
        with st.form("churn_prediction_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                tenure = st.slider("Customer Tenure (months)", 1, 72, 12,
                                 help="How long the customer has been with the company")
                monthly_charges = st.number_input("Monthly Charges ($)", 20.0, 120.0, 65.0,
                                                help="Customer's monthly bill amount")
                total_charges = st.number_input("Total Charges ($)", 20.0, 8500.0, 2000.0,
                                              help="Total amount charged to customer")
                contract = st.selectbox("Contract Type", 
                                       ["Month-to-month", "One year", "Two year"],
                                       help="Contract duration")
            
            with col2:
                internet_service = st.selectbox("Internet Service", 
                                              ["DSL", "Fiber optic", "No"],
                                              help="Type of internet service")
                payment_method = st.selectbox("Payment Method", 
                                            ["Electronic check", "Mailed check", 
                                             "Bank transfer (automatic)", "Credit card (automatic)"],
                                            help="How customer pays their bill")
                tech_support = st.selectbox("Tech Support", ["Yes", "No"],
                                          help="Does customer have tech support")
                multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"],
                                            help="Does customer have multiple phone lines")
            
            # Predict button
            submitted = st.form_submit_button("🔮 Assess Churn Risk", type="primary")
            
            if submitted:
                # Calculate churn risk using business logic
                risk_score = self.calculate_manual_churn_risk(
                    tenure, monthly_charges, total_charges, contract, 
                    internet_service, payment_method, tech_support, multiple_lines
                )
                
                # Display results
                self.display_manual_churn_results(risk_score, tenure, monthly_charges, contract)
    
    def calculate_manual_churn_risk(self, tenure, monthly_charges, total_charges, contract, 
                                   internet_service, payment_method, tech_support, multiple_lines):
        """Calculate churn risk score for manual input"""
        risk_score = 20  # Base risk
        
        # Tenure factor
        if tenure < 6:
            risk_score += 35
        elif tenure < 12:
            risk_score += 25
        elif tenure < 24:
            risk_score += 15
        elif tenure > 48:
            risk_score -= 15
        
        # Monthly charges factor
        if monthly_charges > 100:
            risk_score += 25
        elif monthly_charges > 80:
            risk_score += 15
        elif monthly_charges < 30:
            risk_score -= 10
        
        # Contract factor
        if contract == "Month-to-month":
            risk_score += 30
        elif contract == "One year":
            risk_score += 5
        elif contract == "Two year":
            risk_score -= 20
        
        # Service factors
        if internet_service == "Fiber optic":
            risk_score += 15
        elif internet_service == "No":
            risk_score -= 5
        
        if "Electronic check" in payment_method:
            risk_score += 15
        elif "automatic" in payment_method:
            risk_score -= 10
        
        if tech_support == "No":
            risk_score += 12
        
        if multiple_lines == "No":
            risk_score += 5
        
        return max(0, min(100, risk_score))
    
    def display_manual_churn_results(self, risk_score, tenure, monthly_charges, contract):
        """Display manual churn prediction results"""
        # Determine risk level and color
        if risk_score > 70:
            risk_level = "🔴 HIGH RISK"
            recommendation = "**Immediate intervention required!** Consider retention offers, account manager assignment, or service improvements."
            color = "red"
        elif risk_score > 40:
            risk_level = "🟡 MEDIUM RISK" 
            recommendation = "**Monitor closely and consider proactive engagement.** Survey satisfaction and offer loyalty programs."
            color = "orange"
        else:
            risk_level = "🟢 LOW RISK"
            recommendation = "**Customer appears stable.** Continue regular service and consider upselling opportunities."
            color = "green"
        
        # Display main result
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.metric("Churn Risk Score", f"{risk_score}%")
            st.markdown(f"**Risk Level:** {risk_level}")
        
        with col2:
            # Customer value assessment
            annual_value = monthly_charges * 12
            if tenure > 0:
                ltv_estimate = (monthly_charges * tenure) + (annual_value * 2)  # Current + 2 year projection
            else:
                ltv_estimate = annual_value * 2
            
            st.metric("Est. Customer LTV", f"${ltv_estimate:,.0f}")
            st.metric("Monthly Revenue", f"${monthly_charges:.0f}")
        
        # Recommendation
        st.markdown("#### 💡 Recommendation")
        if risk_score > 70:
            st.error(recommendation)
        elif risk_score > 40:
            st.warning(recommendation)
        else:
            st.success(recommendation)
        
        # Risk factors breakdown
        with st.expander("🔍 Risk Factors Analysis"):
            st.write("**Key risk factors for this customer:**")
            
            factors = []
            if tenure < 12:
                factors.append(f"📅 **Short tenure** ({tenure} months) - New customers are more likely to churn")
            if monthly_charges > 80:
                factors.append(f"💰 **High monthly charges** (${monthly_charges}) - Price sensitivity concern")
            if contract == "Month-to-month":
                factors.append("📋 **Month-to-month contract** - No long-term commitment")
            if monthly_charges > 100:
                factors.append("💸 **Premium pricing tier** - Higher churn risk")
            
            if factors:
                for factor in factors:
                    st.write(f"- {factor}")
            else:
                st.write("- ✅ No major risk factors identified")
            
            # Protective factors
            protective = []
            if tenure > 24:
                protective.append(f"⏰ **Long tenure** ({tenure} months) - Established relationship")
            if contract == "Two year":
                protective.append("📋 **Long-term contract** - Commitment to stay")
            if monthly_charges < 50:
                protective.append(f"💰 **Affordable pricing** (${monthly_charges}) - Good value")
            
            if protective:
                st.write("**Protective factors:**")
                for factor in protective:
                    st.write(f"- {factor}")
        
        # Suggested actions
        if risk_score > 40:
            with st.expander("🎯 Suggested Actions"):
                st.write("**Immediate actions to reduce churn risk:**")
                
                actions = [
                    "📞 Schedule proactive customer service call",
                    "📊 Conduct customer satisfaction survey",
                    "💝 Offer personalized retention discount",
                    "📱 Provide account manager contact",
                    "🔧 Review and optimize service plan"
                ]
                
                if contract == "Month-to-month":
                    actions.append("📋 Incentivize longer-term contract")
                if monthly_charges > 80:
                    actions.append("💰 Review pricing options or bundle discounts")
                
                for action in actions:
                    st.write(f"- {action}")
        
        # ROI calculation for retention
        if risk_score > 50:
            retention_cost = min(monthly_charges * 2, 200)  # Max 2 months or $200
            potential_loss = ltv_estimate * (risk_score / 100)
            roi = (potential_loss - retention_cost) / retention_cost * 100
            
            st.info(f"""
            📊 **Retention ROI Analysis:**
            - Estimated retention cost: ${retention_cost:.0f}
            - Potential revenue loss: ${potential_loss:,.0f}
            - ROI of retention effort: {roi:.0f}%
            """)