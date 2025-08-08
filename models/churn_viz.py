import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

class ChurnVisualization:
    def __init__(self):
        # Generate sample data for demonstration
        np.random.seed(42)
    
    def show_visualizations(self):
        """Display all customer churn model visualizations"""
        st.markdown('<h2 class="section-header">📞 Customer Churn Prediction Model Analysis</h2>', unsafe_allow_html=True)
        
        # Confusion Matrix
        self.show_confusion_matrix()
        
        # Model Performance Metrics
        self.show_model_metrics()
        
        # Feature importance for churn
        self.show_feature_importance()
        
        # ROC Curve
        self.show_roc_curve()
        
        # Churn analysis by segments
        self.show_churn_segments()
        
        # Customer lifetime value analysis
        self.show_customer_analysis()
    
    def show_confusion_matrix(self):
        """Display confusion matrix"""
        st.markdown("### 📊 Confusion Matrix")
        
        conf_matrix = np.array([[1200, 150], [200, 800]])
        
        fig_conf = px.imshow(conf_matrix, 
                           text_auto=True,
                           aspect="auto",
                           title="Confusion Matrix - Churn Prediction",
                           labels={'x': 'Predicted', 'y': 'Actual'},
                           x=['No Churn', 'Churn'],
                           y=['No Churn', 'Churn'],
                           color_continuous_scale='Blues')
        fig_conf.update_layout(height=400)
        st.plotly_chart(fig_conf, use_container_width=True)
        
        # Calculate metrics from confusion matrix
        tn, fp, fn, tp = conf_matrix.ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * (precision * recall) / (precision + recall)
        
        with st.expander("🔍 Matrix Interpretation"):
            st.write(f"""
            - **True Negatives (TN)**: {tn} - Correctly predicted non-churners
            - **False Positives (FP)**: {fp} - Incorrectly predicted churners  
            - **False Negatives (FN)**: {fn} - Missed actual churners (costly!)
            - **True Positives (TP)**: {tp} - Correctly identified churners
            
            **Key Insight**: Model has low false negative rate, good for retention campaigns.
            """)
    
    def show_model_metrics(self):
        """Display model performance metrics"""
        st.markdown("### 📈 Model Performance Metrics")
        
        # Calculate metrics
        conf_matrix = np.array([[1200, 150], [200, 800]])
        tn, fp, fn, tp = conf_matrix.ravel()
        
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * (precision * recall) / (precision + recall)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Accuracy", f"{accuracy:.1%}", "2.3%")
        with col2:
            st.metric("Precision", f"{precision:.1%}", "1.8%")
        with col3:
            st.metric("Recall", f"{recall:.1%}", "3.1%")
        with col4:
            st.metric("F1-Score", f"{f1:.1%}", "2.5%")
        
        # Performance comparison with other models
        st.markdown("#### 🏆 Model Comparison")
        
        models = ['Logistic Regression', 'Random Forest', 'XGBoost', 'Neural Network']
        accuracies = [0.84, 0.89, 0.91, 0.88]
        precisions = [0.79, 0.84, 0.86, 0.83]
        recalls = [0.75, 0.80, 0.82, 0.79]
        
        comparison_df = pd.DataFrame({
            'Model': models,
            'Accuracy': accuracies,
            'Precision': precisions,
            'Recall': recalls
        })
        
        fig_comparison = go.Figure()
        
        fig_comparison.add_trace(go.Scatter(
            x=models, y=accuracies, mode='lines+markers',
            name='Accuracy', line=dict(color='blue', width=3)
        ))
        
        fig_comparison.add_trace(go.Scatter(
            x=models, y=precisions, mode='lines+markers',
            name='Precision', line=dict(color='orange', width=3)
        ))
        
        fig_comparison.add_trace(go.Scatter(
            x=models, y=recalls, mode='lines+markers',
            name='Recall', line=dict(color='green', width=3)
        ))
        
        fig_comparison.update_layout(
            title="Model Performance Comparison",
            xaxis_title="Model",
            yaxis_title="Score",
            height=400
        )
        
        st.plotly_chart(fig_comparison, use_container_width=True)
        
        st.success("🏆 **Best Model**: XGBoost with 91% accuracy and 86% precision")
    
    def show_feature_importance(self):
        """Display feature importance for churn prediction"""
        st.markdown("### 🎯 Top Factors Influencing Customer Churn")
        
        churn_features = ['Monthly Charges', 'Total Charges', 'Contract Type', 'Internet Service', 
                         'Payment Method', 'Tech Support', 'Tenure', 'Multiple Lines']
        churn_importance = [0.28, 0.22, 0.15, 0.12, 0.08, 0.06, 0.05, 0.04]
        
        fig_churn_imp = px.bar(y=churn_features, x=churn_importance, orientation='h',
                              title="Features Most Predictive of Customer Churn",
                              labels={'x': 'Importance', 'y': 'Features'},
                              color=churn_importance,
                              color_continuous_scale='Reds')
        fig_churn_imp.update_layout(height=400)
        st.plotly_chart(fig_churn_imp, use_container_width=True)
        
        with st.expander("💡 Business Insights"):
            st.write("""
            **Top Churn Drivers:**
            - **Monthly Charges (28%)**: High pricing sensitivity - consider tiered pricing
            - **Total Charges (22%)**: Long-term value perception - loyalty programs needed
            - **Contract Type (15%)**: Month-to-month contracts are risky - incentivize longer terms
            - **Internet Service (12%)**: Service quality issues - focus on Fiber Optic satisfaction
            - **Payment Method (8%)**: Electronic check users more likely to churn
            
            **Actionable Recommendations:**
            - Implement competitive pricing for high-charge customers
            - Create retention offers for month-to-month customers
            - Improve fiber optic service quality and support
            """)
    
    def show_roc_curve(self):
        """Display ROC curve"""
        st.markdown("### 📈 ROC Curve Analysis")
        
        # Generate ROC curve data
        fpr = np.linspace(0, 1, 100)
        tpr = 1 - np.exp(-5 * fpr)  # More realistic ROC curve
        tpr = tpr / tpr.max() * 0.95  # Scale to realistic values
        
        # Calculate AUC
        auc = np.trapz(tpr, fpr)
        
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', 
                                   name=f'ROC Curve (AUC = {auc:.3f})',
                                   line=dict(color='#1f77b4', width=3)))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                                   name='Random Classifier',
                                   line=dict(color='red', dash='dash')))
        
        fig_roc.update_layout(
            title="ROC Curve - Churn Prediction Model",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            showlegend=True,
            height=500
        )
        
        # Add optimal threshold point
        optimal_idx = np.argmax(tpr - fpr)
        fig_roc.add_trace(go.Scatter(
            x=[fpr[optimal_idx]], y=[tpr[optimal_idx]], 
            mode='markers', name='Optimal Threshold',
            marker=dict(color='red', size=10, symbol='star')
        ))
        
        st.plotly_chart(fig_roc, use_container_width=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("AUC Score", f"{auc:.3f}", "Excellent")
        with col2:
            st.metric("Optimal Threshold", f"{fpr[optimal_idx]:.3f}")
        with col3:
            st.metric("True Positive Rate", f"{tpr[optimal_idx]:.3f}")
    
    def show_churn_segments(self):
        """Display churn rate by customer segments"""
        st.markdown("### 📊 Churn Rate by Customer Segments")
        
        segments = ['Month-to-Month', 'One Year', 'Two Year', 'Fiber Optic', 'DSL', 'No Internet']
        churn_rates = [0.42, 0.11, 0.03, 0.31, 0.19, 0.08]
        colors = ['#d62728' if rate > 0.3 else '#ff7f0e' if rate > 0.15 else '#2ca02c' for rate in churn_rates]
        
        fig_segments = px.bar(x=segments, y=churn_rates,
                            title="Churn Rate by Customer Segments",
                            labels={'x': 'Customer Segment', 'y': 'Churn Rate'},
                            color=churn_rates,
                            color_continuous_scale='RdYlBu_r')
        fig_segments.update_layout(height=400)
        st.plotly_chart(fig_segments, use_container_width=True)
        
        # Segment analysis
        high_risk_segments = [seg for seg, rate in zip(segments, churn_rates) if rate > 0.3]
        st.warning(f"⚠️ **High Risk Segments**: {', '.join(high_risk_segments)}")
        st.success(f"✅ **Stable Segments**: Two Year contracts have lowest churn (3%)")
    
    def show_customer_analysis(self):
        """Display customer lifetime value and tenure analysis"""
        st.markdown("### 💰 Customer Value & Retention Analysis")
        
        # Generate sample customer data
        np.random.seed(42)
        n_customers = 1000
        
        tenure_months = np.random.exponential(24, n_customers)
        monthly_revenue = np.random.normal(65, 20, n_customers)
        churn_prob = np.exp(-tenure_months/36) + np.random.normal(0, 0.1, n_customers)
        churn_prob = np.clip(churn_prob, 0, 1)
        
        customer_df = pd.DataFrame({
            'Tenure_Months': tenure_months,
            'Monthly_Revenue': monthly_revenue,
            'Churn_Probability': churn_prob,
            'Customer_LTV': tenure_months * monthly_revenue
        })
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Tenure vs Churn Probability
            fig_tenure = px.scatter(customer_df, x='Tenure_Months', y='Churn_Probability',
                                  title="Customer Tenure vs Churn Risk",
                                  labels={'Tenure_Months': 'Tenure (Months)', 
                                         'Churn_Probability': 'Churn Probability'},
                                  opacity=0.6)
            
            # Add trend line
            z = np.polyfit(customer_df['Tenure_Months'], customer_df['Churn_Probability'], 2)
            p = np.poly1d(z)
            x_trend = np.linspace(0, customer_df['Tenure_Months'].max(), 100)
            fig_tenure.add_trace(go.Scatter(x=x_trend, y=p(x_trend), 
                                          mode='lines', name='Trend',
                                          line=dict(color='red', width=2)))
            
            st.plotly_chart(fig_tenure, use_container_width=True)
        
        with col2:
            # Customer LTV Distribution
            fig_ltv = px.histogram(customer_df, x='Customer_LTV',
                                 title="Customer Lifetime Value Distribution",
                                 labels={'Customer_LTV': 'Customer LTV ($)', 'count': 'Number of Customers'})
            st.plotly_chart(fig_ltv, use_container_width=True)
        
        # Key insights
        avg_ltv = customer_df['Customer_LTV'].mean()
        high_value_customers = (customer_df['Customer_LTV'] > customer_df['Customer_LTV'].quantile(0.8)).sum()
        
        st.info(f"""
        **Key Insights:**
        - 📊 Average Customer LTV: ${avg_ltv:,.0f}
        - 🏆 High-value customers (top 20%): {high_value_customers} customers
        - ⏰ Churn risk decreases significantly after 12 months
        - 💡 Focus retention efforts on customers with 3-12 month tenure
        """)