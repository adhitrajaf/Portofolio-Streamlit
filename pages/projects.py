import streamlit as st

def show():
    """Display the projects page"""
    st.markdown('<h1 class="main-header">My Projects</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    Explore my diverse portfolio of **Machine Learning**, **IoT**, and **Industrial Automation** projects. 
    Each project demonstrates practical application of cutting-edge technologies to solve real-world challenges.
    """)
    
    # Project 1: House Price Prediction
    st.markdown('<div class="project-card">', unsafe_allow_html=True)
    st.markdown("""
    ## 🏠 House Price Prediction - Advanced Regression
    
    **Objective**: Predict house prices using advanced machine learning regression techniques
    
    **Key Features**:
    - 📊 Comprehensive data analysis with 79 explanatory variables
    - 🤖 Multiple ML models: Linear Regression, Random Forest, XGBoost, Neural Networks
    - 📈 Feature engineering and selection optimization
    - 🎯 Model evaluation using RMSE, MAE, and R² metrics
    - 🚀 Interactive prediction interface via Streamlit
    
    **Technologies**: Python, Scikit-learn, XGBoost, Pandas, NumPy, Matplotlib, Seaborn, Streamlit
    
    **Dataset**: Kaggle House Prices Competition
    """)
    
    # col1, col2 = st.columns(2)
    # with col1:
    #     if st.button("🔗 View on GitHub", key="house_github"):
    #         st.write("GitHub link would redirect here")
    # with col2:
    #     if st.button("🎮 Try Live Demo", key="house_demo"):
    #         st.write("Navigate to ML Prediction page to test this model!")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Project 2: Customer Churn Prediction
    st.markdown('<div class="project-card">', unsafe_allow_html=True)
    st.markdown("""
    ## 📞 Telco Customer Churn Prediction
    
    **Objective**: Predict customer churn for telecommunications company to improve retention strategies
    
    **Key Features**:
    - 🎯 Binary classification to identify at-risk customers
    - 📊 Feature analysis: demographics, services, billing patterns
    - ⚖️ Handling class imbalance with SMOTE and stratified sampling  
    - 🤖 Ensemble models: Random Forest, Gradient Boosting, Neural Networks
    - 📈 Model interpretability with SHAP values
    - 💼 Business impact analysis and retention recommendations
    
    **Technologies**: Python, Scikit-learn, SHAP, Plotly, Streamlit, Imbalanced-learn
    
    **Business Impact**: Potential 15-20% reduction in churn rate through targeted interventions
    """)
    
    # col1, col2 = st.columns(2)
    # with col1:
    #     if st.button("🔗 View on GitHub", key="churn_github"):
    #         st.write("GitHub link would redirect here")
    # with col2:
    #     if st.button("🎮 Try Live Demo", key="churn_demo"):
    #         st.write("Navigate to ML Prediction page to test this model!")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Project 3: IoT Air Pressure Balancing System
    st.markdown('<div class="project-card">', unsafe_allow_html=True)
    st.markdown("""
    ## 🏭 IoT Air Pressure Balancing System (Industry Project)
    
    **Objective**: Automated air pressure balancing system for 27-room pharmaceutical factory
    
    **Key Achievements**:
    - 🏭 **Real-world Implementation**: Deployed in PT Inti Utama Solusindo factory
    - 📡 **IoT Integration**: Real-time monitoring of air quality across 27 rooms
    - ⚙️ **PLC Control**: Siemens PLCs for precision machinery control
    - 🔧 **Arduino Integration**: Industrial-grade sensors with microcontrollers
    - 🛡️ **Safety Critical**: Ensuring consistent air quality for personnel safety
    - 📊 **Data Collection**: Continuous monitoring and logging for compliance
    
    **Technologies**: Siemens PLCs, Arduino Uno, Industrial Sensors, C++, Python, IoT Protocols
    
    **Business Impact**: Enhanced workplace safety and regulatory compliance
    """)
    
    # col1, col2 = st.columns(2)
    # with col1:
    #     if st.button("🔗 Technical Details", key="iot_details"):
    #         st.write("Detailed technical documentation available")
    # with col2:
    #     if st.button("📋 Case Study", key="iot_case"):
    #         st.write("Industry case study documentation")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Project 4: Computer Vision Quality Control
    st.markdown('<div class="project-card">', unsafe_allow_html=True)
    st.markdown("""
    ## 👁️ Computer Vision Quality Control System
    
    **Objective**: Automated sorting of empty tablet containers using YOLOv8 object detection
    
    **Key Features**:
    - 🎯 **YOLOv8 Implementation**: State-of-the-art object detection for defect identification
    - 🔧 **Hardware Integration**: Pneumatic-hydraulic sorting mechanism
    - ⚡ **Real-time Processing**: High-speed sorting to match production line speed  
    - 📦 **Quality Assurance**: Preventing defective products from reaching customers
    - 🤖 **Automation**: Reduced manual labor and human error
    - 📈 **Performance**: Improved production efficiency and reduced returns
    
    **Technologies**: YOLOv8, OpenCV, Python, Computer Vision, Industrial Automation
    
    **Business Impact**: Significant reduction in product returns due to defects
    """)
    
    # col1, col2 = st.columns(2)
    # with col1:
    #     if st.button("🔗 View Implementation", key="cv_impl"):
    #         st.write("Computer vision implementation details")
    # with col2:
    #     if st.button("📊 Performance Metrics", key="cv_metrics"):
    #         st.write("Model accuracy and speed benchmarks")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Project 5: Final Thesis Project
    st.markdown('<div class="project-card">', unsafe_allow_html=True)
    st.markdown("""
    ## 🎓 Automatic Damper System for Factory Sterilization (Final Thesis)
    
    **Objective**: Prototype design for preventing cross-contamination in factory environments
    
    **Innovation Highlights**:
    - 🛡️ **Contamination Prevention**: Critical system for pharmaceutical manufacturing
    - ⚡ **Rapid Response**: Simultaneous pressure balancing between rooms and corridors
    - 🎯 **Precision Control**: Maintaining stringent sterilization standards
    - 🏭 **Industrial Application**: Designed for PT. XYZ manufacturing facility
    - 📊 **Performance Validation**: Successful prototype testing and validation
    - 🔬 **Research Contribution**: Novel approach to factory room sterilization
    
    **Technologies**: Embedded Systems, Sensor Networks, Control Systems, MATLAB/Simulink
    
    **Academic Achievement**: Contributed to Cum Laude graduation with 3.80 GPA
    """)
    
    # col1, col2 = st.columns(2)
    # with col1:
    #     if st.button("📄 Read Thesis", key="thesis_read"):
    #         st.write("Full thesis document available")
    # with col2:
    #     if st.button("🎬 Demo Video", key="thesis_demo"):
    #         st.write("Prototype demonstration video")
    
    st.markdown('</div>', unsafe_allow_html=True)