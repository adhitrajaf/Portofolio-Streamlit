import streamlit as st
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# Import pages
from pages import home, about_me, projects, model_visualization, ml_prediction
from utils import page_config, custom_css

# Page configuration
page_config.setup_page_config()

# Apply custom CSS
custom_css.apply_custom_css()

# Sidebar navigation
st.sidebar.title("🎯 Navigation")
page = st.sidebar.selectbox(
    "Choose a page:",
    ["🏠 Home", "👨‍💻 About Me", "🚀 My Projects", "📊 Model Visualization", "🔮 ML Prediction"]
)

# Route to appropriate page
if page == "🏠 Home":
    home.show()
elif page == "👨‍💻 About Me":
    about_me.show()
elif page == "🚀 My Projects":
    projects.show()
elif page == "📊 Model Visualization":
    model_visualization.show()
elif page == "🔮 ML Prediction":
    ml_prediction.show()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; margin-top: 2rem;'>
    <p>🚀 Built with Streamlit | 💼 Adhitya Fajar Rachmadi | 🎯 Machine Learning Portfolio</p>
    <p>📧 rachmadiadhityafajar@gmail.com | 🎓 Currently learning at Dibimbing.id</p>
</div>
""", unsafe_allow_html=True)