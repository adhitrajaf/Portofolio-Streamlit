import streamlit as st
from PIL import Image

def show():
    """Display the home page"""
    st.markdown('<h1 class="main-header">My Portfolio with Streamlit</h1>', unsafe_allow_html=True)
    
    # Hero section dengan layout samping
    col1, col2 = st.columns([1, 2])  # Rasio 1:2 untuk foto dan teks
    
    with col1:
        st.markdown("### 📸 Profile Photo")
        # Path ke foto profile
        profile_photo_path = "images/profile_photo.jpg"
        
        try:
            # Buka dan resize foto untuk kualitas HD
            from PIL import Image
            img = Image.open(profile_photo_path)
            # Resize dengan LANCZOS untuk kualitas terbaik
            img_resized = img.resize((350, 450), Image.LANCZOS)
            st.image(img_resized, caption="Adhitya Fajar Rachmadi", use_container_width=True)  
        except FileNotFoundError:
            st.image("https://via.placeholder.com/350x450.png?text=Photo+Not+Found", 
                    caption="Profile photo not found", use_column_width=True)
        except Exception as e:
            st.error(f"Error loading image: {e}")
    
    with col2:
        # Welcome message
        st.markdown("""
        ## 🌟 Welcome to My Data Science & Machine Learning Portfolio!
        
        Hi there! I'm **Adhitya Fajar Rachmadi**, a passionate **Machine Learning Enthusiast** and **Fresh Electrical Engineering Graduate** 
        currently enhancing my skills through an intensive **Machine Learning & AI Bootcamp at Dibimbing.id**.
        
        ### 🚀 Current Focus:
        Transitioning from **Electrical Engineering** to **Data Science & Machine Learning**, combining my technical background 
        with cutting-edge AI technologies to solve real-world problems.
        
        ### 🎯 What I Bring:
        - Strong **analytical thinking** from engineering background
        - Hands-on experience with **IoT projects**
        - Passionate about **data-driven solutions**
        - Continuous learner in **AI/ML technologies**
        """)
    
    # Separator
    st.markdown("---")
    
    # What You'll Find Here section
    st.markdown("""
    ## 🔍 What You'll Find Here:
    """)
    
    # Feature cards dalam 2 kolom
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **📖 About Me**  
        Learn about my background, education, and professional journey
        
        **💼 My Projects**  
        Explore my machine learning projects including house price prediction and customer churn analysis
        """)
    
    with col2:
        st.markdown("""
        **📊 Model Visualization**  
        Interactive visualizations of data and model performance
        
        **🔮 ML Prediction**  
        Live prediction interface for my trained models
        """)
    
    # Key highlights
    st.markdown("---")
    st.markdown("## 📈 Key Highlights")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("🎓 GPA", "3.80", "Cum Laude")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("📚 TOEFL Score", "597", "English Proficient")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("💼 Experience", "2+", "Years Industry")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("🏆 Projects", "5+", "ML & IoT")
        st.markdown('</div>', unsafe_allow_html=True)