import streamlit as st

def show():
    """Display the about me page"""
    st.markdown('<h1 class="main-header">About Me</h1>', unsafe_allow_html=True)
    
    # Personal Info Section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<h2 class="section-header">🎯 Professional Summary</h2>', unsafe_allow_html=True)
        st.markdown("""
        **Fresh Electrical Engineering Graduate** with a **3.80 GPA (Cum Laude)** from Universitas Sebelas Maret, 
        currently enhancing expertise through a **Machine Learning & AI Bootcamp at Dibimbing.id**. 
        
        With hands-on experience in **IoT systems**, **industrial automation**, and **computer vision**, 
        I'm passionate about applying **machine learning** and **data science** to solve complex real-world problems.
        
        ### 🌟 Key Strengths:
        - **Academic Excellence**: Top GPA achiever in semesters 5 & 7
        - **Industry Experience**: Practical exposure at PPSDM Migas Cepu and PT Inti Utama Solusindo
        - **Leadership**: Former Chairman of Student Association of Electrical Engineering
        - **Technical Innovation**: Designed automatic damper systems and computer vision solutions
        """)
    
    with col2:
        st.markdown('<h2 class="section-header">📧 Contact Info</h2>', unsafe_allow_html=True)
        st.markdown("""
        **📍 Location**: Pacitan, East Java, Indonesia
        
        **📧 Email**: rachmadiadhityafajar@gmail.com
        
        **💼 LinkedIn**: [Connect with me](https://www.linkedin.com/in/adhitya-fajar-rachmadi/  )
        
        **📱 WhatsApp**: Available on request
        
        **🌐 JobStreet**: [View Profile](https://id.jobstreet.com/id/profile/adhityafajar-rachmadi-r2vHDJZBQ5)
                    
        **📑 Github**: [GitHub Profile](https://github.com/adhitrajaf)
        """)
    
    # Education Section
    st.markdown('<h2 class="section-header">🎓 Education</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("""
        **Universitas Sebelas Maret – Surakarta, Indonesia**  
        *Bachelor Degree in Electrical Engineering* | **GPA: 3.80 (Cum Laude)**  
        *August 2020 – July 2024*
        
        **Final Project**: *"Prototype Design of an Automatic Damper System for Balancing Process for Factory Room Sterilization at PT. XYZ"*
        - Designed cutting-edge device to prevent cross-contamination between factory rooms
        - Successfully executed rapid and simultaneous pressure balancing
        - Maintained stringent sterilization standards in industrial environments
        """)
    
    # Current Learning
    st.markdown('<h2 class="section-header">📚 Current Learning</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    **Machine Learning and Artificial Intelligence Bootcamp**  
    *Dibimbing.id* | *March 2025 – Ongoing*
    
    🔹 **Python & SQL**: Data management, analysis, and visualization  
    🔹 **Machine Learning**: Classification, regression, clustering, and prediction models  
    🔹 **Deep Learning**: Advanced neural network architectures  
    🔹 **MLOps**: AI model deployment and management with Langchain  
    🔹 **Data Visualization**: Creating compelling data stories
    """)
    
    # Skills Section
    st.markdown('<h2 class="section-header">🛠 Technical Skills</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Programming & Development:**")
        skills_prog = ["Python", "C++", "SQL", "Machine Learning", "Deep Learning", "Computer Vision", "IoT Development"]
        for skill in skills_prog:
            st.markdown(f'<span class="skill-badge">{skill}</span>', unsafe_allow_html=True)
        
        st.markdown("\n\n**Industrial Systems:**")
        skills_ind = ["PLC Programming", "DCS Systems", "SCADA", "Siemens PLCs", "Allen-Bradley", "Yokogawa Centum VP"]
        for skill in skills_ind:
            st.markdown(f'<span class="skill-badge">{skill}</span>', unsafe_allow_html=True)
    
    with col2:
        st.markdown("**Data Science & ML:**")
        skills_ds = ["Pandas", "NumPy", "Scikit-learn", "TensorFlow", "MLOps", "Data Visualization", "Statistical Analysis"]
        for skill in skills_ds:
            st.markdown(f'<span class="skill-badge">{skill}</span>', unsafe_allow_html=True)
        
        st.markdown("\n\n**Soft Skills:**")
        skills_soft = ["Leadership", "Team Management", "Project Management", "Problem Solving", "Communication", "Adaptability"]
        for skill in skills_soft:
            st.markdown(f'<span class="skill-badge">{skill}</span>', unsafe_allow_html=True)