import streamlit as st
import pandas as pd
import os

# Custom imports
from preprocessing.preprocess import preprocess_data, create_input_dict, get_feature_options
from model.predict import predict_depression, get_depression_recommendations, get_emergency_resources
from model.utils import load_and_check_dataset, calculate_prevalence, first_time_setup
from visualizations.viz import (
    plot_depression_distribution, plot_risk_factors,
    plot_correlation_matrix, plot_academic_pressure_vs_depression,
    plot_binary_gauge_chart
)

# Set page config
st.set_page_config(
    page_title="Student Depression Predictor",
    page_icon="😞",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

try:
    local_css("assets/style.css")
except:
    pass

# App title and intro
st.title("Student Depression Predictor")
st.markdown("""
This application predicts depression risk in students based on various factors.
Fill in the form below and click the 'Predict' button to see your results.

**Disclaimer:** This is not a medical diagnostic tool. If you're experiencing mental health issues, 
please consult a healthcare professional.
""")

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Prediction", "About Depression", "Insights", "Help Resources"])

# Check if models are trained
if not os.path.exists("models/best_model.pkl"):
    st.warning("Models not trained yet. Please run setup first.")
    if st.button("Run Setup"):
        with st.spinner("Setting up models... This might take a while."):
            success = first_time_setup()
            if success:
                st.success("Setup complete! Please refresh the page.")
            else:
                st.error("Setup failed. Please check logs.")
    st.stop()

# Data loading
@st.cache_data
def load_data():
    try:
        return load_and_check_dataset("data/student_depression_dataset.csv")
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return None

df = load_data()

# Prediction page
if page == "Prediction":
    st.header("Depression Prediction Tool")
    
    col1, col2 = st.columns(2)
    
    # Get default values and options
    default_values = create_input_dict()
    options = get_feature_options()
    
    # Create form
    with st.form("prediction_form"):
        # Personal info
        with col1:
            st.subheader("Personal Information")
            age = st.slider("Age", 15, 35, default_values['Age'])
            gender = st.radio("Gender", options['Gender'])
            education_level = st.selectbox("Education Level", options['Education Level'])
            
            st.subheader("Academic Information")
            cgpa = st.slider("CGPA", 0.0, 4.0, default_values['CGPA'], 0.1)
            academic_pressure = st.slider("Academic Pressure", 0, 10, default_values['Academic Pressure'])
            study_satisfaction = st.slider("Study Satisfaction", 0, 10, default_values['Study Satisfaction'])
            study_hours = st.slider("Study Hours Per Day", 0, 16, default_values['Work/Study Hours'])
        
        # Health info
        with col2:
            st.subheader("Health Information")
            sleep_duration = st.selectbox("Sleep Duration", options['Sleep Duration'])
            dietary_habits = st.selectbox("Dietary Habits", options['Dietary Habits'])
            
            st.subheader("Mental Health Information")
            financial_stress = st.slider("Financial Stress Level", 0, 10, default_values['Financial Stress'])
            family_history = st.radio("Family History of Mental Illness", options['Family History of Mental Illness'])
            suicidal_thoughts = st.radio("Have you ever had suicidal thoughts?", options['Have you ever had suicidal thoughts ?'])
        
        # Submit button
        submit_button = st.form_submit_button("Predict Depression Risk")
    
    # When form is submitted
    if submit_button:
        # Create input dict
        user_input = {
            'Age': age,
            'Gender': gender,
            'CGPA': cgpa,
            'Sleep Duration': sleep_duration,
            'Financial Stress': financial_stress,
            'Academic Pressure': academic_pressure,
            'Study Satisfaction': study_satisfaction,
            'Work/Study Hours': study_hours,
            'Dietary Habits': dietary_habits,
            'Family History of Mental Illness': family_history,
            'Have you ever had suicidal thoughts ?': suicidal_thoughts,
            'Education Level': education_level
        }
        
        # Get prediction
        result = predict_depression(user_input)
        prediction = result['prediction']
        
        # Show result
        st.header("Prediction Results")
        
        # Display gauge chart
        st.subheader("Depression Risk Assessment")
        fig = plot_binary_gauge_chart(prediction)
        st.pyplot(fig)
        
        # Display text results
        risk_level = "High Risk" if prediction == 1 else "Low Risk" 
        st.subheader(f"Prediction: {risk_level}")
        
        if result['probability'] is not None:
            st.write(f"Confidence: {result['probability']*100:.1f}%")
        
        # Recommendations
        st.subheader("Recommendations")
        recommendations = get_depression_recommendations(prediction)
        for i, rec in enumerate(recommendations, 1):
            st.write(f"{i}. {rec}")
        
        # Warning for detected depression
        if prediction == 1:
            st.error("""
            **IMPORTANT: This prediction indicates a risk of depression.**
            
            Please consult a mental health professional for proper assessment. This is not a medical diagnosis, 
            but your responses suggest you may benefit from professional support.
            
            Scroll down to the "Help Resources" section for mental health resources.
            """)

# About Depression page
elif page == "About Depression":
    st.header("Understanding Student Depression")
    
    st.subheader("What is Depression?")
    st.write("""
    Depression is more than just feeling sad or going through a rough patch. It's a serious mental health condition 
    that requires understanding and medical care. Left untreated, depression can be devastating for those who have it 
    and their families. With proper diagnosis and treatment, many people with depression lead healthy, normal lives.
    """)
    
    st.subheader("Common Signs of Depression in Students")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        * Persistent sad, anxious, or "empty" mood
        * Loss of interest in hobbies and activities
        * Decreased energy, fatigue
        * Difficulty concentrating, remembering, making decisions
        * Insomnia, early-morning awakening, or oversleeping
        """)
    
    with col2:
        st.markdown("""
        * Changes in appetite or unplanned weight changes
        * Thoughts of death or suicide
        * Restlessness or irritability
        * Physical symptoms that don't respond to treatment
        * Social withdrawal and isolation
        """)
    
    st.subheader("Depression Prevalence in Students")
    if df is not None:
        stats = calculate_prevalence(df)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("No Depression", f"{stats['prevalence'].get('No Depression', 0):.1f}%")
        
        with col2:
            st.metric("Depression", f"{stats['prevalence'].get('Depression', 0):.1f}%")
        
        # Plot distribution
        fig = plot_depression_distribution(df)
        st.pyplot(fig)
    
    st.subheader("Risk Factors for Student Depression")
    st.write("""
    Several factors can contribute to depression among students:
    
    * **Academic Pressure**: High expectations, workload, and fear of failure
    * **Financial Stress**: Tuition costs, student loans, and living expenses
    * **Study Satisfaction**: Level of enjoyment and fulfillment from studies
    * **Sleep Disruption**: Irregular sleep schedules and sleep deprivation
    * **Dietary Habits**: Poor diet and nutrition
    * **Family History**: Genetic predisposition to depression
    """)

# Insights page
elif page == "Insights":
    st.header("Data Insights")
    
    if df is not None:
        st.subheader("Depression vs. Academic Pressure")
        fig = plot_academic_pressure_vs_depression(df)
        st.pyplot(fig)
        
        st.subheader("Top Risk Factors")
        fig = plot_risk_factors()
        st.pyplot(fig)
        
        st.subheader("Feature Correlation Matrix")
        fig = plot_correlation_matrix(df)
        st.pyplot(fig)
        
        with st.expander("View Raw Data Sample"):
            st.dataframe(df.head(20))
    else:
        st.error("Dataset not available. Please check if the file exists in the data directory.")

# Help Resources page
elif page == "Help Resources":
    st.header("Mental Health Resources")
    
    st.warning("""
    **Important Note**: If you're experiencing a mental health emergency or having thoughts of harming yourself, 
    please call emergency services immediately or go to your nearest emergency room.
    """)
    
    resources = get_emergency_resources()
    
    for region, contacts in resources.items():
        st.subheader(region)
        
        for contact in contacts:
            st.markdown(f"**{contact['name']}**")
            
            if 'phone' in contact:
                st.markdown(f"📞 {contact['phone']}")
            
            st.markdown(f"🌐 [{contact['website']}]({contact['website']})")
    
    st.subheader("University Resources")
    st.write("""
    Most universities offer free or low-cost counseling services to students. 
    Please check with your university's health services or student affairs office 
    for more information about available resources.
    """)

# Footer
st.markdown("""
---
**Disclaimer**: This application is for educational purposes only and is not a substitute for professional medical advice, 
diagnosis, or treatment. Always seek the advice of a qualified healthcare provider with any questions you may have regarding 
a medical condition.
""")