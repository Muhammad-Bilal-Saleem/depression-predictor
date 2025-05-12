import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

def preprocess_data(df):
    """
    Perform preprocessing steps on the input dataframe.
    
    Args:
        df (pd.DataFrame): Raw student depression dataset
        
    Returns:
        pd.DataFrame: Processed dataframe ready for modeling
    """
    # Make a copy to avoid modifying the original
    df_processed = df.copy()
    
    # Drop unnecessary columns
    if 'id' in df_processed.columns:
        df_processed.drop('id', axis=1, inplace=True)
    
    # Since we're only focusing on students
    if 'Profession' in df_processed.columns:
        df_processed = df_processed[df_processed['Profession'] == 'Student']
        df_processed.drop('Profession', axis=1, inplace=True)
    
    # These columns aren't relevant for students
    if 'Job Satisfaction' in df_processed.columns:
        df_processed.drop('Job Satisfaction', axis=1, inplace=True)
        
    if 'Work Pressure' in df_processed.columns:
        df_processed.drop('Work Pressure', axis=1, inplace=True)
    
    # Drop City column as requested
    if 'City' in df_processed.columns:
        df_processed.drop('City', axis=1, inplace=True)
    
    # Handle missing values in Financial Stress
    if 'Financial Stress' in df_processed.columns:
        df_processed = df_processed[df_processed['Financial Stress'] != '?']
        df_processed['Financial Stress'] = df_processed['Financial Stress'].astype(float)
    
    # Ensure CGPA is between 0 and 4
    if 'CGPA' in df_processed.columns:
        # Convert CGPA to float if it's not already
        df_processed['CGPA'] = df_processed['CGPA'].astype(float)
        
        # Check if CGPA is on a different scale (like 0-10)
        if df_processed['CGPA'].max() > 4.0:
            # Convert from 0-10 scale to 0-4 scale
            df_processed['CGPA'] = df_processed['CGPA'] / 10 * 4
        
        # Ensure it doesn't exceed boundaries
        df_processed['CGPA'] = df_processed['CGPA'].clip(0.0, 4.0)
    
    # Binary encoding
    binary_mappings = {
        'Family History of Mental Illness': {"Yes": 1, "No": 0},
        'Have you ever had suicidal thoughts ?': {"Yes": 1, "No": 0},
        'Gender': {"Male": 0, "Female": 1}
    }
    
    for col, mapping in binary_mappings.items():
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].map(mapping)
    
    # Label encoding for ordinal categories
    le = LabelEncoder()
    label_encode_cols = ['Sleep Duration', 'Dietary Habits']
    for col in label_encode_cols:
        if col in df_processed.columns:
            df_processed[col] = le.fit_transform(df_processed[col])
    
    # Handle Degree field - categorize into educational levels
    if 'Degree' in df_processed.columns:
        # Map degrees to educational levels
        education_level_map = {
            # High School
            'Class 12': 'High School',
            
            # Bachelor's degrees
            'B.Ed': 'Bachelors',
            'B.Com': 'Bachelors',
            'B.Arch': 'Bachelors',
            'BCA': 'Bachelors',
            'B.Tech': 'Bachelors',
            'BHM': 'Bachelors',
            'BSc': 'Bachelors',
            'B.Pharm': 'Bachelors',
            'BBA': 'Bachelors',
            'LLB': 'Bachelors',
            'BE': 'Bachelors',
            'BA': 'Bachelors',
            'MBBS': 'Bachelors',
            
            # Master's degrees
            'MSc': 'Masters',
            'MCA': 'Masters',
            'M.Tech': 'Masters',
            'M.Ed': 'Masters',
            'M.Com': 'Masters',
            'M.Pharm': 'Masters',
            'MD': 'Masters',
            'MBA': 'Masters',
            'MA': 'Masters',
            'LLM': 'Masters',
            'MHM': 'Masters',
            'ME': 'Masters',
            
            # Doctorate
            'PhD': 'PhD',
            
            # Default
            'Others': 'Others'
        }
        
        # Apply mapping
        df_processed['Education Level'] = df_processed['Degree'].map(education_level_map)
        
        # Fill any missing mappings with 'Others'
        df_processed['Education Level'].fillna('Others', inplace=True)
        
        # Drop the original Degree column
        df_processed.drop('Degree', axis=1, inplace=True)
        
        # One-hot encode the Education Level
        enc = OneHotEncoder(sparse_output=False)
        encoded_arr = enc.fit_transform(df_processed[['Education Level']])
        encoded_df = pd.DataFrame(
            encoded_arr, 
            columns=enc.get_feature_names_out(['Education Level'])
        )
        
        df_processed = df_processed.drop(columns=['Education Level'])
        df_processed = pd.concat([df_processed, encoded_df], axis=1)
    
    # Fill missing values
    df_processed.fillna(df_processed.median(numeric_only=True), inplace=True)  # For numeric columns
    df_processed.fillna(df_processed.mode().iloc[0], inplace=True)  # For categorical columns
    
    return df_processed

def create_input_dict():
    """
    Create a dictionary with default values for user input.
    
    Returns:
        dict: Dictionary of default values for each input field
    """
    return {
        'Age': 20,
        'CGPA': 3.0,
        'Sleep Duration': 'Less than 6 hours',
        'Financial Stress': 5,
        'Academic Pressure': 5,
        'Study Satisfaction': 5,
        'Work/Study Hours': 8,
        'Dietary Habits': 'Moderate',
        'Family History of Mental Illness': 'No',
        'Have you ever had suicidal thoughts ?': 'No',
        'Gender': 'Male',
        'Education Level': 'Bachelors'
    }

def preprocess_user_input(user_input):
    """
    Preprocess user input into format expected by the model.
    
    Args:
        user_input (dict): User input values
        
    Returns:
        pd.DataFrame: Processed user input ready for prediction
    """
    # Convert to DataFrame
    df = pd.DataFrame([user_input])
    
    # Apply all preprocessing steps
    return preprocess_data(df)

def get_feature_options():
    """
    Define options for categorical features.
    
    Returns:
        dict: Dictionary of options for each categorical feature
    """
    return {
        'Sleep Duration': ['Less than 5 hours', '5-6 hours', '6-7 hours', '7-8 hours', 'More than 8 hours'],
        'Dietary Habits': ['Poor', 'Moderate', 'Healthy'],
        'Family History of Mental Illness': ['Yes', 'No'],
        'Have you ever had suicidal thoughts ?': ['Yes', 'No'],
        'Gender': ['Male', 'Female'],
        'Education Level': ['High School', 'Bachelors', 'Masters', 'PhD', 'Others']
    }