import pandas as pd
import numpy as np
import joblib
import os
from preprocessing.preprocess import preprocess_user_input

def predict_depression(user_input, model_name='best_model', model_path='models/'):
    """
    Predict binary depression risk based on user input.
    
    Args:
        user_input (dict): User input values
        model_name (str): Name of the model to use for prediction
        model_path (str): Path to the saved models
        
    Returns:
        dict: Prediction result with probability
    """
    # Load model
    model = joblib.load(os.path.join(model_path, f'{model_name}.pkl'))
    
    # Preprocess user input
    X = preprocess_user_input(user_input)
    
    # Load feature columns
    feature_columns = joblib.load(os.path.join(model_path, 'feature_columns.pkl'))
    
    # Make sure X has all the required columns in the right order
    missing_cols = set(feature_columns) - set(X.columns)
    for col in missing_cols:
        X[col] = 0
    
    X = X[feature_columns]
    
    # Make prediction
    prediction = int(model.predict(X)[0])
    
    # Get probability if available
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(X)[0]
        probability = probabilities[prediction]
    else:
        probability = None
    
    return {
        'prediction': prediction,
        'probability': probability
    }

def get_risk_factors(user_input, top_n=3, model_path='models/'):
    """
    Identify top risk factors for the user.
    
    Args:
        user_input (dict): User input values
        top_n (int): Number of top risk factors to return
        model_path (str): Path to the saved models
        
    Returns:
        list: List of top risk factors
    """
    # Get feature importances
    feature_importance = joblib.load(os.path.join(model_path, 'feature_importance.pkl'))
    
    # Sort by importance
    sorted_features = feature_importance.sort_values('Importance', ascending=False)
    
    # Get human-readable feature names
    top_features = []
    for feature in sorted_features['Feature'].head(top_n):
        # Convert one-hot encoded features back to original names
        if 'City_' in feature:
            feature_name = f"Living in {feature.replace('City_', '')}"
        elif 'Degree_' in feature:
            feature_name = f"Pursuing {feature.replace('Degree_', '')}"
        else:
            # Convert camel case to readable format
            feature_name = ' '.join(feature.split('_'))
        
        top_features.append(feature_name)
    
    return top_features

def get_depression_recommendations(prediction):
    """
    Get recommendations based on depression risk prediction.
    
    Args:
        prediction (int): Predicted depression risk (0=low risk, 1=high risk)
        
    Returns:
        list: List of recommendations
    """
    general_recommendations = [
        "Maintain a consistent sleep schedule",
        "Practice mindfulness and relaxation techniques",
        "Stay physically active",
        "Eat a balanced diet",
        "Connect with friends and family"
    ]
    
    if prediction == 0:  # Low risk
        return general_recommendations + [
            "Continue your current routine",
            "Monitor your mood periodically",
            "Learn stress management techniques"
        ]
    else:  # High risk
        return general_recommendations + [
            "Speak with a mental health professional",
            "Consider joining therapy or support groups",
            "Develop a stress reduction plan",
            "Inform close friends or family about how you're feeling",
            "Contact your university's mental health services"
        ]

def get_emergency_resources():
    """
    Get emergency mental health resources.
    
    Returns:
        dict: Dictionary of emergency resources
    """
    return {
        'Global': [
            {
                'name': 'International Association for Suicide Prevention',
                'website': 'https://www.iasp.info/resources/Crisis_Centres/'
            },
            {
                'name': '7 Cups - Online Therapy & Free Counseling',
                'website': 'https://www.7cups.com/'
            }
        ],
        'USA': [
            {
                'name': 'National Suicide Prevention Lifeline',
                'phone': '1-800-273-8255',
                'website': 'https://suicidepreventionlifeline.org/'
            },
            {
                'name': 'Crisis Text Line',
                'phone': 'Text HOME to 741741',
                'website': 'https://www.crisistextline.org/'
            }
        ],
        'UK': [
            {
                'name': 'Samaritans',
                'phone': '116 123',
                'website': 'https://www.samaritans.org/'
            }
        ],
        'Canada': [
            {
                'name': 'Crisis Services Canada',
                'phone': '1-833-456-4566',
                'website': 'https://www.crisisservicescanada.ca/'
            }
        ],
        'Australia': [
            {
                'name': 'Lifeline Australia',
                'phone': '13 11 14',
                'website': 'https://www.lifeline.org.au/'
            }
        ]
    }