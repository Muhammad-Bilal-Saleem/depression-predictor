import pandas as pd
import numpy as np
import os
import joblib

def load_and_check_dataset(file_path):
    """
    Load the dataset and perform basic checks.
    
    Args:
        file_path (str): Path to dataset file
        
    Returns:
        pd.DataFrame: Loaded dataframe
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at {file_path}")
    
    df = pd.read_csv(file_path)
    
    # Check if required columns exist
    required_columns = ['Depression', 'Academic Pressure', 'Family History of Mental Illness']
    missing_columns = set(required_columns) - set(df.columns)
    
    if missing_columns:
        raise ValueError(f"Dataset missing required columns: {missing_columns}")
    
    return df

def calculate_prevalence(df):
    """
    Calculate depression prevalence statistics.
    
    Args:
        df (pd.DataFrame): Dataset
        
    Returns:
        dict: Dictionary of depression statistics
    """
    # Define depression categories based on binary classification
    # Convert to binary first (assuming threshold of 4)
    binary_depression = (df['Depression'] > 4).astype(int)
    
    # Calculate prevalence
    prevalence = binary_depression.value_counts(normalize=True) * 100
    
    # Create a readable dictionary with 'No Depression' and 'Depression' as keys
    prevalence_dict = {
        'No Depression': prevalence.get(0, 0),
        'Depression': prevalence.get(1, 0)
    }
    
    return {
        'prevalence': prevalence_dict,
        'mean_score': df['Depression'].mean(),
        'median_score': df['Depression'].median()
    }

def save_feature_importance(model_path='models/'):
    """
    Save feature importance from all models.
    
    Args:
        model_path (str): Path to saved models
    """
    from model.train import get_feature_importance
    
    # Models to extract feature importance from
    model_names = ['logistic_regression', 'random_forest', 'xgboost']
    
    # Get feature importance from each model
    all_importance = pd.DataFrame()
    
    for model_name in model_names:
        if os.path.exists(os.path.join(model_path, f'{model_name}.pkl')):
            importance = get_feature_importance(model_name, model_path)
            if all_importance.empty:
                all_importance = importance
            else:
                # Normalize and combine
                importance['Importance'] = importance['Importance'] / importance['Importance'].max()
                all_importance['Importance'] += importance['Importance']
    
    # Average and normalize
    if not all_importance.empty:
        all_importance['Importance'] = all_importance['Importance'] / len(model_names)
        all_importance = all_importance.sort_values('Importance', ascending=False)
        
        # Save
        joblib.dump(all_importance, os.path.join(model_path, 'feature_importance.pkl'))

def first_time_setup(data_path='data/', model_path='models/'):
    """
    Perform first-time setup for the app.
    
    Args:
        data_path (str): Path to data directory
        model_path (str): Path to model directory
        
    Returns:
        bool: True if setup was successful
    """
    try:
        from model.train import train_models
        
        # Create directories if they don't exist
        os.makedirs(data_path, exist_ok=True)
        os.makedirs(model_path, exist_ok=True)
        
        # Check if dataset exists
        dataset_path = os.path.join(data_path, 'student_depression_dataset.csv')
        if not os.path.exists(dataset_path):
            print("Dataset not found. Please place the dataset in the data directory.")
            return False
        
        # Check if models exist
        if os.path.exists(os.path.join(model_path, 'best_model.pkl')):
            print("Models already exist. Skipping training.")
            return True
        
        # Load dataset
        df = load_and_check_dataset(dataset_path)
        
        # Train models
        print("Training models...")
        train_models(df, model_path)
        
        # Save feature importance
        save_feature_importance(model_path)
        
        print("Setup complete!")
        return True
        
    except Exception as e:
        print(f"Setup failed: {str(e)}")
        return False