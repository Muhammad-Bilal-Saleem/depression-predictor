import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

from preprocessing.preprocess import preprocess_data

def train_models(df, save_path='models/'):
    """
    Train multiple models on the dataset and save them to disk.
    
    Args:
        df (pd.DataFrame): Raw student depression dataset
        save_path (str): Path to save trained models
        
    Returns:
        dict: Dictionary containing trained models and their accuracy scores
    """
    # Create models directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    # Preprocess data
    df_processed = preprocess_data(df)
    
    # Convert depression score to binary classification
    # 0 = Low risk (score <= 4)
    # 1 = High risk (score > 4)
    df_processed['Depression'] = (df_processed['Depression'] > 4).astype(int)
    
    # Split into features and target
    X = df_processed.drop(columns=['Depression'])
    y = df_processed['Depression']
    
    # Save feature columns for later use
    joblib.dump(X.columns.tolist(), os.path.join(save_path, 'feature_columns.pkl'))
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    
    # Initialize models
    models = {
        'logistic_regression': LogisticRegression(max_iter=1000, random_state=42),
        'random_forest': RandomForestClassifier(n_estimators=100, max_depth=20, min_samples_leaf=1, random_state=42),
        'xgboost': XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, subsample=0.8, random_state=42),
        'mlp': MLPClassifier(hidden_layer_sizes=(100,), activation='relu', alpha=0.0001, learning_rate_init=0.001, random_state=42, max_iter=300)
    }
    
    results = {}
    
    # Train and evaluate each model
    for name, model in models.items():
        # Train
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Evaluate
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'report': classification_report(y_test, y_pred, output_dict=True)
        }
        
        # Save model
        joblib.dump(model, os.path.join(save_path, f'{name}.pkl'))
        
    # Find best model
    best_model_name = max(results, key=lambda x: results[x]['accuracy'])
    results['best_model'] = best_model_name
    
    # Save best model separately
    joblib.dump(results[best_model_name]['model'], os.path.join(save_path, 'best_model.pkl'))
    
    return results

def load_model(model_name='best_model', model_path='models/'):
    """
    Load a trained model from disk.
    
    Args:
        model_name (str): Name of the model to load
        model_path (str): Path to the saved models
        
    Returns:
        object: Trained model
    """
    return joblib.load(os.path.join(model_path, f'{model_name}.pkl'))

def get_feature_importance(model_name='best_model', model_path='models/'):
    """
    Get feature importance from a trained model.
    
    Args:
        model_name (str): Name of the model to load
        model_path (str): Path to the saved models
        
    Returns:
        pd.DataFrame: DataFrame containing feature importance
    """
    model = load_model(model_name, model_path)
    feature_columns = joblib.load(os.path.join(model_path, 'feature_columns.pkl'))
    
    # Different models have different ways of accessing feature importance
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])
    else:
        return pd.DataFrame({'Feature': feature_columns, 'Importance': np.ones(len(feature_columns))})
    
    return pd.DataFrame({'Feature': feature_columns, 'Importance': importances}).sort_values('Importance', ascending=False)

if __name__ == '__main__':
    # Load data
    df = pd.read_csv('data/student_depression_dataset.csv')
    
    # Train models
    results = train_models(df)
    
    # Print results
    for name, result in results.items():
        if name != 'best_model':
            print(f"{name} accuracy: {result['accuracy']:.4f}")
    
    print(f"Best model: {results['best_model']}")