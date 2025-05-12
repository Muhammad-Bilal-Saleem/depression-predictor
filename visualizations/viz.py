import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import joblib
import os

def plot_depression_distribution(df):
    """
    Plot distribution of binary depression outcomes.
    
    Args:
        df (pd.DataFrame): Dataset containing depression values (0 or 1)
        
    Returns:
        fig: Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Count values of depression
    depression_counts = df['Depression'].value_counts().reset_index()
    depression_counts.columns = ['Depression', 'Count']
    
    # Map the binary values to labels
    depression_counts['Label'] = depression_counts['Depression'].map({0: 'No Depression', 1: 'Depression'})
    
    # Create color mapping
    colors = ['green', 'red']
    
    # Create bar plot
    sns.barplot(data=depression_counts, x='Label', y='Count', palette=colors, ax=ax)
    
    # Add percentage labels on top of bars
    total = len(df)
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x() + p.get_width()/2.,
                height + 0.5,
                f'{height/total*100:.1f}%',
                ha="center")
    
    ax.set_title('Distribution of Depression in Students')
    ax.set_xlabel('')
    ax.set_ylabel('Count')
    
    return fig

def plot_risk_factors(model_path='models/'):
    """
    Plot top risk factors based on feature importance.
    
    Args:
        model_path (str): Path to saved models
        
    Returns:
        fig: Matplotlib figure
    """
    # Load feature importance
    feature_importance = joblib.load(os.path.join(model_path, 'feature_importance.pkl'))
    
    # Get top 10 features
    top_features = feature_importance.head(10).sort_values('Importance')
    
    # Clean up feature names
    def clean_feature_name(name):
        if 'Education Level_' in name:
            return f"Education: {name.replace('Education Level_', '')}"
        else:
            return name.replace('_', ' ')
    
    top_features['Feature'] = top_features['Feature'].apply(clean_feature_name)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.barplot(data=top_features, y='Feature', x='Importance', palette='viridis', ax=ax)
    
    ax.set_title('Top 10 Risk Factors for Student Depression')
    ax.set_xlabel('Relative Importance')
    ax.set_ylabel('')
    
    return fig

def plot_correlation_matrix(df):
    """
    Plot correlation matrix of numerical features.
    
    Args:
        df (pd.DataFrame): Dataset
        
    Returns:
        fig: Matplotlib figure
    """
    # Select only numerical columns
    numeric_df = df.select_dtypes(include=[np.number])
    
    # Calculate correlation matrix
    corr = numeric_df.corr()
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    
    # Draw the heatmap
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)
    
    ax.set_title('Correlation Matrix of Features')
    
    return fig

def plot_academic_pressure_vs_depression(df):
    """
    Plot relationship between academic pressure and depression.
    
    Args:
        df (pd.DataFrame): Dataset
        
    Returns:
        fig: Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create a countplot instead of boxplot for binary target
    # Group by academic pressure and calculate percentage of depression
    ap_groups = df.groupby('Academic Pressure')['Depression'].mean().reset_index()
    ap_groups['Depression %'] = ap_groups['Depression'] * 100
    
    # Plot as bar chart
    sns.barplot(data=ap_groups, x='Academic Pressure', y='Depression %', ax=ax)
    
    ax.set_title('Relationship Between Academic Pressure and Depression')
    ax.set_xlabel('Academic Pressure Level')
    ax.set_ylabel('Depression Rate (%)')
    
    return fig

def plot_binary_gauge_chart(value, title="Depression Risk Assessment"):
    """
    Create a simple gauge chart for binary depression outcome.
    
    Args:
        value (int): Binary value (0 or 1)
        title (str): Title of gauge
        
    Returns:
        fig: Matplotlib figure
    """
    # Set colors and labels based on value
    if value == 0:
        color = 'green'
        risk_level = 'Low Risk'
    else:
        color = 'red'
        risk_level = 'High Risk'
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Hide axis
    ax.axis('off')
    
    # Create a pie chart with one slice as our gauge
    ax.pie([1], colors=[color], wedgeprops=dict(width=0.3), startangle=90, counterclock=False)
    
    # Add text with prediction
    ax.text(0, 0, risk_level, ha='center', va='center', fontsize=24, fontweight='bold', color=color)
    
    # Add title
    ax.set_title(title, pad=20, fontsize=16)
    
    return fig