# Student Depression Prediction Web App

A web application built with Streamlit that predicts depression levels in students based on various factors including academic pressure, lifestyle, and personal background.

## Features

- **Depression Prediction**: Get an assessment of depression risk based on personal inputs
- **Data Visualizations**: Explore insights about depression risk factors and correlations
- **Recommendations**: Receive personalized recommendations based on your depression risk level
- **Mental Health Resources**: Access to a curated list of mental health resources

## Project Structure

```
student_depression_app/
├── app.py               # Main Streamlit application
├── model/              
│   ├── __init__.py
│   ├── train.py         # Model training script
│   ├── predict.py       # Prediction functions
│   └── utils.py         # Helper functions
├── data/
│   └── student_depression_dataset.csv  # Dataset
├── preprocessing/
│   └── preprocess.py    # Data preprocessing functions
├── visualizations/
│   └── viz.py           # Visualization functions
├── models/              # Saved model files
│   └── .gitkeep
├── assets/              # CSS, images, etc.
│   └── style.css
├── requirements.txt     # Project dependencies
└── README.md            # Project documentation
```

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd student_depression_app
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the dependencies:
```bash
pip install -r requirements.txt
```

4. Place the dataset in the `data` directory:
```bash
mkdir -p data
# Copy your student_depression_dataset.csv to the data directory
```

## Running the App

Run the Streamlit app with:

```bash
streamlit run app.py
```

The first time you run the app, it will automatically train the models, which may take a few minutes.

## Usage

1. Navigate to the "Prediction" page
2. Fill in the form with your personal information
3. Click the "Predict Depression Level" button
4. View your results and recommendations

## Deployment

This app can be deployed to Streamlit Cloud, Heroku, or any other platform that supports Python web applications.

### Streamlit Cloud Deployment

1. Push your code to GitHub
2. Go to [Streamlit Cloud](https://streamlit.io/cloud)
3. Connect your GitHub repository
4. Deploy the app

## Models

The app uses several machine learning models to predict depression levels:

- Logistic Regression
- Random Forest
- XGBoost
- Multi-layer Perceptron (Neural Network)

The best performing model is automatically selected during training.

## License

[MIT License](LICENSE)

## Acknowledgements

- Dataset source: [Student Depression Dataset on Kaggle](https://www.kaggle.com/datasets/adilshamim8/student-depression-dataset)
