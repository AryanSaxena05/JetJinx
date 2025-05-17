import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import warnings
import os
warnings.filterwarnings('ignore')

def load_and_preprocess_data():
    """Load and preprocess the flight data"""
    print("Loading data...")
    # Load the data
    df = pd.read_csv('../data/flights.csv')
    
    # For demonstration, we'll use a subset of the data
    df = df.sample(n=100000, random_state=42)
    
    # Create target variable (1 if delay > 15 minutes, 0 otherwise)
    df['IS_DELAYED'] = (df['ARRIVAL_DELAY'] > 15).astype(int)
    
    # Select features
    features = [
        'MONTH', 'DAY_OF_WEEK', 'AIRLINE', 'ORIGIN_AIRPORT', 
        'DESTINATION_AIRPORT', 'SCHEDULED_DEPARTURE', 'DISTANCE'
    ]
    
    X = df[features].copy()
    y = df['IS_DELAYED']
    
    # Convert categorical columns to string type
    categorical_cols = ['AIRLINE', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT']
    for col in categorical_cols:
        X[col] = X[col].astype(str)
    
    # Handle categorical variables
    label_encoders = {}
    
    for col in categorical_cols:
        label_encoders[col] = LabelEncoder()
        X[col] = label_encoders[col].fit_transform(X[col])
    
    # Create models directory if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # Save label encoders for later use
    joblib.dump(label_encoders, 'models/label_encoders.joblib')
    
    # Scale numerical features
    scaler = StandardScaler()
    numerical_cols = ['MONTH', 'DAY_OF_WEEK', 'SCHEDULED_DEPARTURE', 'DISTANCE']
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
    
    # Save scaler for later use
    joblib.dump(scaler, 'models/scaler.joblib')
    
    return X, y

def train_model(X, y):
    """Train the Random Forest model"""
    print("Training model...")
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Random Forest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    print("\nModel Performance:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save the model
    joblib.dump(model, 'models/flight_delay_model.joblib')
    print("\nModel saved successfully!")
    
    return model

def predict_delay(input_data):
    """Predict flight delay for new input"""
    # Load the saved model and preprocessors
    model = joblib.load('models/flight_delay_model.joblib')
    label_encoders = joblib.load('models/label_encoders.joblib')
    scaler = joblib.load('models/scaler.joblib')
    
    # Convert input to DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Convert categorical columns to string type
    categorical_cols = ['AIRLINE', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT']
    for col in categorical_cols:
        input_df[col] = input_df[col].astype(str)
    
    # Transform categorical variables
    for col in categorical_cols:
        input_df[col] = label_encoders[col].transform(input_df[col])
    
    # Scale numerical features
    numerical_cols = ['MONTH', 'DAY_OF_WEEK', 'SCHEDULED_DEPARTURE', 'DISTANCE']
    input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])
    
    # Make prediction
    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)
    
    return prediction[0], probability[0][1]

def main():
    # Load and preprocess data
    X, y = load_and_preprocess_data()
    
    # Train model
    model = train_model(X, y)
    
    # Example of how to use the prediction function
    print("\nExample prediction:")
    example_input = {
        'MONTH': 7,
        'DAY_OF_WEEK': 3,
        'AIRLINE': 'AA',
        'ORIGIN_AIRPORT': 'JFK',
        'DESTINATION_AIRPORT': 'LAX',
        'SCHEDULED_DEPARTURE': 1200,
        'DISTANCE': 2475
    }
    
    prediction, probability = predict_delay(example_input)
    print(f"\nPrediction for example flight: {'Delayed' if prediction == 1 else 'Not Delayed'}")
    print(f"Probability of delay: {probability:.2%}")

if __name__ == "__main__":
    main() 