from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load("customer_satisfaction_model.pkl")

# Define categorical columns based on dataset
categorical_cols = ["Customer Gender", "Product Purchased", "Ticket Type", "Ticket Status", "Ticket Priority", "Ticket Channel"]

@app.route('/')
def home():
    return "Customer Satisfaction Prediction API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json()
        
        # Convert input data to DataFrame
        df = pd.DataFrame([data])
        
        # One-Hot Encode categorical variables
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
        
        # Ensure input matches model's training columns
        missing_cols = set(model.feature_names_in_) - set(df.columns)
        for col in missing_cols:
            df[col] = 0  # Add missing columns with default value 0
        
        # Reorder columns to match model training data
        df = df[model.feature_names_in_]
        
        # Convert data to numeric format
        df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
        
        # Make prediction
        prediction = model.predict(df)
        
        # Return result as JSON
        return jsonify({'Customer Satisfaction Prediction': float(prediction[0])})
    
    except Exception as e:
        return jsonify({'error': str(e)})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)