from flask import Flask, request, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression  # Replace with your actual model
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load('fraud_detection_model.pkl')  # Replace with the actual filename

# def preprocess_and_predict(input_data):
#     # Convert input data to DataFrame
#     input_df = pd.DataFrame([input_data])
    
#     # Drop irrelevant columns if present
#     if 'TransactionID' in input_df.columns:
#         input_df = input_df.drop(columns=['TransactionID'])
#     if 'TransactionDate' in input_df.columns:
#         input_df = input_df.drop(columns=['TransactionDate'])
    
#     # Convert categorical columns to numerical using one-hot encoding
#     input_df = pd.get_dummies(input_df, columns=['TransactionType', 'Location'], drop_first=True)
    
#     # Ensure the input data has the same columns as the training data
#     training_columns = model.feature_names_in_
#     input_df = input_df.reindex(columns=training_columns, fill_value=0)
    
#     # Make prediction
#     prediction = model.predict(input_df)
#     probability = model.predict_proba(input_df)
    
#     return prediction[0], probability[0][1]


def preprocess_and_predict(input_data):
    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Drop irrelevant columns if present
    if 'TransactionID' in input_df.columns:
        input_df = input_df.drop(columns=['TransactionID'])
    if 'TransactionDate' in input_df.columns:
        input_df = input_df.drop(columns=['TransactionDate'])
    
    # Convert categorical columns to numerical using one-hot encoding
    input_df = pd.get_dummies(input_df, columns=['TransactionType', 'Location'], drop_first=True)
    
    # Ensure the input data has the same columns as the training data
    # Create a DataFrame with the same columns as the training data
    training_columns = model.feature_names_in_
    input_df = input_df.reindex(columns=training_columns, fill_value=0)
    
    # Make prediction
    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)
    
    # Convert to native Python types
    prediction = int(prediction[0])
    fraud_probability = float(probability[0][1])
    
    return prediction, fraud_probability

@app.route('/predict', methods=['POST'])
def predict_fraud():
    try:
        data = request.get_json()
        print(data)
        print(type(data['Amount']))
        prediction, fraud_probability = preprocess_and_predict(data)
        return jsonify({'prediction': prediction, 'fraud_probability': fraud_probability})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)