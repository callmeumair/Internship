from flask import Flask, request, jsonify
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = joblib.load('models/churn_model.joblib')

@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint
    """
    return jsonify({'status': 'healthy'})

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint for making predictions
    """
    try:
        # Get data from request
        data = request.get_json()
        
        # Extract features
        features = {
            'age': float(data.get('age')),
            'tenure': float(data.get('tenure')),
            'monthly_charges': float(data.get('monthly_charges')),
            'total_charges': float(data.get('total_charges')),
            'gender': data.get('gender'),
            'internet_service': data.get('internet_service'),
            'contract': data.get('contract'),
            'payment_method': data.get('payment_method')
        }
        
        # Preprocess the input (similar to training preprocessing)
        processed_features = preprocess_input(features)
        
        # Make prediction
        prediction = model.predict_proba(processed_features.reshape(1, -1))[0]
        
        # Return prediction
        response = {
            'churn_probability': float(prediction[1]),
            'prediction': 'Likely to Churn' if prediction[1] > 0.5 else 'Not Likely to Churn'
        }
        
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        return jsonify({'error': str(e)}), 400

def preprocess_input(features):
    """
    Preprocess the input features to match the training data format
    """
    # Create feature vector (this should match your training preprocessing)
    # This is a simplified version - you should match your training preprocessing exactly
    feature_vector = np.array([
        features['age'],
        features['tenure'],
        features['monthly_charges'],
        features['total_charges']
    ])
    
    # Scale numerical features
    scaler = StandardScaler()
    feature_vector = scaler.fit_transform(feature_vector.reshape(1, -1))
    
    return feature_vector.flatten()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True) 