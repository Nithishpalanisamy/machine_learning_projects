from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd

# Load the model
with open('house_price_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load dataset for feature reference
data = pd.read_csv('HousingData.csv')
feature_names = data.drop('MEDV', axis=1).columns.tolist()

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        input_data = np.array([data['features']])
        
        # Check if the input features match expected dimensions
        if len(input_data[0]) != len(feature_names):
            return jsonify({'error': 'Invalid number of features. Expected {}'.format(len(feature_names))})
        
        prediction = model.predict(input_data)
        return jsonify({'prediction': prediction[0]})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
