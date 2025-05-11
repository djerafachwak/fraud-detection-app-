from flask import Flask, request, jsonify
import pandas as pd
import joblib
from Preprocessing import full_preprocessing


app = Flask(__name__)
model = joblib.load("Fraud_det.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    data = pd.read_excel(file)
    processed_data = full_preprocessing(data)
    predictions = model.predict(processed_data)

    return jsonify({"predictions": predictions.tolist()})

if __name__ == '__main__':
    app.run(debug=True, port=5000)