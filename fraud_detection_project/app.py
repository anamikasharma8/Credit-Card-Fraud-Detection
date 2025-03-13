from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = joblib.load("fraud_model_det.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json()

        # Convert to DataFrame
        df = pd.DataFrame(data, index=[0])  

        # Make prediction
        prediction = model.predict(df)[0]

        # Return response
        if prediction == 1:
            result = {"status": "🚨 Fraud Detected!"}
        else:
            result = {"status": "✅ Legit Transaction"}

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
