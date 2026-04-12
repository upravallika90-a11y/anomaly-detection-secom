from flask import Flask, request, jsonify
import pickle
import numpy as np

# Initialize app
app = Flask(__name__)

# Load model
model = pickle.load(open("model/model.pkl", "rb"))

@app.route("/")
def home():
    return "Anomaly Detection API is running!"

# Prediction API
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Convert input to numpy array
        features = np.array(data["features"]).reshape(1, -1)

        prediction = model.predict(features)

        # Convert output
        result = "Anomaly" if prediction[0] == -1 else "Normal"

        return jsonify({
            "prediction": result
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        })

if __name__ == "__main__":
    app.run(debug=True)