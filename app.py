import pickle
import os
import logging
from flask import Flask, request, render_template, flash, redirect, url_for, jsonify

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Required for using Flask's flash messages

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load the model
MODEL_PATH = "model/iris_model.pkl"

def load_model():
    """Loads the ML model from file."""
    if not os.path.exists(MODEL_PATH):
        logging.error("Model file not found. Please train the model first.")
        raise FileNotFoundError("Model file not found. Train the model using 'train.py'.")
    
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

try:
    model = load_model()
    logging.info("Model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading model: {e}")
    model = None

# Define class mapping
IRIS_CLASSES = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}

def predict_iris(features):
    """Predicts the iris class based on input features."""
    try:
        prediction = model.predict([features])[0]
        return IRIS_CLASSES.get(prediction, "Unknown")
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return "Error in prediction"

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        try:
            # Get input from form
            features = [
                float(request.form["sepal_length"]),
                float(request.form["sepal_width"]),
                float(request.form["petal_length"]),
                float(request.form["petal_width"]),
            ]
            predicted_class = predict_iris(features)
            return render_template("index.html", prediction_text=f"Predicted Iris Class: {predicted_class}")
        except ValueError:
            flash("Invalid input. Please enter numeric values for all features.", "error")
            return redirect(url_for("home"))
    
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict_api():
    """API endpoint for model prediction."""
    try:
        data = request.get_json()
        features = [
            float(data["sepal_length"]),
            float(data["sepal_width"]),
            float(data["petal_length"]),
            float(data["petal_width"]),
        ]
        predicted_class = predict_iris(features)
        return jsonify({"prediction": predicted_class})
    except (ValueError, KeyError):
        return jsonify({"error": "Invalid input format. Please provide all four feature values."}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
