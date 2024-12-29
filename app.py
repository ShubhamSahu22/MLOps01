import pickle
import os
from flask import Flask, request, render_template, flash, redirect, url_for

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Required for using Flask's flash messages

# Load the model
MODEL_PATH = "model/iris_model.pkl"
if not os.path.exists(MODEL_PATH):
    raise Exception(
        "Model file not found. Make sure to train the model by running 'train.py'."
    )

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# Home route to display the form
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        try:
            # Get the input features from the form
            features = [
                float(request.form["sepal_length"]),
                float(request.form["sepal_width"]),
                float(request.form["petal_length"]),
                float(request.form["petal_width"]),
            ]

            # Make a prediction using the model
            prediction = model.predict([features])[0]

            # Map the numeric class to the iris class name
            iris_classes = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}
            predicted_class = iris_classes.get(prediction, "Unknown")

            return render_template(
                "index.html", prediction_text=f"Predicted Iris Class: {predicted_class}"
            )

        except ValueError:
            flash("Invalid input. Please enter numeric values for all features.", "error")
            return redirect(url_for("home"))

    return render_template("index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

