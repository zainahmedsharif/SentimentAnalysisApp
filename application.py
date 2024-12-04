from flask import Flask, request, jsonify
import joblib
import boto3
import os

# Initialize the Flask app
application = Flask(__name__)

# S3 Configuration
BUCKET_NAME = "sagemaker-studio-314146328089-acwxlhob99r"
MODEL_KEY = "models/naive_bayes_model.joblib"
LOCAL_MODEL_PATH = "model.joblib"

# Function to download and load the model
def load_model_locally():
    """
    Loads the model from a local file using joblib.

    Assumes the model is already in .joblib format and present in the same directory.
    """
    import joblib
    import os

    # Define the local model path
    local_file_name = "naive_bayes_model.joblib"

    # Check if the model file exists
    if not os.path.exists(local_file_name):
        raise FileNotFoundError(f"Model file {local_file_name} not found in the local directory.")

    print("Loading the model from local file...")
    # Load the joblib model
    model = joblib.load(local_file_name)

    print("Model loaded successfully.")
    return model

# Load the model at startup
model_pipeline = load_model_locally()

# Label mapping
label_mapping = {0: "negative", 1: "positive"}

@application.route('/predict', methods=['POST'])
def predict_sentiment():
    try:
        # Get text input from the request
        data = request.get_json()
        input_text = data.get('text', '')

        # Ensure text is provided
        if not input_text:
            return jsonify({"error": "Text input is required"}), 400

        # Predict sentiment
        prediction = model_pipeline.predict([input_text])
        sentiment = label_mapping.get(prediction[0], "unknown")

        # Return prediction as JSON
        return jsonify({"sentiment": sentiment})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the app
if __name__ == '__main__':
    application.run(host="0.0.0.0", port=8080)