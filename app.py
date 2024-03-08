from flask import Flask, request, jsonify
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load the pre-trained model
model_file_path = 'random_forest_model.pkl'
try:
    model = joblib.load(model_file_path)
except Exception as e:
    print(f"An error occurred while loading the model: {e}")
    model = None

# Load label encoders
label_encoder_gender = LabelEncoder()
label_encoder_sleep_disorder = LabelEncoder()
label_encoder_occupation = LabelEncoder()  # Add label encoder for Occupation

# Load the dataset (for label encoder fitting)
file_path = 'Sleep_health_and_lifestyle_dataset.csv'
df = pd.read_csv(file_path)

# Fit label encoders
label_encoder_gender.fit(df['Gender'])
label_encoder_sleep_disorder.fit(df['Sleep Disorder'])
label_encoder_occupation.fit(df['Occupation'])  # Fit label encoder for Occupation

# Endpoint for prediction
@app.route('/predictsleep', methods=['POST'])
def predict_sleep():
    if model is None:
        return jsonify({'error': 'Model not loaded!'}), 500

    data = request.json
    app.logger.info("Received data: %s", data)

    # Check if all required fields are present in the input data
    required_fields = ['Age', 'Gender', 'DailySteps', 'Occupation']
    if not all(field in data for field in required_fields):
        return jsonify({'error': 'Missing required fields'}), 400

    try:
        # Prepare input data for prediction
        age = float(data['Age'])
        gender = data['Gender']
        daily_steps = float(data['DailySteps'])
        occupation = data['Occupation']

        app.logger.info("Received input: Age=%s, Gender=%s, DailySteps=%s, Occupation=%s", age, gender, daily_steps, occupation)

        # Convert categorical variables to numerical using label encoding
        gender_encoded = label_encoder_gender.transform([gender])[0]
        occupation_encoded = label_encoder_occupation.transform([occupation])[0]

        app.logger.info("Encoded Gender: %s", gender_encoded)
        app.logger.info("Encoded Occupation: %s", occupation_encoded)

        # Make predictions for new data
        prediction = model.predict([[age, gender_encoded, daily_steps, occupation_encoded]])

        app.logger.info("Prediction: %s", prediction)

        # Mapping numeric labels to categories
        predicted_category = label_encoder_sleep_disorder.inverse_transform(prediction)

        app.logger.info("Predicted category: %s", predicted_category)

        return jsonify({'predicted_sleep_disorder': predicted_category[0]})
    except Exception as e:
        app.logger.error("Prediction error: %s", e)
        return jsonify({'error': f'Prediction error: {e}'}), 400

# Route for the root URL
@app.route('/')
def index():
    return 'Welcome to the Sleep Disorder Prediction API!'

if __name__ == '__main__':
    app.run(debug=True)
