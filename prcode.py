import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib

# Load the dataset from the CSV file
file_path = 'Sleep_health_and_lifestyle_dataset.csv'
df = pd.read_csv(file_path)

# Select relevant columns including the "Occupation" column
selected_columns = ['Age', 'Gender', 'DailySteps', 'Occupation', 'Sleep Disorder']
df = df[selected_columns]

# Drop rows with missing values in the selected columns
df.dropna(subset=selected_columns, inplace=True)

# Convert categorical variables to numerical using label encoding
label_encoder_gender = LabelEncoder()
label_encoder_sleep_disorder = LabelEncoder()
label_encoder_occupation = LabelEncoder()  # Add label encoder for Occupation

df['Gender'] = label_encoder_gender.fit_transform(df['Gender'])
df['Sleep Disorder'] = label_encoder_sleep_disorder.fit_transform(df['Sleep Disorder'])
df['Occupation'] = label_encoder_occupation.fit_transform(df['Occupation'])  # Encode Occupation

# Split the data into features (X) and target variable (y)
X = df[['Age', 'Gender', 'DailySteps', 'Occupation']]
y = df['Sleep Disorder']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a RandomForestClassifier model
model = RandomForestClassifier()

# Train the model
model.fit(X_train, y_train)

# Print message indicating that the model is trained
print('Model trained successfully!')

# Evaluate the model on the test set
y_pred = model.predict(X_test)

# Display accuracy and classification report
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
print('Classification Report:')
print(classification_report(y_test, y_pred))

# Interactive prediction loop
while True:
    # Take user input for new data
    age = float(input("Enter Age: "))
    gender = input("Enter Gender (e.g., 'Male', 'Female'): ")
    daily_steps = float(input("Enter Daily Steps: "))
    occupation = input("Enter Occupation: ")

    # Encode categorical variables
    gender_encoded = label_encoder_gender.transform([gender])[0]
    occupation_encoded = label_encoder_occupation.transform([occupation])[0]

    # Prepare input data for prediction
    new_data = [[age, gender_encoded, daily_steps, occupation_encoded]]

    # Make predictions for new data
    prediction = model.predict(new_data)

    # Mapping numeric labels to categories
    predicted_category = label_encoder_sleep_disorder.inverse_transform(prediction)

    print(f'Predicted Sleep Disorder: {predicted_category[0]}')

    # Ask the user if they want to make another prediction
    another_prediction = input("Do you want to make another prediction? (yes/no): ")
    if another_prediction.lower() != 'yes':
        break

# Save the trained model
model_file_path = "random_forest_model.pkl"
full_model_path = os.path.join(os.getcwd(), model_file_path)
try:
    joblib.dump(model, full_model_path)
    print("Model saved successfully as 'random_forest_model.pkl'")
except Exception as e:
    print(f"An error occurred while saving the model: {e}")
