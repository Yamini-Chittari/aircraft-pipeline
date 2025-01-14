from model.prediction import maintenance_model, lifespan_model
from model.recommendation import generate_recommendation
import mlflow
import joblib
import pandas as pd
import re

# Function to sanitize parameter names
def sanitize_param_name(param_name):
    # Replace invalid characters with underscores
    return re.sub(r'[^a-zA-Z0-9_ ]', '_', param_name)

# Load the pre-trained models (assuming they are saved as .pkl files)
maintenance_model = joblib.load('models/maintenance_classifier_model.pkl')
lifespan_model = joblib.load('models/remaining_life_regressor_model.pkl')

# Ensure the experiment is created or set up (if not already done)
mlflow.set_experiment('aircraft_maintenance_experiment')

# Log the models to MLflow
with mlflow.start_run():
    mlflow.sklearn.log_model(maintenance_model, artifact_path="maintenance_classifier_model")
    mlflow.sklearn.log_model(lifespan_model, artifact_path="remaining_life_regressor_model")
print('Models logged successfully.')

# Simulate input data (you can use real data here)
user_data = pd.DataFrame({
    'Temperature (°C)': [50],
    'Pressure (bar)': [10],
    'Vibration (mm/s)': [1],
    'Working_Hours': [1000]
})

# Make predictions using the models
predicted_maintenance = maintenance_model.predict(user_data)[0]
predicted_lifespan = lifespan_model.predict(user_data)[0]

# Generate recommendation
failure_type = "None" if predicted_maintenance == 0 else "Turbine Overheat"
flight_type = "commercial"
flight_number = 1234
recommendation = generate_recommendation(failure_type, flight_type, flight_number, predicted_lifespan)

# Log the prediction to MLflow
with mlflow.start_run():
    mlflow.log_param(sanitize_param_name("Temperature (°C)"), user_data['Temperature (°C)'][0])
    mlflow.log_param(sanitize_param_name("Pressure (bar)"), user_data['Pressure (bar)'][0])
    mlflow.log_param(sanitize_param_name("Vibration (mm/s)"), user_data['Vibration (mm/s)'][0])
    mlflow.log_param(sanitize_param_name("Working Hours"), user_data['Working_Hours'][0])
    mlflow.log_param("Prediction", recommendation)

print('Prediction logged successfully.')
