import random
import joblib
import pandas as pd
import mlflow
from model.recommendation import generate_recommendation
from mlflow_utils.mlflow_logging import log_model, log_prediction

# Define the experiment name
experiment_name = "aircraft_maintenance_experiment"

# Create or set the experiment
mlflow.set_experiment(experiment_name)

# Load the models
maintenance_model = joblib.load('models/maintenance_classifier_model.pkl')
lifespan_model = joblib.load('models/remaining_life_regressor_model.pkl')

def make_predictions(input_data):
    """
    Make predictions based on input data and generate recommendations.

    Args:
        input_data (pd.DataFrame): Input data with columns ['Temperature (°C)', 'Pressure (bar)', 'Vibration (mm/s)', 'Working_Hours'].

    Returns:
        dict: Generated recommendation based on prediction results.
    """
    # Ensure input_data has the correct columns
    X_input = input_data[['Temperature (°C)', 'Pressure (bar)', 'Vibration (mm/s)', 'Working_Hours']]

    # Predict maintenance and lifespan using the models
    predicted_maintenance = maintenance_model.predict(X_input)[0]
    predicted_lifespan = lifespan_model.predict(X_input)[0]

    # Generate failure type based on predicted maintenance
    failure_type = "None" if predicted_maintenance == 0 else random.choice(
        ['Turbine Overheat', 'Pressure Loss', 'Valve Stuck'])

    # Generate random flight type and flight number
    flight_type = random.choice(['commercial', 'cargo', 'private', 'long-haul', 'short-haul'])
    flight_number = random.randint(1000, 9999)

    # Generate recommendation using the failure type, flight type, and predicted lifespan
    recommendation = generate_recommendation(failure_type, flight_type, flight_number, predicted_lifespan)

    # Log the prediction details with MLflow
    log_prediction(input_data, recommendation, "Maintenance Prediction Model")

    return recommendation

# Sample input data to test prediction
input_data = pd.DataFrame({
    'Temperature (°C)': [75],
    'Pressure (bar)': [10],
    'Vibration (mm/s)': [0.5],
    'Working_Hours': [1500]
})

# Call the prediction function
recommendation = make_predictions(input_data)

# Print the recommendation
print(recommendation)
