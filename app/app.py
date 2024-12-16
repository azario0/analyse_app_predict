from flask import Flask, render_template, request
import joblib
import pandas as pd
from datetime import datetime

app = Flask(__name__)

# Load models and encoders
rf_model = joblib.load('rf_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoders = joblib.load('label_encoders.pkl')

# Get label encoder classes
le_location = label_encoders['location_id']
le_season = label_encoders['season']
le_peak = label_encoders['peak_hour_flag']
le_sensor = label_encoders['sensor_noise_flag']
le_allocation = label_encoders['resource_allocation']

@app.route('/')
def index():
    # Get unique values for dropdowns
    locations = le_location.classes_
    seasons = le_season.classes_
    return render_template('index.html', locations=locations, seasons=seasons)

@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve form data
    timestamp = request.form['timestamp']
    location = request.form['location_id']
    visitor_count = float(request.form['visitor_count'])
    temperature = float(request.form['temperature'])
    air_quality = float(request.form['air_quality_index'])
    noise_level = float(request.form['noise_level'])
    visitor_satisfaction = float(request.form['visitor_satisfaction'])
    season = request.form['season']
    peak = request.form['peak_hour_flag']
    sensor = request.form['sensor_noise_flag']
    allocation = request.form['resource_allocation']
    
    # Process timestamp
    dt = datetime.fromisoformat(timestamp)
    hour = dt.hour
    dayofweek = dt.weekday()
    
    # Encode categorical variables
    location_encoded = le_location.transform([location])[0]
    season_encoded = le_season.transform([season])[0]
    peak_encoded = le_peak.transform([peak])[0]
    sensor_encoded = le_sensor.transform([sensor])[0]
    allocation_encoded = le_allocation.transform([allocation])[0]
    
    # Create feature array
    features = [
        hour, dayofweek, visitor_count, temperature, air_quality,
        noise_level, visitor_satisfaction, 0, 0,  # t_sne_dim1 and t_sne_dim2 set to 0
        location_encoded, season_encoded, peak_encoded, sensor_encoded, allocation_encoded
    ]
    
    # Scale features
    features_scaled = scaler.transform([features])
    
    # Make prediction
    prediction = rf_model.predict(features_scaled)[0]
    
    return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)