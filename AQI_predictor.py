import requests
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from typing import Dict, Any, List, Optional, Tuple

class WeatherAQIPredictor:
    """
    A class to fetch weather and air quality data, and predict future AQI.
    """
    def _init_(self, api_key: str):
        """
        Initializes the predictor with the OpenWeatherMap API key.
        """
        self.api_key = api_key
        self.base_url_weather = "http://api.openweathermap.org/data/2.5/weather"
        self.base_url_forecast = "http://api.openweathermap.org/data/2.5/forecast"
        self.base_url_aqi = "http://api.openweathermap.org/data/2.5/air_pollution"
        self.base_url_geo = "http://api.openweathermap.org/geo/1.0/direct"
        self.historical_data = []

    # ==============================================================================
    # 1. DATA INGESTION
    # - Collects historical air quality data from various sources (e.g., API).
    # - Preprocesses the data by handling missing values, outliers, etc.
    # ==============================================================================

    def get_coordinates(self, city: str) -> Optional[Tuple[float, float]]:
        """Gets latitude and longitude for a given city."""
        params = {'q': city, 'limit': 1, 'appid': self.api_key}
        try:
            response = requests.get(self.base_url_geo, params=params)
            response.raise_for_status()  # Raises an HTTPError for bad responses
            data = response.json()
            if data:
                return data[0]['lat'], data[0]['lon']
            else:
                print(f"Error: Could not find coordinates for {city}.")
                return None
        except requests.exceptions.RequestException as e:
            print(f"Error fetching coordinates: {e}")
            return None

    def fetch_current_weather(self, city: str) -> Optional[Dict[str, Any]]:
        """Fetches current weather data for a city."""
        params = {'q': city, 'appid': self.api_key, 'units': 'metric'}
        try:
            response = requests.get(self.base_url_weather, params=params)
            response.raise_for_status()
            data = response.json()
            return {
                'city': data['name'],
                'temperature': data['main']['temp'],
                'humidity': data['main']['humidity'],
                'pressure': data['main']['pressure'],
                'wind_speed': data['wind']['speed'],
                'description': data['weather'][0]['description'],
            }
        except requests.exceptions.RequestException as e:
            print(f"Error fetching current weather: {e}")
            return None

    def fetch_current_aqi(self, lat: float, lon: float) -> Optional[Dict[str, Any]]:
        """Fetches current Air Quality Index (AQI) data."""
        params = {'lat': lat, 'lon': lon, 'appid': self.api_key}
        try:
            response = requests.get(self.base_url_aqi, params=params)
            response.raise_for_status()
            data = response.json()['list'][0]
            return {
                'aqi': data['main']['aqi'],
                'pm2_5': data['components']['pm2_5'],
                'pm10': data['components']['pm10'],
                'co': data['components']['co'],
            }
        except requests.exceptions.RequestException as e:
            print(f"Error fetching AQI data: {e}")
            return None

    def fetch_weather_forecast(self, city: str) -> Optional[List[Dict[str, Any]]]:
        """Fetches 5-day weather forecast (at 3-hour intervals)."""
        params = {'q': city, 'appid': self.api_key, 'units': 'metric'}
        try:
            response = requests.get(self.base_url_forecast, params=params)
            response.raise_for_status()
            data = response.json()
            forecasts = []
            for item in data['list']:
                forecasts.append({
                    'timestamp': datetime.fromtimestamp(item['dt']),
                    'temperature': item['main']['temp'],
                    'humidity': item['main']['humidity'],
                    'pressure': item['main']['pressure'],
                    'wind_speed': item['wind']['speed'],
                })
            return forecasts
        except requests.exceptions.RequestException as e:
            print(f"Error fetching weather forecast: {e}")
            return None
            
    # ==============================================================================
    # 2. FEATURE ENGINEERING
    # - Extracts relevant features from the preprocessed data.
    # - Creates new features that might be useful for prediction.
    # ==============================================================================
    
    def prepare_data_for_model(self) -> Optional[Tuple[pd.DataFrame, pd.Series]]:
        """
        Prepares the collected historical data for model training.
        This is where preprocessing and feature engineering would occur.
        """
        if len(self.historical_data) < 10:
            print("Not enough historical data for prediction. Need at least 10 readings.")
            return None
        
        df = pd.DataFrame(self.historical_data)

        # --- Preprocessing (Example) ---
        # Handle missing values if any, though our current collection method doesn't produce them.
        df.dropna(inplace=True)

        # --- Feature Extraction ---
        # Select the features (independent variables) and the target (dependent variable).
        features = ['temperature', 'humidity', 'pressure', 'wind_speed']
        target = 'aqi'
        
        # --- New Feature Creation (Example) ---
        # In a real-world scenario with timestamps, you could add features like:
        # df['hour_of_day'] = df['timestamp'].dt.hour
        # df['day_of_week'] = df['timestamp'].dt.dayofweek
        # df['temp_humidity_interaction'] = df['temperature'] * df['humidity']
        
        X = df[features]
        y = df[target]
        
        return X, y

    # ==============================================================================
    # 3. MODEL TRAINING & PREDICTION
    # ==============================================================================

    def train_aqi_model(self) -> Optional[RandomForestRegressor]:
        """Trains a model to predict AQI based on historical data."""
        prepared_data = self.prepare_data_for_model()
        if prepared_data is None:
            return None
        
        X, y = prepared_data
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Initialize and train the model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate the model's performance on the test set
        score = model.score(X_test, y_test)
        print(f"Model R² Score: {score:.3f}")
        
        return model

    def predict_future_aqi(self, model: RandomForestRegressor, future_weather: Dict[str, Any]) -> Optional[int]:
        """Predicts future AQI using a trained model and future weather data."""
        if model is None:
            print("Model is not trained. Cannot make a prediction.")
            return None
            
        # Ensure the input data has the same feature structure
        features = ['temperature', 'humidity', 'pressure', 'wind_speed']
        X_pred = pd.DataFrame([future_weather])[features]
        
        predicted_aqi = model.predict(X_pred)[0]
        return round(predicted_aqi)


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    # IMPORTANT: Replace with your own OpenWeatherMap API key.
    API_KEY = "ba2346b5c1a8f36abc1f72abcae9c739" 
    
    predictor = WeatherAQIPredictor(api_key=API_KEY)
    city = "Delhi"
    
    # --- 1. Fetch Current Weather ---
    print("=" * 50)
    print(f"FETCHING CURRENT WEATHER FOR {city.upper()}")
    print("=" * 50)
    current_weather = predictor.fetch_current_weather(city)
    if current_weather:
        print(f"Temperature: {current_weather['temperature']}°C")
        print(f"Humidity: {current_weather['humidity']}%")
        print(f"Description: {current_weather['description']}")

    # --- 2. Fetch Current AQI ---
    print("\n" + "=" * 50)
    print(f"FETCHING CURRENT AIR QUALITY FOR {city.upper()}")
    print("=" * 50)
    coords = predictor.get_coordinates(city)
    if coords:
        lat, lon = coords
        current_aqi = predictor.fetch_current_aqi(lat, lon)
        if current_aqi:
            aqi_labels = {1: "Good", 2: "Fair", 3: "Moderate", 4: "Poor", 5: "Very Poor"}
            aqi_level = current_aqi['aqi']
            print(f"AQI Level: {aqi_level} ({aqi_labels.get(aqi_level, 'Unknown')})")
            print(f"PM2.5: {current_aqi['pm2_5']} μg/m³")

    # --- 3. AQI Prediction Demo ---
    print("\n" + "=" * 50)
    print("AQI PREDICTION DEMO")
    print("=" * 50)
    print("NOTE: Using simulated historical data for this demonstration.")
    
    # Simulate collecting data over time for demonstration purposes.
    # In a real application, you would collect and store this data periodically.
    for _ in range(50):
        simulated_data = {
            'temperature': 25 + np.random.randn() * 5,
            'humidity': 50 + np.random.randn() * 15,
            'pressure': 1010 + np.random.randn() * 5,
            'wind_speed': 5 + np.random.randn() * 2,
            'aqi': np.random.randint(1, 6) # Random AQI between 1 (Good) and 5 (Very Poor)
        }
        predictor.historical_data.append(simulated_data)
        
    # Train the model on the simulated historical data
    model = predictor.train_aqi_model()
    
    # Predict AQI based on the current weather conditions
    if model and current_weather:
        predicted_value = predictor.predict_future_aqi(model, current_weather)
        print(f"\nPredicted AQI based on current weather: {predicted_value}")

    # --- 4. Weather Forecast ---
    print("\n" + "=" * 50)
    print("WEATHER FORECAST (Next 12 hours)")
    print("=" * 50)
    forecasts = predictor.fetch_weather_forecast(city)
    if forecasts:
        for forecast in forecasts[:4]: # Show the next 4 intervals (12 hours)
            print(f"{forecast['timestamp'].strftime('%Y-%m-%d %H:%M')}: {forecast['temperature']:.1f}°C")

if _name_ == "_main_":
    main()