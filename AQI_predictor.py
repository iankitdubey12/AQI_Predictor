import requests
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import tkinter as tk
from tkinter import ttk, messagebox

class WeatherAQIPredictor:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url_geo = "http://api.openweathermap.org/geo/1.0/direct"
        self.base_url_aqi_forecast = "http://api.openweathermap.org/data/2.5/air_pollution/forecast"
        self.base_url_aqi_history = "http://api.openweathermap.org/data/2.5/air_pollution/history"
        self.base_url_aqi_current = "http://api.openweathermap.org/data/2.5/air_pollution"

    def get_coordinates(self, city: str) -> Optional[Tuple[float, float]]:
        params = {'q': city, 'limit': 1, 'appid': self.api_key}
        try:
            response = requests.get(self.base_url_geo, params=params)
            response.raise_for_status()
            data = response.json()
            if data:
                return data[0]['lat'], data[0]['lon']
            return None
        except requests.exceptions.RequestException as e:
            print(f"Error fetching coordinates: {e}")
            return None

    def fetch_aqi_history(self, lat: float, lon: float, days: int = 3) -> Optional[List[Dict[str, Any]]]:
        end_time = int(datetime.now().timestamp())
        start_time = int((datetime.now() - timedelta(days=days)).timestamp())
        params = {'lat': lat, 'lon': lon, 'start': start_time, 'end': end_time, 'appid': self.api_key}
        try:63
            response = requests.get(self.base_url_aqi_history, params=params)
            response.raise_for_status()
            data = response.json()
            history = []
            for entry in data.get("list", []):
                history.append({
                    "timestamp": datetime.fromtimestamp(entry["dt"]),
                    "aqi": entry["main"]["aqi"],
                    "pm2_5": entry["components"]["pm2_5"],
                    "pm10": entry["components"]["pm10"],
                    "co": entry["components"]["co"],
                })
            return history
        except requests.exceptions.RequestException as e:
            print(f"Error fetching AQI history: {e}")
            return None

    def fetch_aqi_current(self, lat: float, lon: float) -> Optional[Dict[str, Any]]:
        params = {'lat': lat, 'lon': lon, 'appid': self.api_key}
        try:
            response = requests.get(self.base_url_aqi_current, params=params)
            response.raise_for_status()
            data = response.json()
            entries = data.get("list", [])
            if not entries:
                return None
            entry = entries[0]
            return {
                "timestamp": datetime.fromtimestamp(entry["dt"]),
                "aqi": entry["main"]["aqi"],
                "pm2_5": entry["components"]["pm2_5"],
                "pm10": entry["components"]["pm10"],
                "co": entry["components"]["co"],
            }
        except requests.exceptions.RequestException as e:
            print(f"Error fetching current AQI: {e}")
            return None

    def fetch_aqi_forecast(self, lat: float, lon: float) -> Optional[List[Dict[str, Any]]]:
        params = {'lat': lat, 'lon': lon, 'appid': self.api_key}
        try:
            response = requests.get(self.base_url_aqi_forecast, params=params)
            response.raise_for_status()
            data = response.json()
            forecasts = []
            for entry in data.get("list", []):
                forecasts.append({
                    "timestamp": datetime.fromtimestamp(entry["dt"]),
                    "aqi": entry["main"]["aqi"],
                    "pm2_5": entry["components"]["pm2_5"],
                    "pm10": entry["components"]["pm10"],
                    "co": entry["components"]["co"],
                })
            return forecasts
        except requests.exceptions.RequestException as e:
            print(f"Error fetching AQI forecast: {e}")
            return None
        
class AQIGUI:
    def __init__(self, root, api_key):
        self.root = root
        self.root.title("Air Quality Index (AQI) Predictor")
        self.predictor = WeatherAQIPredictor(api_key)

        self.city_label = ttk.Label(root, text="Enter City:")
        self.city_label.grid(row=0, column=0, padx=5, pady=5)
        self.city_entry = ttk.Entry(root, width=30)
        self.city_entry.grid(row=0, column=1, padx=5, pady=5)

        self.fetch_button = ttk.Button(root, text="Fetch AQI", command=self.fetch_and_display)
        self.fetch_button.grid(row=0, column=2, padx=5, pady=5)

        self.text = tk.Text(root, width=100, height=20)
        self.text.grid(row=1, column=0, columnspan=3, padx=10, pady=10)

    def fetch_and_display(self):
        city = self.city_entry.get().strip()
        if not city:
            messagebox.showerror("Input Error", "Please enter a city name.")
            return

        coords = self.predictor.get_coordinates(city)
        if not coords:
            messagebox.showerror("Error", f"Could not find coordinates for '{city}'.")
            return

        lat, lon = coords
        history_data = self.predictor.fetch_aqi_history(lat, lon, days=3) or []
        current_data = self.predictor.fetch_aqi_current(lat, lon)
        current_list = [current_data] if current_data else []
        forecast_data = self.predictor.fetch_aqi_forecast(lat, lon) or []
        forecast_one_day = forecast_data[:24]
        all_data = history_data + current_list + forecast_one_day

        if not all_data:
            messagebox.showinfo("No Data", "No AQI data available.")
            return

        df = pd.DataFrame(all_data)
        df["date"] = df["timestamp"].dt.date
        aqi_labels = {1: "Good", 2: "Fair", 3: "Moderate", 4: "Poor", 5: "Very Poor"}

        self.text.delete(1.0, tk.END)
        self.text.insert(tk.END, f"Air Quality Data for {city.title()}:\n")
        self.text.insert(tk.END, "-"*80 + "\n")
        for entry in all_data:
            ts = entry["timestamp"].strftime("%Y-%m-%d %H:%M")
            self.text.insert(tk.END,
                             f"{ts} → AQI {entry['aqi']} ({aqi_labels.get(entry['aqi'], 'Unknown')}) | "
                             f"PM2.5={entry['pm2_5']} µg/m³ | PM10={entry['pm10']} µg/m³ | CO={entry['co']} µg/m³\n")

        daily_avg = df.groupby("date")["aqi"].mean().round(1)
        plt.figure(figsize=(10, 5))
        plt.plot(daily_avg.index, daily_avg.values, marker='o', color='crimson', linewidth=2)
        plt.title(f"AQI for Past 3 Days, Today, and Next Day - {city.title()}", fontsize=14, fontweight='bold')
        plt.xlabel("Date", fontsize=12)
        plt.ylabel("Average AQI (1–5)", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        for i, val in enumerate(daily_avg.values):
            plt.text(daily_avg.index[i], val + 0.1, str(val), ha='center', fontsize=10)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    API_KEY = "ba2346b5c1a8f36abc1f72abcae9c739"  
    root = tk.Tk()
    gui = AQIGUI(root, API_KEY)
    root.mainloop()
