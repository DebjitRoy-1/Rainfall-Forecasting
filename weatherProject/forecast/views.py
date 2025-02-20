from django.shortcuts import render
from django.http import HttpRequest, HttpResponse
import requests
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_squared_error
from datetime import datetime, timedelta
import pytz
import os
import logging

API_KEY = 'db5885c16b69e40376c7bb6b887a18da'
BASE_URL = 'https://api.openweathermap.org/data/2.5/'

# Configure logging for better accuracy tracking
logging.basicConfig(level=logging.DEBUG)

def get_current_weather(city):
    url = f"{BASE_URL}weather?q={city}&appid={API_KEY}&units=metric"
    response = requests.get(url)
    data = response.json()

    if response.status_code != 200 or 'main' not in data:
        raise ValueError(f"Error fetching weather data for {city}")

    # Ensure high precision with rounded values
    return {
        'city': data['name'],
        'current_temp': round(data['main']['temp'],),
        'feels_like': round(data['main']['feels_like'],),
        'temp_min': round(data['main']['temp_min'], 2),
        'temp_max': round(data['main']['temp_max'], 2),
        'humidity': round(data['main']['humidity'], 1),
        'description': data['weather'][0]['description'],
        'country': data['sys']['country'],
        'wind_gust_dir': data['wind']['deg'],
        'pressure': round(data['main']['pressure'], 2),  # Pressure with more precision
        'Wind_Gust_Speed': round(data['wind']['speed'], 2),  # Wind speed with 2 decimal places
        'clouds': data['clouds']['all'],
        'Visibility': data['visibility'],  # Visibility in meters
        'timezone': data['timezone'],  # Timezone offset in seconds
    }

def read_historical_data(filename):
    df = pd.read_csv(filename)
    df = df.dropna()
    df = df.drop_duplicates()

    if df.empty:
        raise ValueError("The dataset is empty after cleaning.")

    # Check for significant outliers or incorrect data in historical dataset
    logging.debug(f"Data shape after cleaning: {df.shape}")
    
    return df

def prepare_data(data):
    le = LabelEncoder()
    try:
        data['WindGustDir'] = le.fit_transform(data['WindGustDir'])
        data['RainTomorrow'] = le.fit_transform(data['RainTomorrow'])
    except ValueError as e:
        logging.error(f"Label encoding failed: {e}")
        raise ValueError("Label encoding failed.")

    return data[['MinTemp', 'MaxTemp', 'WindGustDir', 'WindGustSpeed', 'Humidity', 'Pressure', 'Temp']], data['RainTomorrow'], le

def train_rain_model(X, y):
    if X.size == 0 or y.size == 0:
        raise ValueError("Input data is empty for training the model.")
    
    x_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=200, random_state=42)  # Increased estimators for stability
    model.fit(x_train, y_train)
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    logging.debug("Mean Squared Error for Rain Model: %.2f", mse)
    
    return model

def prepare_regression_data(data, feature):
    X, y = [], []
    for i in range(len(data) - 1):
        X.append(data[feature].iloc[i])
        y.append(data[feature].iloc[i + 1])
    
    X = np.array(X).reshape(-1, 1)
    y = np.array(y)
    
    return X, y

def train_regression_model(X, y):
    if X.size == 0 or y.size == 0:
        raise ValueError("Input data for regression model is empty.")
    
    model = RandomForestRegressor(n_estimators=200, random_state=42)  # More estimators for better accuracy
    model.fit(X, y)
    
    return model

def predict_future(model, current_value):
    predictions = [current_value]
    for i in range(5):
        next_value = model.predict(np.array([[predictions[-1]]]))
        predictions.append(next_value[0])
    
    return predictions[1:]

def weather_view(request: HttpRequest):
    try:
        if request.method == 'POST':
            city = request.POST.get('city')
            current_weather = get_current_weather(city)

            # Load historical data
            csv_path = os.path.join('../weather.csv')  # Ensure the path to the CSV is correct
            historical_data = read_historical_data(csv_path)

            # Prepare and train the rain prediction model
            X, y, le = prepare_data(historical_data)
            rain_model = train_rain_model(X, y)

            # Map wind direction to compass points
            wind_deg = current_weather['wind_gust_dir'] % 360
            compass_points = [
                ("N", 0, 11.25), ("NNE", 11.25, 33.75), ("NE", 33.75, 56.25),
                ("ENE", 56.25, 78.75), ("E", 78.75, 101.25), ("ESE", 101.25, 123.75),
                ("SE", 123.75, 146.25), ("SSE", 146.25, 168.75), ("S", 168.75, 191.25),
                ("SSW", 191.25, 213.75), ("SW", 213.75, 236.25), ("WSW", 236.25, 258.75),
                ("W", 258.75, 281.25), ("WNW", 281.25, 303.75), ("NW", 303.75, 326.25),
                ("NNW", 326.25, 348.75)
            ]
            compass_direction = next((point for point, start, end in compass_points if start <= wind_deg < end), None)
            if compass_direction is None:
                logging.warning(f"Wind direction {wind_deg} not mapped to compass points.")
                compass_direction_encoded = -1  # Default if mapping fails
            else:
                try:
                    compass_direction_encoded = le.transform([compass_direction])[0]
                except ValueError:
                    compass_direction_encoded = -1

            # Process current weather data
            current_data = {
                'MinTemp': current_weather['temp_min'],
                'MaxTemp': current_weather['temp_max'],
                'WindGustDir': compass_direction_encoded,
                'WindGustSpeed': current_weather['Wind_Gust_Speed'],
                'Humidity': current_weather['humidity'],
                'Pressure': current_weather['pressure'],
                'Temp': current_weather['current_temp']
            }

            current_df = pd.DataFrame([current_data])

            # Rain Prediction
            rain_prediction = rain_model.predict(current_df)[0]

            # Prepare and train regression models for temperature and humidity
            X_temp, y_temp = prepare_regression_data(historical_data, 'Temp')
            X_hum, y_hum = prepare_regression_data(historical_data, 'Humidity')

            temp_model = train_regression_model(X_temp, y_temp)
            hum_model = train_regression_model(X_hum, y_hum)

            # Predict future temperature and humidity
            future_temp = predict_future(temp_model, current_weather['temp_min'])
            future_humidity = predict_future(hum_model, current_weather['humidity'])

            # Get the timezone offset from the OpenWeatherMap response
            timezone_offset = current_weather['timezone']

            # Calculate the time based on UTC offset
            utc_time = datetime.utcnow().replace(tzinfo=pytz.utc)
            local_time = utc_time + timedelta(seconds=timezone_offset)

            # Get next 5 hours in 12-hour format
            future_times = [(local_time + timedelta(hours=i)).strftime("%I:%M %p") for i in range(5)]

            # Store each value separately
            time1, time2, time3, time4, time5 = future_times
            temp1, temp2, temp3, temp4, temp5 = future_temp
            hum1, hum2, hum3, hum4, hum5 = future_humidity

            # Pass data to template
            context = {
                'location': city,
                'current_temp': current_weather['current_temp'],
                'MinTemp': current_weather['temp_min'],
                'MaxTemp': current_weather['temp_max'],
                'feels_like': current_weather['feels_like'],
                'humidity': current_weather['humidity'],
                'clouds': current_weather['clouds'],
                'description': current_weather['description'],
                'city': current_weather['city'],
                'country': current_weather['country'],
                'time': local_time.strftime("%I:%M %p"),
                'date': local_time.strftime("%B %d, %Y"),
                'wind': f"{current_weather['Wind_Gust_Speed']} m/s",
                'pressure': f"{current_weather['pressure']} hPa",
                'visibility': f"{current_weather['Visibility']} meters",
                'time1': time1,
                'time2': time2,
                'time3': time3,
                'time4': time4,
                'time5': time5,
                'temp1': f"{round(temp1, 1)}",
                'temp2': f"{round(temp2, 1)}",
                'temp3': f"{round(temp3, 1)}",
                'temp4': f"{round(temp4, 1)}",
                'temp5': f"{round(temp5, 1)}",
                'hum1': f"{round(hum1, 1)}",
                'hum2': f"{round(hum2, 1)}",
                'hum3': f"{round(hum3, 1)}",
                'hum4': f"{round(hum4, 1)}",
                'hum5': f"{round(hum5, 1)}",
            }


            return render(request, 'weather.html', context)
    except Exception as e:
        logging.error(f"Error in weather view: {str(e)}")
        return HttpResponse(f"An error occurred: {str(e)}", status=500)

    return render(request, 'weather.html')


