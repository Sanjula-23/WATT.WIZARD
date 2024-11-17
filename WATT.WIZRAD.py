# This is the basic project without GUI. You can try this version in your terminal 
# Dinalofcl - 2024

import sys
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Energy data file collecting process and Error handeling
def collect_EnergyData(file_path):
    try:
        data = pd.read_csv(file_path, parse_dates=['timestamps'])
        data = data.set_index('timestamps')
        return data
    except FileNotFoundError:
        print(f"System Error! The system was unable to locate the file path: {file_path}")
    except pd.errors.EmptyDataError:
        print(f"File Error! This file ({file_path}) is empty!")
    except Exception as expt:
        print(f"Error reading the CSV File: {expt}")
        return None


# Data cleaning process 
def clean_data(data):
    if 'consumption' in data.columns:
        data['consumption'] = data['consumption'].clip(lower=0)
         
        mean_value = data['consumption'].mean()
        data['consumption'] = data['consumption'].fillna(mean_value)

        data.loc[data['consumption'] > 10000, 'consumption'] = mean_value

        Q1 = data['consumption'].quantile(0.25)
        Q3 = data['consumption'].quantile(0.75)
        IQR = Q3 -Q1
        upper_threshold = Q3 + 1.5 * IQR

        data['consumption'] = np.where(data['consumption'] > upper_threshold, mean_value, data['consumption'])
        return data
    
    else:
        print("Column 'consumption' not found in the data. ")
        return data


# Collected Energy data analysing process
def analyze_consumption(data):
    hourly_consumption = data['consumption'].resample('h').sum()
    print("\n<< Daily Consumption Report >>")
    print("\n-------------------------------------")
    print(hourly_consumption)

    print("\n")
    
    max_hourly_consumption = hourly_consumption.max()
    max_hours = hourly_consumption.idxmax()
    print(f"1. Highest Usage is on: {max_hourly_consumption} kWh on {max_hours}")

    print("\n")

    min_daily_consumption = hourly_consumption.min()
    min_hours = hourly_consumption.idxmin()
    print(f"2. Lowest Usage is on: {min_daily_consumption} kWh on {min_hours}")

    print("\n")

    total_consumption = hourly_consumption.sum()
    print(f"3. Total Consumption of the day: {total_consumption} kWh")

    print("\n")

    max_percentage = (max_hourly_consumption / total_consumption) * 100
    print(f"4. Maximum Day Consumption as a percentage of Total: {max_percentage:.3f}%")


# Energy data prediction process - Machine learning model uses here
def predict_demand(data):

    data['hour'] = data.index.hour
    data['days_of_week'] = data.index.dayofweek
    data['month'] = data.index.month
    data['week_of_year'] = data.index.isocalendar().week

    features = ['hour', 'days_of_week', 'month', 'week_of_year']
    X = data[features]
    y = data['consumption']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

    model = RandomForestRegressor(n_estimators=100,max_features= None,max_depth=10,min_samples_split=2, min_samples_leaf=1,bootstrap=True, random_state=100) # Pre trained Values using google colab
    model.fit(X_train, y_train)      # ML Model

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred) 
    
    print("\n")
    print(f"5. Mean Absolute Error Value is: {mae}")
    print("\n")
    print(f"6. R Squared Value is: {r2}") 

    return model

# Future data prediction process
def predict_future_consumption(model, cleaned_data):
    last_timestamp = cleaned_data.index[-1]
    print("\n")

    while True:
        try:
            future_hours = int(input("Please input the number of hours you wish to forecast. The system is ready to calculate your future energy values? "))
            if future_hours <= 0:
                raise ValueError("Number of hours must be a positive value.")
            break
        except ValueError as expt:
            print(f"Invalid input: {expt}. Please enter a positive integer. ")

    future_data = []

    for i in range(1, future_hours + 1):
        next_hour = last_timestamp + pd.Timedelta(hours=i)
        future_data.append({
            'hour': next_hour.hour,
            'days_of_week': next_hour.dayofweek,
            'month': next_hour.month,
            'week_of_year': next_hour.isocalendar().week 

        })

    future_data = pd.DataFrame(future_data)
    future_data.index = [last_timestamp + pd.Timedelta(hours=i) for i in range(1, future_hours + 1)]

    future_consumption = model.predict(future_data)
    future_consumption = np.clip(future_consumption, a_min=0, a_max=None)

    print("\n")
    print("7. Predicted Consumption for next hours: ")
    print()
    for time, consumption in zip(future_data.index, future_consumption):
        print(f" {time}: {consumption:.2f} kWh")

    return future_consumption, future_data


# Suggestions for Higher energy data periods 
def suggest_methods_to_reduce(high_usage_periods):
    print("\n<< Here are some tips to reduce energy consumption during high-usage periods: Suggested by WATT.WIZARD >>")
    print("\nGeneral Tip: Try to store excess energy during low-usage periods and use it during peak hours. Ensure that co-workers are aware of energy-saving practices.")
    
    suggestions = {
        "Early Morning (00:00 - 05:59)": [
            "--> Schedule non-essential processes for off-peak hours.",
            "--> Use energy-efficient lighting with motion sensors.",
            "--> Optimize HVAC system for minimal nighttime operations."
        ],
        "Morning (06:00 - 10:59)": [
            "-->Implement staggered start times for equipment.",
            "-->Utilize natural lighting where possible.",
            "--> Encourage energy-conscious behavior among early shift workers."
        ],
        "Afternoon (11:00 - 15:59)": [
            "--> Conduct regular maintenance to ensure equipment efficiency.",
            "--> Use smart power strips to reduce phantom energy consumption.",
            "--> Optimize production schedules to avoid simultaneous operation of high-energy equipment."
        ],
        "Evening (16:00 - 18:59)": [
            "--> Implement demand response strategies during grid peak times.",
            "--> Use energy storage systems to offset high demand.",
            "--> Encourage telecommuting or flexible hours to reduce facility energy use."
        ],
        "Night (19:00 - 23:59)": [
            "--> Automate shutdown procedures for non-essential equipment.",
            "--> Use timer controls for exterior lighting.",
            "--> Conduct energy-intensive processes during off-peak hours if possible."
        ]
    }

    high_usage_times = set()
    for time in high_usage_periods.index:
        if 0 <= time.hour < 6:
            high_usage_times.add("Early Morning (00:00 - 05:59)")
        elif 6 <= time.hour < 11:
            high_usage_times.add("Morning (06:00 - 10:59)")
        elif 11 <= time.hour < 16:
            high_usage_times.add("Afternoon (11:00 - 15:59)")
        elif 16 <= time.hour < 19:
            high_usage_times.add("Evening (16:00 - 18:59)")
        else:
            high_usage_times.add("Night (19:00 - 23:59)")

    for time_period in high_usage_times:
        print(f"\n{time_period}:")
        for suggestion in suggestions[time_period]:
            print(f"{suggestion}")
    

# Plotted graph for present data and future data using matplotlib
def plot_consumption(data, future_data, future_consumption):
    plt.figure(figsize=(16, 8))
    plt.plot(data.index, data['consumption'], label='Historical Data')
    plt.plot(future_data.index, future_consumption, 'r-.', label='Predicted Data')
    plt.title('Energy Consumption Graph: Based on Given and the Predicted Values\n Dinalofcl - WATT.WIZARD')
    plt.xlabel('Time')
    plt.ylabel('Consumption in kWh')
    plt.legend()
    plt.grid(True)
    plt.show()


# Main function 
def main():
    print("\n")
    file_path = input("Hello User! Please enter the full path to the CSV file: ")
    raw_data = collect_EnergyData(file_path)

    if raw_data is not None:
        
        cleaned_data = clean_data(raw_data)

        analyze_consumption(cleaned_data)

        model = predict_demand(cleaned_data)

        future_consumption, future_data = predict_future_consumption(model, cleaned_data)

        high_usage_threshold = np.percentile(future_consumption, 90)
        high_usage_periods = future_data[future_consumption > high_usage_threshold]
        suggest_methods_to_reduce(high_usage_periods) 

        print("\n")
        print("We are creating the graphical user interface. Hold On....", end='', flush=True)
        time.sleep(3)
        sys.stdout.write('\r' + ' ' * 60 + '\r')
        sys.stdout.flush()

        plot_consumption(cleaned_data, future_data, future_consumption)

    else:
        print("Failed to collect and process data.")

if __name__ == "__main__":
    main()