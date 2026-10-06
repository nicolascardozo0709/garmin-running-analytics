"""
Data Loader & Preprocessing Module
Extracts, cleans, and standardizes multi-year wearable running activities.
"""

import pandas as pd
import numpy as np


def parse_duration_to_seconds(duration_str):
    """Converts duration strings like '58:40' or '1:14:22' into total seconds."""
    if pd.isna(duration_str):
        return np.nan
    parts = str(duration_str).strip().split(':')
    try:
        if len(parts) == 2:
            minutes, seconds = map(float, parts)
            return minutes * 60 + seconds
        elif len(parts) == 3:
            hours, minutes, seconds = map(float, parts)
            return hours * 3600 + minutes * 60 + seconds
    except ValueError:
        return np.nan
    return np.nan


def parse_pace_to_decimal_minutes(pace_str):
    """Converts pace string like '5:37' into decimal minutes per km (e.g., 5.616 min/km)."""
    if pd.isna(pace_str):
        return np.nan
    parts = str(pace_str).strip().split(':')
    try:
        if len(parts) == 2:
            minutes, seconds = map(float, parts)
            return minutes + (seconds / 60.0)
    except ValueError:
        return np.nan
    return np.nan


def load_and_preprocess_running_data(csv_path):
    """
    Loads raw cardioActivities CSV, isolates running sessions,
    normalizes time features, and performs biometric feature engineering.
    """
    df = pd.read_csv(csv_path)

    # 1. Filter for running activities
    df_runs = df[df['Type'].str.lower() == 'running'].copy()

    # 2. DateTime processing
    df_runs['Date'] = pd.to_datetime(df_runs['Date'])
    df_runs = df_runs.sort_values('Date').reset_index(drop=True)
    df_runs['Year'] = df_runs['Date'].dt.year
    df_runs['Month'] = df_runs['Date'].dt.to_period('M')
    df_runs['DayOfWeek'] = df_runs['Date'].dt.day_name()

    # 3. Time & Pace conversions
    df_runs['Duration_Seconds'] = df_runs['Duration'].apply(parse_duration_to_seconds)
    df_runs['Duration_Minutes'] = df_runs['Duration_Seconds'] / 60.0
    df_runs['Pace_Min_Per_Km'] = df_runs['Average Pace'].apply(parse_pace_to_decimal_minutes)

    # 4. Standardize numeric columns
    numeric_cols = ['Distance (km)', 'Average Speed (km/h)', 'Calories Burned', 'Climb (m)', 'Average Heart Rate (bpm)']
    for col in numeric_cols:
        df_runs[col] = pd.to_numeric(df_runs[col], errors='coerce')

    # Remove extreme outliers or misrecorded runs (< 0.5 km or erroneous sensor data)
    df_runs = df_runs[(df_runs['Distance (km)'] >= 0.5) & (df_runs['Distance (km)'] <= 50.0)]
    df_runs = df_runs[(df_runs['Pace_Min_Per_Km'] >= 3.0) & (df_runs['Pace_Min_Per_Km'] <= 12.0)]

    # 5. Heart Rate Imputation
    # For older sessions before heart rate strap was purchased, impute using pace-distance correlation
    mean_hr_by_speed = df_runs.groupby(pd.qcut(df_runs['Average Speed (km/h)'], q=5, duplicates='drop'))['Average Heart Rate (bpm)'].transform('median')
    df_runs['HR_Imputed'] = df_runs['Average Heart Rate (bpm)'].fillna(mean_hr_by_speed)
    df_runs['HR_Imputed'] = df_runs['HR_Imputed'].fillna(df_runs['Average Heart Rate (bpm)'].median())

    # 6. Biometric Feature Engineering
    # Aerobic Decoupling / Cardiac Cost Index (Speed vs Heart Rate efficiency)
    df_runs['Cardiac_Cost_Index'] = (df_runs['Average Speed (km/h)'] / df_runs['HR_Imputed']) * 100.0

    # Heart Rate Training Zones (Standard Karvonen 5-Zone Model)
    def assign_hr_zone(hr):
        if hr < 130:
            return 'Zone 1 (Recovery)'
        elif hr < 148:
            return 'Zone 2 (Aerobic Base)'
        elif hr < 162:
            return 'Zone 3 (Tempo)'
        elif hr < 175:
            return 'Zone 4 (Threshold)'
        else:
            return 'Zone 5 (Anaerobic / Max)'

    df_runs['HR_Zone'] = df_runs['HR_Imputed'].apply(assign_hr_zone)

    # Rolling Metrics (30-day rolling average pace and mileage)
    df_runs.set_index('Date', inplace=True)
    df_runs['Rolling_30D_Mileage_Km'] = df_runs['Distance (km)'].rolling('30D').sum()
    df_runs['Rolling_30D_Avg_Pace'] = df_runs['Pace_Min_Per_Km'].rolling('30D').mean()
    df_runs['Rolling_30D_Cardiac_Efficiency'] = df_runs['Cardiac_Cost_Index'].rolling('30D').mean()
    df_runs.reset_index(inplace=True)

    return df_runs


if __name__ == '__main__':
    data_path = r"C:\Users\nicoc\.gemini\antigravity\scratch\garmin-running-analytics\data\cardioActivities.csv"
    clean_df = load_and_preprocess_running_data(data_path)
    print(f"Processed {len(clean_df)} running sessions successfully.")
    print("Columns available:", list(clean_df.columns))
    print(clean_df[['Date', 'Distance (km)', 'Pace_Min_Per_Km', 'HR_Imputed', 'HR_Zone']].head())
