"""
Biometric Analysis & Performance Modeling Module
Computes exercise physiology metrics, aerobic decoupling, and predictive models
using vectorised linear algebra (NumPy OLS implementation).
"""

import pandas as pd
import numpy as np


def calculate_athletic_summary(df):
    """Computes high-level athletic milestones and lifetime stats."""
    total_km = df['Distance (km)'].sum()
    total_hours = df['Duration_Minutes'].sum() / 60.0
    total_elevation = df['Climb (m)'].sum()
    avg_pace = df['Pace_Min_Per_Km'].mean()
    
    # Milestone runs
    runs_5k = df[(df['Distance (km)'] >= 4.9) & (df['Distance (km)'] <= 5.5)]
    fastest_5k = runs_5k['Pace_Min_Per_Km'].min() if not runs_5k.empty else np.nan
    
    runs_10k = df[(df['Distance (km)'] >= 9.8) & (df['Distance (km)'] <= 11.0)]
    fastest_10k = runs_10k['Pace_Min_Per_Km'].min() if not runs_10k.empty else np.nan
    
    longest_run = df['Distance (km)'].max()
    
    # 80/20 Polarized training compliance (Zone 1 and 2 percentage)
    aerobic_base_pct = (df[df['HR_Zone'].isin(['Zone 1 (Recovery)', 'Zone 2 (Aerobic Base)'])].shape[0] / len(df)) * 100.0

    return {
        "Total_Distance_Km": round(total_km, 2),
        "Total_Hours_Trained": round(total_hours, 1),
        "Total_Elevation_Gain_M": int(total_elevation),
        "Average_Pace_Min_Km": round(avg_pace, 2),
        "Fastest_5K_Pace": round(fastest_5k, 2) if not np.isnan(fastest_5k) else "N/A",
        "Fastest_10K_Pace": round(fastest_10k, 2) if not np.isnan(fastest_10k) else "N/A",
        "Longest_Run_Km": round(longest_run, 2),
        "Aerobic_Base_Percentage": round(aerobic_base_pct, 1)
    }


class WearablePaceRegressor:
    """
    Multivariate Ordinary Least Squares (OLS) Regressor for Kinematic Pace Modeling.
    Predicts running pace (min/km) based on Distance, Elevation Climb, and Heart Rate.
    """
    def __init__(self):
        self.weights = None
        self.feature_names = ['Distance (km)', 'Climb (m)', 'Heart_Rate_BPM']
        
    def fit(self, X, y):
        # Add bias (intercept) term
        X_design = np.c_[np.ones(X.shape[0]), X]
        # Solve Normal Equations: beta = (X^T * X)^(-1) * X^T * y
        self.weights, _, _, _ = np.linalg.lstsq(X_design, y, rcond=None)
        return self
        
    def predict(self, X):
        X_arr = np.asarray(X)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)
        X_design = np.c_[np.ones(X_arr.shape[0]), X_arr]
        return X_design @ self.weights


def train_pace_prediction_model(df):
    """
    Splits wearable telemetry into train/test sets, fits the OLS model,
    and returns comprehensive statistical evaluation metrics.
    """
    features = ['Distance (km)', 'Climb (m)', 'HR_Imputed']
    target = 'Pace_Min_Per_Km'
    
    clean_data = df.dropna(subset=features + [target]).copy()
    
    # 80/20 Deterministic Train/Test Split
    np.random.seed(42)
    indices = np.random.permutation(len(clean_data))
    split_point = int(len(clean_data) * 0.8)
    train_idx, test_idx = indices[:split_point], indices[split_point:]
    
    X_train = clean_data[features].iloc[train_idx].values
    y_train = clean_data[target].iloc[train_idx].values
    X_test = clean_data[features].iloc[test_idx].values
    y_test = clean_data[target].iloc[test_idx].values
    
    # Train model
    model = WearablePaceRegressor()
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Metrics
    mae = np.mean(np.abs(y_test - y_pred))
    rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
    ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
    ss_res = np.sum((y_test - y_pred) ** 2)
    r2 = 1.0 - (ss_res / ss_tot)
    
    return {
        "model": model,
        "MAE_seconds_per_km": round(mae * 60.0, 1), # In seconds
        "RMSE_seconds_per_km": round(rmse * 60.0, 1),
        "R2_Score": round(r2, 3),
        "Intercept": round(model.weights[0], 4),
        "Coefficients": {
            "Distance (km)": round(model.weights[1], 4),
            "Climb (m)": round(model.weights[2], 5),
            "Heart Rate (bpm)": round(model.weights[3], 4)
        }
    }
