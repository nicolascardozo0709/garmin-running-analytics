"""
Main Execution Script
Runs the end-to-end wearable sports analytics and machine learning pipeline.
"""

import os
from src.data_loader import load_and_preprocess_running_data
from src.biometric_analysis import calculate_athletic_summary, train_pace_prediction_model
from src.visualizer import generate_all_plots


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, 'data', 'cardioActivities.csv')
    output_dir = os.path.join(base_dir, 'outputs')
    
    print("=" * 68)
    print("   WEARABLE BIOMETRICS & RUNNING PERFORMANCE ANALYTICS PIPELINE")
    print("=" * 68)
    
    # 1. Pipeline: Load & Preprocess
    print("\n[1/4] Loading and cleaning wearable running telemetry...")
    df = load_and_preprocess_running_data(data_path)
    print(f"      -> Successfully parsed {len(df)} workouts across {df['Year'].nunique()} years.")
    
    # 2. Pipeline: Athletic Summary
    print("\n[2/4] Computing physiological & performance milestones...")
    summary = calculate_athletic_summary(df)
    for k, v in summary.items():
        print(f"      - {k.replace('_', ' ')}: {v}")
        
    # 3. Pipeline: Machine Learning Pace Modeling
    print("\n[3/4] Fitting Multivariate OLS Pace Prediction Model (NumPy)...")
    ml_results = train_pace_prediction_model(df)
    print(f"      - Model R² Score: {ml_results['R2_Score']}")
    print(f"      - Mean Absolute Error (MAE): {ml_results['MAE_seconds_per_km']} seconds/km")
    print(f"      - Root Mean Squared Error (RMSE): {ml_results['RMSE_seconds_per_km']} seconds/km")
    print("      - Model Equation Coefficients:")
    print(f"         * Intercept: {ml_results['Intercept']}")
    for feat, coef in ml_results['Coefficients'].items():
        print(f"         * {feat}: {coef}")
        
    # Example prediction: 10K run with 80m elevation at 155 bpm
    example_input = [[10.0, 80.0, 155.0]]
    pred_pace = ml_results['model'].predict(example_input)[0]
    pred_min = int(pred_pace)
    pred_sec = int((pred_pace - pred_min) * 60)
    print(f"\n      [+] Real-world Inference Example:")
    print(f"          Target: 10 km | Climb: 80 m | HR: 155 bpm")
    print(f"          -> Predicted Pace: {pred_min}:{pred_sec:02d} min/km ({round(pred_pace, 2)} min/km)")
        
    # 4. Pipeline: Visualizations
    print("\n[4/4] Rendering high-resolution figures for GitHub README...")
    plots = generate_all_plots(df, output_dir)
    for p in plots:
        print(f"      -> Saved: {os.path.basename(p)}")
        
    print("\n" + "=" * 68)
    print("  Pipeline execution completed successfully!")
    print(f"  Visuals ready in: {output_dir}")
    print("=" * 68)


if __name__ == '__main__':
    main()
