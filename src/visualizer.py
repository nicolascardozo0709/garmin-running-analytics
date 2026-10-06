"""
Visualizer Module
Generates production-grade publication figures for GitHub README and technical portfolio.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Set style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8


def generate_all_plots(df, output_dir):
    """Generates all analytics plots and saves them as high-resolution PNGs."""
    os.makedirs(output_dir, exist_ok=True)
    generated_files = []

    # -------------------------------------------------------------
    # 1. Timeline of Aerobic Efficiency & 30-Day Rolling Mileage
    # -------------------------------------------------------------
    fig, ax1 = plt.subplots(figsize=(12, 5), dpi=300)
    
    color_mileage = '#3498db'
    color_efficiency = '#e74c3c'
    
    ax1.set_title('Longitudinal Aerobic Adaptation: 30-Day Rolling Mileage vs Cardiac Efficiency', fontsize=13, fontweight='bold', pad=15)
    ax1.plot(df['Date'], df['Rolling_30D_Mileage_Km'], color=color_mileage, linewidth=2, label='30-Day Rolling Mileage (km)')
    ax1.set_xlabel('Timeline', fontsize=11, labelpad=10)
    ax1.set_ylabel('30-Day Volume (km)', color=color_mileage, fontsize=11, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor=color_mileage)
    
    ax2 = ax1.twinx()
    ax2.plot(df['Date'], df['Rolling_30D_Cardiac_Efficiency'], color=color_efficiency, linewidth=2, linestyle='--', label='Cardiac Cost Index (Speed/HR * 100)')
    ax2.set_ylabel('Cardiac Efficiency Index', color=color_efficiency, fontsize=11, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor=color_efficiency)
    ax2.grid(False) # avoid clutter
    
    fig.tight_layout()
    path1 = os.path.join(output_dir, 'aerobic_efficiency_timeline.png')
    fig.savefig(path1, bbox_inches='tight')
    plt.close(fig)
    generated_files.append(path1)

    # -------------------------------------------------------------
    # 2. Polarized Training Distribution (Heart Rate Zones)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    zone_order = ['Zone 1 (Recovery)', 'Zone 2 (Aerobic Base)', 'Zone 3 (Tempo)', 'Zone 4 (Threshold)', 'Zone 5 (Anaerobic / Max)']
    palette = ['#2ecc71', '#3498db', '#f39c12', '#e67e22', '#e74c3c']
    
    zone_counts = df['HR_Zone'].value_counts().reindex(zone_order).fillna(0)
    bars = ax.bar(zone_counts.index, zone_counts.values, color=palette, edgecolor='#333333', linewidth=0.7)
    
    for bar in bars:
        height = bar.get_height()
        pct = (height / len(df)) * 100.0
        ax.annotate(f'{int(height)} runs\n({pct:.1f}%)',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9.5, fontweight='bold')
                    
    ax.set_title('Exercise Intensity Distribution: Karvonen Heart Rate Zones (80/20 Rule Analysis)', fontsize=13, fontweight='bold', pad=15)
    ax.set_ylabel('Number of Workouts', fontsize=11, fontweight='bold')
    ax.set_xlabel('Biometric Cardiovascular Zone', fontsize=11, labelpad=10)
    ax.set_ylim(0, max(zone_counts.values) * 1.18)
    
    fig.tight_layout()
    path2 = os.path.join(output_dir, 'heart_rate_zones_distribution.png')
    fig.savefig(path2, bbox_inches='tight')
    plt.close(fig)
    generated_files.append(path2)

    # -------------------------------------------------------------
    # 3. Distance vs Pace Colored by Heart Rate
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    # Normalized bubble sizes (between 25 and 180 pt)
    bubble_sizes = np.clip(df['Calories Burned'] / 8.0, 25, 200)
    scatter = ax.scatter(df['Distance (km)'], df['Pace_Min_Per_Km'], 
                         c=df['HR_Imputed'], cmap='viridis', 
                         s=bubble_sizes, alpha=0.65, edgecolors='none')
    
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label('Average Heart Rate (BPM)', fontsize=11, fontweight='bold')
    
    ax.set_title('Kinematic & Biometric Dispersion: Running Distance vs Pace (Bubble Size = Calories)', fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel('Distance (km)', fontsize=11, fontweight='bold', labelpad=10)
    ax.set_ylabel('Average Pace (min/km)', fontsize=11, fontweight='bold', labelpad=10)
    
    # Invert y-axis so faster pace (lower min/km) is at the top
    ax.invert_yaxis()
    
    fig.tight_layout()
    path3 = os.path.join(output_dir, 'distance_vs_pace_biometrics.png')
    fig.savefig(path3, bbox_inches='tight')
    plt.close(fig)
    generated_files.append(path3)

    return generated_files
