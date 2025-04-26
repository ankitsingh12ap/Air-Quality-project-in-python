import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from scipy import stats
#
# Load data (replace with your file path)
df = pd.read_csv('Air_Quality.csv')

# --------------------------
# Function 1: Clean Data
# --------------------------
def clean_data(df, pollutant_name='Nitrogen dioxide (NO2)'):
    """
    Clean the dataset by:
    1. Filtering for a specific pollutant
    2. Handling datetime conversion
    3. Removing invalid values
    """
    df = df.copy()
    
    # Filter for specified pollutant
    df = df[df['Name'] == pollutant_name]
    
    # Convert dates
    df['Start_Date'] = pd.to_datetime(df['Start_Date'], errors='coerce')
    
    # Remove invalid entries
    df = df.dropna(subset=['Data Value', 'Start_Date'])
    df = df[df['Data Value'] >= 0]
    
    # Extract year from Time Period
    df['Year'] = df['Time Period'].str.extract('(\d{4})').astype(float)
    
    return df

# --------------------------
# Function 2: Perform EDA
# --------------------------
def perform_eda(df):
    """Generate exploratory visualizations and analysis"""
    print("Basic Info:")
    print(df.info())
    print("\nDescriptive Statistics:")
    print(df.describe())

    # Plot 1: Distribution of Data Values
    plt.figure(figsize=(10,6))
    sns.histplot(df['Data Value'], kde=True)
    plt.title(f"Distribution of {df['Name'].iloc[0]} Levels")
    plt.xlabel(f"{df['Measure'].iloc[0]} ({df['Measure Info'].iloc[0]})")
    plt.show()

    # Plot 2: Trends Over Time
    plt.figure(figsize=(12,6))
    sns.lineplot(data=df, x='Year', y='Data Value', ci=None)
    plt.title(f"{df['Name'].iloc[0]} Levels Over Years")
    plt.show()

    # Plot 3: Top 10 Locations
    top_locations = df['Geo Place Name'].value_counts().nlargest(10).index
    plt.figure(figsize=(12,6))
    sns.boxplot(data=df[df['Geo Place Name'].isin(top_locations)], 
                x='Geo Place Name', y='Data Value')
    plt.xticks(rotation=45)
    plt.title(f"{df['Name'].iloc[0]} Levels by Top Locations")
    plt.show()

    # Plot 4: Seasonal Comparison (Interactive)
    seasonal_df = df[df['Time Period'].str.contains('Winter|Summer')]
    seasonal_df['Season'] = seasonal_df['Time Period'].str.split().str[0]
    fig = px.box(seasonal_df, x='Season', y='Data Value', 
                 title=f"{df['Name'].iloc[0]} Levels by Season")
    fig.show()

    # Plot 5: Geographic Heatmap (if coordinates available)
    # (Add mapping logic if geographic coordinates exist in data)

# --------------------------
# Function 3: Hypothesis Testing
# --------------------------
def hypothesis_test(df):
    """Compare pollutant levels between winter and summer seasons"""
    seasonal_df = df[df['Time Period'].str.contains('Winter|Summer', na=False)]
    seasonal_df['Season'] = seasonal_df['Time Period'].str.split().str[0]
    
    winter = seasonal_df[seasonal_df['Season'] == 'Winter']['Data Value']
    summer = seasonal_df[seasonal_df['Season'] == 'Summer']['Data Value']
    
    # Check normality
    _, p_normal = stats.normaltest(seasonal_df['Data Value'])
    
    if p_normal < 0.05:  # Non-parametric test
        test_result = stats.mannwhitneyu(winter, summer)
        test_type = 'Mann-Whitney U'
    else:  # Parametric test
        test_result = stats.ttest_ind(winter, summer)
        test_type = 'T-test'
    
    print(f"\nHypothesis Test ({test_type}):")
    print(f"Statistic: {test_result.statistic:.4f}")
    print(f"P-value: {test_result.pvalue:.4f}")
    
    if test_result.pvalue < 0.05:
        print("Significant difference between seasons (p < 0.05)")
    else:
        print("No significant difference between seasons")

# --------------------------
# Master Function
# --------------------------
def run_analysis(pollutant='Nitrogen dioxide (NO2)'):
    """Run complete analysis pipeline"""
    df_clean = clean_data(df, pollutant)
    perform_eda(df_clean)
    hypothesis_test(df_clean)

# --------------------------
# Execute Analysis
# --------------------------
# Example usage:
run_analysis('Nitrogen dioxide (NO2)')
# run_analysis('Fine particles (PM 2.5)')  # For PM2.5 analysis
