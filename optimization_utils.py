import pandas as pd
import os

# Load the precomputed optimization results once
csv_path = os.path.join(os.path.dirname(__file__), 'optimization_results.csv')
df_results = pd.read_csv(csv_path)

# Find the plateau value (first effort budget where TotalDelayReduced stops increasing)
df_results['DeltaDelay'] = df_results['TotalDelayReduced'].diff().fillna(df_results['TotalDelayReduced'].iloc[0])
epsilon = 0.001
plateau_row = df_results[df_results['DeltaDelay'] < epsilon].head(1)
if not plateau_row.empty:
    plateau_budget = int(plateau_row['EffortBudget'].iloc[0])
else:
    plateau_budget = df_results['EffortBudget'].max()

# Delay types
DELAY_TYPES = ['AIRLINE_DELAY', 'LATE_AIRCRAFT_DELAY', 'AIR_SYSTEM_DELAY', 'SECURITY_DELAY', 'WEATHER_DELAY']

def get_optimization_result(effort_budget):
    # If above plateau, use plateau row
    if effort_budget > plateau_budget:
        row = df_results[df_results['EffortBudget'] == plateau_budget].iloc[0]
    else:
        row = df_results[df_results['EffortBudget'] == effort_budget].iloc[0]
    delay_by_type = {dt: round(row[dt], 2) for dt in DELAY_TYPES}
    total_delay_reduced = round(row['TotalDelayReduced'], 2)
    result = {
        'total_delay_reduced': total_delay_reduced,
        'delay_by_type': delay_by_type
    }
    return result, plateau_budget 