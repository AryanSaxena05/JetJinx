import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

# Paths to data
monthly_delay_csv = 'flights.csv'  # Update if needed
optimization_csv = 'optimization_results.csv'

# Output directory for HTML plots
output_dir = 'static/insights_plots'
os.makedirs(output_dir, exist_ok=True)

# --- Visualization 1: Monthly Average Arrival Delay ---
def plot_monthly_avg_delay():
    df = pd.read_csv(monthly_delay_csv, usecols=["MONTH", "ARRIVAL_DELAY"])
    df = df[df["ARRIVAL_DELAY"] < 500]  # Remove outliers
    df = df.dropna(subset=["ARRIVAL_DELAY"])
    monthly_delay = df.groupby("MONTH")["ARRIVAL_DELAY"].mean().reset_index()
    fig = px.bar(monthly_delay, x="MONTH", y="ARRIVAL_DELAY", title="Average Arrival Delay by Month", labels={"ARRIVAL_DELAY": "Avg Arrival Delay (min)"})
    fig.write_html(os.path.join(output_dir, 'monthly_avg_delay.html'), include_plotlyjs='cdn')

# --- Visualization 2: Optimization Delay Reduction by Type (Stacked Area) ---
def plot_optimization_stack():
    df = pd.read_csv(optimization_csv)
    delay_types = ["AIRLINE_DELAY", "LATE_AIRCRAFT_DELAY", "AIR_SYSTEM_DELAY", "SECURITY_DELAY", "WEATHER_DELAY"]
    fig = go.Figure()
    for dt in delay_types:
        fig.add_trace(go.Scatter(
            x=df["EffortBudget"],
            y=df[dt],
            mode='lines',
            stackgroup='one',
            name=dt.replace('_', ' ').title()
        ))
    fig.update_layout(title="Delay Minutes Allocated by Delay Type Across Effort Budgets", xaxis_title="Effort Budget", yaxis_title="Minutes of Delay Reduced")
    fig.write_html(os.path.join(output_dir, 'optimization_stack.html'), include_plotlyjs='cdn')

# --- Visualization 3: Regime Bar Chart (Optional) ---
def plot_regime_bar():
    df = pd.read_csv(optimization_csv)
    delay_types = ["AIRLINE_DELAY", "LATE_AIRCRAFT_DELAY", "AIR_SYSTEM_DELAY", "SECURITY_DELAY", "WEATHER_DELAY"]
    if 'RegimeID' not in df.columns:
        return  # Only plot if regime info is present
    avg_alloc_per_regime = (
        df.groupby("RegimeID")[delay_types]
        .mean()
        .reset_index()
        .round(2)
    )
    regime_labels = [f"Regime {int(rid)}" for rid in avg_alloc_per_regime["RegimeID"]]
    fig = go.Figure()
    for dt in delay_types:
        fig.add_trace(go.Bar(
            x=regime_labels,
            y=avg_alloc_per_regime[dt],
            name=dt.replace('_', ' ').title()
        ))
    fig.update_layout(barmode='stack', title="Average Delay Minutes by Type Within Each Regime", xaxis_title="Regime", yaxis_title="Avg Minutes Reduced")
    fig.write_html(os.path.join(output_dir, 'regime_bar.html'), include_plotlyjs='cdn')

if __name__ == '__main__':
    plot_monthly_avg_delay()
    plot_optimization_stack()
    plot_regime_bar()
    print(f"Plots saved to {output_dir}") 