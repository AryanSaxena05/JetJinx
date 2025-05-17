import pandas as pd

def get_monthly_avg_delay():
    df = pd.read_csv('flights.csv', usecols=["MONTH", "ARRIVAL_DELAY"])
    df = df[df["ARRIVAL_DELAY"] < 500]  # Remove outliers
    df = df.dropna(subset=["ARRIVAL_DELAY"])
    monthly_delay = df.groupby("MONTH")["ARRIVAL_DELAY"].mean().reset_index()
    return monthly_delay.to_dict(orient='records') 