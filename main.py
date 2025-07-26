import numpy as np
import pandas as pd
from src.forecast_method import ForecastMethods


if __name__ == "__main__":
    df = pd.read_csv('data/NA_removed.csv', index_col=0, parse_dates=True)
    df.index.name = 'Date'
    df.index = pd.to_datetime(df.index)
    total = df['total']
    predict = ForecastMethods(total, h=5)
    predict.RunAll()