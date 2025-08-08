
import pandas as pd
from src.forecast_method import ForecastMethods, measure_error


if __name__ == "__main__":
    # sample data
    start_date = '2022-01-01'
    sample_data = [
        2747.26, 8752.47, 5924.90, 4342.00, 30363.86, 13771.12, 8849.39, 5699.26,
        4933.24, 8788.62, 6027.33, 6520.33, 12349.15, 9675.69, 6456.26, 5518.01,
        3820.18, 3851.67, 7797.29, 5508.97, 5110.00, 13753.07, 6443.38, 6026.16,
        8788.62, 6027.33, 6520.33, 8849.39, 6443.38]
    total = pd.Series(sample_data, index=pd.date_range(start=start_date, periods=len(sample_data), freq='ME'))
    train = total[:-5]  # use all but the last 5 for training
    test = total[-5:]  # last 5 for testing
    # example usage
    frequency = 'ME'  # Monthly frequency (end of month)
    predict = ForecastMethods(train, h=5, frequency=frequency)
    results = predict.RunAll()
    for method, result in results.items():
        print(f"Method: {method}")
        print(f"Error: {measure_error(result, test)}\n")
