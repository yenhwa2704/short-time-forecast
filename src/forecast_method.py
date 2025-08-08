"""
ForecastMethods class for time series forecasting
This module provides a unified interface for applying multiple time series forecasting methods
to a given training dataset. In addition to methods for naive forecasting, statistical models and econometric models,
grey models and fuzzy time series models are also implemented to deal with short-term forecasting tasks.
"""
import os
import math
import json
import warnings
import numpy as np
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import SeasonalNaive, AutoARIMA
from sktime.forecasting.ets import AutoETS
from statsmodels.tsa.exponential_smoothing.ets import ETSModel


class ForecastMethods:
    """
    ForecastMethods provides a unified interface for applying multiple time series forecasting methods
    to a given training dataset.

    This class implements a variety of forecasting techniques,
    including simple statistical methods, econometric models, grey models, and fuzzy time series models.
    It is designed to facilitate the comparison and combination of different forecasting approaches.

    Attributes:
        trains (pd.Series): The training time series data.
        h (int): The forecast horizon (number of periods to predict).
        ahead_idx (pd.DatetimeIndex): The index for the forecasted periods.

    Methods:
        RunAll():
            Runs all implemented forecasting methods and returns their forecasts in a DataFrame.

        NAIVE():
            Forecasts using the last observed value (naive method).

        AVG():
            Forecasts using the mean of the training data.

        SNAIVE(season_length=12, frequency='M', n_job=1):
            Forecasts using the Seasonal Naive method.

        ARIMA(season_length=12, frequency='M', n_job=1):
            Forecasts using the AutoARIMA model.

        ETS():
            Forecasts using the automatic Exponential Smoothing (ETS) model.

        ETS_manual():
            Forecasts using manually specified ETS models and selects the best based on AICc.

        SES():
            Forecasts using Simple Exponential Smoothing (SES).

        GM11():
            Forecasts using the GM(1,1) grey model.

        BGM11():
            Forecasts using the BGM(1,1) grey model with fuzzy membership.

        RBGM11():
            Forecasts using the robust BGM(1,1) grey model.

        FTS_Chen(m=10):
            Forecasts using Chen's Fuzzy Time Series method.

        FTS_My(m=10):
            Forecasts using a custom Fuzzy Time Series method.

    Notes:
        - Some methods require external libraries (e.g., StatsForecast, AutoARIMA, AutoETS, ETSModel).
        - The class is intended for use with univariate time series data.
        - Forecasts are returned as pandas.Series or pandas.DataFrame indexed by the forecast horizon.
    """

    def __init__(self, trains: pd.Series, h: int = 5, frequency: str = 'ME'):
        self.trains = trains
        self.h = h
        self.frequency = frequency
        self.forecast_dtime = pd.to_datetime(trains.index[-1]) + pd.DateOffset(months=1)
        self.ahead_idx = pd.date_range(self.forecast_dtime, periods=h, freq=frequency)

    @staticmethod
    def triang_cdf(x, params):
        """
        Compute the cumulative distribution function (CDF) of a triangular distribution.

        Parameters:
            x (float): The value at which to evaluate the CDF.
            params (list or tuple): The parameters of the triangular distribution (a, c, b),
                                    where a is the lower limit, c is the mode, and b is the upper limit.

        Returns:
            float: The CDF value at x.
        """
        a, c, b = params
        if x < a:
            return 0.0
        elif a <= x < c:
            return ((x - a) ** 2) / ((b - a) * (c - a))
        elif c <= x <= b:
            return 1.0 - ((b - x) ** 2) / ((b - a) * (b - c))
        else:
            warnings.warn("Value out of bounds for triangular distribution CDF.")
            return 1.0

    def RunAll(self) -> pd.DataFrame:
        '''
        run all forecasting methods, return a dataframe -> shape: (h, num(methods))
        '''
        frct_df = pd.DataFrame([], index=self.ahead_idx)
        # simple methods
        frct_df['NAIVE'] = self.NAIVE()
        frct_df['AVG'] = self.AVG()
        frct_df['SNAIVE'] = self.SNAIVE()

        # econometrics methods
        frct_df['ARIMA'] = self.ARIMA()
        frct_df['ETS'] = self.SES()
        # frct_df['ETS'] = self.ETS_manual()

        # grey models
        frct_df['GM11'] = self.GM11()
        frct_df['BGM11'] = self.BGM11()
        frct_df['RBGM11'] = self.RBGM11()

        # FTS
        frct_df['FTS_Chen'] = self.FTS_Chen()
        frct_df['FTS_My'] = self.FTS_My()

        # Combinations
        frct_df['EQUAL'] = frct_df.mean(axis=1)
        select_models = ['SNAIVE', 'ARIMA', 'RBGM11', 'FTS_My']
        frct_df['SELECT'] = frct_df[select_models].mean(axis=1)
        # prevent to occur error when calculating error measures
        frct_df.replace(0, 1, inplace=True)

        return frct_df

    # ==================simple methods==================
    def NAIVE(self) -> pd.Series:
        return pd.Series([self.trains.iloc[-1]] * self.h, index=self.ahead_idx)

    def AVG(self) -> pd.Series:
        return pd.Series([self.trains.mean()] * self.h, index=self.ahead_idx)

    def SNAIVE(self, season_length: int = 12, frequency: str = 'M', n_job: int = 1) -> pd.Series:
        """
        Generates forecasts using the Seasonal Naive (SNAIVE) method.

        Args:
            season_length (int, optional): The number of periods in a season. Default is 12.
            frequency (str, optional): The frequency of the time series (e.g., 'M' for monthly). Default is 'M'.
            n_job (int, optional): The number of parallel jobs to run. Default is 1.

        Returns:
            pandas.Series: Forecasted values indexed by the forecast horizon.
        """
        train_data = self.trains.copy()
        train_data = pd.DataFrame(train_data).reset_index()
        train_data.columns = ['ds', 'y']
        train_data['unique_id'] = 0
        train_data = train_data[['unique_id', 'ds', 'y']]

        fcst = StatsForecast(
            models=[SeasonalNaive(season_length=season_length)],
            freq=frequency,
            n_jobs=n_job
        )
        prediction = fcst.forecast(df=train_data, h=self.h)
        prediction = prediction.set_index('ds')['SeasonalNaive']
        prediction.index = self.ahead_idx
        return prediction
    # ==================simple methods==================

    # ==================traditional methods==================
    def ARIMA(self, season_length=12, frequency='M', n_job=1) -> pd.Series:
        """
        Fits an AutoARIMA model to the training data and generates forecasts.

        Parameters:
            season_length (int, optional): The number of periods in a seasonal cycle. Default is 12.
            frequency (str, optional): The frequency of the time series data (e.g., 'M' for monthly). Default is 'M'.
            n_job (int, optional): The number of parallel jobs to run. Default is 1.

        Returns:
            pandas.Series: Forecasted values indexed by the forecast horizon.
        """
        train_data = self.trains.copy()
        train_data = pd.DataFrame(train_data).reset_index()
        train_data.columns = ['ds', 'y']
        train_data['unique_id'] = 0
        train_data = train_data[['unique_id', 'ds', 'y']]

        fcst = StatsForecast(
            models=[AutoARIMA(season_length)],
            freq=frequency,
            n_jobs=n_job
        )
        try:
            prediction = fcst.forecast(df=train_data, h=self.h)
            prediction = prediction.set_index('ds')['AutoARIMA']
        except Exception as e:
            warnings.warn(f"ARIMA model fitting failed: {e}, using AVG instead.")
            prediction = self.AVG()
            prediction = prediction.to_frame(name='AutoARIMA')
        
        prediction.index = self.ahead_idx
        return prediction

    def ETS(self):
        """
        Fits an Exponential Smoothing State Space Model (ETS) to the training data and generates forecasts.

        This method attempts to automatically select and fit an ETS model to the training data using the AutoETS class.
        If the ETS model fitting fails, it falls back to using the average (AVG) method for prediction and logs a message.

        Returns:
            pandas.Series: Forecasted values indexed by the forecast horizon.
        """
        train_data = self.trains.copy()
        train_data.index = range(0, len(train_data))
        try:
            auto_ets = AutoETS(auto=True, sp=12).fit(y=train_data)
            prdct = auto_ets.predict(list(range(0, self.h)))
        except Exception:
            prdct = self.AVG()
            print(getattr(train_data, 'name', ''), "failed to predict with ETS, use AVG.")
        prdct.index = self.ahead_idx
        return prdct

    def ETS_manual(self):
        """
        Fits three different Exponential Smoothing (ETS) models to the training data
        and selects the best model based on the lowest AICc value.
        The models fitted are:
            - Additive error, additive trend, additive seasonal (AAA)
            - Additive error, additive trend, no seasonal (AAN)
            - Additive error, no trend, additive seasonal (ANA)
        Returns:
            pandas.Series: Forecasted values indexed by the forecast horizon.
        """
        train_data = self.trains.copy()
        aaa = ETSModel(train_data, error='add', trend='add', seasonal='add', seasonal_periods=12).fit()
        aan = ETSModel(train_data, error='add', trend='add', seasonal=None).fit()
        ana = ETSModel(train_data, error='add', trend=None, seasonal='add', seasonal_periods=12).fit()
        aiccs = [aaa.aicc, aan.aicc, ana.aicc]
        min_idx = np.argmin(aiccs)
        pred = [aaa, aan, ana][min_idx].forecast(self.h)
        return pred

    def SES(self):
        """
        Performs Simple Exponential Smoothing (SES) forecasting using an additive error and trend model.
        """
        train_data = self.trains.copy()
        aan = ETSModel(train_data, error='add', trend='add', seasonal=None).fit()
        pred = aan.forecast(self.h)
        pred.index = self.ahead_idx
        return pred

    # ==================traditional methods==================

    # ==================STS: Grey Models==================
    def GM11(self):
        """
        Implements the GM(1,1) Grey Model for time series forecasting.

        Returns:
            pandas.Series: Forecasted values indexed by the forecast horizon.
        """
        cum_series = self.trains.cumsum()
        # Calculate the mean sequence (background value)
        near_mean = ((cum_series.shift(-1) + cum_series) / 2)[:-1].to_list()
        Y = np.asarray(self.trains[:-1]).reshape(-1, 1)
        B = np.column_stack((-np.array(near_mean), np.ones(len(near_mean))))
        # Least squares estimation of parameters
        beta = np.linalg.lstsq(B, Y, rcond=None)[0].flatten()
        a, b = beta
        first_val = self.trains.iloc[0]
        # GM(1,1) model formula
        frct = [first_val]
        for k in range(1, len(self.trains) + self.h):
            frct.append((first_val - b / a) * np.exp(-a * k) + b / a)
        # Restore to original series
        origin_prct = [frct[0]] + [frct[i] - frct[i - 1] for i in range(1, len(frct))]
        return pd.Series(origin_prct[-self.h:], index=self.ahead_idx)

    def _fuzzy_membership(self, values, robust=False):
        """
        Compute fuzzy membership values for a series.
        If robust=True, outliers are assigned membership 0.
        """
        Q1 = values.quantile(0.25)
        Q2 = values.median()
        Q3 = values.quantile(0.75)
        if robust:
            LL = max(Q1 - 1.5 * (Q3 - Q1), values.min())
            UL = min(Q3 + 1.5 * (Q3 - Q1), values.max())
        else:
            LL = min(Q1 - 1.5 * (Q3 - Q1), values.min())
            UL = max(Q3 + 1.5 * (Q3 - Q1), values.max())
        MF = []
        for val in values:
            if robust and (val <= LL or val >= UL):
                MF.append(0)
            elif val == Q2:
                MF.append(1)
            elif val < Q2:
                MF.append((val - LL) / (Q2 - LL) if Q2 != LL else 0)
            else:
                MF.append((UL - val) / (UL - Q2) if UL != Q2 else 0)
        return MF, LL, UL, Q2

    def _GM_fuzzy(self, robust=False):
        """
        Generalized fuzzy GM(1,1) model for both BGM11 and RBGM11.
        """
        cum_series = self.trains.cumsum()
        Y = np.asarray(self.trains[:-1]).reshape(-1, 1)
        MF, LL, UL, Q2 = self._fuzzy_membership(self.trains, robust=robust)
        bg_value = [
            cum_series.iloc[i - 1] + MF[i] * self.trains.iloc[i]
            for i in range(1, len(MF))
        ]
        bg_mat = np.column_stack((-np.array(bg_value), np.ones(len(bg_value))))
        beta = np.linalg.lstsq(bg_mat, Y, rcond=None)[0].flatten()
        a, b = beta
        const = b / a if a != 0 else 0
        first_val = self.trains.iloc[0]
        frct = [first_val]
        for k in range(1, len(self.trains) + self.h):
            frct.append((first_val - const) * math.exp(-a * k) + const)
        origin_prct = [frct[0]] + [frct[i] - frct[i - 1] for i in range(1, len(frct))]
        return pd.Series(origin_prct[-self.h:], index=self.ahead_idx)

    def BGM11(self):
        """
        Implements the BGM(1,1) Grey Model with fuzzy membership for time series forecasting.
        """
        return self._GM_fuzzy(robust=False)

    def RBGM11(self):
        """
        Implements the Robust BGM(1,1) Grey Model for time series forecasting.
        """
        return self._GM_fuzzy(robust=True)
    # ==================STS: Grey Models==================

    # ==================STS: Fuzzy Models==================
    def FTS_Chen(self, m=10):
        """
        Implements Chen's Fuzzy Time Series (FTS) forecasting method.

        This method partitions the historical data range into fuzzy sets, fuzzifies the data,
        constructs fuzzy logical relationships, and generates forecasts based on these relationships.

        Args:
            m (int, optional): Number of fuzzy sets to partition the data into. Default is 10.

        Returns:
            pandas.Series: Forecasted values indexed by the forecast horizon.
        """
        lb, ub = self.trains.min(), self.trains.max()
        his_data = self.trains.tolist()
        radius = (ub - lb) / (m * 2)
        # Define fuzzy sets as triangular membership functions
        fuzzy_list = [
            [lb - radius + 2 * i * radius, lb + radius + 2 * i * radius, lb + 3 * radius + 2 * i * radius]
            for i in range(m)
        ]

        # Fuzzify the historical data
        location = [-1] * len(his_data)
        for j, val in enumerate(his_data):
            if val <= lb:
                location[j] = 0
            elif val >= ub:
                location[j] = m - 1
            else:
                for i in range(m):
                    cdf = self.triang_cdf(val, fuzzy_list[i])
                    if 0.125 <= cdf <= 0.875:
                        location[j] = i
                        break

        # Build fuzzy logical relationships
        fuzzy_dict = {i: [] for i in range(m)}
        for i in range(len(his_data) - 1):
            fuzzy_dict[location[i]].append(location[i + 1])

        # Forecasting
        frct_result = [his_data[0]]
        for current in range(len(his_data) + self.h - 1):
            if current < len(his_data):
                vl = his_data[current]
            else:
                vl = frct_result[-1]

            belong = int(m / 2)
            if vl < lb:
                belong = 0
            elif vl > ub:
                belong = m - 1
            else:
                for i in range(m):
                    cdf = self.triang_cdf(vl, fuzzy_list[i])
                    if 0.125 <= cdf <= 0.875:
                        belong = i
                        break

            next_states = set(fuzzy_dict[belong])
            possible = len(next_states)
            if possible == 1:
                frct = fuzzy_list[list(next_states)[0]][1]
            elif possible == 0:
                frct = fuzzy_list[belong][1]
            else:
                frct = sum(fuzzy_list[j][1] for j in next_states) / possible
            frct_result.append(frct)

        return pd.Series(frct_result[-self.h:], index=self.ahead_idx)

    def FTS_My(self, m=10):
        iqr = self.trains.quantile(.75) - self.trains.quantile(.25)
        lb = self.trains.quantile(.25) - 1.5 * iqr
        ub = self.trains.quantile(.75) + 1.5 * iqr
        his_data = self.trains.to_list()
        radius = (ub - lb) / (m * 2)
        fuzzy_list = [
            [lb - 1 * radius + 2 * i * radius, lb + 1 * radius + 2 * i * radius, lb + 3 * radius + 2 * i * radius]
            for i in range(m)]

        location = [-1] * len(his_data)
        for j in range(len(his_data)):
            if his_data[j] <= lb:
                location[j] = 0
            elif his_data[j] >= ub:
                location[j] = m - 1
            else:
                for i in range(0, m):
                    cdf = self.triang_cdf(his_data[j], fuzzy_list[i])
                    if 0.125 <= cdf <= 0.875:
                        location[j] = i
                        break

        fuzzy_dict = {i: [] for i in range(0, m)}
        for i in range(0, len(his_data) - 1):
            fuzzy_dict[location[i]].append(location[i + 1])

        frct_result = [his_data[0]]
        for current in range(len(his_data) + self.h - 1):
            if current < len(his_data):
                vl = his_data[current]
            else:
                vl = frct_result[-1]

            belong = int(m / 2)
            if vl < lb:
                belong = 0
            elif vl > ub:
                belong = m - 1
            else:
                for i in range(0, m):
                    cdf = self.triang_cdf(vl, fuzzy_list[i])
                    if 0.125 <= cdf <= 0.875:
                        belong = i
                        break
            possible = len(set(fuzzy_dict[belong]))
            if possible == 1:
                frct = fuzzy_list[(fuzzy_dict[belong][0])][1]
            elif possible == 0:
                frct = fuzzy_list[belong][1]
            else:
                frct = 0
                for j in set(fuzzy_dict[belong]):
                    frct += (fuzzy_list[j][1] / possible)
            frct_result.append(frct)

        return pd.Series(frct_result[-self.h:], self.ahead_idx)
    # ==================STS: Fuzzy Models==================
