import os
import math
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsforecast import StatsForecast
from statsforecast.models import SeasonalNaive, AutoARIMA
from sktime.forecasting.ets import AutoETS

# from hts.hierarchy import HierarchyTree
from statsmodels.tsa.exponential_smoothing.ets import ETSModel


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
        return 1.0


class ForecastMethods:
    # ====================model function====================
    #  required input: trains (pd.Series) -> training data, h (int) -> forecasting lengths
    #  output: pd.Series -> forecasting results (length=h)
    #  some functions may be allowed to adjust their parameters
    # ====================model function====================

    def __init__(self, trains: pd.Series, h: int = 5):
        self.trains = trains
        self.h = h
        self.ahead_idx = pd.date_range('2022-01-01', periods=h, freq='M')

    def RunAll(self):  # run all forecasting methods, return a dataframe -> shape: (h, num(methods))
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
        frct_df.replace(0, 1, inplace=True)  # prevent to occur error when calculating error measures

        return frct_df

    # ==================simple methods==================
    def NAIVE(self):
        return pd.Series([self.trains.iloc[-1]] * self.h, index=self.ahead_idx)

    def AVG(self):
        return pd.Series([self.trains.mean()] * self.h, index=self.ahead_idx)

    def SNAIVE(self, season_length=12, frequency='M', n_job=1):
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
    def ARIMA(self, season_length=12, frequency='M', n_job=1):
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
        prediction = fcst.forecast(df=train_data, h=self.h)
        prediction = prediction.set_index('ds')['AutoARIMA']
        prediction.index = self.ahead_idx
        return prediction

    def ETS(self):
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
        train_data = self.trains.copy()
        aaa = ETSModel(train_data, error='add', trend='add', seasonal='add', seasonal_periods=12).fit()
        aan = ETSModel(train_data, error='add', trend='add', seasonal=None).fit()
        ana = ETSModel(train_data, error='add', trend=None, seasonal='add', seasonal_periods=12).fit()
        aiccs = [aaa.aicc, aan.aicc, ana.aicc]
        min_idx = np.argmin(aiccs)
        pred = [aaa, aan, ana][min_idx].forecast(self.h)
        return pred

    def SES(self):
        train_data = self.trains.copy()
        aan = ETSModel(train_data, error='add', trend='add', seasonal=None).fit()
        pred = aan.forecast(self.h)
        pred.index = self.ahead_idx
        return pred

    # ==================traditional methods==================

    # ==================STS: Grey Models==================
    def GM11(self):
        cum_series = self.trains.cumsum()
        near_mean = list(((cum_series.shift(-1) + cum_series) / 2)[:-1])
        Y = np.mat(list(self.trains[:-1])).T
        B = [-near_mean[i] for i in range(len(near_mean))]
        B = np.mat([B, np.ones(len(B))]).T
        beta = np.linalg.inv(B.T.dot(B)).dot(B.T).dot(Y)
        beta = np.array(beta).reshape(2)
        const = beta[1] / beta[0]
        first_val = self.trains.iloc[0]
        frct = [first_val] + [(first_val - const) * math.exp(-beta[0] * k) + const for k in range(1, len(self.trains) + self.h)]
        origin_prct = [first_val] + [frct[i] - frct[i - 1] for i in range(1, len(self.trains) + self.h)]
        return pd.Series(origin_prct[-self.h:], self.ahead_idx)

    def BGM11(self):
        cum_series = self.trains.cumsum()
        Y = np.mat(list(self.trains[:-1])).T
        Q1, Q2, Q3 = self.trains.quantile(0.25), self.trains.median(), self.trains.quantile(0.75)
        LL = min(Q1 - 1.5 * (Q3 - Q1), self.trains.min())
        UL = max(Q3 + 1.5 * (Q3 - Q1), self.trains.max())
        MF = []
        for i in self.trains:
            if i == Q2:
                MF.append(1)
            elif i < Q2:
                MF.append((i - LL) / (Q2 - LL))
            else:
                MF.append((UL - i) / (UL - Q2))
        bg_value = [cum_series.iloc[i - 1] + MF[i] * self.trains.iloc[i] for i in range(1, len(MF))]
        bg_mat = [-bg_value[i] for i in range(len(bg_value))]
        bg_mat = np.mat([bg_mat, np.ones(len(bg_mat))]).T
        beta = np.linalg.inv(bg_mat.T.dot(bg_mat)).dot(bg_mat.T).dot(Y)
        beta = np.array(beta).reshape(2)
        const = beta[1] / beta[0]
        frct = [self.trains.iloc[0]] + [(self.trains.iloc[0] - const) * math.exp(-beta[0] * k) + const
                                        for k in range(1, len(self.trains) + self.h)]
        origin_prct = [frct[0]] + [frct[i] - frct[i - 1] for i in range(1, len(self.trains) + self.h)]
        return pd.Series(origin_prct[-self.h:], self.ahead_idx)

    def RBGM11(self):
        cum_series = self.trains.cumsum()
        Y = np.mat(list(self.trains[:-1])).T
        Q1, Q2, Q3 = self.trains.quantile(0.25), self.trains.median(), self.trains.quantile(0.75)
        LL = max(Q1 - 1.5 * (Q3 - Q1), self.trains.min())
        UL = min(Q3 + 1.5 * (Q3 - Q1), self.trains.max())
        MF = []
        for i in self.trains:
            if i <= LL or i >= UL:
                MF.append(0)
            elif i == Q2:
                MF.append(1)
            elif i < Q2:
                MF.append((i - LL) / (Q2 - LL))
            else:
                MF.append((UL - i) / (UL - Q2))
        bg_value = [cum_series.iloc[i - 1] + MF[i] * self.trains.iloc[i] for i in range(1, len(MF))]
        bg_mat = [-bg_value[i] for i in range(len(bg_value))]
        bg_mat = np.mat([bg_mat, np.ones(len(bg_mat))]).T
        beta = np.linalg.inv(bg_mat.T.dot(bg_mat)).dot(bg_mat.T).dot(Y)
        beta = np.array(beta).reshape(2)
        const = beta[1] / beta[0]
        frct = [self.trains.iloc[0]] + [(self.trains.iloc[0] - const) * math.exp(-beta[0] * k) + const
                                        for k in range(1, len(self.trains) + self.h)]
        origin_prct = [frct[0]] + [frct[i] - frct[i - 1] for i in range(1, len(self.trains) + self.h)]
        return pd.Series(origin_prct[-self.h:], self.ahead_idx)
    # ==================STS: Grey Models==================

    # ==================STS: Fuzzy Models==================
    def FTS_Chen(self, m=10):
        lb, ub = self.trains.min(), self.trains.max()
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
                    cdf = triang_cdf(his_data[j], fuzzy_list[i])
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
                    cdf = triang_cdf(vl, fuzzy_list[i])
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
                    cdf = triang_cdf(his_data[j], fuzzy_list[i])
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
                    cdf = triang_cdf(vl, fuzzy_list[i])
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

