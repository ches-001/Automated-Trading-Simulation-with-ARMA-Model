import numpy as np
import pandas as pd
from typing import Union
from sklearn.linear_model import LinearRegression


class AutoRegressionModel:
    r"""
    Auto Regression Model
    """
    def __init__(self, base_regressor=LinearRegression(), nlags:int=2):
        self.nlags = nlags
        self.base_regressor = base_regressor
        self.n_per_sample_ = self.nlags
        
    def fit(self, X:pd.Series):
        data_df = pd.DataFrame()
        
        for i in range(1, self.nlags+1):
            data_df[f'lag_{i}'] = X.shift(i)

        data_df['target'] = X
        data_df = data_df.dropna()
        
        X = data_df.drop('target', axis=1).values
        Y = data_df['target'].values.reshape(-1)
        
        self.base_regressor.fit(X, Y)

    def predict(self, X:Union[np.ndarray, pd.Series])->np.ndarray:
        if isinstance(X, pd.Series):
            X = X.values
        X = X.reshape(1, -1)
        prediction = self.base_regressor.predict(X)
        return prediction.reshape(-1)


class MovingAverageModel:
    r"""
    Moving Average Model
    """
    def __init__(self, base_regressor=LinearRegression(), estimator_lags=2, nlags:int=2):
        self.nlags = nlags
        self.base_regressor = base_regressor
        self.estimator_lags = estimator_lags
        self.n_per_sample_ = self.nlags + self.estimator_lags - 1
        self.ar_estimator = self.ar_estimator = AutoRegressionModel(base_regressor, self.estimator_lags)


    def fit(self, X:pd.Series):
        self.ar_estimator.fit(X)

        data_df = pd.DataFrame()
        estimates = X.rolling(self.estimator_lags).apply(lambda x: self.ar_estimator.predict(x.values))
        errors = X - estimates

        for i in range(1, self.nlags+1):
            data_df[f'error_lag_{i-1}'] = errors.shift(i)
        
        data_df['target'] = X
        data_df = data_df.dropna()

        X = data_df.drop('target', axis=1).values
        Y = data_df['target'].values.reshape(-1)

        self.base_regressor.fit(X, Y)


    def predict(self, X:Union[np.ndarray, pd.Series])->np.ndarray:
        if not isinstance(X, pd.Series):
            X = pd.Series(X)

        estimates = X.rolling(self.estimator_lags).apply(lambda x : self.ar_estimator.predict(x.values))
        errors = X - estimates
        errors = errors.dropna().values.reshape(1, -1)
        
        prediction = self.base_regressor.predict(errors)
        return prediction.reshape(-1)


class ARMA:
    r"""
    AutoRegression Moving Average Model
    """
    def __init__(self, pq:tuple, base_regressor=LinearRegression(),  estimator_lags:int=2):
        self.base_regressor = base_regressor
        self.p, self.q = pq
        self.estimator_lags = estimator_lags
        self.n_per_sample_ = self.estimator_lags + self.q - 1
        self.ar_estimator = AutoRegressionModel(base_regressor, self.estimator_lags)
        self.coef_ = None
        self.intercept_ = None

        if self.p >= self.estimator_lags:
             raise ValueError(f'p({self.p}) must be less than estimator_lag({self.estimator_lags})')


    def _getLags(self, series:pd.Series, nlags:int, lag_type:str)->pd.DataFrame:
        df = pd.DataFrame()
        for i in range(1, nlags+1):
            df[f'{lag_type}_{i}'] = series.shift(i)
        return df


    def fit(self, X:pd.Series):
        self.ar_estimator.fit(X)

        estimates = X.rolling(self.estimator_lags).apply(lambda x: self.ar_estimator.predict(x.values))
        errors = X - estimates

        X_lag = self._getLags(X, self.p, 'x_lag')
        error_lags = self._getLags(errors, self.q, 'error_lag')

        data_df = pd.concat((X_lag, error_lags), axis=1)
        data_df['target'] = X
        
        #drop na values
        data_df = data_df.dropna()
        X = data_df.drop('target', axis=1).values
        Y = data_df['target'].values.reshape(-1)

        self.base_regressor.fit(X, Y)


    def predict(self, X:Union[np.ndarray, pd.Series])->np.ndarray:
        if not isinstance(X, pd.Series):
            X = pd.Series(X)

        points = None

        if self.q > 0:
            estimates = X.rolling(self.estimator_lags).apply(lambda x : self.ar_estimator.predict(x.values))
            errors = X - estimates

            if self.p > 0:
                X = X.iloc[-self.p:]

                data_df = pd.DataFrame(data={'x':X, 'errors':errors})
                points = np.concatenate((
                    data_df['x'].dropna().values, data_df['errors'].dropna().values
                )).reshape(1, -1)

            else:
                points = errors.dropna().values.reshape(1, -1)
        
        else:
            points = X.iloc[-self.p:].values.dropna().reshape(1, -1)

        prediction = self.base_regressor.predict(points)
        return prediction.reshape(-1)