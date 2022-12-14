from ts_model import ARMA
from pipeline import PipeLine
from api_utils import TwelveDataApiUtils
from sklearn.linear_model import LinearRegression

if __name__ == '__main__':
    stock_symbol = 'MSFT'
    interval = '1min'
    n_samples = 5000
    active_duration = 3600 #1 hour

    api_utils = TwelveDataApiUtils(stock_symbol, interval, n_samples)
    regressor = LinearRegression()
    arma_model = ARMA((2, 1), base_regressor=regressor, estimator_lags=3)
    pipeline = PipeLine(arma_model, api_utils, session_duration=active_duration)
    pipeline.eventLoop()