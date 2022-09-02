from ts_model import ARMA
from pipeline import PipeLine
from api_utils import TwelveDataApiUtils

if __name__ == '__main__':
    stock_symbol = 'MSFT'
    interval = '1min'
    n_samples = 5000

    api_utils = TwelveDataApiUtils(stock_symbol, interval, n_samples)
    arma_model = ARMA((2, 1), 3)
    pipeline = PipeLine(arma_model, api_utils)
    pipeline.eventLoop()