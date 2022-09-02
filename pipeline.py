import pandas as pd
import json, os, time
from requests import ConnectionError
from ts_model import ARMA
from api_utils import TwelveDataApiUtils
from datetime import datetime

class PipeLine:
    def __init__(self, model:ARMA, td:TwelveDataApiUtils):
        self.model = model
        self.td = td

        self.date_ = datetime.now().strftime('%Y-%m-%d')
        self.dir_name_ = 'logs'
        self.log_filename_ = f'log-{self.date_}.json'
        self.closing_prices_ = None
        self.MakeNewDataRecord()
        self.getClosingPrices()
        self.trainModel()

        self.info_to_log_ = None
        

    def MakeNewDataRecord(self)->None:
        return self.td.writeData()


    def getClosingPrices(self)->None:
        data_file = os.path.join(self.td.dir_name_, self.td.filename)

        data = None
        with open(data_file, 'r') as f:
            data = json.loads(f.read())
        f.close

        self.closing_prices_ = pd.Series([float(i['close']) for i in reversed(data)])
        self.closing_prices_.index = [i['datetime'] for i in reversed(data)]
        self.closing_prices_.index = self.closing_prices_.index.astype('datetime64[ns]')


    def trainModel(self, use_all=False)->None:
        if not use_all:
            return self.model.fit(self.closing_prices_.iloc[:-1])

        return self.model.fit(self.closing_prices_)


    def makePrediction(self, timestamp)->dict:
        n = self.model.estimator_lags + self.model.q - 1
        _input = self.closing_prices_.iloc[-n-1:-1]
        prediction = self.model.predict(_input)
        
        response = {
            'datetime':timestamp,
            'prediction':prediction[0],
        }
        return response


    def logInfo(self, record)->None:
        if not os.path.isdir(self.dir_name_):
            os.mkdir(self.dir_name_)

        log_file = os.path.join(self.dir_name_, self.log_filename_)
        if not os.path.isfile(log_file):
            json.dump([record], open(log_file, 'w'))

        else:
            data = json.load(open(log_file, 'r'))
            data = [record] + data
            json.dump(data, open(log_file, 'w'))
    

    def getPredictionStatus(self)->dict:
        pred_outcome = self.info_to_log_['prediction'] > self.info_to_log_['previous_price']
        actual_outcome = self.info_to_log_['actual_price'] > self.info_to_log_['previous_price']

        prediction_status = {
            'accurate_prediction': actual_outcome == pred_outcome,
            'predicted_to_rise': pred_outcome,
            'rise_status': actual_outcome,
        }
        return prediction_status


    def eventLoop(self)->None:        
        cycles = 0
        
        while True:
            current_quote = None
            try:
                current_quote = self.td.getQuote()
            except ConnectionError:
                print('connection error, retrying...')
                continue
            
            """if not current_quote['is_market_open']:
                self.MakeNewDataRecord()
                self.trainModel(use_all=True)
                print('The Market is currently closed')
                break"""
            
            last_timestamp = self.closing_prices_.index[-1]
            quote_timestamp = pd.Timestamp(current_quote['datetime'])

            if (quote_timestamp != last_timestamp) or cycles==0:
                self.MakeNewDataRecord()
                self.getClosingPrices()
                
                previous_timestamp = self.closing_prices_.index[-2].__str__()
                previous_price = self.closing_prices_.iloc[-2]

                pred_data = self.makePrediction(quote_timestamp.__str__())
                prediction_timestamp = pred_data['datetime']
                prediction = pred_data['prediction']
                
                prediction_status = None
                if self.info_to_log_ is not None:
                    self.info_to_log_.update(actual_price=previous_price)
                    prediction_status = self.getPredictionStatus()
                    self.logInfo(self.info_to_log_)
                
                print(
                    f'prediction status at {previous_timestamp}: {prediction_status}\n',
                    f'Price at {previous_timestamp}: USD {previous_price}\n',
                    f'Prediction at {prediction_timestamp}: USD {prediction}\n\n'
                )

                self.info_to_log_ = {
                    'previous_timestamp':previous_timestamp,
                    'previous_price': previous_price,
                    'prediction_timestamp':prediction_timestamp,
                    'prediction':prediction,
                    'actual_price':None,
                }
                #cycles+=1
            time.sleep(10)