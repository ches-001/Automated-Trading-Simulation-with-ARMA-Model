import pandas as pd
import json, os, time
from requests import ConnectionError
from ts_model import ARMA
from api_utils import TwelveDataApiUtils
from datetime import datetime
import matplotlib.pyplot as plt

class PipeLine:
    def __init__(self, model:ARMA, td:TwelveDataApiUtils, session_duration:int=None):
        self.model = model
        self.td = td
        self.session_duration = session_duration
        self.date_ = datetime.now().strftime('%Y-%m-%d')
        self.dir_name_ = 'logs'
        self.log_filename_ = f'log-{self.td.filename}-{self.date_}.json'
        self.closing_prices_ = None
        self.MakeNewDataRecord()
        self.getClosingPrices()
        self.trainModel()
        self.start_time_ = self.getCurrentTime()
        self.stop_time_ = self.start_time_ + pd.to_timedelta(self.session_duration, unit='S')\
        if self.session_duration else None
        self.info_to_log_ = None
        self.profit_loss_count_ = {'profit':0, 'loss':0}
        

    def getCurrentTime(self)->pd.Timestamp:
        return pd.Timestamp(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))


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


    def makePrediction(self, timestamp:str)->dict:
        n = self.model.n_per_sample_
        _input = self.closing_prices_.iloc[-n-1:-1]
        prediction = self.model.predict(_input)
        
        response = {
            'datetime':timestamp,
            'prediction':prediction[0],
        }
        return response


    def logInfo(self, record:dict)->None:
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

    
    def recordProfitLossCount(self, profit:bool)->None:
        if profit:
            self.profit_loss_count_['profit'] += 1
        else:
            self.profit_loss_count_['loss'] += 1

    
    def finalEvents(self)->None:
        self.MakeNewDataRecord()
        self.trainModel(use_all=True)
        #plot profit and loss counts in bar chart
        plt.bar(self.profit_loss_count_.keys(), self.profit_loss_count_.values(), width=0.5)


    def eventLoop(self)->None:        
        cycles = 0
        loop_delay = 10
        
        while True:
            if self.stop_time_:
                if self.getCurrentTime() >= self.stop_time_:
                    print(f'>>>>> session duration has elapsed at {self.getCurrentTime()}')
                    self.finalEvents()
                    break

            current_quote = None
            try:
                current_quote = self.td.getQuote()
            except ConnectionError:
                print('>>>>> connection error, retrying...')
                continue
            
            if not current_quote['is_market_open']:
                print('>>>>> The Market is currently closed')
                if cycles != 0:
                    self.finalEvents()
                break
            
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
                    self.recordProfitLossCount(prediction_status['accurate_prediction'])
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
                cycles+=1
            time.sleep(loop_delay)