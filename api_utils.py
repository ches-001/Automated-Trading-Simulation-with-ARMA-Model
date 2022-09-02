import requests, json, os
from dotenv import load_dotenv

load_dotenv('.env')

class TwelveDataApiUtils:
    def __init__(self, symbol:str, interval:str, outputsize:int):
        self.symbol = symbol
        self.interval = interval
        self.outputsize = outputsize
        self.api_domain = 'https://twelve-data1.p.rapidapi.com/'

        self.headers = {
            "X-RapidAPI-Key": os.getenv('X-RapidAPI-Key'),
            "X-RapidAPI-Host": os.getenv('X-RapidAPI-Host')
        }

        self.querystring = {
            'symbol':self.symbol,
            'interval':self.interval,
            'outputsize':self.outputsize,
            'format':'json',
        }

        self.dir_name_ = 'data'

        self.filename = '{symbol}-{interval}.json'.format(
            symbol=self.symbol.replace('/', '-'), interval=self.interval)


    def writeData(self)->None:
        if not os.path.isdir(self.dir_name_): os.mkdir(self.dir_name_)
        
        file = os.path.join(self.dir_name_, self.filename)
        if os.path.isfile(file): os.remove(file)

        url = f'{self.api_domain}time_series'
        r = requests.get(url, headers=self.headers, params=self.querystring)
        content = json.loads(r.text)

        self.raiseErrorOnBadRequest(content)
        
        with open(file, 'w') as f:
            json.dump(content['values'], f)
        f.close()


    def appendData(self, new_content)->None:
        file = os.path.join(self.dir_name_, self.filename)

        if not os.path.isfile(file): self.writeData()

        data = json.load(open(file, 'r'))

        if data[0]['datetime'] != new_content['datetime']:
            content = {
                'datetime': new_content['datetime'],
                'open': new_content['open'],
                'high': new_content['high'],
                'low': new_content['low'],
                'close': new_content['close'],
                'volume':new_content['volume']
            }
            json.dump([content] + data, open(file, 'w'))


    def getCurrentPrice(self)->dict:
        url = f'{self.api_domain}price'
        r = requests.get(url, headers=self.headers, params=self.querystring)
        content = json.loads(r.text)

        self.raiseErrorOnBadRequest(content)
        return content


    def getQuote(self)->dict:
        url = f'{self.api_domain}quote'
        r = requests.get(url, headers=self.headers, params=self.querystring)
        content = json.loads(r.text)

        self.raiseErrorOnBadRequest(content)
        return content

    
    def raiseErrorOnBadRequest(self, request_content:dict)->None:
        if 'status' in request_content.keys() and request_content['status'] == 'error':
            raise ValueError(request_content)