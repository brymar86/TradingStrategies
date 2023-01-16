# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 12:05:51 2022

@author: user
"""


from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from binance.client import Client
import pandas as pd
import numpy as np
import json
import ta
from backtesting.lib import crossover
from decouple import config
import time
import orjson
import websocket
import pandas as pd
from threading import Thread
from datetime import datetime
from traceback import format_exc

api_key = config("API_KEY") # these are testnet keys
secret_key = config("SECRET_KEY") # these are testnet keys

# THE WS Class Code was assisted with the help of my great frend Georgie who helped me understand 
# the Binance Websocket API Ingestion procedures and dictionary mappings. 

class WS(Thread):
    def __init__(self, symbols: list, timeframes: list, use_testnet=False):
        '''
        Opens Binance WS klines stream for multiple symbols and timeframes.
        Valid timeframes: 1s, 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M
        Streams format: <symbol>@kline_<interval>
        Ref: https://binance-docs.github.io/apidocs/spot/en/#kline-candlestick-streams
        
        Args:
            symbols: List of symbols to get data for
            timeframes: List of timeframes to get data in
            logger: Logger object, set to None to use basic print for logging
            use_testnet: Use testnet data
        
        Attributes:
            data: Dict where candles data for each symbol and timeframe is stored.
            Each kay of data is a symbol and the value is a dict where each key is a timeframe.
            Each timeframe value is a dict where keys are UNIX timestamps of the candles and
            values are lists of open, high, low, close, volume. The current candle is constantly updated.
            Example structure:
            {
                'BTCUSDT': {
                    '1m': {
                        1662556860000: [18849.36, 18855.11, 18841.22, 18849.5, 142.32401]
                        1662556870000: [18849.36, 18855.11, 18841.22, 18849.5, 142.32401]
                        }
                }, 
                'ETHUSDT': {
                    '1m': {
                        1662556860000: [1535.98, 1536.44, 1534.33, 1536.43, 932.4534]
                        1662556870000: [1535.98, 1536.44, 1534.33, 1536.43, 932.4534]
                    }
                }
            }
        '''
        super().__init__()
        
        self.ws = None
        if use_testnet:
            self.ws_url = 'wss://testnet.binance.vision/stream?streams='
        else:
            self.ws_url = 'wss://stream.binance.com:9443/stream?streams='
        
        self.data = {}
        self.streams = []
        for symbol in symbols:
            self.data[symbol] = {}
            for timeframe in timeframes:
                self.ws_url += f'{symbol.lower()}@kline_{timeframe.lower()}/'
                self.data[symbol][timeframe] = {}
        self.ws_url = self.ws_url.rstrip('/')
        
        self.connected = False
        self.daemon = True  # If set to True the thread will not block execution.
    
    def log(self, msg, level='INFO'):
        print(f'{datetime.now()} [{level.upper()}] {msg}')
    
    def connect(self):
        self.ws = websocket.WebSocketApp(
            url=self.ws_url,
            on_message=self.on_message,
            on_close=self.on_close,
            on_error=self.on_error
        )
        self.connected = True
    
    def reconnect(self):
        del self.ws
        self.ws = None
        self.connect
    
    def start_stream(self):
        self.ws.run_forever()
    
    def on_message(self, ws, msg):
        '''
        Receives a WS messages and saves the data in self.data
        Kline message format:
        {
            "stream": "ethusdt@kline_1m",
            data: {
                "e": "kline",     // Event type
                "E": 123456789,   // Event time
                "s": "BNBBTC",    // Symbol
                "k": {
                    "t": 123400000, // Kline start time
                    "T": 123460000, // Kline close time
                    "s": "ETHUSDT",  // Symbol
                    "i": "1m",      // Interval
                    "f": 100,       // First trade ID
                    "L": 200,       // Last trade ID
                    "o": "0.0010",  // Open price
                    "c": "0.0020",  // Close price
                    "h": "0.0025",  // High price
                    "l": "0.0015",  // Low price
                    "v": "1000",    // Base asset volume
                    "n": 100,       // Number of trades
                    "x": false,     // Is this kline closed?
                    "q": "1.0000",  // Quote asset volume
                    "V": "500",     // Taker buy base asset volume
                    "Q": "0.500",   // Taker buy quote asset volume
                    "B": "123456"   // Ignore
                }
            }
        }
        '''
        msg = orjson.loads(msg)
        # here I'm literally just parsing the data given I have a ws stream in the proper format.
        if 'stream' in msg and 'kline' in msg['stream']:
            symbol = msg['data']['s']
            timeframe = msg['stream'].split('_')[1]
            kline = msg['data']['k']
            # setting the value to open, high, low, close, volume from the kline dictionary
            self.data[symbol][timeframe][kline['t']] = [float(kline['o']), float(kline['h']), float(kline['l']), float(kline['c']), float(kline['v'])]
        else:
            self.log(f'WS message not kline: {msg}')

    def on_error(self, ws, error):
        self.log(error)

    def on_close(self, msg):
        self.log('WS connection closing', 'WARNING')
        self.connected = False

    def get_candles(self, symbol, timeframe):
        if not self.data[symbol.upper()][timeframe]:
            return pd.DataFrame(columns=['Time','Open', 'High', 'Low', 'Close', 'Volume'])
        df = pd.DataFrame(self.data[symbol.upper()][timeframe]).transpose()
        df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        df.index = pd.to_datetime(df.index, unit='ms')
        df['Time'] = df.index
        return df

    def run(self):
        while True:
            try:
                self.log('Starting stream')
                self.connect()
                self.start_stream()
            except:
                self.log(format_exc(), 'error')
                del self.ws
                self.ws = None
                time.sleep(1)


class BinanceData:
    
    
    client = Client() # historical prices.  Force a decision to API TESTNET or Live
    
            
    @classmethod
    def client_live(cls, api_live, api_secret_live):
        
        """In case you need to switch to different keys
        
        Feed your real api, and secret keys for Live Trading."""
        
        cls.client = Client(api_live, api_secret_live)
    
    @classmethod
    def client_testnet(cls, api, api_secret):
        """Use testnet.binance.vision
        
        us user tld = 'us' .
        
        """
        cls.client = Client(api, api_secret, testnet = True) 
    
    def dataprocess(self, frame):
        #just slicing the columns we want
        frame = frame.iloc[:, :6]
        frame.columns = ['Time', 'Open', 'High', 'Low', 'Close', 'Volume']  
        frame = frame.set_index('Time')               
        frame = frame.astype(float)
        frame.index = pd.to_datetime(frame.index, unit = 'ms')
        # frame['Ticker'] = self.symbol + 'USDT'
        return frame

    # NOTE can definitely merge these data gathering functions
    def getdailydata(self, symbol: str, baseasset: str , start:str):
        """Pulling Binance API data
    
        symbol = Ticker 
        baseasset = base asset of choise. (USDT often)
        Your just need to add the symbol 
    
        """
        frame = pd.DataFrame(self.client.get_historical_klines(symbol + baseasset , '1d', start))
        return self.dataprocess(frame)

    def gethourdata(self, symbol:str , baseasset:str, lookback:str):
        
        """This is a way to get hour data:
        
        symbol = ticker
        lookback = how long the timer period you're sampling'
        """
        if isinstance(lookback, int): 
            lookback = str(lookback)
        
        frame = pd.DataFrame(self.client.get_historical_klines(symbol + baseasset, '1h', lookback + 'hour ago UTC'))# YOU HAVE TO DO THIS OR ERROR
        return self.dataprocess(frame)
    
    def getminutedata(self, symbol:str, baseasset:str, lookback: str):
        
        if isinstance(lookback, int):
            lookback = str(lookback)
        
        frame = pd.DataFrame(self.client.get_historical_klines(symbol + baseasset, '1m', lookback + 'min ago UTC'))
        return self.dataprocess(frame)
        
    
    def websocket(self, symbols: list, timeframes: list, use_testnet: bool = False):
        
        """
        
        This method creates an instance of the WS class that handles websocket data stream for multiple symbols and timeframes.
        Args:
            symbols: List of symbols to get data for, e.g. ['BTCUSDT', 'ETHUSDT']
            timeframes: List of timeframes to get data for, e.g. ['30m', '1h']
            use_testnet: If set to True the data source will be Binance Testnet
        
        Notes:
            1. Websocket data is streaming, e.g. when don't get any historical data from it.
            2. To get the candles dataframe use get_websocket_candles() method.
            3. Call websocket only once on initialization.
        """
        self.ws = WS(symbols=symbols, timeframes=timeframes, use_testnet=use_testnet)
        self.ws.start()
    
    
    def get_websocket_candles(self, symbol: str, timeframe: str):
        '''
        Get candles data collected by the websocket object.
        Returns Pandas DataFrame in the same format as getminutedata and gethourdata getdailydata methods.
        Args:
            symbol: The symbol to get data for. Must be one of the symbols passed to the websocket() method.
            timeframe: The timeframe to get data for. Must be one of the timeframes passed to the websocket() method.
        '''
        return self.ws.get_candles(symbol=symbol, timeframe=timeframe)


# if __name__ == '__main__':
    # Test daily data
    # bin_trade = BinanceData()
    # bin_trade.getdailydata('BTC', 'USDT' , '2019-01-01')

    # Test websocket data ==> Georgi These Need now to go to BUSD
    # bin_trade = BinanceData()
    # bin_trade.websocket(symbols=['BTCBUSD', 'ETHBUSD'], timeframes=['1m'])

    # As the WS data collector is running on it's separate thread in parallel we can have a loop in which we periodically fetch the collected data and print it.
    # while True:
    #     print(bin_trade.get_websocket_candles('BTCBUSD', '1m'))
    #     print(bin_trade.get_websocket_candles('ETHBUSD', '1m'))
    #     time.sleep(5)
