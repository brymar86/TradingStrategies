# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 16:46:28 2022

@author: user
"""
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from binance.client import Client
import ta
import pandas as pd
import numpy as np
import json
import ta
from backtesting.lib import crossover
from decouple import config
import time
import time
import orjson
import websocket
import pandas as pd
from threading import Thread
from datetime import datetime
from traceback import format_exc



class BinanceData:
    
    """
    This is the Binance Data for public Github.  I have websockets upon request.
    
    
    """
    client = Client()
    
            
    @classmethod
    def client_live(cls, api, api_secret):
        
        """In case you need to switch to different keys
        
        Feed your real api, and secret keys for Live Trading."""
        
        cls.client = Client(api, api_secret)
    
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
        