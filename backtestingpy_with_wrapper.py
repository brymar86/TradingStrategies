# -*- coding: utf-8 -*-
"""
Created on Sun Sep 11 16:59:59 2022

@author: user
"""
# know where you store Python 

   # Import Libraries
from binance.client import Client
#from keys_secrets import Binance_API#
import numpy as np
import pandas as pd
import yfinance as yf
import ta
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from datetime import date, timedelta, datetime
import seaborn as sns
import matplotlib.pyplot as plt
import math
from BinanceData import BinanceData

#Parent Dir to path to make it possible to import BinanceData and BinanceTrader.

import sys
sys.path.append("../")
from BinanceData import BinanceData
from BinanceTrader import BinanceTrader

client = Client()

# C:\Users\user\anaconda3\pkgs\pip-22.1.2-py38haa95532_0

# =============================================================================
# This is a different way of doing the same thing I did in TradePro_plus_BY.py
# This is how one can use the backtesting.py to backtest a strategy and then 
# pass said strategy into a trader class object given that the object is set up to 
# handle dictonary
# =============================================================================

#instantiate data
data = BinanceData()

btc_daily = data.getdailydata('BTC', 'USDT',  '2019-01-01') # from jan 1 2019
btc_hourly = data.gethourdata('BTC', 'USDT', 4320) # 4 months
btc_minute = data.getminutedata('BTC', 'USDT', 10080) # 2 weeks

#Add signals, take_profit_price and stop_loss_price to all dataframes so the strategy can assign values to those columns
btc_daily['signal'] = None
btc_daily['take_profit_price'] = None
btc_daily['stop_loss_price'] = None

btc_hourly['signal'] = None
btc_hourly['take_profit_price'] = None
btc_hourly['stop_loss_price'] = None

btc_minute['signal'] = None
btc_minute['take_profit_price'] = None
btc_minute['stop_loss_price'] = None


# =============================================================================
#
# =============================================================================
# This is backtesting.py ONLY! IF YOU JUST WANT TO USE BACKTESTING.PY'S Backtest
# =============================================================================

#TradePro Class
class TradePro(Strategy):
    
    """backtesting,py Class object.  
    
     Here we are using this to build our Strategy and Perform a backtest only. 
    
    https://kernc.github.io/backtesting.py/doc/backtesting/backtesting.html#backtesting.backtesting.Strategy.I
    
    """
    window_ema1 = 8
    window_ema2 = 14
    window_ema3 = 50
    
    def init(self):
        
        """Initialize the strategy. Override this method. Declare indicators (with Strategy.I()). 
        Precompute what needs to be precomputed or can be precomputed in a vectorized fashion 
        before the strategy starts."""
        
        # use these three guys in the classes below
        close = self.data.Close
        high = self.data.High
        low = self.data.Low
        
        # using oop to use the ta library and bactetsing class together!
        self.stoch_k = self.I(ta.momentum.stochrsi_k, pd.Series(close))
        self.stoch_d = self.I(ta.momentum.stochrsi_d, pd.Series(close))
        self.EMA_8 = self.I(ta.trend.ema_indicator, pd.Series(close), window = self.window_ema1)
        self.EMA_14 = self.I(ta.trend.ema_indicator, pd.Series(close), window = self.window_ema2)
        self.EMA_50 = self.I(ta.trend.ema_indicator, pd.Series(close), window = self.window_ema3)
        
        # The indicator provide an indication of the degree of price volatility. 
        # Strong moves, in either direction, are often accompanied by large ranges,
        # or large True Ranges.
        self.atr = self.I(ta.volatility.average_true_range, pd.Series(high), pd.Series(low), pd.Series(close))
        
    def next(self):
        
        """Main strategy runtime method, called as each new Strategy.data instance 
        (row; full candlestick bar) becomes available. This is the main method where strategy 
        decisions upon data precomputed in Strategy.init() take place."""
        
        
        price = self.data.Close[-1]
        # check for stochastic_rsi_kline crossing over stochastic rsi d line and price beating ema8, ema14, ema 50
        if (crossover(self.stoch_k, self.stoch_d) and 
            price > self.EMA_8 and 
            self.EMA_8 > self.EMA_14 and
            self.EMA_14 > self.EMA_50):
            sl = price - self.atr*3 # stop loss if 3 x biger than 14 day window of high/low/close volatility.
            tp = price + self.atr*2 # not getting greedy we take 2 times the average true vol. 
            # self.buy(sl = sl, tp = tp) # if this is not in the backtest will not run. THIS HAS TO BE OFF IF 
            # PASSING TO MOST TRADING OJECTS
            
            # I am commenting out the self.buy method call and adding the signal, tp price and sl price to self.data 
            
            self.data.signal[-1] = 'buy'
            self.data.take_profit_price[-1] = tp
            self.data.stop_loss_price[-1] = sl


# bt = Backtest(btc_daily, TradePro, cash = 30000, commission=.001)
# bt.run() # works with    self.buy(sl = sl, tp = tp)

bt_h = Backtest(btc_hourly, TradePro, cash = 30000, commission=.001)
bt_h.run()

# =============================================================================
# Creating a Wrapper Class object 
# =============================================================================


class StrategyWrapper:
    
    def __init__(self, strategy):
        
        self.strategy = strategy

    def signals(self, data):
        
        """
        Receives a dictionary containing historical data as Pandas DataFrame. 
        Example format of input:
            
            Quesiton what does the Dataframe look like?

        
        {
            'BTCUSDT': df, 
            'ETHUSDT': df
        }
        
        Returns a signal dictionary:
        {
            'BTCUSDT':{
                'signal': 'buy',
                'take_profit_price': 100000
                'take_profit_price': 1000
            },
            'ETHUSDT':{
                'signal': 'sell',
                'take_profit_price': 100000
                'take_profit_price': 1000
            }
        }
        """

        result = {}
        for symbol, df in data.items():
            # Add signal, take_profit_price and stop_loss_price to input data
            df['signal'] = None
            df['take_profit_price'] = None
            df['stop_loss_price'] = None
            
            # Run backtest
            backtest_result = Backtest(df, TradePro, cash=10_000, commission=.001)
            
            
            signal = backtest_result._data.iloc[-1]['signal']
            take_profit_price = backtest_result._data.iloc[-1]['take_profit_price']
            stop_loss_price = backtest_result._data.iloc[-1]['stop_loss_price']
            
            # Check if the latest row of the _data DataFrame contains a signal. If so add it to the results
            if signal:
                result[symbol] = {'signal': signal, 'take_profit_price': take_profit_price, 'stop_loss_price': stop_loss_price}
            
            return result
        
strategy = StrategyWrapper(TradePro)  #


# =============================================================================
# testing strategy outside of the BinanceTrader Class for verification:
# =============================================================================

data = BinanceData()
btc_hourly = data.gethourdata('BTC', 'USDT', 1000)
signals = strategy.signals({'BTCUSDT': btc_hourly})
print(signals)

# =============================================================================
# Optional.  probably going to use the former as we will need to use 
# class method in testnet api or live api keys to pass into 
# binancetrader 
# =============================================================================
# trader = BinanceTrader(
#         symbols=['BNB/BUSD', 'ETH/BUSD'],
#         timeframe='1m',
#         strategy=strategy,
#         strategy_name='TradePro_backtest-TEST',
#         historical_data_period=100,
#         trade_size_percent=5, # explain
#         take_profit_percent=10, # Can this be something like buyprice * 2vol instead of a percent.
#         stop_loss_percent=10  # can this be trailing? 
#     )

# trader.Strategy_Loop()
