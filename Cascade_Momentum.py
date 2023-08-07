 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 15:06:49 2023

@author: Blacksheep, CFA

This class was inspired by the functional CODE algovibes YouTube Channel provided

UPDATE Aug/6/2023: This will break as the site used to handle survivorship biased changed their website so we can't scrape it.

"""



import os
import csv
import time
import pickle
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
import datetime as dt
from pathlib import Path
import pandas_datareader
import statsmodels.api as sm
import matplotlib.pyplot as plt
from nasdaq_2016 import Nasdaq2016
from scipy.stats import ttest_ind
from datetime import date, timedelta
import pandas_datareader.data as reader
from pandas.tseries.offsets import MonthEnd
from statsmodels.sandbox.regression.predstd import wls_prediction_std




class CascadeMomentum:
    
    def __init__(self, start_date: str = '2016-01-01', qqq_start: str = '2016-01-01'):
        
        """I'm setting the start_date to 2016-01-01 as that's the most relevant date given our 
        data.  You can set it to whatever date you want but we have history only till 2016"""
        
        self.start_date = start_date
        self.qqq_start = qqq_start
        
        # ONly want the data once         
        nasdaq_data = Nasdaq2016()
        nasdaq_tickers = nasdaq_data.tickers_2016()
        self.tickers = nasdaq_tickers['tickers']
        self.df_isactive_b = nasdaq_tickers['df_isactive']
        
        self.price_df = yf.download(self.tickers, start=self.start_date)['Adj Close']
        self.price = self.price_df.dropna(axis=1, how='all')
        self.qqq = yf.download('QQQ', start=self.qqq_start)['Adj Close']
        
        self.df_isactive = self.df_boolean()
        
#        I moved the rolling returns to a dictionary to make it more flexible. However the use of those parameters
        # is hard coded in the get_top() method so it relays on those rolling return periods specifically so it doesn't make sense
        # to create a init variable for those parameters.

        self.rolling_returns = {
            '12_M': self.get_rolling_ret(12, 'M'),
            '6_M': self.get_rolling_ret(6, 'M'),
            '3_M': self.get_rolling_ret(3, 'M')
        }
        self.mtl = self.return_df()
        
    def df_boolean(self) -> pd.DataFrame:
        
        """I'm creating a boolean dataframe that provides me ticker mappings back 
        to 2016 of stocks that I pulled today that were in and out of the Nasdaq 
        since 2016."""
        
        # price dataframe returns = values, columns = stocks, rows = dates
        df = self.price
        # boolean = values, columns = dates, rows = 2016 Nasdaq Tickers
        df_isactive = self.df_isactive_b
        # this is a mask.  If the tickers in the price dataframe are tickers in boolean dataframe, keep them. (Mask version of dropna)
        index = df_isactive.index.isin(df.columns)
        # now apply mask to the boolean matrix to have a boolean matrix only of tickers that are existing in both price and boolean matrices
        df_isactive = df_isactive.loc[index]
        
        return df_isactive
    
    def return_df(self, period: str = 'M') -> pd.DataFrame:
        
        """
        This is going to create the dataframe of 1+r for each stock.  can be 
        used to calculate the get_rolling return and performance metrics
        
        index: datetime objects
        columns: stocks/tickers
        values: 1 + r levels of the stock. 
        
        """
        df = self.price
        df.index = pd.to_datetime(df.index)
        # here i'm calculating the (1+r) return function. 
        mtl = (df.pct_change() + 1)[1:].resample(period).prod()
        
        return mtl

    def get_rolling_ret(self, window: int, period: str) -> pd.DataFrame:
        
        """I will provide a bit of flexibility if one doesn't want 
        a monthy rolling cascade otherwise It will assume you want 
        a monthly rolling cascade function"""
        # [georgi] I added the period to the return_df() call
        mtl = self.return_df(period=period)

        return mtl.rolling(window).apply(np.prod)
    
    def filter_return(self, date: str, df_cascade: pd.DataFrame) -> pd.DataFrame:
        
        """
        date: date you want to start strategy evaluation at.  Should be 
        between orgination_date of historical_stocks (2016) < date < Today
        
        df_boolean: df_isactive_cleaned this should be from the method
        cascade_df: these will be the ret_12, ret_6, ret_3 dataframes
        """
        
        df_boolean = self.df_isactive
        #this is goin to return an integer
        ix_ = np.argmax(pd.to_datetime(date) >= df_boolean.columns)
        #create the mapping of stocks that exist
        df_boolean = df_boolean[df_boolean.iloc[:, ix_]]
        return df_cascade[df_boolean]
    
    def get_top(self, date: str) -> pd.Series:
         
        ret_12, ret_6, ret_3 = self.rolling_returns['12_M'], self.rolling_returns['6_M'], self.rolling_returns['3_M']
        ret_12m, ret_6m, ret_3m = self.filter_return(date, ret_12), self.filter_return(date, ret_6), self.filter_return(date, ret_3)
        #note df.loc[date].nlargest(50) takes the df and transposes the tickers to the index
        top_50 = ret_12m.loc[date].nlargest(50).index
        top_30 = ret_6m.loc[date, top_50].nlargest(30).index
        top_10 = ret_3m.loc[date, top_30].nlargest(10).index
        # this is a list of tickers that is being returned. 
        return top_10
    
    def point_performance(self, date: str) -> pd.DataFrame:
    
        """portfolio point performance for a given index date location
        within the 1_r matrix."""
    
        mtl = self.return_df()
    
        portfolio = mtl.loc[date:, self.get_top(date)][1:2]
        
        # This is equal weighted and that's fine for now but I  need to make 
        # certain portfolio weighting decisions GIVEN I have a signal coming 
        # from conditionals i'm going to make from 12-2 momentum and NLP sentiment
        # of the top 10 stocks. 
        
    
        return portfolio.mean(axis=1)
    
#     it may be good idea to add option to save the plot to png file.
    def performance_historical_plot(self, start_date='2016-01-31'):
        
        """Given NO SIGNAL and NO conditionals this is the return of 
        the signal in isolation if no 12-2, and NLP sentiment score would be applied
        to the top 10 stocks. """
        
        mtl = self.return_df()
        qqq = self.qqq
        
        rets = []
        # mtl columns are symbols (AAPL, ABNB...) and the index is dates
        # for date in mtl[start_date].index:
        for date in mtl[mtl.index >= pd.to_datetime(start_date)].index:
            pf_per = self.point_performance(date)
            rets.append(pf_per)

        qqq = qqq[qqq.index >= pd.to_datetime(start_date)]
        

        rets_fin = pd.concat(rets)
        rets_fin = pd.DataFrame(rets_fin)
        plt.plot(rets_fin.index, rets_fin.cumprod(), label="Cascade Returns")
        plt.plot(qqq.index, (qqq.pct_change() + 1).cumprod(), label="Nasdaq Returns")
        plt.xlabel('Time')
        plt.ylabel('1+R base level')
        plt.title('Cumulative Return')
        plt.legend()
        plt.show()  


 
#I am adding simple test here that will also serve as a test
if __name__ == '__main__':
    cm = CascadeMomentum()
    cm.df_boolean()
    df = cm.return_df()
    print(df)
    print(cm.get_rolling_ret(window=5, period='M'))
    print(cm.filter_return(date='2016-01-31', df_cascade=df))
    print(cm.get_top(date='2016-01-31'))
    print(cm.point_performance(date='2016-01-31'))
    cm.performance_historical_plot(start_date='2016-01-31')
