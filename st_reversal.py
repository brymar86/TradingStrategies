#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 14 13:24:11 2023

@author: user
"""

import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class ReversalStrategy:
    
    
    def __init__(self, start_date='2015-01-01'):
        
        """
        How to make the Reversal Strategy for monthly sample checking 3 stocks
        """
        
        stocks = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0].Symbol
        sp_tickers = stocks.to_list()
        df = yf.download(sp_tickers, start=start_date)['Adj Close']
        df.index = pd.to_datetime(df.index)
        ret_df = df.pct_change()
        ret_df = ret_df.iloc[1:]
        log_ret = np.log(1 + ret_df)
        self.mtl_ret = (ret_df + 1).resample('M').prod()

    def worst_performers(self, date:str, stocks:int):
        
        all_ = self.mtl_ret.loc[date]
        worst = all_.nsmallest(stocks)
        relevant_ret = self.mtl_ret[worst.name:][1:2][worst.index]
        rev_returns = (relevant_ret).mean(axis=1).values[0]
        return rev_returns
    
    def best_performance(self, date:str, stocks:int):
        
        all_ = self.mtl_ret.loc[date]
        best = all_.nlargest(stocks)
        relevant_ret = self.mtl_ret[best.name:][1:2][best.index]
        rev_returns = (relevant_ret).mean(axis=1).values[0]
        return rev_returns

    def calculate_returns(self, stocks:int, reversal = True):
        
        if reversal:
            
            returns = []
            for date in self.mtl_ret.index[:-1]:
                returns.append(self.worst_performers(date, stocks))
            
        else:
             returns = []
             for date in self.mtl_ret.index[:-1]:
                 returns.append(self.best_performance(date, stocks))
                 
        return returns

    def plot_cumulative_returns(self, returns:list):
        
        pd.Series(returns, index=self.mtl_ret.index[:-1]).cumprod().plot()
        plt.xlabel('date')
        plt.ylabel('level')
        plt.title("CumRet_Reversal")
        plt.show()



if __name__ == '__main__':
    strategy = ReversalStrategy()
    returns = strategy.calculate_returns(3)
    strategy.plot_cumulative_returns(returns)
