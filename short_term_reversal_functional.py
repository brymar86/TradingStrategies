#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 14 05:28:42 2023

@author: AlgoVibes sketch to Class
"""

import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

stocks = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0].Symbol
# check
stocks.head(5)
# take stocks to list
#elements as tickers as elements within a list we can batch feed API
sp_tickers = stocks.to_list()
# Pull stocks
df = yf.download(sp_tickers, start = '2015-01-01')['Adj Close']
# to datetime
df.index = pd.to_datetime(df.index)
# convert to regular prices
ret_df = df.pct_change()
#log rets
ret_df=ret_df.iloc[1:]
log_ret = np.log(1 + ret_df)

#lets take basic monthly data returns. I can subtract this to get montly ret but this base is helpful
mtl_ret = (ret_df + 1).resample('M').prod()
# mtl_ret.index[0]
# Timestamp('2015-01-31 00:00:00', freq='M')
# worst performers on january 31st 2015
worst = mtl_ret.loc[mtl_ret.index[0]].nsmallest(3)
print(worst)
print(f"\nPrint variable.Name: {worst.name}")
#look these are the names so you can slice these like a dataframe
print(f"\n Worst index: {worst.index}")
# This is just the index. worst name is an index value so I can slice indices
mtl_ret[worst.name: "2015-05-31"]
# here he slides one roww.  you have to slice one row in this manner to grape one index 
# Trick, I'm taking the index of the first date/worst three stocks then taking the next month
# This is literally just the next months row. 
print(mtl_ret[worst.name:][1:2])
#three things happening, df slice via index, then index slice next row, then column slice
#  df[index:][1:2][colNames] HERE YOU SEE THE REVERSAL PERFORMANCE FOR THE NEXT MONTH
mtl_ret[worst.name:][1:2][worst.index]
# now lets grab the return
reversal_stocks = mtl_ret[worst.name:][1:2][worst.index]
# this takes the return of the previous months worst portfolio returns and gets the average next month return
reversal_mean = (reversal_stocks).mean(axis =1).values[0]

# =============================================================================
# Funcitonal Script 
# =============================================================================
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

stocks = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0].Symbol
# check
stocks.head(5)
# take stocks to list
#elements as tickers as elements within a list we can batch feed API
sp_tickers = stocks.to_list()
# Pull stocks
df = yf.download(sp_tickers, start = '2015-01-01')['Adj Close']
# to datetime
df.index = pd.to_datetime(df.index)
# convert to regular prices
ret_df = df.pct_change()
#log rets
ret_df=ret_df.iloc[1:]
log_ret = np.log(1 + ret_df)

#lets take basic monthly data returns. I can subtract this to get montly ret but this base is helpful
mtl_ret = (ret_df + 1).resample('M').prod()


def worst_performers(date, stocks):
    """
    This is how to run the reversal strategy
    """
    # this is the index
    all_ = mtl_ret.loc[date]
    worst = all_.nsmallest(stocks)
    # df[index:][1:2][colNames] HERE YOU SEE THE REVERSAL PERFORMANCE FOR THE NEXT MONTH
    relevant_ret = mtl_ret[worst.name:][1:2][worst.index]
    rev_returns = (relevant_ret).mean(axis=1).values[0]
    return rev_returns

worst_performers('2015-01-31',3)

# return calcs
returns = []
for date in mtl_ret.index[:-1]:
    returns.append(worst_performers(date, 3))
    
    
pd.Series(returns).prod()

pd.Series(returns, index = mtl_ret.index[:-1]).cumprod().plot()
plt.xlabel('date')
plt.ylabel('level')
plt.title("CumRet_Reversal")
plt.show()



def best_performance(date, stocks):
    """
    This is how to run the reversal strategy
    """
    # this is the index
    all_ = mtl_ret.loc[date]
    worst = all_.nlargest(stocks)
    # df[index:][1:2][colNames] HERE YOU SEE THE REVERSAL PERFORMANCE FOR THE NEXT MONTH
    relevant_ret = mtl_ret[worst.name:][1:2][worst.index]
    rev_returns = (relevant_ret).mean(axis=1).values[0]
    return rev_returns



# return calcs
returns = []
for date in mtl_ret.index[:-1]:
    returns.append(best_performance(date, 3))
    
    
pd.Series(returns).prod()

pd.Series(returns, index = mtl_ret.index[:-1]).cumprod().plot()
plt.xlabel('date')
plt.ylabel('level')
plt.title("CumRet_Best")
plt.show()















    
# =============================================================================
#     
# class ShortermReversal:
#     
#     def __init__
#     
#     
#     
#     
# =============================================================================
    
    
    
    
    
    