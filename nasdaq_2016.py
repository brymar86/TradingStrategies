#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 14:41:58 2023

@author: Blacksheep, CFA MFE
"""

 #!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pickle
import pandas as pd
import numpy as np
import yfinance as yf
import datetime as dt
import time
import matplotlib.pyplot as plt


class Nasdaq2016:
    
    """
    This class is used for handling survivorship bias.  it uses Siblis Research information of yearly 
    constiuents that are found within the nasdaq.
    
    It is used hand in hand with several strategies that are or will be uploaded to TradingStrategies 
    Repository. 
    
    """
    
    def __init__(self):
        
        self.df_boolean = self.get_data()
        
    def get_data(self):
        
        try:
            # this is our 21016 Nasdaq where we are pulling the tickers that existed in the nasdaq
            df_boolean = pd.read_html('https://siblisresearch.com/data/historical-components-nasdaq/')[0]
            
        except Exception as e:
            print(f"An error occurred while parsing the data: {e}")
            df_boolean = None
        return df_boolean
    
    def tickers_2016(self):
        
        if self.df_boolean is None:
            
            print("Data is not available.")
            
            return None
        #pull in dataframe
        df_isactive = self.df_boolean
        # set index as ticker
        df_isactive.set_index('Ticker', inplace = True)
        # get only the most relevant columns
        df_isactive = df_isactive[df_isactive.columns[2:]]
        # ensure columnspace in df_isactive is a datetime object
        df_isactive.columns = pd.to_datetime(df_isactive.columns, errors='coerce')
        # create the boolean mapping of True False instead of value/nan
        df_isactive = df_isactive=='X'
        # send the 2016 tickers in the rowspace into a list 
        tickers = df_isactive.index.to_list()
        # store tickers and store df_isactive in a dictionary
        dictionary = {'tickers': tickers, 'df_isactive': df_isactive}
        
        return dictionary
    
if __name__ == "__main__":

    result = Nasdaq2016()
    print(result)
    dictionary = result.tickers_2016()
    print(dictionary['tickers'])

