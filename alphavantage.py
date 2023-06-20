#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 16:29:34 2023

@author: user
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 10:37:05 2021

@author: Aspiring Quants. 
"""
from datetime import datetime
import pandas as pd
import numpy as np
import requests
import dotenv
import os





class AlphaVantageApi:
    
    def __init__(self, api_key: str, crypto_currency_list_file_name: str='cryptoccy_list.txt'):
        '''
        AlphaVantage API client
        
        api_key: the AlphaVantage API key
        '''
        self.api_key = api_key
        
        # Try to load the cryptocurrency list file and print warning if this fails.
        # In case of failure the get_daily_exchange_rates() method will relay on the "market" argument
        try:
            with open(crypto_currency_list_file_name) as f:
                self.cryptoccy_list =  f.read().splitlines()
        except FileNotFoundError:
            print(f'WARNING: {crypto_currency_list_file_name} is missing')
            self.cryptoccy_list = ''

    def get_live_updates(self, symbol: str) -> pd.DataFrame:
        '''
        Price and volume data for a symbol.
        
        symbol: symbol to get data for
        
        Returned dataframe has index:
        'symbol', 'open', 'high', 'low', 'price', 'volume', 'latest trading day', 'previous close', 'change', 'change percent'
        and column "values" so it can be accessed by column and index, e.g. df['values']['price']
        
        Ref: https://www.alphavantage.co/documentation/#latestprice
        '''
        api_url = f'https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={self.api_key}'
        raw_df = requests.get(api_url).json()
        attributes = {'attributes':['symbol', 'open', 'high', 'low', 'price', 'volume', 'latest trading day', 'previous close', 'change', 'change percent']}
        attributes_df = pd.DataFrame(attributes)
        values = []
        for i in list(raw_df['Global Quote']):
            values.append(raw_df['Global Quote'][i])
        values_df = pd.DataFrame(values).rename(columns = {0:'values'})
        frames = [attributes_df, values_df]
        df = pd.concat(frames, axis = 1, join = 'inner').set_index('attributes')
        return df
    
    def get_intraday_data(self, symbol: str, interval: str) -> pd.DataFrame:
        '''
        Pull Intraday Data
        
        symbol: symbol to get data for
        interval: one of 1min, 5min, 15min, 30min, 60min
        
        Ref: https://www.alphavantage.co/documentation/#intraday
        '''
    
        api_url = f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval={interval}&apikey={self.api_key}'
        raw_df = requests.get(api_url).json()
        df = pd.DataFrame(raw_df[f'Time Series ({interval})']).T
        df = df.rename(columns = {'1. open': 'open', '2. high': 'high', '3. low': 'low', '4. close': 'close', '5. volume': 'volume'})
        for i in df.columns:
            df[i] = df[i].astype(float)
        df.index = pd.to_datetime(df.index)
        df = df.iloc[::-1]
        return df
    
    def get_historical_data(self, symbol: str, data_type: str, start_date: datetime = None, 
                            end_date: datetime = None, convert_timeframe: str = None, multiple=False) -> pd.DataFrame:
        '''
        Pull Historical Data in daily, weekly, or monthly adjusted timeframe
        
        symbol: symbol to get data for
        data_type: daily, weekly, monthly
        start_date: starting datetime for the data
        end_date: ending datetime for the data
        convert_timeframe: timeframe to convert the data to, must be higher than the original, e.g., 2w if original is weekly
        
        multiple: Referring to if you feed this call a list of string tickers:
            ex:
                tickers = ['AAPL', 'MSFT']
        
        Ref: https://www.alphavantage.co/documentation/#dailyadj
        

        '''
        if multiple:
            
            data = {}
            for ticker in symbol:
                
                if data_type == 'daily':
                    df = self.get_time_series_daily_adjusted(symbol=ticker)
                elif data_type == 'weekly':
                    df = self.get_time_series_weekly_adjusted(symbol=ticker)
                elif data_type == 'monthly':
                    df = self.get_time_series_monthly_adjusted(symbol=ticker)
                
                if start_date:
                    df = df[df.index >= start_date]
            
                if end_date:
                    df = df[df.index <= end_date]
            
                if convert_timeframe:
                    df = df.resample(convert_timeframe).agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'adj close': 'last', 'volume': 'sum'}).dropna(subset=['open'])
                
                    data[ticker] = df
                
                if 'Time Series (Daily)' not in raw_df:
                    
                    print(f'Unexpected API response for symbol {symbol}: {raw_df}')
                    return pd.DataFrame()  # or however you want to handle this

        
        else:
            
            if data_type == 'daily':
                df = self.get_time_series_daily_adjusted(symbol=symbol)
            elif data_type == 'weekly':
                df = self.get_time_series_weekly_adjusted(symbol=symbol)
            elif data_type == 'monthly':
                df = self.get_time_series_monthly_adjusted(symbol=symbol)
                
            if start_date:
                df = df[df.index >= start_date]
        
            if end_date:
                df = df[df.index <= end_date]
            
            if convert_timeframe:
                df = df.resample(convert_timeframe).agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'adj close': 'last', 'volume': 'sum'}).dropna(subset=['open'])
        
        return df

    
    def get_time_series_daily_adjusted(self, symbol: str) -> pd.DataFrame:
        '''
        Get daily historical data
        symbol: symbol to get data for
        Ref: https://www.alphavantage.co/documentation/#dailyadj
        '''
        api_url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={symbol}&apikey={self.api_key}&outputsize=full'
        raw_df = requests.get(api_url).json()
        df = pd.DataFrame(raw_df['Time Series (Daily)']).T
        df = df.rename(columns = {'1. open': 'open', '2. high': 'high', '3. low': 'low', '4. close': 'close', '5. adjusted close': 'adj close', '6. volume': 'volume'})
        for i in df.columns:
            df[i] = df[i].apply(self.try_float)
        df.index = pd.to_datetime(df.index)
        df = df.iloc[::-1].drop(['7. dividend amount', '8. split coefficient'], axis = 1, errors='ignore')
        return df
    
    def get_time_series_weekly_adjusted(self, symbol: str) -> pd.DataFrame:
        '''
        Get weekly historical data
        symbol: symbol to get data for
        Ref: https://www.alphavantage.co/documentation/#weeklyadj
        '''
        api_url = f'https://www.alphavantage.co/query?function=TIME_SERIES_WEEKLY_ADJUSTED&symbol={symbol}&apikey={self.api_key}&outputsize=full'
        raw_df = requests.get(api_url).json()
        df = pd.DataFrame(raw_df['Weekly Adjusted Time Series']).T
        df = df.rename(columns = {'1. open': 'open', '2. high': 'high', '3. low': 'low', '4. close': 'close', '5. adjusted close': 'adj close', '6. volume': 'volume'})
        for i in df.columns:
            df[i] = df[i].apply(self.try_float)
        df.index = pd.to_datetime(df.index)
        df = df.iloc[::-1].drop(['7. dividend amount', '8. split coefficient'], axis = 1, errors='ignore')
        return df
    
    def get_time_series_monthly_adjusted(self, symbol: str) -> pd.DataFrame:
        '''
        Get monthly historical data
        symbol: symbol to get data for
        Ref: https://www.alphavantage.co/documentation/#monthlyadj
        '''
        api_url = f'https://www.alphavantage.co/query?function=TIME_SERIES_MONTHLY_ADJUSTED&symbol={symbol}&apikey={self.api_key}&outputsize=full'
        raw_df = requests.get(api_url).json()
        df = pd.DataFrame(raw_df['Monthly Adjusted Time Series']).T
        df = df.rename(columns = {'1. open': 'open', '2. high': 'high', '3. low': 'low', '4. close': 'close', '5. adjusted close': 'adj close', '6. volume': 'volume'})
        for i in df.columns:
            df[i] = df[i].apply(self.try_float)
        df.index = pd.to_datetime(df.index)
        df = df.iloc[::-1].drop(['7. dividend amount', '8. split coefficient'], axis = 1, errors='ignore')
        return df
    
    def get_daily_exchange_rates(self, symbol: str, base_currency: str, market: str = None) -> pd.DataFrame:
        '''
        Pull Daily Exchange Rates
        
        symbol: symbol to get data for
        base_currency: base currency to get data for
        market: the market to get data for, can be "fx", "crypto" or None. If it is None self.cryptoccy_list will be used to determine the market
        
        Ref: https://www.alphavantage.co/documentation/#currency-daily
             https://www.alphavantage.co/documentation/#fx-daily
        '''
        
        # Make sure we have market OR self.cryptoccy_list provided
        assert market or self.cryptoccy_list
        
        if (market == 'crypto') or (symbol in self.cryptoccy_list):
            api_url = f'https://www.alphavantage.co/query?function=DIGITAL_CURRENCY_DAILY&symbol={symbol}&market={base_currency}&apikey={self.api_key}&outputsize=full'
            interval = '(Digital Currency Daily)'
        else:
            api_url = f'https://www.alphavantage.co/query?function=FX_DAILY&from_symbol={base_currency}&to_symbol={symbol}&apikey={self.api_key}&outputsize=full'
            interval = 'FX (Daily)'
        
        raw_df = requests.get(api_url).json()
        df = pd.DataFrame(raw_df[f'Time Series {interval}']).T
        df = df.rename(columns = lambda x: x.split(' ', 1)[1])
        for i in df.columns:
            df[i] = df[i].astype(float)
        df.index = pd.to_datetime(df.index)
        return df
    
    def get_treasury_yields(self, interval: str, maturity: str) -> pd.DataFrame:
        '''
        Pull Treasury Yields
        
        interval: daily, weekly, or monthly
        maturity: 3month, 2year, 5year, 7year, 10year, or 30year
        
        Returns dataframe with index date and a column "value" containing the value
        
        Ref: https://www.alphavantage.co/documentation/#treasury-yield
        '''
        
        api_url = f'https://www.alphavantage.co/query?function=TREASURY_YIELD&interval={interval}&maturity={maturity}&apikey={self.api_key}'
        raw_df = requests.get(api_url).json()
        #print(raw_df['data'])
        df = pd.DataFrame(raw_df['data'])
        for i in df.columns[1:]:
            df[i] = df[i].apply(self.try_float)
        df.set_index('date', inplace=True)
        df.index = pd.to_datetime(df.index)
        return df
    
    def get_financial_statements(self, function_code: int, symbol: str) -> pd.DataFrame:
        '''
        Get Financial Statements and Company Overview
        
        symbol: symbol to get data for
        function_code: {0: 'INCOME_STATEMENT', 1: 'BALANCE_SHEET', 2: 'CASH_FLOW', 3: 'OVERVIEW'}
        
        Ref: https://www.alphavantage.co/documentation/#income-statement
             https://www.alphavantage.co/documentation/#balance-sheet
             https://www.alphavantage.co/documentation/#company-overview
        '''
        switcher = {0: 'INCOME_STATEMENT', 1: 'BALANCE_SHEET', 2: 'CASH_FLOW', 3: 'OVERVIEW'}
        function = switcher.get(function_code)
        api_url = f'https://www.alphavantage.co/query?function={function}&symbol={symbol}&apikey={self.api_key}'
        raw_df = requests.get(api_url).json()
        #print(raw_df)
        if function_code in list(switcher.keys())[:3]:
            df = pd.DataFrame(raw_df['annualReports'])
        elif function_code == 3:
            df = pd.DataFrame([raw_df])     
        return df

    @staticmethod
    def try_float(value):
        '''Try to convert value to float, return original value on error'''
        try:
            return float(value)
        except:
            return value


if __name__ == '__main__':
    # We get the API key from a .enf file containing a variable called ALPHAVANTAGE_API_KEY
    api_key = os.environ.get('ALPHAVANTAGE_API_KEY')
    
    av = AlphaVantageApi(api_key=api_key)
    symbol = "AAPL"
    data_type = "daily"
    start_date_str = "2021-01-01"
    end_date_str = "2022-01-01"
    interval = '5min'
    tickers = ['AAPL', 'MSFT']
        
    # # Get daily historical data for AAPL
    df = av.get_historical_data(symbol=symbol, data_type= data_type, start_date=start_date_str, end_date=end_date_str)
    print(df)
    
    # # Get balance sheet financial statement for AAPL
    df = av.get_financial_statements(function_code=0, symbol=symbol)
    print(df)
    
    # # Get intraday data for AAPL
    df = av.get_intraday_data(symbol=symbol, interval=interval)
    print(df)
    
    # Get weekly historical data for AAPL and convert it to 2w timeframe
    df = av.get_historical_data(symbol=symbol, data_type='weekly', convert_timeframe='2d')
    print(df)
    
    df = av.get_historical_data(symbol=tickers, data_type = 'daily', start_date= "2021-01-01", multiple = True)
    
    


# api_url = f'https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={api_key}'
# raw_df = requests.get(api_url).json()
# #I always use this to check api JSON 
# print(f"This is what you can make the columns out of: {raw_df['Global Quote']}")
# print( raw_df['Global Quote']['01. symbol'])
# # build up attributes needed, I'm thinking that these would be the best for intra-day information.
# attributes = {'attributes':['symbol', 'open', 'high', 'low', 'price', 'volume', 'latest trading day', 'previous close', 'change', 'change percent']}
# attributes_df = pd.DataFrame(attributes)
# values = []

# for ii in list(raw_df['Global Quote']):
#     values.append(raw_df['Global Quote'][ii])
# def get_live_updates(api_key, symbol):
    
#     api_url = f'https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={api_key}'
#     raw_df = requests.get(api_url).json()
#     attributes = {'attributes':['symbol', 'open', 'high', 'low', 'price', 'volume', 'latest trading day', 'previous close', 'change', 'change percent']}
#     attributes_df = pd.DataFrame(attributes)
#     values = []
#     for i in list(raw_df['Global Quote']):
#         values.append(raw_df['Global Quote'][i])
#     values_df = pd.DataFrame(values).rename(columns = {0:'values'})
#     frames = [attributes_df, values_df]
#     df = pd.concat(frames, axis = 1, join = 'inner').set_index('attributes')
#     return df

# print(get_live_updates(API_KEY, symbol))
