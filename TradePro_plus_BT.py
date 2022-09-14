# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 18:44:49 2022

@author: user
"""
import yfinance as yf
from BinanceData import BinanceData
import matplotlib.pyplot as plt
import seaborn as sns
import json
import pandas as pd
import numpy as np
import datetime as dt
from binance.client import Client
from binance.exceptions import * 
import ta
from backtesting.lib import crossover
from decouple import config
from backtesting.lib import Strategy
from scipy.stats import norm
import scipy.stats as sp
from matplotlib import pylab as plt


class TradePro_Daily:
        
    
    def __init__(self, symbol:str, baseasset:str, start:str):
        
        # This changes slightly on final version
        # This is a slow strategy, i'm getting information and deciding on the close on the next open.
        
        self.symbol = symbol
        self.baseasset = baseasset
        self.start = start
        
        obj = BinanceData()
        self.df = obj.getdailydata(self.symbol, self.baseasset, self.start)# this assumes BinanceTrader specified right way. 
        
        if self.df.empty:
            print('No Data Pulled')
            
        else:
            # When I instantiate a strategy class object I want these called. 
            self.Build_Dataframe()
            self.Generate_Signals()
            self.Loop_It()

        
    def Build_Dataframe(self, ema1 = 8, ema2= 14, ema3 = 50):
        
        """Build appropriate columns for dataframe for analysis."""
        
        self.df['stochrsi_k'] = ta.momentum.stochrsi_k(self.df.Close)
        self.df['stochrsi_d'] = ta.momentum.stochrsi_d(self.df.Close)
        
        for ii in (ema1,ema2,ema3):
            self.df['EMA_'+str(ii)] = ta.trend.ema_indicator(self.df.Close, window = ii)    
            # this is that average_true_range of volatility function
        self.df['atr'] = ta.volatility.average_true_range(self.df.High, self.df.Low, self.df.Close)
        self.df['cross'] = (self.df['stochrsi_k'] > self.df['stochrsi_d']) & (self.df['stochrsi_k'] > self.df['stochrsi_d']).diff() 
        self.df.dropna(inplace = True)
        
    def Generate_Signals(self):
        
        """Generating Signals"""
        
        conditions = [(self.df.cross) & (self.df.Close > self.df.EMA_8) & (self.df.EMA_8 > self.df.EMA_14) & \
                      (self.df.EMA_14 > self.df.EMA_50),# End of buy condtions.
                      
                      (self.df.Close >= (self.df.atr*2)) | (self.df.Close <= (self.df.atr*3))] # Take profit and stop loss included for backtest
            
        choices = ['Buy', 'Sell']
        
        self.df['signal'] = np.select(conditions, choices,0)
        
        self.df.signal = self.df.signal.shift() # you have to do this to directly access the buy price on the next days open. you accomplish this by shifing. 
        self.df.dropna(inplace = True)
        # check here.  you need to adjust .diff() if you have multiple buys 
        
    def Loop_It(self):
        
                
        """I have to think now HOW TO PUT IN HERE THE Signal Column."""
        
        position = False
        buydates, selldates = [], [] # ok since we are storing these via index I can .loc these. 
        
        for index, row in self.df.iterrows():
            
            if not position and row['signal'] == 'Buy': #not in position and we hit a buy, buy it!
                position = True
                buydates.append(index) # which is the date stamp of a buy sent to list
                  
            if position and row['signal'] == 'Sell': # if you're in a position and the signal= sell you need ot sell it.
                position = False # now i'm no longer in the position.
                selldates.append(index) # Timestamp of a sell date. 
        # this is actually just the buy and sell arrays so that you have the prices to calculate profits
        # and drawdowns. 
        self.buy_arr = self.df.loc[buydates].Open # I ALWAYS BUY AND SELL AT OPEN
        self.sell_arr = self.df.loc[selldates].Open #Notice using self allows me to access this for analysis.
        
        
    def signal(self):
        
        """To pass dictionary object to BinanceTrader Class""" 
        
        symbol = self.symbol
        signal_dictionary = {}
        
        signal = self.df.iloc[-1]['signal']
        # minimum take profit and stol loss values. 
        take_profit_price = self.df.iloc[-1]['Close'] + self.df.iloc[-1]['atr'] * 2 # hardcoded value
        stop_loss = self.df.iloc[-1]['Close'] - self.df.iloc[-1]['atr']*3
        
        if signal == 'Buy': 
            signal_dictionary[symbol] = {'signal': signal, 'take_profit_price': take_profit_price, 'stop_loss_price': stop_loss,
                                             'Date': self.df.iloc[-1].name}
        else:
            signal_dictionary[symbol] = {'signal': signal, 'Sell Price' : self.df.iloc[-1]['Close'],
                                         'Date': self.df.iloc[-1].name}
        
        return signal_dictionary
        
    
    
class Backtest(TradePro_Daily):
    
    def __init__(self, symbol:str, baseasset:str, start:str, Distribution:str, capital_at_risk:int):
        
        super().__init__(symbol, baseasset, start)
        
        self.capital_at_risk = capital_at_risk
        # options ['Gaussian', 'Exponential', 'Gamma', 'Student-T', ]
        self.Distribution = Distribution  # select distribution for strategy HPR once known. 
        
        if isinstance(capital_at_risk, str):
            int(capital_at_risk)
        #instantiate Class
        obj = BinanceData()
        self.buy_hold = obj.getdailydata(self.symbol, self.baseasset, self.start)
        
        if self.buy_hold.empty:
            print('No Data pulled')
        else:
            self.buy_hold = self.buy_hold
         
        
    def calc_profit(self):
        """Filters out open trades and then calculates profits of closed trades.
        
        This returns a numpy array of holding period returns."""
        
        if self.buy_arr.index[-1] > self.sell_arr.index[-1]:
            self.buy_arr = self.buy_arr[:-1]
            # this will caclulate each buy and sell signals holding period return for a buy and sell signal
        return (self.sell_arr.values - self.buy_arr.values)/self.buy_arr.values 
    
    def Annualized_Strategy_Ret(self):
        
        """
        
        Lets get Annualized Return to apply to Sharpe and Sortino Ratios
        
        https://www.nasdaq.com/articles/annualized-return-vs-cumulative-return-2015-11-03
        
        """
        HPR = self.calc_profit()        
        period_start = self.buy_arr.index[0]
        period_end = self.sell_arr.index[-1]
        days_in_period = (period_end - period_start).days
        Rc = pd.Series(HPR + 1).cumprod()-1
        n = days_in_period/365
    
        Ra = ((1 + Rc)**(1/n))-1
        # annualized return (without compounding)
        return Ra
    
    def Daily_Ret(self):
        
        """Daily Return"""
                
        HPR = self.calc_profit()        
        period_start = self.buy_arr.index[0]
        period_end = self.sell_arr.index[-1]
        days_in_period = (period_end - period_start).days
        Rc = pd.Series(HPR + 1).cumprod()-1
        n = days_in_period
    
        Rdaily = ((1 + Rc)**(1/n))-1
        
        return Rdaily

    def Geometric_Return(self):
        """Calculates the total Strategy Return and Buy and Hold return.
        
        ONLY use this for log returns. """ 
    
        holding_period_rets = self.calc_profit() # holding period rets
        add_1 = [ii + 1 for ii in holding_period_rets] # adding one to holding return array
        # this is scipys way of calculating and rooting the return. need to -1
        geo_ret = sp.mstats.gmean(add_1) -1
        geo_ret = np.round(geo_ret, 3)
        dictionary = {"Geometric Strategy Return (%)": f"{geo_ret} %"} 
        
        return dictionary
    
    def Max_Liklihood(self):
        
        
        """Estimate the Max_Liklihood of your holding period returns, calc_profits
        return Distribution. """
         
        HPR = self.calc_profit()
        pass
                          
 
    def Historical_VaR(self, VaR_Level = 5):
        
        """ Historical Var.  Set initally at 95% VaR but can switch to 99% VaR by switching 5 to 1.
        since we are using np.percentile we feed an integer value of '5' not .05. 
        
        Historical VaR makes no Assumption about the Strategies Return Distribution. 
        
        """
        capital_at_risk = self.capital_at_risk
        returns = self.calc_profit()
        VaR_historical_strat_risk = np.percentile(returns, VaR_Level)
        VaR = np.round(VaR_historical_strat_risk * 100, 3)
        historical_VaR = (VaR_historical_strat_risk * capital_at_risk)
        historical_VaR = np.round(historical_VaR,3)
        
        dictionary_hist = {'Historical VaR Loss %': f"{VaR} %" , \
                           "Historical VaR (USD)": f" ${historical_VaR}"}       
        return dictionary_hist
    
    def Parametric_VaR(self, VaR_Level = .05):
        
        """This is the Standard VaR leveraging Mean-Variance analysis of the strategies risk. 
        
        Here we will use the Strategies historical return and Standard deviation to assess the VaR.
        
        This assume Gaussian Distribution of returns of the Strategy (not the asset being traded).
        
        IF other method called, sp.distribution.ppf ==> this is the inverted CDF,
        F^-1 of the cumulative distribtuion function to find the left of the curve. 
        
        """
        capital_at_risk = self.capital_at_risk
        returns = self.Daily_Ret()
        Distribution = self.Distribution
        mu = returns.mean() # you must use these estimates for parametric mu
        sigma = returns.std() # take holding period sigma for parametric vol estimate.
        
        if Distribution == 'Gaussian':
            
            Parametric_VaR = norm.ppf(VaR_Level, mu, sigma)
            
        elif Distribution == 'Exponential':
            
            Parametric_VaR = sp.expon.ppf(VaR_Level, loc = mu, scale = sigma)
            
        elif Distribution == 'Gamma':
            
            Parametric_VaR = sp.gamma.ppf(VaR_Level, loc = mu, scale = sigma)
        
        elif Distribution == "Student-T":
            
            Parametric_VaR = sp.t.ppf(.05, loc = mu, scale = sigma)
            
        else:
            print("Please place a relevant distribution to calculate Daily-VaR")
            
            
        VaR = np.round(Parametric_VaR*100, 3)
        VaR_risk = Parametric_VaR*capital_at_risk
        VaR_risk = np.round(VaR_risk, 3)
        dictionary = {'Daily VaR (%)': f"{VaR} %",\
                                      "Daily VaR (USD)":f" ${VaR_risk}" }
        return dictionary
    
    def Monte_Carlo_VaR(self, num_reps = 10000, n = 1000, VaR_Level = 5):
        
        
        """Computes a simulated example of Risk. We are going to use a Gaussian distribution
        for a sanity check.
        
        Future updates would include Max Liklihood of strategry returns on various distributions
        to then proceed to Monte_Carlo_simulated once the appropriate distribution is achieved.  """
        
        capital_at_risk = self.capital_at_risk
        returns = self.calc_profit()
        # Distribution = self.Distribution: we need to add this functionality to the method. 
        
        sim_data = pd.DataFrame([])
        mu = returns.mean() # i'm taking average holding period returns of Strategy as the 1st moment appx
        sigma = returns.std() 
        
        # i'm taking std() of holding period returns as 2nd moment appx
        temp = pd.DataFrame(np.random.normal(mu, sigma, num_reps)) # i'm assuming normal dist.  this can be changed. 
        sim_data = pd.concat([sim_data, temp], axis = 0)
        sim_data.columns = ['Strategy Sim']

        MC_percentile = []
        MC_Value = [] 
        # I don't use the ppf. it's similar to historical Var. i just
        # want he left tail of what I simmulated. 
        MC_percentile.append(np.percentile(sim_data, VaR_Level)) # FIND THE LEFT TAIL at 95%
        percentage = MC_percentile[0] * 100 # get this to percentage to show in dictionary
        rounded_perct =  np.round(percentage, 3) # round percentage so its not obnoxious
        MC_Value.append(MC_percentile[0]*capital_at_risk) # this is the value given capital we risked

        MCS_Value =  np.round(MC_Value[0],3) # this is us rounding the value of the risk so it's not too big. 


        dictionary_mcs = {'MCS_VaR_Loss (%)': f'{rounded_perct} % ',
                  "MCS_VaR_Strategy (USD)" : f' ${MCS_Value}'}
        # df_VaR = pd.DataFrame(dictionary, index = ['MCS_VaR_GaussianDistribution'])
        
        return dictionary_mcs
    
    
    def ES_Parametric(self, VaR_Level = .05):
        
        """
        Here we are taking the parametric/Mean-Variance Expected Shortfall of the Strategy
        
        Math = 1/(1-alpha)*integral from alpha to 1(VaR_{u})du (convert to latex)
        
        """
        capital_at_risk = self.capital_at_risk
        returns = self.calc_profit()
        #Distribution = self.Distribution: we need to add this aspect for ES
        mu = returns.mean()
        sigma = returns.std()
        alpha = -norm.pdf(VaR_Level, mu, sigma)
        VaR_param = (capital_at_risk*alpha)
        # i'm taking an integral here. assuming normal dist with where it's centered at mu and sigma
        Expectation_est = norm.expect(lambda x: x, lb = norm.ppf(1-VaR_Level,mu,sigma),loc = mu, scale = sigma)
        ES_param = (1/VaR_Level)* capital_at_risk * Expectation_est
        ES_param =  -1*np.round(ES_param,3)
        dictionary = {"Expected_Shortfall (USD)": f"${ES_param}"}
        return dictionary
    
    
    
    def Sortino_ratio(self, risk_free = 0):
        """
        This is the Sortino Ratio Calculation.
        Assumptions is that the risk free rate is zero.
        
        Recall: Calc profits give us holding period
        returns for EACH period a Buy and Sell signal was executed. 
        I take cumulative returns to get me to the point of time today to get
        the real value of the return for the asset Thus my Total Return is 
        used in the numerator.
        
        
        """
        
        returns = self.Annualized_Strategy_Ret()
        mu = returns.mean()
        #calculate downside deviation step by step
        sort_setup = np.minimum(0, returns - risk_free)**2 # summation i=1 to n, min(0, Xi-rf)^2
        sort_setup2 = np.mean(sort_setup) # 1/n portion of equation
        sort_downside_dev = np.sqrt(sort_setup2)
        
        sortino_ratio = np.mean(returns - risk_free)/sort_downside_dev
        dictionary = {'Strategy Annualized Sortino Ratio': sortino_ratio}
        return dictionary
    
    def Sharpe_Ratio(self, risk_free = 0):
        
        """
        This is the Classical Sharpe Ratio:
            
        E[R] - rf/sigma
            
        By definition I'm taking the the total return as historical Sharpe 
        should take the point in time the strategy started to the date today 
        that we are checking.  
            
        """
            
        annualized_rets = self.Annualized_Strategy_Ret()
        #Geogi. I'm calculating the average return
        mu = annualized_rets.mean()
        sigma = annualized_rets.std()
        sharpe_ratio = (mu - risk_free)/sigma
        dictionary = {'Strategy Annualized Sharpe Ratio': sharpe_ratio}
        return dictionary
    
    
    
    def Buy_Hold_Strategy(self, VaR_Level = .05, rf = 0):
        

        """This is a buy and hold on the Asset/Asset Class passively
        """
        
        capital_at_risk = self.capital_at_risk
        # just a holding period return if held for the ENTIRE DURATION. Verified correct
        asset_returns = (self.buy_hold.Close[-1] - self.buy_hold.Close[0])/self.buy_hold.Close[0]
        bh_return = np.round(asset_returns*100, 3)
        
        # calculating the daily returns. 
        returns_daily = self.buy_hold.Close.pct_change()
        mu_daily = returns_daily.mean()
        sigma_daily = returns_daily.std()
        mu = mu_daily * 365 # crypto doesn't sleep!
        sigma = sigma_daily * np.sqrt(365) # this is crypto we run 365
        B_H_Sharpe = (mu-rf)/sigma
        B_H_Sharpe = np.round(B_H_Sharpe, 3)
        
        # Calculating the B&H Sortino
        sort_setup = np.minimum(0, returns_daily - rf)**2 # summation i=1 to n, min(0, Xi-rf)^2
        sort_setup2 = np.mean(sort_setup) # 1/n portion of equation
        sort_downside_dev = np.sqrt(sort_setup2)
        sort_sigma = sort_downside_dev*np.sqrt(365)
        sort_mu = returns_daily.mean() * 365
        B_H_Sortino = (sort_mu - rf)/sort_sigma
        B_H_Sortino = np.round(B_H_Sortino, 3)
        
        # Calc Mean_Variance/Daily VaR for Strategy
        Parametric_VaR = norm.ppf(VaR_Level, mu_daily, sigma_daily)
        VaR = np.round(Parametric_VaR*100,3)
        VaR_risk = Parametric_VaR*capital_at_risk
        VaR_risk = np.round(VaR_risk, 3)
        
        # Buy & Hold Geometric Return Daily returns estimate. 
        #ONLY REVLEVANT FOR LOG RETURNS. 
        # returns_daily = returns_daily.dropna()
        # add_1 = [ii + 1 for ii in returns_daily] # adding one to holding return array
        # # this is scipys way of calculating and rooting the return. need to -1
        # geo_ret = sp.mstats.gmean(add_1) -1
        # geo_ret = np.round(geo_ret, 3) 
               
        dictionary = {'Buy & Hold Cumulative Return': f" {bh_return} % ", \
                      "Buy&Hold VaR (%)": f"{VaR} %", \
                      "Buy&Hold Daily VaR (USD)": f"${VaR_risk} ",\
                      "Buy&Hold Annualized Sharpe ": f"{B_H_Sharpe}", \
                      "Buy&Hold Annualized Sortino ": f"{B_H_Sortino}" }
            
        return dictionary
   
        
    def Metrics(self, logs = False):
        
        capital_at_risk = self.capital_at_risk
        
        """Capital_at_Risk is the amount of capital being run in the strategy. 
        Will need to be computed to provide VaR metrics the proper information"""
        
        HPR = self.calc_profit()
        max_drawdown = HPR.max() - HPR.min()
        max_drawdown = np.round(max_drawdown*100, 3)
        e_curve = (pd.Series(HPR) + 1).prod()
        cum_ret = np.round((e_curve-1)*100,3)
        
        
        array = HPR
        winners = [ii for ii in array if ii >= 0] 
        win_rate = np.round((len(winners)/len(array))*100,3)
        
        
        dictionary = {'Strategy Max_Drawdown':f"{max_drawdown} %", \
                      'Strategy Growth': e_curve, \
                      "Win Rate(%)": f"{win_rate} %", \
                      "Cumulative Return": f"{cum_ret} %"} 
        if logs:
            
            dictionary.update(self.Geometric_Return())
            
        else:
            pass 
        
        dictionary.update(self.Sortino_ratio())
        dictionary.update(self.Sharpe_Ratio())
        dictionary.update(self.Parametric_VaR())
        dictionary.update(self.Historical_VaR())
        dictionary.update(self.Monte_Carlo_VaR())
        dictionary.update(self.ES_Parametric())
        dictionary.update(self.Buy_Hold_Strategy())
        df = pd.DataFrame(dictionary, index = ['Risk Metrics'])
        
        return df.T
    
    
    def Buys_and_Sell_Chart(self):
        
        plt.figure(figsize = (10, 5))
        plt.plot(self.df.Close)
        plt.scatter(self.buy_arr.index, self.buy_arr.values, marker = '^', c = 'g')
        plt.scatter(self.sell_arr.index, self.sell_arr.values, marker = 'v', c = 'red')
        plt.xlabel('Time')
        plt.ylabel('Asset Closing Price')
        plt.title('Buy & Sell Marks for {}'.format(self.symbol)) 
        
    def Strategy_Cumulative_Growth(self, stratey_name:str):
        """Plots the Cumulative return of the strategy."""
        
        HPR = self.calc_profit()
        index = self.sell_arr.index
        cum_ret = (pd.Series(HPR) + 1).cumprod()
        plt.figure(figsize = (10,5))
        plt.plot(index, cum_ret)
        plt.xlabel('Time')
        plt.ylabel('Strategy Growth Level')
        plt.title('Strategy Performance for {}'.format(stragegy_name))
        
        
        


instance = TradePro_Daily('BTC', 'USDT', '2019-01-01')
instance.df
instance_bt = Backtest('BTC', 'USDT', '2019-01-01', 'Gaussian', 10000)
instance_bt.Metrics()
instance_bt.Buys_and_Sell_Chart()


