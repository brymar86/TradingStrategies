#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 13:08:34 2022

@author: user
"""
import pickle
import pandas as pd
import numpy as np
import yfinance as yf
import os
import datetime
import time
import matplotlib.pyplot as plt
import wrds
from pandas.tseries.offsets import *
from datetime import date, timedelta
import pandas_datareader
import csv
from pathlib import Path
from scipy.stats import ttest_ind
import statsmodels.api as sm
import warnings
import pandas_datareader.data as reader
import pandas.tseries.offsets



warnings.filterwarnings('ignore')


data_folder = r"/home/user/Documents/GitKraken_pulls/AFP_Project"
id_wrds = 'blacksheep'


# =============================================================================
# Data Ingestion from CRSP 
# =============================================================================
# CRSP
conn = wrds.Connection(wrds_username = id_wrds)
conn.create_pgpass_file()
df = conn.raw_sql("""
                  SELECT a.date, a.permno, b.ticker, b.comnam, b.shrcd, b.exchcd, 
                  a.ret, a.shrout, a.prc
                  FROM crspq.msf AS a
                  LEFT JOIN crspq.msenames AS b
                  ON a.permno = b.permno
                  AND b.namedt <= a.date
                  AND a.date <= b.nameendt
                  WHERE a.date BETWEEN '01/01/1920' and '06/30/2022' 
                  """ )   # Where is usually in relation to time.

conn.close()
# get CRSP delisted stock returns
conn = wrds.Connection(wrds_username = id_wrds)
df_dl = conn.raw_sql("""
                     SELECT permno, dlret, dlretx, dlstdt, dlstcd
                     FROM crspq.msedelist
                     """)
conn.close()
# get compustat data
conn = wrds.Connection(wrds_username=id_wrds)
cstat = conn.raw_sql("""
                      SELECT a.gvkey, a.datadate,a.fyear, a.ceq,                        
                      b.sic, b.gsubind, b.gind, b.year1, b.naics 
                      FROM comp.funda AS a             
                      LEFT JOIN comp.names AS b
                      on a.gvkey = b.gvkey
                      WHERE indfmt='INDL'
                      AND datafmt ='STD'
                      AND popsrc ='D'
                      AND consol ='C'
                      """)
conn.close()
# CRSP 
# also to confirm these are columns being called from 
conn = wrds.Connection(wrds_username=id_wrds)
crsp_cstat = conn.raw_sql("""
                      SELECT gvkey, lpermno AS permno, 
                      lpermco AS permco,
                      linktype, linkprim, liid, linkdt, linkenddt
                      FROM crspq.ccmxpf_linktable
                      WHERE (linkprim = 'C' or linkprim = 'P')
                      AND (linktype = 'LU' or linktype = 'LC' or linktype = 'LD' or
                      linktype = 'LN' or linktype = 'LS' or linktype = 'LX')
                      """)
conn.close()

                        


df.to_pickle(data_folder + 'df.pkl')
df_dl.to_pickle(data_folder + 'df_dl.pkl')
cstat.to_pickle(data_folder + 'cstat.pkl')
crsp_cstat.to_pickle(data_folder + 'crsp_cstat.pkl')


# =============================================================================
# Cleaning Data
# =============================================================================
df = pd.read_pickle(data_folder + 'df.pkl')
df_dl= pd.read_pickle(data_folder + 'df_dl.pkl')
cstat = pd.read_pickle(data_folder + 'cstat.pkl')
crsp_cstat = pd.read_pickle(data_folder + 'crsp_cstat.pkl')
# get stock index returns 
#target= pd.read_pickle(data_folder + 'st_rev_final2')
ff_st_rev = pd.read_csv('F-F_ST_Reversal_Factor.csv') # Via fama french website


# change data types returns
def clean_crsp(df, df_dl):
    
    """Clean Crsp Dataframe"""
    
    df['date']= pd.to_datetime(df['date'], format='%Y-%m-%d')
    df['permno'] = df['permno'].astype(int)
    df = df.sort_values(by = ['date', 'permno'])
    # change data types delisted returns
    df_dl['permno'] = df_dl['permno'].astype(int)
    df_dl.rename(columns = {'dlstdt':'date'}, inplace = True)
    df_dl['date']= pd.to_datetime(df_dl['date'], format='%Y-%m-%d')
    # merge regular returns dataframe with delisted returns one
    df2 = pd.merge(df, df_dl, how = 'left', on =['permno', 'date'])
    # filter shares code to only 1 for first digit (common shares) 
    # 0 or 1 for second digit (no further def, or no futher def needed)
    df2 = df2[(df2['shrcd'] == 10) | (df2['shrcd'] == 11)]
    # filter exchange code to major stock exchanges - NYSE (1), NYSE(MKT) (2), or NASDAQ (3)
    df2 = df2[(df2['exchcd'] == 1) | (df2['exchcd'] == 2) | (df2['exchcd'] == 3)]
    
    # merge actual returns and delisted ones. Filter out dates with no returns
    df2['ret'] = df2['ret'].fillna(0)
    df2['dlret'] = df2['dlret'].fillna(0)
    df2['ret_c'] = ((1+df2['ret'])*(1+df2['dlret'])) -1
    # df2['noret'] = np.where((df2['ret'] != 0) | (df2['dlret'] != 0), 1, 0)
    df2 = df2.sort_values(by = ['date', 'permno'])
    # construct market cap
    df2['mktcap'] = df2['prc'].abs()*df2['shrout'] 
    # construct lagged market cap
    df2 = df2.sort_values(by = ['permno', 'date']).reset_index().drop('index', axis = 1).copy()
    df2['p_date'] = df2['date'].dt.year*12 + df2['date'].dt.month # what is this?
    df2['lag'] = df2['p_date'].diff(1) == 1
    df2.loc[df2[df2['permno'].diff(1) != 0].index,['lag']] = False 
    df2['mktcapl'] = df2[['permno', 'mktcap']].groupby('permno').shift(1)
    df2.loc[df2[df2['lag'] == False].index, ['mktcapl']] = np.nan # setting np.nan at that location

    # construct lagged return/ same process as above
    df2['ret_l'] = df2[['permno', 'ret_c']].groupby('permno').shift(1)
    df2.loc[df2[df2['lag'] == False].index, ['ret_l']] = np.nan
    return df2


df2 = clean_crsp(df, df_dl)

# =============================================================================
# Find NYSE BreakPOints for Size and Momentum 
# output 6 groups.  small stock returns, large returns, low mom, high mom, mom
# reguar returns
# =============================================================================

def Size_Momentum(df2):
    
    """
    Create Size and Momentum DF's by filtering break points.
    
    Take cleaned pandas dataframe object and return
    a dataframe with small and large return return vectors
    as defined by a median marketcap for that date,
    a high and low return vector for returns that were
    either above the 70th percentile or below the 30th percentile
    
    Lastly this function also filters out any months without a lagged
    marketcap or return
    
    """
    # use 1 for NYSE firms and filter out any missing lagged market caps and returns

    df_nyse = df2[(df2['exchcd'] == 1) & (df2['mktcapl'].notnull()) & 
                  (df2['ret_l'].notnull())].reset_index(drop=True)
    # find the median lagged market cap each (so we can sort firms into large and small)
    df_nyse_size = df_nyse.groupby(['date'])['mktcapl'].median().to_frame().reset_index().rename(
            columns = {'mktcapl':'med_mkt_cap'})
    # find the 30th and 70th percentile of the past month's returns (so we can sort by momentum)
    df_nyse_mom = df_nyse.groupby(['date'])['ret_l'].quantile([0.3,0.7]).to_frame().reset_index()
    # just adjusting data so 30th %ile and 70th %ile are in their own columns
    df_nyse_mom = df_nyse_mom.pivot(index = 'date', columns= 'level_1', 
                                    values='ret_l').reset_index().rename(
                                            columns = {0.3:'mom30', 0.7:'mom70'})
    # merge the median mkt cap and momentum 30 and 70 %iles back into the main dataset

    df3 = df2.merge(df_nyse_size, how = 'left', on = 'date').merge(df_nyse_mom, how = 'left', on = 'date')
    # filter out any months without a lagged market cap or return
    df3 = df3[(df3['mktcapl'].notnull()) & (df3['ret_l'].notnull())].reset_index(drop=True)
    # create new column that says whether firm is small or large for that month
    df3['sm_lg'] = np.where(df3['mktcapl'] < df3['med_mkt_cap'], 'sm', 'lg')    
    # create new column that says whether firm is low (l), medium (m), or high (h) momentum based on previous month ret
    df3['mom'] = np.where(df3['ret_l'] < df3['mom30'], 'l', np.where(df3['ret_l'] > df3['mom70'], 'h', 'm'))
    
    return df3

df3 = Size_Momentum(df2)
# =============================================================================
# Calculate Value Weighted Returns for Factor
# =============================================================================

def value_weights(df3):
    """ Calculate value weighted returns for Factors
    
    bring in self.df from before.
    """
# calculate total mkt cap WITHIN EACH OF THE 6 GROUPS FOR EACH MONTH SINCE THEY ARE EACH A PORTFOLIO
# this is a temp dataframe to use with sums of the lagged values to apply the value weights.
    tot_mktcap = df3.groupby(['date', 'sm_lg', 'mom'])['mktcapl'].sum().to_frame().reset_index().rename(
    columns = {'mktcapl':'tot_mktcap'})  
    # without referrin to that its a lagged t-1 value.
    # merge with existing data
    df4 = df3.merge(tot_mktcap, how = 'left', on = ['date', 'sm_lg', 'mom'])
    # calculate weights for each stock each month (dividing by lagged tot_mktcap value)
    df4['wt'] = df4['mktcapl'] / df4['tot_mktcap']
    # calculate weighted returns for each stock each onth
    df4['wtdret'] = df4['wt'] * df4['ret_c'] # Q:is ret_c the true price return for each PERMNO?
    # groupby to get total return for each portfolio each month
    #   mom IS ALL the momentum returns
    st_rev = df4.groupby(['date', 'sm_lg', 'mom'])['wtdret'].sum().to_frame().reset_index()
    # assign portfolios to the high (small high, or large high) or low portfolio (small low or large low)
    st_rev['port'] = np.where(st_rev['mom'] == 'h', 'high', np.where(st_rev['mom'] == 'l', 'low', np.nan))
    # only keep returns for high and low portfolio (we don't need firms between 30th and 70th momentum %ile)
    st_rev2 = st_rev[(st_rev['port'] == 'low') | (st_rev['port'] == 'high')].reset_index(drop=True)
    # to calculate st factor we do st = (1/2 (Small Low + Big Low) - 1/2(Small High + Big High))
    # so first take the average of the low and high ports
    st_rev_final = st_rev2.groupby(['date', 'port'])['wtdret'].mean().to_frame().reset_index()
    # rearrange data so low and high portfolios are each a column
    st_rev_final2 = st_rev_final.pivot(index = 'date', columns= 'port', values='wtdret').reset_index()
    # subtract low - high portfolio
    st_rev_final2['st_rev'] = st_rev_final2['low'] - st_rev_final2['high']
    return st_rev_final2



st_rev_final2 = value_weights(df3)



# =============================================================================
# Compare with Fama French Short-Term Reversal Factor
# =============================================================================

def F_F_corr(st_reversal, ff_st_rev):
    
    """   Find correlation with F_F to verify what we pulled from WRDS 
    and what F_F puts in their own database as the true Short Term
    Reversal Factor.  """

    # add month and year columns so we can more easily merge with FF data
    st_reversal['year'] =  st_reversal['date'].astype(str).str[:4]
    st_reversal['month'] = st_reversal['date'].astype(str).str[5:7]
    
    # add in fama french data from ken french website
    ff_st_rev['year'] = ff_st_rev['date'].astype(str).str[:4]
    ff_st_rev['month'] = ff_st_rev['date'].astype(str).str[4:]
    ff_st_rev = ff_st_rev.rename(columns = {'ST_Rev': 'FF_st_rev'})

    # merge my data with FF data
    st_reversal3 = st_reversal.merge(ff_st_rev[['year', 'month', 'FF_st_rev']],\
                                      how = 'left', on = ['year', 'month'])

    st_rev_final3 = st_reversal3[['date', 'st_rev', 'FF_st_rev']] 
    # turn my returns in % to match FF data
    st_rev_final3['st_rev'] = st_rev_final3['st_rev']*100
    st_rev_final3['FF_st_rev'] = st_rev_final3['FF_st_rev'].astype(float)
    # add column for differences in our returns
    st_rev_final3['diff'] = (st_rev_final3['st_rev'] - st_rev_final3['FF_st_rev']).abs()
    # calculate correlation between our returns and FF 
    corr = np.corrcoef(st_rev_final3['st_rev'], st_rev_final3['FF_st_rev'])
    cor = (corr[1][0]).round(5)
    x = "Correlation between our created short-term reversal factor and fama-french one: " + str(cor)
    return (x, st_rev_final3)

cor = F_F_corr(st_rev_final2, ff_st_rev)[0]
st_reversal_3 = F_F_corr(st_rev_final2, ff_st_rev)[1]


# =============================================================================
# Compare iwth other FF Factors
# =============================================================================

def compare_FF(st_rev_final2, ff_st_rev, plot1 = False, plot2 = False):
    
    st_rev_final3 =  F_F_corr(st_rev_final2, ff_st_rev)[1]
    FF = pandas_datareader.famafrench.FamaFrenchReader('F-F_Research_Data_Factors', start = '1900', end = '2023')
    FF2 = FF.read()[0]/100
    FF3 = FF2.reset_index()
    FF3['Mkt'] = FF3['Mkt-RF'] + FF3['RF']
    # this is specific to Fama French Datetime Objects
    FF3['date']= pd.DataFrame(FF3[['Date']].values.astype('datetime64[ns]')) + MonthEnd(0)
    FF3 = FF3[:-1]
    FF3 = FF3.drop(columns=['Date'])
    # brings us to 1926-07-31
    FF3 = FF3.iloc[:1152]
    st_rev_final4 = st_rev_final3.copy()
    # this is the same date as the datatreaders FF values whe iloc 5 with st_reversal
    st_rev_final4 = st_rev_final4.iloc[5:].reset_index(drop=True)
    st_rev_final4 = pd.concat([st_rev_final4, FF3[['SMB', 'HML', 'Mkt-RF', 'RF']]], axis=1)
    st_rev_final4['FF_st_rev'] = st_rev_final4['FF_st_rev'] /100
    st_rev_final4['FF_st_rev_exc'] = st_rev_final4['FF_st_rev'] - st_rev_final4['RF']
    st_rev_final4['SMB_exc'] = st_rev_final4['SMB'] - st_rev_final4['RF']
    st_rev_final4['HML_exc'] = st_rev_final4['HML'] - st_rev_final4['RF']
    st_rev_final4['st_rev_exc'] = (st_rev_final4['st_rev']/100) - st_rev_final4['RF']
    
    if plot1:
        
    
        st_rev_final5 = st_rev_final4
        plt.plot(st_rev_final5['date'], (st_rev_final5['FF_st_rev_exc'] + 1).cumprod(), label='FF_st_rev') 
        plt.plot(st_rev_final5['date'], (st_rev_final5['st_rev_exc'] + 1).cumprod(), label='st_rev') 
        # plt.plot(st_rev_final5['date'], (st_rev_final5['Mkt-RF'] + 1).cumprod(), label='mkt') 
        plt.plot(st_rev_final5['date'], (st_rev_final5['SMB_exc'] + 1).cumprod(), label='smb') 
        plt.plot(st_rev_final5['date'], (st_rev_final5['HML_exc'] + 1).cumprod(), label='hml') 
        plt.xlabel('Date')
        plt.ylabel('Return')
        plt.title("Short-Term Reversal Cumulative Excess Return Over Time")
        plt.legend()
#        plt.show()
        
    elif plot2:
        
        st_rev_final5 = st_rev_final4
        plt.plot(st_rev_final5['date'], st_rev_final5['FF_st_rev'], label='st_rev') 
        plt.xlabel('Date')
        plt.ylabel('Return')
        plt.title("Short-Term Reversal Returns Over Time")
        plt.legend()
                                    
    return (st_rev_final4)

compare_FF(st_rev_final2, ff_st_rev, plot1 = True)




# =============================================================================
# Add in compustat data to form portfolios based on 4 llevels of GICS
# =============================================================================
#You have to change this from drews stuff. 
def crsp_compu_link(df3, cstat, crsp_cstat):

    """ Lets Link Compustat and CUSIP!
    dataframe DF3 needs to be in the form of df3 as described above
    pre-mean-reverted return.  
   """
     #  lagged t-1 value of market cap for value weighted return.
    tot_mktcap = df3.groupby(['date', 'sm_lg', 'mom'])['mktcapl'].sum().to_frame().reset_index().rename(
    columns = {'mktcapl':'tot_mktcap'}) 
    # merge with existing data
    df4 = df3.merge(tot_mktcap, how = 'left', on = ['date', 'sm_lg', 'mom'])
    # calculate weights for each stock each month (dividing by lagged tot_mktcap value)
    df4['wt'] = df4['mktcapl'] / df4['tot_mktcap']
    # calculate weighted returns for each stock each onth
    df4['wtdret'] = df4['wt'] * df4['ret_c'] # Q:is ret_c the true price return for each PERMNO?
    # Form link from Compustat Data to CRSP data: NOTICE gvkey is your mapping
    cstatlink = cstat[['gvkey', 'datadate', 'sic', 'gsubind']].merge(crsp_cstat[['gvkey', 'permno', 'permco']],
                 how = 'left', on = 'gvkey').rename(
                         columns = {'datadate': 'date'})
    # Get rid OF NULL permnos and typecast as integer
    cstatlink['permno'] = cstatlink['permno'].fillna(0)
    cstatlink['permno'] = cstatlink['permno'].astype(np.int32)
    # merge gics codes into large dataset
    ind1 = df4[['date', 'permno', 'ret_c', 'mktcapl']].merge(
    cstatlink[['permno', 'gsubind']], how='left', on='permno').drop_duplicates(
    subset = ['date', 'permno']).reset_index(drop=True)

    # create columns for each of the 4 levels of gics classification
    ind1['sector'] = ind1['gsubind'].astype(str).str[:2]
    ind1['indgroup'] = ind1['gsubind'].astype(str).str[:4]
    ind1['industry'] = ind1['gsubind'].astype(str).str[:6]
    ind1 = ind1.rename(columns = {'gsubind':'subind'})
    
    ind1 = ind1[ind1['subind'].notnull()]
    # this takes  each stocks mktvalue wt-1,i/ marketvalue sum(w_(t-1,subind))  Same logic for the rest
    ind1['subind_wt'] = ind1.groupby(['date', 'subind'])['mktcapl'].transform(lambda x: x/x.sum())
    ind1['sect_wt'] = ind1.groupby(['date', 'sector'])['mktcapl'].transform(lambda x: x/x.sum())
    ind1['indgrp_wt'] = ind1.groupby(['date', 'indgroup'])['mktcapl'].transform(lambda x: x/x.sum())
    ind1['ind_wt'] = ind1.groupby(['date', 'industry'])['mktcapl'].transform(lambda x: x/x.sum())

    # calculate weighted returns within classification group
    # this is each stocks return for that given sub/industry/sector/ect
    ind1['subind_wt_ret'] = ind1['subind_wt'] * ind1['ret_c'] 
    ind1['sect_wt_ret'] = ind1['sect_wt'] * ind1['ret_c'] 
    ind1['indgrp_wt_ret'] = ind1['indgrp_wt'] * ind1['ret_c'] 
    ind1['ind_wt_ret'] = ind1['ind_wt'] * ind1['ret_c'] 


    # calculate value-weighted means of each classification group

    subind_rets = ind1.groupby(['date', 'subind'])['subind_wt_ret'].sum().to_frame().reset_index().rename(
            columns = {'subind_wt_ret':'subind_mean'})
    sect_rets = ind1.groupby(['date', 'sector'])['sect_wt_ret'].sum().to_frame().reset_index().rename(
            columns = {'sect_wt_ret':'sect_mean'})
    indgrp_rets = ind1.groupby(['date', 'indgroup'])['indgrp_wt_ret'].sum().to_frame().reset_index().rename(
            columns = {'indgrp_wt_ret':'indgrp_mean'})
    ind_rets = ind1.groupby(['date', 'industry'])['ind_wt_ret'].sum().to_frame().reset_index().rename(
            columns = {'ind_wt_ret':'ind_mean'})
    
    ind1 = ind1.merge(subind_rets, how = 'left', on =['date', 'subind']).merge(
                        sect_rets, how = 'left', on =['date', 'sector']).merge(
                        indgrp_rets, how = 'left', on =['date', 'indgroup']).merge(
                        ind_rets, how = 'left', on =['date', 'industry'])
    
    # combine industry means back into main dataset
    df5 = df4.merge(ind1[['date', 'permno', 'sector', 'indgroup', 'industry','subind','sect_mean', 'indgrp_mean', 
                      'ind_mean',  'subind_mean']], how='left', on=['date', 'permno'])

    # calculate adjusted returns for each stock by subtracting out sector, industry group, industry, subindustry means

    df5['sect_wtd'] = df5['wtdret'] - (df5['wt'] *df5['sect_mean'])
    df5['indgrp_wtd'] = df5['wtdret'] - (df5['wt'] *df5['indgrp_mean'])
    df5['ind_wtd'] = df5['wtdret'] - (df5['wt'] *df5['ind_mean'])
    df5['subind_wtd'] = df5['wtdret'] - (df5['wt'] *df5['subind_mean'])

    return df5


df5 = crsp_compu_link(df3, cstat, crsp_cstat)
df5




    
    










#
#df['date']= pd.to_datetime(df['date'], format='%Y-%m-%d')
#df['permno'] = df['permno'].astype(int)
#df = df.sort_values(by = ['date', 'permno'])
## change data types delisted returns
#df_dl['permno'] = df_dl['permno'].astype(int)
#df_dl.rename(columns = {'dlstdt':'date'}, inplace = True)
#df_dl['date']= pd.to_datetime(df_dl['date'], format='%Y-%m-%d')
## merge regular returns dataframe with delisted returns one
#df2 = pd.merge(df, df_dl, how = 'left', on =['permno', 'date'])
## filter shares code to only 1 for first digit (common shares) 
## 0 or 1 for second digit (no further def, or no futher def needed)
#df2 = df2[(df2['shrcd'] == 10) | (df2['shrcd'] == 11)]
## filter exchange code to major stock exchanges - NYSE (1), NYSE(MKT) (2), or NASDAQ (3)
#df2 = df2[(df2['exchcd'] == 1) | (df2['exchcd'] == 2) | (df2['exchcd'] == 3)]
#    
## merge actual returns and delisted ones. Filter out dates with no returns
#df2['ret'] = df2['ret'].fillna(0)
#df2['dlret'] = df2['dlret'].fillna(0)
#df2['ret_c'] = ((1+df2['ret'])*(1+df2['dlret'])) -1
#    # df2['noret'] = np.where((df2['ret'] != 0) | (df2['dlret'] != 0), 1, 0)
#df2 = df2.sort_values(by = ['date', 'permno'])
## construct market cap
#df2['mktcap'] = df2['prc'].abs()*df2['shrout'] 
#df2 = df2.sort_values(by = ['permno', 'date']).reset_index().drop('index', axis = 1).copy()
## This is a measure of extra caution below.  We want the last trading on the last day of the month
## in this sense we use pythons dt and boolean logic to create several filters at lagged marketcap
## and with 'ret_l' to ensure we are taking the LAST Trading day.  
#df2['p_date'] = df2['date'].dt.year*12 + df2['date'].dt.month 
#df2['lag'] = df2['p_date'].diff(1) == 1
#df2.loc[df2[df2['permno'].diff(1) != 0].index,['lag']] = False 
# 
#df2['mktcapl'] = df2[['permno', 'mktcap']].groupby('permno').shift(1)
#df2.loc[df2[df2['lag'] == False].index, ['mktcapl']] = np.nan # setting np.nan at that location
#
## construct lagged return/ same process as abov
#df2['ret_l'] = df2[['permno', 'ret_c']].groupby('permno').shift(1)
#df2.loc[df2[df2['lag'] == False].index, ['ret_l']] = np.nan
##creating the breakpoint as the median NYSE Market Equity Value
#df_nyse = df2[(df2['exchcd'] == 1) & (df2['mktcapl'].notnull()) & 
#                  (df2['ret_l'].notnull())].reset_index(drop=True)
## find the median lagged market cap each (so we can sort firms into large and small)
#df_nyse_size = df_nyse.groupby(['date'])['mktcapl'].median().to_frame().reset_index().rename(
#        columns = {'mktcapl':'med_mkt_cap'})
## find the 30th and 70th percentile of the past month's returns (so we can sort by momentum)
#df_nyse_mom = df_nyse.groupby(['date'])['ret_l'].quantile([0.3,0.7]).to_frame().reset_index()
## just adjusting data so 30th %ile and 70th %ile are in their own columns
#df_nyse_mom = df_nyse_mom.pivot(index = 'date', columns= 'level_1', 
#                                values='ret_l').reset_index().rename(
#                                        columns = {0.3:'mom30', 0.7:'mom70'})
## merge the median mkt cap and momentum 30 and 70 %iles back into the main dataset. Double merge both
#df3 = df2.merge(df_nyse_size, how = 'left', on = 'date').merge(df_nyse_mom, how = 'left', on = 'date')
## filter out any months without a lagged market cap or return
#df3 = df3[(df3['mktcapl'].notnull()) & (df3['ret_l'].notnull())].reset_index(drop=True)
## create new column that says whether firm is small or large for that month
#df3['sm_lg'] = np.where(df3['mktcapl'] < df3['med_mkt_cap'], 'sm', 'lg')    
## create new column that says whether firm is low (l), medium (m), or high (h) momentum based on previous month ret
#df3['mom'] = np.where(df3['ret_l'] < df3['mom30'], 'l', np.where(df3['ret_l'] > df3['mom70'], 'h', 'm'))
##lets create a temp dataframe we will use to build the value weighted return
#tot_mktcap = df3.groupby(['date', 'sm_lg', 'mom'])['mktcapl'].sum().to_frame().reset_index().rename(
#        columns = {'mktcapl':'tot_mktcap'})  
## without referrin to that its a lagged t-1 value.
## merge with existing data
#df4 = df3.merge(tot_mktcap, how = 'left', on = ['date', 'sm_lg', 'mom'])
## calculate weights for each stock each month (dividing by lagged tot_mktcap value)
#df4['wt'] = df4['mktcapl'] / df4['tot_mktcap']
## calculate weighted returns as r_{t} * wt_{t-1}
#df4['wtdret'] = df4['wt'] * df4['ret_c'] 
## groupby and sum to get value-weighted return of the portfolio of stocks each month
#st_rev = df4.groupby(['date', 'sm_lg', 'mom'])['wtdret'].sum().to_frame().reset_index()
## assign portfolios to the high (small high, or large high) or low portfolio (small low or large low)
#st_rev['port'] = np.where(st_rev['mom'] == 'h', 'high', np.where(st_rev['mom'] == 'l', 'low', np.nan))
## only keep returns for high and low portfolio (we don't need firms between 30th and 70th momentum %ile)
#st_rev2 = st_rev[(st_rev['port'] == 'low') | (st_rev['port'] == 'high')].reset_index(drop=True)
## to calculate st factor we do st = (1/2 (Small Low + Big Low) - 1/2(Small High + Big High))
## so first take the average of the low and high ports
#st_rev_final = st_rev2.groupby(['date', 'port'])['wtdret'].mean().to_frame().reset_index()
## rearrange data so low and high portfolios are each a column
#st_rev_final2 = st_rev_final.pivot(index = 'date', columns= 'port', values='wtdret').reset_index()
## subtract low - high portfolio
#st_rev_final2['st_rev'] = st_rev_final2['low'] - st_rev_final2['high']
#
#
## =============================================================================
## New Function make with this Code here
## =============================================================================
#
## add month and year columns so we can more easily merge with ff Data
#st_rev_final2
#st_rev_final2['year'] = st_rev_final2['date'].astype(str).str[:4]
#st_rev_final2['month'] = st_rev_final2['date'].astype(str).str[5:7]
##get year and month at each datapoint
#ff_st_rev = pd.read_csv('F-F_ST_Reversal_Factor.csv')
##ff_st_rev['date'] = pd.to_datetime(ff_st_rev['date'], format='%Y-%m')
#ff_st_rev['year'] = ff_st_rev['date'].astype(str).str[:4]
#ff_st_rev['month'] = ff_st_rev['date'].astype(str).str[4:]
#ff_st_rev = ff_st_rev.rename(columns = {'ST_Rev': 'FF_st_rev'})
#
## merge data with ff data
#st_rev_final3 = st_rev_final2.merge(ff_st_rev[['year', 'month', 'FF_st_rev']], how = 'left',
#                                    on = ['year', 'month'])
## here we are just grouping the returns we made and famma french returns together
#st_rev_final4 = st_rev_final3[['date', 'st_rev', 'FF_st_rev']]
## we have an option.  We will adjust our returns. 
#st_rev_final4['st_rev'] = st_rev_final4['st_rev']*100
## we need to convert FF return values to floats
#st_rev_final4['FF_st_rev'] = st_rev_final4['FF_st_rev'].astype(float)
## add column for differences in our returns
#st_rev_final4['diff'] = (st_rev_final4['st_rev'] - st_rev_final4['FF_st_rev']).abs()
##calculate correlation
#corr = np.corrcoef(st_rev_final4['st_rev'], st_rev_final4['FF_st_rev'])
#cor = (corr[1][0]).round(5)
#print("Correlation between our created short-term reversal factor and fama-french one: " + str(cor))
#
#
## =============================================================================
## funtional form for linkage   
## =============================================================================
#
#
#
#
#
#
#
#
#
#
#
