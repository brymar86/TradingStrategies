# TradingStrategies
I enjoy caputuring ideas in code.  I will update this repositroy with new scripts on various strategies that I find interesting and potentially useful. 

### Purpose

Simply to take interesting strategies and express them in OOP python format or clean funcitonal programming.  In addition I've built my own backtestor and am also experimenting with backtesting.py.  

Trend Strategies, market neutral strategies, and mean-reversion strategies in both equities and crypto are my areas of interest.  Eventually I will include derivative strategies by early 2024.   Please contact me with any questions at bryanmarty86@gmail.com

PS

This code is meant for educational purposes only and is intended everyone if they find it useful. 

Currently:
BInance Data
nasdaq_2019.py
Cascade_Momentum (due to nasdaq_2019) 

Have been deprecated OR the datasource that was scraped has chagned their web pages.  The code is accurate given the datasources would be available.  Scripts leverageing BinanceData Unfortunatley will not work.  The tecniques for handling survivorship bias in nasdaw_2019 will work, one just needs to get the historical data. The real value is how the boolean masking works in filtering the dataframe with the relevant symbols that would have been reflected. 

