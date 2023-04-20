# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 2023

@author: Felipe Segundo Abril Bermúdez
"""

# Libraries ----
import warnings
import numpy as np
import pandas as pd
import yfinance as yf

# Global options ----
warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None
pd.set_option('display.max_columns', None)

# Load of financial time series ----
def load_financial_time_series(
    ticker_dict,
    initial_date="1900-01-01",
    final_date="2023-01-01",
    interval="1d"
):
    """Financial time series download from Yahoo Finance
    Download and process multiple data from Yahoo finance:
        ticker_dict: Dictionary of Yahoo finance tickers (items) and his name (values) for download
        initial_date: Initial date for time series
        final_date: Final date for time series
        interval: Frequency between reported data
    """
    
    # Load and process information ----
    df_financial_time_series = []
    for ticker, ticker_name in ticker_dict.items():
        # Download data ----
        df_local = yf.download(tickers = ticker, start = initial_date, end = final_date, interval = interval)
        df_local["symbol"] = ticker
        df_local["ticker_name"] = ticker_name
        
        # Generate date column and sort by symbol, ticker_name and generated date ----
        df_local["date"] = df_local.index
        df_local = df_local.sort_values(by = ["symbol", "ticker_name", "date"])
        
        # Generate index column ----
        df_local["step"] = np.arange(df_local.shape[0]) - 1

        # Relocate date, symbol, ticker_name and step column ----
        df_local.insert(0, "date", df_local.pop("date"))
        df_local.insert(1, "symbol", df_local.pop("symbol"))
        df_local.insert(2, "ticker_name", df_local.pop("ticker_name"))
        df_local.insert(3, "step", df_local.pop("step"))        

        # Estimate return with close price and profile time series ----
        old_count = df_local.shape[0]
        df_local["return"] = df_local["Close"].diff(periods = 1)
        df_local = df_local[(df_local["return"].notnull() & df_local["return"] != 0)]
        print("- Download {} with initial {} rows and {} rows after profiling".format(ticker, old_count, df_local.shape[0]))

        # Estimate log-return with close price ----
        df_local["log_return"] = np.log(df_local["Close"])
        df_local["log_return"] = df_local["log_return"].diff(periods = 1)

        # Estimate absolute log-return ----
        df_local["absolute_log_return"] = np.abs(df_local["log_return"])

        # Estimate log-return volatility ----
        df_temp_1 = df_local.rename(columns = {"absolute_log_return" : "temp_1"}).groupby(["symbol"])["temp_1"].max()
        df_local = df_local.merge(df_temp_1, left_on = "symbol", right_on = "symbol")
        df_local["z_score"] = df_local[["absolute_log_return"]].apply(lambda x: (x - np.mean(x)) / np.std(x))
        df_local["log_volatility"] = np.sqrt(np.abs(df_local["z_score"])) / df_local["temp_1"]

        # Replace NaN with zeros ----
        df_local["log_return"] = df_local["log_return"].fillna(0)
        df_local["absolute_log_return"] = df_local["absolute_log_return"].fillna(0)
        df_local["log_volatility"] = df_local["log_volatility"].fillna(0)

        # Estimate cumulative sum of log-return, absolute log-return and log-return volatility ----
        df_local["cum_log_return"] = df_local["log_return"].cumsum()
        df_local["cum_absolute_log_return"] = df_local["absolute_log_return"].cumsum()
        df_local["cum_log_volatility"] = df_local["log_volatility"].cumsum()

        # Estimate cumulative mean of log-return, absolute log-return and log-return volatility ----
        df_local["cummean_log_return"] = df_local["log_return"].rolling(df_local.shape[0], min_periods = 1).mean()
        df_local["cummean_absolute_log_return"] = df_local["absolute_log_return"].rolling(df_local.shape[0], min_periods = 1).mean()
        df_local["cummean_log_volatility"] = df_local["log_volatility"].rolling(df_local.shape[0], min_periods = 1).mean()
        
        # Estimate cumulative variance of log-return, absolute log-return and log-return volatility ----
        df_local["cumvariance_log_return"] = df_local["log_return"].rolling(df_local.shape[0], min_periods = 2).var()
        df_local["cumvariance_absolute_log_return"] = df_local["absolute_log_return"].rolling(df_local.shape[0], min_periods = 2).var()
        df_local["cumvariance_log_volatility"] = df_local["log_volatility"].rolling(df_local.shape[0], min_periods = 2).var()
        
        # Final merge of data
        df_financial_time_series.append(df_local.fillna(0))
        print("- Processed {} : {}".format(ticker, ticker_name))
        
    df_financial_time_series = pd.concat(df_financial_time_series)
    
    del [df_financial_time_series["temp_1"], df_financial_time_series["z_score"]]
    
    return df_financial_time_series