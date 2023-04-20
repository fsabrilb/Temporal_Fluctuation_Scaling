# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 2023

@author: Felipe Abril Bermúdez
"""

# Libraries ----
import logging
import warnings
import numpy as np
import pandas as pd
import estimate_optimal_window as eow

from tqdm import tqdm
from functools import partial
from scipy.optimize import curve_fit

# Global options ----
warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None
pd.set_option('display.max_columns', None)

# Temporal fluctuation scaling (TFS) ----
def temporal_fluctuation_scaling(mean, coefficient_tfs, exponent_tfs):
    """Estimation of temporal fluctuation scaling (TFS)
    Estimation of temporal fluctuation scaling:
        mean: Mean of a sample in time t
        coefficient_tfs: Coefficient for estimation of TFS as power law
        exponent tfs: Exponent for estimation of TFS as power law
    """    
    return coefficient_tfs * np.power(mean, exponent_tfs)

# Estimate mean and variance parameters ----
def estimate_tfs_parameters_local(
    df_fts,
    p_norm,
    log_path,
    log_filename,
    verbose,
    arg_list
):
    """Estimation of parameters local
    Estimation of parameters of temporal fluctuation scaling:
        df_fts: Dataframe with multiple financial time series
        p_norm: p-norm selection
        log_path: Logs path
        log_filename: log filename for output
        verbose: verbose
        arg_list[0]: Symbol to filter in time series
        arg_list[1]: Number of steps used to filter data
    """
    
    # Definition of arg_list components ----
    symbol = arg_list[0]
    n_step = arg_list[1]
    
    try:
        # Filtration of information ----
        df_fts = df_fts[((df_fts["step"] <= n_step) & (df_fts["symbol"] == symbol))]

        # Estimation of parameters (TFS) ----
        popt_1, pcov_1 = curve_fit(temporal_fluctuation_scaling, df_fts["cummean_log_return"], df_fts["cumvariance_log_return"])
        popt_2, pcov_2 = curve_fit(temporal_fluctuation_scaling, df_fts["cummean_absolute_log_return"], df_fts["cumvariance_absolute_log_return"])
        popt_3, pcov_3 = curve_fit(temporal_fluctuation_scaling, df_fts["cummean_log_volatility"], df_fts["cumvariance_log_volatility"])

        # Estimation of uncertainty in estimate parameters (TFS) ----
        ee_1 = np.sqrt(np.diag(pcov_1))
        ee_2 = np.sqrt(np.diag(pcov_2))
        ee_3 = np.sqrt(np.diag(pcov_3))
        
        # Estimation of value with estimated parameters (TFS) ----
        estimated_1 = temporal_fluctuation_scaling(df_fts["cummean_log_return"], *popt_1).values
        estimated_2 = temporal_fluctuation_scaling(df_fts["cummean_absolute_log_return"], *popt_2).values
        estimated_3 = temporal_fluctuation_scaling(df_fts["cummean_log_volatility"], *popt_3).values

        # Estimation of average error with residuals (TFS) ----
        ae_1 = eow.estimate_p_norm(x = df_fts["cumvariance_log_return"].values, y = estimated_1, p = p_norm)
        ae_2 = eow.estimate_p_norm(x = df_fts["cumvariance_absolute_log_return"].values, y = estimated_2, p = p_norm)
        ae_3 = eow.estimate_p_norm(x = df_fts["cumvariance_log_volatility"].values, y = estimated_3, p = p_norm)
        
        # Estimation of R squared (TFS) ----
        r2_1 = eow.estimate_coefficient_of_determination(y = df_fts["cumvariance_log_return"].values, y_fitted = estimated_1)
        r2_2 = eow.estimate_coefficient_of_determination(y = df_fts["cumvariance_absolute_log_return"].values, y_fitted = estimated_2)
        r2_3 = eow.estimate_coefficient_of_determination(y = df_fts["cumvariance_log_volatility"].values, y_fitted = estimated_3)

        # Final dataframe with regressions ----
        df_parameters = pd.DataFrame(
            {
                "symbol" : [symbol, symbol, symbol],
                "max_step" : [df_fts["step"].max(), df_fts["step"].max(), df_fts["step"].max()],
                "time_series" : ["log-return", "absolute log-return", "log-return volatility"],
                "p_norm" : [p_norm, p_norm, p_norm],
                "coefficient_tfs" : [popt_1[0], popt_2[0], popt_3[0]],
                "error_coefficient_tfs" : [ee_1[0], ee_2[0], ee_3[0]],
                "exponent_tfs" : [popt_1[1], popt_2[1], popt_3[1]],
                "error_exponent_tfs" : [ee_1[1], ee_2[1], ee_3[1]],
                "average_error_tfs" : [ae_1, ae_2, ae_3],
                "rsquared_tfs" : [r2_1, r2_2, r2_3]
            },
            index = [0, 1, 2]
        )
        
        # Function development ----
        if verbose >= 1:
            with open("{}/{}.txt".format(log_path, log_filename), "a") as file:
                file.write("Estimated TFS parameters for {} with {} max steps and {}-norm\n".format(symbol, n_step, p_norm))

    except Exception as e:
        # Final dataframe with regressions ----
        df_parameters = pd.DataFrame(
            {
                "symbol" : [symbol, symbol, symbol],
                "max_step" : [df_fts["step"].max(), df_fts["step"].max(), df_fts["step"].max()],
                "time_series" : ["log-return", "absolute log-return", "log-return volatility"],
                "p_norm" : [p_norm, p_norm, p_norm],
                "coefficient_tfs" : [0, 0, 0],
                "error_coefficient_tfs" : [0, 0, 0],
                "exponent_tfs" : [0, 0, 0],
                "error_exponent_tfs" : [0, 0, 0],
                "average_error_tfs" : [0, 0, 0],
                "rsquared_tfs" : [0, 0, 0]
            },
            index = [0, 1, 2]
        )
        
        # Function development ----
        if verbose >= 1:
            with open("{}/{}.txt".format(log_path, log_filename), "a") as file:
                file.write("No estimated TFS parameters for {} with {} max steps and {}-norm\n".format(symbol, n_step, p_norm))
                file.write("{}\n".format(e))
        
    return df_parameters

# Estimate mean and variance parameters ----
def estimate_tfs_parameters(
    df_fts,
    minimal_steps=30,
    p_norm=2,
    log_path="../logs",
    log_filename="log_tfs_evolution",
    verbose=1,
    tqdm_bar=True
):
    """Estimation of parameters global
    Estimation of parameters of temporal fluctuation scaling in parallel loop:
        df_fts: Dataframe with multiple financial time series
        minimal_steps: Minimum points used for regression of temporal fluctuation scaling (TFS)
        p_norm: p-norm selection
        log_path: Logs path
        log_filename: log filename for output
        verbose: verbose
        tqdm_bar: Progress bar flag
    """
    
    # Auxiliary function for estimation of mean and variance parameters ----
    fun_local = partial(
        estimate_tfs_parameters_local,
        df_fts,
        p_norm,
        log_path,
        log_filename,
        verbose
    )
    
    # Definition of arg_list sampling ----
    arg_list = df_fts[df_fts["step"] >= minimal_steps][["symbol", "step"]].values.tolist()
    
    # Parallel loop for mean and variance parameters estimation ----
    df_fts_parameters = eow.parallel_run(fun = fun_local, arg_list = arg_list, tqdm_bar = tqdm_bar)
    df_fts_parameters = pd.concat(df_fts_parameters).reset_index()
    df_fts_parameters = df_fts_parameters[df_fts_parameters["max_step"] != 0]
    del [df_fts_parameters["index"]]
    
    return df_fts_parameters
