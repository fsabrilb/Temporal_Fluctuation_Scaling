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

from tqdm import tqdm
from scipy.stats import norm
from functools import partial
from scipy.optimize import curve_fit
from multiprocessing import Pool, cpu_count

# Global options ----
warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None
pd.set_option('display.max_columns', None)

# Theoretical evolution of the mean with temporal fluctuation (TFS) hamiltonian of parameter b0 and constant drift ----
def mean_evolution_0(t, cumulant_1, b0, drift_params):
    """Estimation of mean evolution
    Estimation of mean evolution with TFS hamiltonian:
        t: Time
        cumulant_1: Cumulant 1 of cumulant generating function
        b0: Parameter of TFS hamiltonian
        drift_params: Parameter to model the stochastic drift as a constant value
    """
    # Estimate mean
    z = cumulant_1 * t + b0 * np.log(1 + t) + drift_params
    
    return z

# Theoretical evolution of the mean with temporal fluctuation (TFS) hamiltonian of parameter b0 ----
def mean_evolution(t, cumulant_1, b0, *drift_params):
    """Estimation of mean evolution
    Estimation of mean evolution with TFS hamiltonian:
        t: Time
        cumulant_1: Cumulant 1 of cumulant generating function
        b0: Parameter of TFS hamiltonian
        drift_params: Parameters to model the stochastic drift as a polynomial of arbitrary degree
    """
    # Estimate stochastic drift
    rx = np.sum([p * (t**i) for i, p in enumerate(drift_params)])
    
    # Estimate mean
    z = cumulant_1 * t + b0 * np.log(1 + t) + rx
    
    return z

# Theoretical evolution of the variance with temporal fluctuation (TFS) hamiltonian of parameter b0 and constant drift ----
def variance_evolution_0(t, cumulant_1, cumulant_2, b0, drift_params):
    """Estimation of mean evolution
    Estimation of mean evolution with TFS hamiltonian:
        t: Time
        cumulant_1: Cumulant 1 of cumulant generating function
        cumulant_2: Cumulant 2 of cumulant generating function
        b0: Parameter of TFS hamiltonian
        drift_params: Parameters to model the stochastic drift as a constant value
    """
    # Estimate variance
    z = cumulant_2 * t - cumulant_1 * b0 * t * np.log(1 + t) - drift_params * b0 * np.log(1 + t) - (b0 * np.log(1 + t))**2
    
    return z

# Theoretical evolution of the variance with temporal fluctuation (TFS) hamiltonian of parameter b0 ----
def variance_evolution(t, cumulant_1, cumulant_2, b0, *drift_params):
    """Estimation of mean evolution
    Estimation of mean evolution with TFS hamiltonian:
        t: Time
        cumulant_1: Cumulant 1 of cumulant generating function
        cumulant_2: Cumulant 2 of cumulant generating function
        b0: Parameter of TFS hamiltonian
        drift_params: Parameters to model the stochastic drift as a polynomial of arbitrary degree
    """
    # Estimate stochastic drift
    rx = np.sum([p * (t**i) for i, p in enumerate(drift_params)])
    
    # Estimate variance
    z = cumulant_2 * t - cumulant_1 * b0 * t * np.log(1 + t) - rx * b0 * np.log(1 + t) - (b0 * np.log(1 + t))**2
    
    return z

# Estimation of p-norm ----
def estimate_p_norm(x, y, p):
    if p == 0:
        z = np.exp(0.5 * np.mean(np.log(np.power(np.abs(x-y), 2))))
    else:
        z = np.power(np.abs(x - y), 1 / p)
    return np.mean(z)

# Estimation of coefficient of determination R2 ----
def estimate_coefficient_of_determination(y, y_fitted):
    return 1 - np.sum(np.power(y - y_fitted, 2)) / np.sum(np.power(y - np.mean(y), 2))

# Estimate mean and variance parameters ----
def estimate_mean_variance_parameters_local(
    df_fts,
    drift_params_degree,
    p_norm,
    log_path,
    log_filename,
    verbose,
    arg_list
):
    """Estimation of parameters local
    Estimation of parameters of mean and variance evolution:
        df_fts: Dataframe with multiple financial time series
        drift_params_degree: Degree of polynomial associated with stochastic drift
        p_norm: p-norm selection
        log_path: Logs path
        log_filename: log filename for output
        verbose: verbose
        arg_list[0]: Symbol to filter in time series
        arg_list[1]: Window size used to filter data
    """
    
    # Definition of arg_list components (Add 1 parameter for d+1 coefficients and 2 for cumulant_1 and b0) ----
    dd = drift_params_degree
    symbol = arg_list[0]
    window_size = arg_list[1]
    
    try:
        # Filtration of information ----
        df_fts = df_fts[((df_fts["step"]%window_size == 0) & (df_fts["symbol"] == symbol))]

        # Estimation of parameters ----
        if dd == 0:
            # Estimation of parameters (Mean) ----
            popt_1, pcov_1 = curve_fit(mean_evolution_0, df_fts["step"], df_fts["cummean_log_return"])
            popt_2, pcov_2 = curve_fit(mean_evolution_0, df_fts["step"], df_fts["cummean_absolute_log_return"])
            popt_3, pcov_3 = curve_fit(mean_evolution_0, df_fts["step"], df_fts["cummean_log_volatility"])

            # Estimation of parameters (Variance) ----
            popt_4, pcov_4 = curve_fit(variance_evolution_0, df_fts["step"], df_fts["cumvariance_log_return"])
            popt_5, pcov_5 = curve_fit(variance_evolution_0, df_fts["step"], df_fts["cumvariance_absolute_log_return"])
            popt_6, pcov_6 = curve_fit(variance_evolution_0, df_fts["step"], df_fts["cumvariance_log_volatility"])
            
            # Estimation of value with estimated parameters (Mean) ----
            estimated_1 = mean_evolution_0(df_fts["step"], *popt_1).values
            estimated_2 = mean_evolution_0(df_fts["step"], *popt_2).values
            estimated_3 = mean_evolution_0(df_fts["step"], *popt_3).values

            # Estimation of value with estimated parameters (Variance) ----
            estimated_4 = variance_evolution_0(df_fts["step"], *popt_4).values
            estimated_5 = variance_evolution_0(df_fts["step"], *popt_5).values
            estimated_6 = variance_evolution_0(df_fts["step"], *popt_6).values
        else:
            # Estimation of parameters (Mean) ----
            popt_1, pcov_1 = curve_fit(mean_evolution, df_fts["step"], df_fts["cummean_log_return"], p0 = [1] * (dd + 3))
            popt_2, pcov_2 = curve_fit(mean_evolution, df_fts["step"], df_fts["cummean_absolute_log_return"], p0 = [1] * (dd + 3))
            popt_3, pcov_3 = curve_fit(mean_evolution, df_fts["step"], df_fts["cummean_log_volatility"], p0 = [1] * (dd + 3))

            # Estimation of parameters (Variance) ----
            popt_4, pcov_4 = curve_fit(variance_evolution, df_fts["step"], df_fts["cumvariance_log_return"], p0 = [1] * (dd + 4))
            popt_5, pcov_5 = curve_fit(variance_evolution, df_fts["step"], df_fts["cumvariance_absolute_log_return"], p0 = [1] * (dd + 4))
            popt_6, pcov_6 = curve_fit(variance_evolution, df_fts["step"], df_fts["cumvariance_log_volatility"], p0 = [1] * (dd + 4))
            
            # Estimation of value with estimated parameters (Mean) ----
            estimated_1 = mean_evolution(df_fts["step"], *popt_1).values
            estimated_2 = mean_evolution(df_fts["step"], *popt_2).values
            estimated_3 = mean_evolution(df_fts["step"], *popt_3).values

            # Estimation of value with estimated parameters (Variance) ----
            estimated_4 = variance_evolution(df_fts["step"], *popt_4).values
            estimated_5 = variance_evolution(df_fts["step"], *popt_5).values
            estimated_6 = variance_evolution(df_fts["step"], *popt_6).values

        # Estimation of uncertainty in estimate parameters (Mean) ----
        ee_1 = np.sqrt(np.diag(pcov_1))
        ee_2 = np.sqrt(np.diag(pcov_2))
        ee_3 = np.sqrt(np.diag(pcov_3))

        # Estimation of uncertainty in estimated parameters (Variance) ----
        ee_4 = np.sqrt(np.diag(pcov_4))
        ee_5 = np.sqrt(np.diag(pcov_5))
        ee_6 = np.sqrt(np.diag(pcov_6))
        
        # Estimation of R squared (Mean) ----
        r2_1 = estimate_coefficient_of_determination(y = df_fts["cummean_log_return"].values, y_fitted = estimated_1)
        r2_2 = estimate_coefficient_of_determination(y = df_fts["cummean_absolute_log_return"].values, y_fitted = estimated_2)
        r2_3 = estimate_coefficient_of_determination(y = df_fts["cummean_log_volatility"].values, y_fitted = estimated_3)

        # Estimation of R squared (Variance) ----
        r2_4 = estimate_coefficient_of_determination(y = df_fts["cumvariance_log_return"].values, y_fitted = estimated_4)
        r2_5 = estimate_coefficient_of_determination(y = df_fts["cumvariance_absolute_log_return"].values, y_fitted = estimated_5)
        r2_6 = estimate_coefficient_of_determination(y = df_fts["cumvariance_log_volatility"].values, y_fitted = estimated_6)
        
        # Estimation of p-mean absolute error (MAE_p) with residuals (Mean) ----
        ae_1 = estimate_p_norm(x = df_fts["cummean_log_return"].values, y = estimated_1, p = p_norm)
        ae_2 = estimate_p_norm(x = df_fts["cummean_absolute_log_return"].values, y = estimated_2, p = p_norm)
        ae_3 = estimate_p_norm(x = df_fts["cummean_log_volatility"].values, y = estimated_3, p = p_norm)
        
        # Estimation of p-mean absolute error (MAE_p) with residuals (Variance) ----
        ae_4 = estimate_p_norm(x = df_fts["cumvariance_log_return"].values, y = estimated_4, p = p_norm)
        ae_5 = estimate_p_norm(x = df_fts["cumvariance_absolute_log_return"].values, y = estimated_5, p = p_norm)
        ae_6 = estimate_p_norm(x = df_fts["cumvariance_log_volatility"].values, y = estimated_6, p = p_norm)

        # Final dataframe with regressions ----
        df_parameters = pd.DataFrame(
            {
                "symbol" : [symbol, symbol, symbol],
                "window_size" : [window_size, window_size, window_size],
                "drift_degree" : [drift_params_degree, drift_params_degree, drift_params_degree],
                "step" : [df_fts["step"].max(), df_fts["step"].max(), df_fts["step"].max()],
                "time_series" : ["log-return", "absolute log-return", "log-return volatility"],
                "p_norm" : [p_norm, p_norm, p_norm]
            },
            index = [0, 1, 2]
        )
        
        # Mean parameters
        for i in np.arange(dd + 3):
            if i == 0:
                column_name = "cumulant_1_mean"
            elif i == 1:
                column_name = "tfs_param_mean"
            else:
                column_name = "drift_coefficient_{}_mean".format(i - 2)
            
            df_parameters[column_name] = [popt_1[i], popt_2[i], popt_3[i]]
            df_parameters["error_{}".format(column_name)] = [ee_1[i], ee_2[i], ee_3[i]]
        df_parameters["average_error_mean"] = [ae_1, ae_2, ae_3]
        df_parameters["rsquared_mean"] = [r2_1, r2_2, r2_3]
        
        # Variance parameters
        for i in np.arange(dd + 4):
            if i == 0:
                column_name = "cumulant_1_variance"
            elif i == 1:
                column_name = "cumulant_2_variance"
            elif i == 2:
                column_name = "tfs_param_variance"
            else:
                column_name = "drift_coefficient_{}_variance".format(i - 2)
            
            df_parameters[column_name] = [popt_4[i], popt_5[i], popt_6[i]]
            df_parameters["error_{}".format(column_name)] = [ee_4[i], ee_5[i], ee_6[i]]
        df_parameters["average_error_variance"] = [ae_4, ae_5, ae_6]
        df_parameters["rsquared_variance"] = [r2_4, r2_5, r2_6]
        
        # Function development ----
        if verbose >= 1:
            with open("{}/{}.txt".format(log_path, log_filename), "a") as file:
                file.write("Estimated parameters for {} with {} window size and {}-norm\n".format(symbol, window_size, p_norm))

    except Exception as e:
        # Final dataframe with regressions ----
        df_parameters = pd.DataFrame(
            {
                "symbol" : [symbol, symbol, symbol],
                "window_size" : [window_size, window_size, window_size],
                "drift_degree" : [drift_params_degree, drift_params_degree, drift_params_degree],
                "step" : [0, 0, 0],
                "time_series" : ["log-return", "absolute log-return", "log-return volatility"],
                "p_norm" : [p_norm, p_norm, p_norm]
            },
            index = [0, 1, 2]
        )
        
        # Mean parameters
        for i in np.arange(dd + 3):
            if i == 0:
                column_name = "cumulant_1_mean"
            elif i == 1:
                column_name = "tfs_param_mean"
            else:
                column_name = "drift_coefficient_{}_mean".format(i - 2)
            
            df_parameters[column_name] = [0, 0, 0]
            df_parameters["error_{}".format(column_name)] = [0, 0, 0]
        df_parameters["average_error_mean"] = [0, 0, 0]
        df_parameters["rsquared_mean"] = [0, 0, 0]
        
        # Variance parameters
        for i in np.arange(dd + 4):
            if i == 0:
                column_name = "cumulant_1_variance"
            elif i == 1:
                column_name = "cumulant_2_variance"
            elif i == 2:
                column_name = "tfs_param_variance"
            else:
                column_name = "drift_coefficient_{}_variance".format(i - 2)
            
            df_parameters[column_name] = [0, 0, 0]
            df_parameters["error_{}".format(column_name)] = [0, 0, 0]
        df_parameters["average_error_variance"] = [0, 0, 0]
        df_parameters["rsquared_variance"] = [0, 0, 0]
        
        # Function development ----
        if verbose >= 1:
            with open("{}/{}.txt".format(log_path, log_filename), "a") as file:
                file.write("No estimated parameters for {} with {} window size and {}-norm\n".format(symbol, window_size, p_norm))
                file.write("{}\n".format(e))
        
    return df_parameters

# Deployment of parallel run in function of arguments list ----
def parallel_run(
    fun,
    arg_list,
    tqdm_bar=False
):
    """Parallel run
    Implement parallel run in arbitrary function with input arg_list:
        fun: Function to implement in parallel
        arg_list: List of arguments to pass in function
        tqdm_bar: Progress bar flag
    """
    
    if tqdm_bar:
        m = []
        with Pool(processes = cpu_count()) as p:
            with tqdm(total = len(arg_list), ncols = 60) as pbar:
                for _ in p.imap(fun, arg_list):
                    m.append(_)
                    pbar.update()
            p.terminate()
            p.join()
    else:
        p = Pool(processes = cpu_count())
        m = p.map(fun, arg_list)
        p.terminate()
        p.join() 
    return m

# Estimate representative sample size ----
def estimate_sample_size(
    n_population,
    sample_error=0.02,
    feature_percentage=0.5,
    confidence_interval=0.95
):
    """Estimation of sample representative size
    Estimation of sample size:
        n_population: Number of observations in the population
        sample_error: Error interval for extrapolating results to the total population
        feature_percentage: Percentage of population with the feature of interest (p*(1-p) is maximum with p=0.5)
        confidence_interval: Confidence interval defined with the tails of distribution
    """
    
    z_alpha = norm.ppf(1 - (1 - confidence_interval) / 2)
    z = n_population * (1 / (1 + (sample_error * sample_error * (n_population - 1)) / (z_alpha * z_alpha * feature_percentage * (1 - feature_percentage))))
    return np.ceil(z)

# Estimate mean and variance parameters ----
def estimate_mean_variance_parameters(
    df_fts,
    drift_params_degree,
    minimal_points=10,
    p_norm=2,
    log_path="../logs",
    log_filename="log_optimal_window",
    verbose=1,
    tqdm_bar=True
):
    """Estimation of parameters global
    Estimation of parameters of mean and variance evolution in parallel loop:
        df_fts: Dataframe with multiple financial time series
        drift_params_degree: Degree of polynomial associated with stochastic drift
        minimal_points: Minimum points used for regression in mean and variance
        p_norm: p-norm selection
        log_path: Logs path
        log_filename: log filename for output
        verbose: verbose
        tqdm_bar: Progress bar flag
    """
    
    # Auxiliary function for estimation of mean and variance parameters ----
    fun_local = partial(
        estimate_mean_variance_parameters_local,
        df_fts,
        drift_params_degree,
        p_norm,
        log_path,
        log_filename,
        verbose
    )
    
    # Definition of arg_list sampling ----
    arg_list = df_fts.value_counts(["symbol"]).rename_axis("symbol").reset_index(name = "counts")
    arg_list["counts"] = np.ceil(arg_list["counts"] / minimal_points)
    arg_list = df_fts.merge(arg_list, left_on = "symbol", right_on = "symbol")
    arg_list = arg_list[arg_list["step"] < arg_list["counts"]][["symbol", "step"]].values.tolist()
    
    # Parallel loop for mean and variance parameters estimation ----
    df_fts_parameters = parallel_run(fun = fun_local, arg_list = arg_list, tqdm_bar = tqdm_bar)
    df_fts_parameters = pd.concat(df_fts_parameters).reset_index()
    df_fts_parameters = df_fts_parameters[((df_fts_parameters["step"] != 0) & (df_fts_parameters["window_size"] != 0))]
    del [df_fts_parameters["index"]]
    
    return df_fts_parameters

# Estimate optimal window ----
def estimate_optimal_window(
    df_fts_parameters
):
    """Estimation of optimal window
    Estimation of optimal window:
        df_fts_parameters: Dataframe with multiple regression obtained with estimate_mean_variance_parameters
    """
   
    # Optimum window per symbol for mean evolution ----
    df_optimal_mean = (
        df_fts_parameters[df_fts_parameters["rsquared_mean"] == df_fts_parameters.groupby(["symbol", "time_series"])["rsquared_mean"].transform("max")]
            .sort_values(["symbol", "time_series", "rsquared_mean", "step"], ascending = [True, True, False, False])
            .drop_duplicates(["symbol", "time_series", "rsquared_mean"])
            .rename(columns = {"window_size" : "window_size_mean", "step" : "step_mean"})
            .drop(list(df_fts_parameters.filter(regex = "_variance")), axis = 1)
    )

    # Optimum window per symbol for variance evolution ----
    df_optimal_variance = (
        df_fts_parameters[df_fts_parameters["rsquared_variance"] == df_fts_parameters.groupby(["symbol", "time_series"])["rsquared_variance"].transform("max")]
            .sort_values(["symbol", "time_series", "rsquared_variance", "step"], ascending = [True, True, False, False])
            .drop_duplicates(["symbol", "time_series", "rsquared_variance"])
            .rename(columns = {"window_size" : "window_size_variance", "step" : "step_variance"})
            .drop(list(df_fts_parameters.filter(regex = "_mean")), axis = 1)
    )
    
    # Merge all data in one dataframe ----    
    df_optimal = df_optimal_mean.merge(
        df_optimal_variance,
        left_on = ["symbol", "time_series", "p_norm", "drift_degree"],
        right_on = ["symbol", "time_series", "p_norm", "drift_degree"]
    )
    
    return df_optimal