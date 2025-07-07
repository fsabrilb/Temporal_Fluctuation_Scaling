# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 2023

@author: Felipe Abril Bermúdez
"""

# Libraries ----
import re
import sys
import warnings
import matplotlib
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mtick
import matplotlib.colors as mcolors
import estimate_optimal_window as eow
import matplotlib.transforms as mtransforms

from math import log10, floor
from scipy.optimize import curve_fit

# Global options ----
warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None
pd.set_option('display.max_columns', None)

colors = ["blue", "green", "red", "purple", "orange", "brown", "pink", "olive", "gray", "cyan"]

# Estimate mean and variance parameters ----
def prepare_data(
    df_fts,
    df_optimal,
    interval
):
    """Preparation of data for plotting
    Join original data with optimal window size data:
        df_fts: Dataframe with multiple financial time series
        df_optimal: Dataframe with optimal window size per financial time series
        interval: Select transformation for estimation of time between minimum and maximum date
    """
    
    # Estimation of interval of time for each ticker ----
    df_fts["min_date"] = pd.to_datetime(df_fts.groupby(["symbol"])["date"].transform("min"), errors = "coerce", infer_datetime_format = True)
    df_fts["max_date"] = pd.to_datetime(df_fts.groupby(["symbol"])["date"].transform("max"), errors = "coerce", infer_datetime_format = True)
    
    interval_dict = {"years" : "Y", "months" : "M", "weeks" : "W", "days" : "D", "hours" : "h", "minutes" : "m", "seconds" : "s", "milliseconds" : "ms"}
    df_fts["duration"] = (df_fts["max_date"] - df_fts["min_date"]) / np.timedelta64(1, interval_dict[interval])
    df_dates = df_fts[["symbol", "min_date", "max_date", "duration"]].drop_duplicates(["symbol", "min_date", "max_date", "duration"])
    
    # Log-return data ----
    df_logr = (
        df_fts[["date", "symbol", "step", "cummean_log_return", "cumvariance_log_return"]]
            .rename(columns = {"cummean_log_return" : "cummean", "cumvariance_log_return" : "cumvariance"})
    )
    df_logr["time_series"] = "log-return"
    
    # Absolute log-return data ----
    df_loga = (
        df_fts[["date", "symbol", "step", "cummean_absolute_log_return", "cumvariance_absolute_log_return"]]
            .rename(columns = {"cummean_absolute_log_return" : "cummean", "cumvariance_absolute_log_return" : "cumvariance"})
    )
    df_loga["time_series"] = "absolute log-return"
    
    # Log-return volatility data ----
    df_logv = (
        df_fts[["date", "symbol", "step", "cummean_log_volatility", "cumvariance_log_volatility"]]
            .rename(columns = {"cummean_log_volatility" : "cummean", "cumvariance_log_volatility" : "cumvariance"})
    )
    df_logv["time_series"] = "log-return volatility"
    
    # Merge final data ----
    df_plot_data = (
        pd.concat([df_logr, df_loga, df_logv])
            .merge(df_optimal, left_on = ["symbol", "time_series"], right_on = ["symbol", "time_series"])
            .merge(df_dates, left_on = ["symbol"], right_on = ["symbol"])
    )
        
    return df_plot_data

# Plot evolution of mean and variance ----
def plot_optimal_window(
    df_fts,
    df_optimal,
    interval,
    symbols,
    width,
    height,
    all_data = False,
    markersize = 2,
    fontsize_labels = 13.5,
    fontsize_legend = 11.5,
    usetex = False,
    n_cols = 4,
    n_x_breaks = 10,
    n_y_breaks = 10,
    fancy_legend = True,
    dpi = 150,
    save_figures = True,
    output_path = "../output_files",
    information_name = "",
    input_generation_date = "2023-03-28"
):
    """Preparation of data for plotting
    Join original data with optimal window size data:
        df_fts: Dataframe with multiple financial time series
        df_optimal: Dataframe with optimal window size per financial time series
        interval: Select transformation for estimation of time between minimum and maximum date
        symbols: Symbols of the financial time series plotted
        width: Width of final plot
        height: Height of final plot
        all_data: Flag for selection of plot all data or only data every optimal_window step
        markersize: Marker size as in plt.plot()
        fontsize_labels: Font size in axis labels
        fontsize_legend: Font size in legend
        usetex: Use LaTeX for renderized plots
        n_cols: Number of columns in legend
        n_x_breaks: Number of divisions in x-axis
        n_y_breaks: Number of divisions in y-axis
        fancy_legend: Fancy legend output
        dpi: Dot per inch for output plot
        save_figures: Save figures flag
        output_path: Output path where figures is saved
        information_name: Name of the output plot
        input_generation_date: Date of generation (control version)
    """
    
    # Plot data and define loop over symbols ----
    df_fts = df_fts[df_fts["symbol"].isin(symbols)]
    df_optimal = df_optimal[df_optimal["symbol"].isin(symbols)]
    df_graph = prepare_data(df_fts = df_fts, df_optimal = df_optimal, interval = interval)
    loop_index = sorted(df_graph["symbol"].unique().tolist())
    
    # Begin plot inputs ----
    matplotlib.rcParams.update(
        {
            "font.family": "serif",
            "text.usetex": usetex,
            "pgf.rcfonts": False
        }
    )
    
    fig1, ax1 = plt.subplots(len(loop_index), 3)
    fig2, ax2 = plt.subplots(len(loop_index), 3)
    fig1.set_size_inches(w = width, h = height)
    fig2.set_size_inches(w = width, h = height)
    counter = 0
        
    for i in loop_index:
        counter_i = 0
        for j in sorted(df_graph[df_graph["symbol"] == i]["time_series"].unique().tolist()):
            # Filter information ----
            df_aux = df_graph[((df_graph["symbol"] == i) & (df_graph["time_series"] == j))]
        
            # Parameters ----
            dates_j = pd.to_datetime(df_aux["date"], errors = "coerce")
            time_labels = pd.date_range(start = dates_j.min(), end = dates_j.max(), periods = n_x_breaks).strftime("%Y-%m-%d")
            ave_mean_j = df_aux["average_error_mean"]
            ave_variance_j = df_aux["average_error_variance"]
            window_mean_j = df_aux["window_size_mean"].unique()[0]
            window_variance_j = df_aux["window_size_variance"].unique()[0]
            dd = df_aux["drift_degree"].unique()[0]
            
            # Extract empirical data ----
            if all_data:
                dates_mean_j = pd.to_datetime(df_aux["date"], errors = "coerce")
                dates_variance_j = pd.to_datetime(df_aux["date"], errors = "coerce")
                mean_j = df_aux["cummean"]
                variance_j = df_aux["cumvariance"]
                
            else:
                dates_mean_j = pd.to_datetime(df_aux[df_aux["step"]%window_mean_j == 0]["date"], errors = "coerce")
                dates_variance_j = pd.to_datetime(df_aux[df_aux["step"]%window_variance_j == 0]["date"], errors = "coerce")
                mean_j = df_aux[df_aux["step"]%window_mean_j == 0]["cummean"]
                variance_j = df_aux[df_aux["step"]%window_variance_j == 0]["cumvariance"]
            
            # Extract fitting data ----
            if dd == 0:
                popt_mean, pcov_mean = curve_fit(
                    eow.mean_evolution_0,
                    df_aux[df_aux["step"]%window_mean_j == 0]["step"],
                    df_aux[df_aux["step"]%window_mean_j == 0]["cummean"]
                )
                popt_variance, pcov_variance = curve_fit(
                    eow.variance_evolution_0,
                    df_aux[df_aux["step"]%window_variance_j == 0]["step"],
                    df_aux[df_aux["step"]%window_variance_j == 0]["cumvariance"]
                )
            else:
                popt_mean, pcov_mean = curve_fit(
                    eow.mean_evolution,
                    df_aux[df_aux["step"]%window_mean_j == 0]["step"],
                    df_aux[df_aux["step"]%window_mean_j == 0]["cummean"],
                    p0 = [1] * (dd + 3)
                )
                popt_variance, pcov_variance = curve_fit(
                    eow.variance_evolution,
                    df_aux[df_aux["step"]%window_variance_j == 0]["step"],
                    df_aux[df_aux["step"]%window_variance_j == 0]["cumvariance"],
                    p0 = [1] * (dd + 4)
                )    
            
            # Extract parameters of error ----
            error_mean = np.sqrt(np.diag(pcov_mean))
            error_mean[np.isinf(error_mean)] = 0
            lower_mean = popt_mean - error_mean
            upper_mean = popt_mean + error_mean
            
            error_variance = np.sqrt(np.diag(pcov_variance))
            error_variance[np.isinf(error_variance)] = 0
            lower_variance = popt_variance - error_variance
            upper_variance = popt_variance + error_variance
            
            # Extract theoretical values ----
            if dd == 0:
                # Theoretical mean ----
                mean_prome = eow.mean_evolution_0(df_aux["step"], *popt_mean)
                mean_lower = eow.mean_evolution_0(df_aux["step"], *lower_mean)
                mean_upper = eow.mean_evolution_0(df_aux["step"], *upper_mean)

                # Theoretical variance ----
                variance_prome = eow.variance_evolution_0(df_aux["step"], *popt_variance)
                variance_lower = eow.variance_evolution_0(df_aux["step"], *lower_variance)
                variance_upper = eow.variance_evolution_0(df_aux["step"], *upper_variance)
                
                # Estimation of R squared (Mean) ----
                mean_aux = eow.mean_evolution_0(df_aux[df_aux["step"]%window_mean_j == 0]["step"], *popt_mean)
                r2_mean = df_aux[df_aux["step"]%window_mean_j == 0]["cummean"]
                r2_mean = np.sum(np.power(r2_mean.values - mean_aux.values, 2)) / np.sum(np.power(r2_mean.values - np.mean(r2_mean), 2))
                r2_mean = 1 - r2_mean

                # Estimation of R squared (Variance) ----
                variance_aux = eow.variance_evolution_0(df_aux[df_aux["step"]%window_variance_j == 0]["step"], *popt_variance)
                r2_variance = df_aux[df_aux["step"]%window_variance_j == 0]["cumvariance"]
                r2_variance = np.sum(np.power(r2_variance.values - variance_aux.values, 2)) / np.sum(np.power(r2_variance.values - np.mean(r2_variance), 2))
                r2_variance = 1 - r2_variance
                
            else:
                # Theoretical mean ----
                mean_prome = eow.mean_evolution(df_aux["step"], *popt_mean)
                mean_lower = eow.mean_evolution(df_aux["step"], *lower_mean)
                mean_upper = eow.mean_evolution(df_aux["step"], *upper_mean)

                # Theoretical variance ----
                variance_prome = eow.variance_evolution(df_aux["step"], *popt_variance)
                variance_lower = eow.variance_evolution(df_aux["step"], *lower_variance)
                variance_upper = eow.variance_evolution(df_aux["step"], *upper_variance)
                
                # Estimation of R squared (Mean) ----
                mean_aux = eow.mean_evolution(df_aux[df_aux["step"]%window_mean_j == 0]["step"], *popt_mean)
                r2_mean = df_aux[df_aux["step"]%window_mean_j == 0]["cummean"]
                r2_mean = np.sum(np.power(r2_mean.values - mean_aux.values, 2)) / np.sum(np.power(r2_mean.values - np.mean(r2_mean), 2))
                r2_mean = 1 - r2_mean

                # Estimation of R squared (Variance) ----
                variance_aux = eow.variance_evolution(df_aux[df_aux["step"]%window_variance_j == 0]["step"], *popt_variance)
                r2_variance = df_aux[df_aux["step"]%window_variance_j == 0]["cumvariance"]
                r2_variance = np.sum(np.power(r2_variance.values - variance_aux.values, 2)) / np.sum(np.power(r2_variance.values - np.mean(r2_variance), 2))
                r2_variance = 1 - r2_variance
            
            # Plot graphs ----
            if len(loop_index) == 1:
                # Plot graph (Mean) ----
                plot_1 = ax1[counter_i].plot(
                    dates_mean_j,
                    mean_j,
                    alpha = 1,
                    zorder = 2,
                    color = "black",
                    marker = "o",
                    linestyle = "",
                    label = "empirical data",
                    markersize = markersize
                )
                ax1[counter_i].plot(dates_j, mean_prome, alpha = 1, zorder = 2, color = colors[counter_i], linewidth = 3, label = "fitting")
                ax1[counter_i].fill_between(
                    dates_j,
                    mean_lower,
                    mean_upper,
                    where = ((mean_upper >= mean_lower) & (mean_upper >= mean_prome) & (mean_prome >= mean_lower)),
                    alpha = 0.19,
                    facecolor = colors[counter_i],
                    interpolate = True
                )
                ax1[counter_i].tick_params(which = "major", direction = "in", top = True, right = True, labelsize = fontsize_labels, length = 12)
                ax1[counter_i].tick_params(which = "minor", direction = "in", top = True, right = True, labelsize = fontsize_labels, length = 6)
                ax1[counter_i].xaxis.set_major_locator(mdates.AutoDateLocator(maxticks = len(time_labels)))
                ax1[counter_i].xaxis.set_minor_locator(mdates.AutoDateLocator(maxticks = (4 * len(time_labels))))
                ax1[counter_i].yaxis.set_major_locator(mtick.MaxNLocator(n_y_breaks))
                ax1[counter_i].yaxis.set_minor_locator(mtick.MaxNLocator(5 * n_y_breaks))
                ax1[counter_i].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
                ax1[counter_i].yaxis.set_major_formatter(mtick.FormatStrFormatter("%.1e"))
                ax1[counter_i].tick_params(axis = "x", labelrotation = 90)
                ax1[counter_i].set_xlabel("Date", fontsize = fontsize_labels)        
                ax1[counter_i].set_ylabel("Mean - {}".format(j.capitalize()), fontsize = fontsize_labels)
                ax1[counter_i].legend(fancybox = fancy_legend, shadow = True, ncol = n_cols, fontsize = fontsize_legend)
                ax1[counter_i].set_title(
                    r"({}) $n_s={}$, $MAE_p={}$, $R^2={}\%$".format(chr(counter_i + 65), window_mean_j, round(ave_mean_j.min(), 5), round(r2_mean * 1e2, 2)),
                    loc = "left",
                    y = 1.005,
                    fontsize = fontsize_labels
                )

                # Plot graph (Variance) ----
                plot_2 = ax2[counter_i].plot(
                    dates_variance_j,
                    variance_j,
                    alpha = 1,
                    zorder = 2,
                    color = "black",
                    marker = "o",
                    linestyle = "",
                    label = "empirical data",
                    markersize = markersize
                )
                ax2[counter_i].plot(dates_j, variance_prome, alpha = 1, zorder = 2, color = colors[counter_i + 3], linewidth = 3, label = "fitting")
                ax2[counter_i].fill_between(
                    dates_j,
                    variance_lower,
                    variance_upper,
                    where = ((variance_upper >= variance_lower) & (variance_upper >= variance_prome) & (variance_prome >= variance_lower)),
                    alpha = 0.19,
                    facecolor = colors[counter_i + 3],
                    interpolate = True
                )
                ax2[counter_i].tick_params(which = "major", direction = "in", top = True, right = True, labelsize = fontsize_labels, length = 12)
                ax2[counter_i].tick_params(which = "minor", direction = "in", top = True, right = True, labelsize = fontsize_labels, length = 6)
                ax2[counter_i].xaxis.set_major_locator(mdates.AutoDateLocator(maxticks = len(time_labels)))
                ax2[counter_i].xaxis.set_minor_locator(mdates.AutoDateLocator(maxticks = (4 * len(time_labels))))
                ax2[counter_i].yaxis.set_major_locator(mtick.MaxNLocator(n_y_breaks))
                ax2[counter_i].yaxis.set_minor_locator(mtick.MaxNLocator(5 * n_y_breaks))
                ax2[counter_i].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
                ax2[counter_i].yaxis.set_major_formatter(mtick.FormatStrFormatter("%.1e"))
                ax2[counter_i].tick_params(axis = "x", labelrotation = 90)
                ax2[counter_i].set_xlabel("Date", fontsize = fontsize_labels)        
                ax2[counter_i].set_ylabel("Variance - {}".format(j.capitalize()), fontsize = fontsize_labels)
                ax2[counter_i].legend(fancybox = fancy_legend, shadow = True, ncol = n_cols, fontsize = fontsize_legend)
                ax2[counter_i].set_title("({})".format(chr(counter_i + 65)), loc = "left", y = 1.005)
                ax2[counter_i].set_title(
                    r"({}) $n_s={}$, $MAE_p={}$, $R^2={}\%$".format(chr(counter_i + 65), window_variance_j, round(ave_variance_j.min(), 5), round(r2_variance * 1e2, 2)),
                    loc = "left",
                    y = 1.005,
                    fontsize = fontsize_labels
                )
            else:
                # Plot graph (Mean) ----
                plot_1 = ax1[counter, counter_i].plot(
                    dates_mean_j,
                    mean_j,
                    alpha = 1,
                    zorder = 2,
                    color = "black",
                    marker = "o",
                    linestyle = "",
                    label = "empirical data",
                    markersize = markersize
                )
                ax1[counter, counter_i].plot(dates_j, mean_prome, alpha = 1, zorder = 2, color = colors[counter_i], linewidth = 3, label = "fitting")
                ax1[counter, counter_i].fill_between(
                    dates_j,
                    mean_lower,
                    mean_upper,
                    where = ((mean_upper >= mean_lower) & (mean_upper >= mean_prome) & (mean_prome >= mean_lower)),
                    alpha = 0.19,
                    facecolor = colors[counter_i],
                    interpolate = True
                )
                ax1[counter, counter_i].tick_params(which = "major", direction = "in", top = True, right = True, labelsize = fontsize_labels, length = 12)
                ax1[counter, counter_i].tick_params(which = "minor", direction = "in", top = True, right = True, labelsize = fontsize_labels, length = 6)
                ax1[counter, counter_i].xaxis.set_major_locator(mdates.AutoDateLocator(maxticks = len(time_labels)))
                ax1[counter, counter_i].xaxis.set_minor_locator(mdates.AutoDateLocator(maxticks = (4 * len(time_labels))))
                ax1[counter, counter_i].yaxis.set_major_locator(mtick.MaxNLocator(n_y_breaks))
                ax1[counter, counter_i].yaxis.set_minor_locator(mtick.MaxNLocator(5 * n_y_breaks))
                ax1[counter, counter_i].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
                ax1[counter, counter_i].yaxis.set_major_formatter(mtick.FormatStrFormatter("%.1e"))
                ax1[counter, counter_i].tick_params(axis = "x", labelrotation = 90)
                ax1[counter, counter_i].set_xlabel("Date", fontsize = fontsize_labels)        
                ax1[counter, counter_i].set_ylabel("Mean - {}".format(j.capitalize()), fontsize = fontsize_labels)
                ax1[counter, counter_i].legend(fancybox = fancy_legend, shadow = True, ncol = n_cols, fontsize = fontsize_legend)
                ax1[counter, counter_i].set_title(
                    r"({}) $n_s={}$, $MAE_p={}$, $R^2={}\%$".format(chr(counter_i + 65), window_mean_j, round(ave_mean_j.min(), 5), round(r2_mean * 1e2, 2)),
                    loc = "left",
                    y = 1.005,
                    fontsize = fontsize_labels
                )

                # Plot graph (Variance) ----
                plot_2 = ax2[counter, counter_i].plot(
                    dates_variance_j,
                    variance_j,
                    alpha = 1,
                    zorder = 2,
                    color = "black",
                    marker = "o",
                    linestyle = "",
                    label = "empirical data",
                    markersize = markersize
                )
                ax2[counter, counter_i].plot(dates_j, variance_prome, alpha = 1, zorder = 2, color = colors[counter_i + 3], linewidth = 3, label = "fitting")
                ax2[counter, counter_i].fill_between(
                    dates_j,
                    variance_lower,
                    variance_upper,
                    where = ((variance_upper >= variance_lower) & (variance_upper >= variance_prome) & (variance_prome >= variance_lower)),
                    alpha = 0.19,
                    facecolor = colors[counter_i + 3],
                    interpolate = True
                )
                ax2[counter, counter_i].tick_params(which = "major", direction = "in", top = True, right = True, labelsize = fontsize_labels, length = 12)
                ax2[counter, counter_i].tick_params(which = "minor", direction = "in", top = True, right = True, labelsize = fontsize_labels, length = 6)
                ax2[counter, counter_i].xaxis.set_major_locator(mdates.AutoDateLocator(maxticks = len(time_labels)))
                ax2[counter, counter_i].xaxis.set_minor_locator(mdates.AutoDateLocator(maxticks = (4 * len(time_labels))))
                ax2[counter, counter_i].yaxis.set_major_locator(mtick.MaxNLocator(n_y_breaks))
                ax2[counter, counter_i].yaxis.set_minor_locator(mtick.MaxNLocator(5 * n_y_breaks))
                ax2[counter, counter_i].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
                ax2[counter, counter_i].yaxis.set_major_formatter(mtick.FormatStrFormatter("%.1e"))
                ax2[counter, counter_i].tick_params(axis = "x", labelrotation = 90)
                ax2[counter, counter_i].set_xlabel("Date", fontsize = fontsize_labels)        
                ax2[counter, counter_i].set_ylabel("Variance - {}".format(j.capitalize()), fontsize = fontsize_labels)
                ax2[counter, counter_i].legend(fancybox = fancy_legend, shadow = True, ncol = n_cols, fontsize = fontsize_legend)
                ax2[counter, counter_i].set_title(
                    r"({}) $n_s={}$, $MAE_p={}$, $R^2={}\%$".format(chr(counter_i + 65), window_variance_j, round(ave_variance_j.min(), 5), round(r2_variance * 1e2, 2)),
                    loc = "left",
                    y = 1.005,
                    fontsize = fontsize_labels
                )

            # Function development ----
            counter_i += 1
            print("Generated plot for {} and time series {}".format(i, j))
        
        counter += 1
    
    fig1.tight_layout()
    fig2.tight_layout()
    if save_figures:
        plt.show()
        fig1.savefig(
            "{}/{}_mean_evolution_{}.png".format(output_path, information_name, re.sub("-", "", input_generation_date)),
            bbox_inches = "tight",
            facecolor = fig1.get_facecolor(),
            transparent = False,
            pad_inches = 0.03,
            dpi = dpi
        )
        plt.show()
        fig2.savefig(
            "{}/{}_variance_evolution_{}.png".format(output_path, information_name, re.sub("-", "", input_generation_date)),
            bbox_inches = "tight",
            facecolor = fig2.get_facecolor(),
            transparent = False,
            pad_inches = 0.03,
            dpi = dpi
        )
    plt.close()
    plt.close()
    
    return df_graph

# Resume evolution of mean and variance ----
def resume_optimal_window(
    df_optimal,
    symbols_order,
    rank=2,
    precision=3,
    output_path = "../output_files",
    input_generation_date = "2023-03-28"
):
    """Preparation of data for resume table
    Join original data with optimal window size data:
        df_optimal: Dataframe with optimal window size per financial time series
        symbols_order: Symbols order of the financial time series
        rank: Integer used for center the number of orders of magnitude respect to minimum value
        precision: Number of digits in final output
        output_path: Output path where figures is saved
        input_generation_date: Date of generation (control version)
    """
    
    # Auxiliary function for round to n significant numbers and extract order of magnitude ----
    def round_to_n(x, n):
        order_magnitude = floor(log10(abs(x)))
        rounded_number = round(x, -int(order_magnitude) + n) / (10.0 ** order_magnitude)
        return order_magnitude, rounded_number
    
    # Extract data ----
    df_optimal = df_optimal[df_optimal["symbol"].isin(symbols_order)].replace(np.nan, 10 ** -100)
    df_optimal.sort_values(["symbol"], inplace = True)
    
    if df_optimal["drift_degree"].unique()[0] == 0:
        # Mean - Order of magnitude for average error, parameters and error in parameters ----
        df_optimal["average_error_mean_om"] = df_optimal["average_error_mean"].apply(lambda x: round_to_n(x, precision)[0])
        df_optimal["cumulant_1_mean_om"] = df_optimal["cumulant_1_mean"].apply(lambda x: round_to_n(x, precision)[0])
        df_optimal["tfs_param_mean_om"] = df_optimal["tfs_param_mean"].apply(lambda x: round_to_n(x, precision)[0])
        df_optimal["drift_coefficient_0_mean_om"] = df_optimal["drift_coefficient_0_mean"].apply(lambda x: round_to_n(x, precision)[0])
        df_optimal["error_cumulant_1_mean_om"] = df_optimal["error_cumulant_1_mean"].apply(lambda x: round_to_n(x, precision)[0])
        df_optimal["error_tfs_param_mean_om"] = df_optimal["error_tfs_param_mean"].apply(lambda x: round_to_n(x, precision)[0])
        df_optimal["error_drift_coefficient_0_mean_om"] = df_optimal["error_drift_coefficient_0_mean"].apply(lambda x: round_to_n(x, precision)[0])
        
        # Mean - Minimum order of magnitude for clear output in rounded numbers ----
        df_optimal["average_error_mean_om_min"] = df_optimal["average_error_mean_om"].min() + rank
        df_optimal["cumulant_1_mean_om_min"] = df_optimal["cumulant_1_mean_om"].min() + rank
        df_optimal["tfs_param_mean_om_min"] = df_optimal["tfs_param_mean_om"].min() + rank
        df_optimal["drift_coefficient_0_mean_om_min"] = df_optimal["drift_coefficient_0_mean_om"].min() + rank
        
        # Mean - Difference in order of magnitude of parameters and error in parameters from minimum order ----
        df_optimal["average_error_mean_om"] = df_optimal["average_error_mean_om"] - df_optimal["average_error_mean_om_min"]
        df_optimal["cumulant_1_mean_om"] = df_optimal["cumulant_1_mean_om"] - df_optimal["cumulant_1_mean_om_min"]
        df_optimal["tfs_param_mean_om"] = df_optimal["tfs_param_mean_om"] - df_optimal["tfs_param_mean_om_min"]
        df_optimal["drift_coefficient_0_mean_om"] = df_optimal["drift_coefficient_0_mean_om"] - df_optimal["drift_coefficient_0_mean_om_min"]
        df_optimal["error_cumulant_1_mean_om"] = df_optimal["error_cumulant_1_mean_om"] - df_optimal["cumulant_1_mean_om_min"]
        df_optimal["error_tfs_param_mean_om"] = df_optimal["error_tfs_param_mean_om"] - df_optimal["tfs_param_mean_om_min"]
        df_optimal["error_drift_coefficient_0_mean_om"] = df_optimal["error_drift_coefficient_0_mean_om"] - df_optimal["drift_coefficient_0_mean_om_min"]
        
        # Mean - Final rounded numbers ----
        df_optimal["average_error_mean_rn"] = df_optimal["average_error_mean"] / (10.0 ** df_optimal["average_error_mean_om_min"])
        df_optimal["cumulant_1_mean_rn"] = df_optimal["cumulant_1_mean"] / (10.0 ** df_optimal["cumulant_1_mean_om_min"])
        df_optimal["tfs_param_mean_rn"] = df_optimal["tfs_param_mean"] / (10.0 ** df_optimal["tfs_param_mean_om_min"])
        df_optimal["drift_coefficient_0_mean_rn"] = df_optimal["drift_coefficient_0_mean"] / (10.0 ** df_optimal["drift_coefficient_0_mean_om_min"])
        df_optimal["error_cumulant_1_mean_rn"] = df_optimal["error_cumulant_1_mean"] / (10.0 ** df_optimal["cumulant_1_mean_om_min"])
        df_optimal["error_tfs_param_mean_rn"] = df_optimal["error_tfs_param_mean"] / (10.0 ** df_optimal["tfs_param_mean_om_min"])
        df_optimal["error_drift_coefficient_0_mean_rn"] = df_optimal["error_drift_coefficient_0_mean"] / (10.0 ** df_optimal["drift_coefficient_0_mean_om_min"])
        
        # Mean - Final resume optimal data ----
        list_mean = ["symbol", "time_series", "cumulant_1_mean_om_min", "tfs_param_mean_om_min", "drift_coefficient_0_mean_om_min", "average_error_mean_om_min"]
        df_optimal_mean_resume = df_optimal[list_mean].sort_values(["symbol"])
        
        df_optimal_mean_resume["Cumulant 1"] = (
            "$" +
            df_optimal["cumulant_1_mean_rn"].apply(lambda x: "{:.{}f}".format(x, precision)) +
            "\pm" +
            df_optimal["error_cumulant_1_mean_rn"].apply(lambda x: "{:.{}f}".format(x, precision)) +
            "$"
        )
        df_optimal_mean_resume["TFS amplitude"] = (
            "$" +
            df_optimal["tfs_param_mean_rn"].apply(lambda x: "{:.{}f}".format(x, precision)) +
            "\pm" +
            df_optimal["error_tfs_param_mean_rn"].apply(lambda x: "{:.{}f}".format(x, precision)) +
            "$"
        )
        df_optimal_mean_resume["Drift"] = (
            "$" +
            df_optimal["drift_coefficient_0_mean_rn"].apply(lambda x: "{:.{}f}".format(x, precision)) +
            "\pm" +
            df_optimal["error_drift_coefficient_0_mean_rn"].apply(lambda x: "{:.{}f}".format(x, precision)) +
            "$"
        )
        df_optimal_mean_resume["MAE_p"] = df_optimal["average_error_mean_rn"].apply(lambda x: "${:.{}f}$".format(x, precision))
        df_optimal_mean_resume["R2"] = df_optimal["rsquared_mean"].apply(lambda x: "${}\%$".format(round(x * 100, 2)))
        
        # Variance - Order of magnitude for average error, parameters and error in parameters ----
        df_optimal["average_error_variance_om"] = df_optimal["average_error_variance"].apply(lambda x: round_to_n(x, precision)[0])
        df_optimal["cumulant_1_variance_om"] = df_optimal["cumulant_1_variance"].apply(lambda x: round_to_n(x, precision)[0])
        df_optimal["cumulant_2_variance_om"] = df_optimal["cumulant_2_variance"].apply(lambda x: round_to_n(x, precision)[0])
        df_optimal["tfs_param_variance_om"] = df_optimal["tfs_param_variance"].apply(lambda x: round_to_n(x, precision)[0])
        df_optimal["drift_coefficient_0_variance_om"] = df_optimal["drift_coefficient_1_variance"].apply(lambda x: round_to_n(x, precision)[0])
        df_optimal["error_cumulant_1_variance_om"] = df_optimal["error_cumulant_1_variance"].apply(lambda x: round_to_n(x, precision)[0])
        df_optimal["error_cumulant_2_variance_om"] = df_optimal["error_cumulant_2_variance"].apply(lambda x: round_to_n(x, precision)[0])
        df_optimal["error_tfs_param_variance_om"] = df_optimal["error_tfs_param_variance"].apply(lambda x: round_to_n(x, precision)[0])
        df_optimal["error_drift_coefficient_0_variance_om"] = df_optimal["error_drift_coefficient_1_variance"].apply(lambda x: round_to_n(x, precision)[0])
        
        # Variance - Minimum order of magnitude for clear output in rounded numbers ----
        df_optimal["average_error_variance_om_min"] = df_optimal["average_error_variance_om"].min() + rank
        df_optimal["cumulant_1_variance_om_min"] = df_optimal["cumulant_1_variance_om"].min() + rank
        df_optimal["cumulant_2_variance_om_min"] = df_optimal["cumulant_2_variance_om"].min() + rank
        df_optimal["tfs_param_variance_om_min"] = df_optimal["tfs_param_variance_om"].min() + rank
        df_optimal["drift_coefficient_0_variance_om_min"] = df_optimal["drift_coefficient_0_variance_om"].min() + rank
        
        # Variance - Difference in order of magnitude of parameters and error in parameters from minimum order ----
        df_optimal["average_error_variance_om"] = df_optimal["average_error_variance_om"] - df_optimal["average_error_variance_om_min"]
        df_optimal["cumulant_1_variance_om"] = df_optimal["cumulant_1_variance_om"] - df_optimal["cumulant_1_variance_om_min"]
        df_optimal["cumulant_2_variance_om"] = df_optimal["cumulant_2_variance_om"] - df_optimal["cumulant_2_variance_om_min"]
        df_optimal["tfs_param_variance_om"] = df_optimal["tfs_param_variance_om"] - df_optimal["tfs_param_variance_om_min"]
        df_optimal["drift_coefficient_0_variance_om"] = df_optimal["drift_coefficient_0_variance_om"] - df_optimal["drift_coefficient_0_variance_om_min"]
        df_optimal["error_cumulant_1_variance_om"] = df_optimal["error_cumulant_1_variance_om"] - df_optimal["cumulant_1_variance_om_min"]
        df_optimal["error_cumulant_2_variance_om"] = df_optimal["error_cumulant_2_variance_om"] - df_optimal["cumulant_2_variance_om_min"]
        df_optimal["error_tfs_param_variance_om"] = df_optimal["error_tfs_param_variance_om"] - df_optimal["tfs_param_variance_om_min"]
        df_optimal["error_drift_coefficient_0_variance_om"] = df_optimal["error_drift_coefficient_0_variance_om"] - df_optimal["drift_coefficient_0_variance_om_min"]
        
        # Variance - Final rounded numbers ----
        df_optimal["average_error_variance_rn"] = df_optimal["average_error_variance"] / (10.0 ** df_optimal["average_error_variance_om_min"])
        df_optimal["cumulant_1_variance_rn"] = df_optimal["cumulant_1_variance"] / (10.0 ** df_optimal["cumulant_1_variance_om_min"])
        df_optimal["cumulant_2_variance_rn"] = df_optimal["cumulant_2_variance"] / (10.0 ** df_optimal["cumulant_2_variance_om_min"])
        df_optimal["tfs_param_variance_rn"] = df_optimal["tfs_param_variance"] / (10.0 ** df_optimal["tfs_param_variance_om_min"])
        df_optimal["drift_coefficient_0_variance_rn"] = df_optimal["drift_coefficient_1_variance"] / (10.0 ** df_optimal["drift_coefficient_0_variance_om_min"])
        df_optimal["error_cumulant_1_variance_rn"] = df_optimal["error_cumulant_1_variance"] / (10.0 ** df_optimal["cumulant_1_variance_om_min"])
        df_optimal["error_cumulant_2_variance_rn"] = df_optimal["error_cumulant_2_variance"] / (10.0 ** df_optimal["cumulant_2_variance_om_min"])
        df_optimal["error_tfs_param_variance_rn"] = df_optimal["error_tfs_param_variance"] / (10.0 ** df_optimal["tfs_param_variance_om_min"])
        df_optimal["error_drift_coefficient_0_variance_rn"] = (
            df_optimal["error_drift_coefficient_1_variance"] / (10.0 ** df_optimal["drift_coefficient_0_variance_om_min"])
        )
        
        # Variance - Final resume optimal data ----
        list_variance = [
            "symbol",
            "time_series",
            "cumulant_1_variance_om_min",
            "cumulant_2_variance_om_min",
            "tfs_param_variance_om_min",
            "drift_coefficient_0_variance_om_min",
            "average_error_variance_om_min"
        ]
        df_optimal_variance_resume = df_optimal[list_variance].sort_values(["symbol"])
        
        df_optimal_variance_resume["Cumulant 1"] = (
            "$" + 
            df_optimal["cumulant_1_variance_rn"].apply(lambda x: "{:.{}f}".format(x, precision)) +
            "\pm" +
            df_optimal["error_cumulant_1_variance_rn"].apply(lambda x: "{:.{}f}".format(x, precision)) +
            "$"
        )
        df_optimal_variance_resume["Cumulant 2"] = (
            "$" + 
            df_optimal["cumulant_2_variance_rn"].apply(lambda x: "{:.{}f}".format(x, precision)) +
            "\pm" +
            df_optimal["error_cumulant_2_variance_rn"].apply(lambda x: "{:.{}f}".format(x, precision)) +
            "$"
        )
        df_optimal_variance_resume["TFS amplitude"] = (
            "$" + 
            df_optimal["tfs_param_variance_rn"].apply(lambda x: "{:.{}f}".format(x, precision)) +
            "\pm" +
            df_optimal["error_tfs_param_variance_rn"].apply(lambda x: "{:.{}f}".format(x, precision)) +
            "$"
        )
        df_optimal_variance_resume["Drift"] = (
            "$" + 
            df_optimal["drift_coefficient_0_variance_rn"].apply(lambda x: "{:.{}f}".format(x, precision)) +
            "\pm" +
            df_optimal["error_drift_coefficient_0_variance_rn"].apply(lambda x: "{:.{}f}".format(x, precision)) +
            "$"
        )
        df_optimal_variance_resume["MAE_p"] = df_optimal["average_error_variance_rn"].apply(lambda x: "${:.{}f}$".format(x, precision))
        df_optimal_variance_resume["R2"] = df_optimal["rsquared_variance"].apply(lambda x: "${}\%$".format(round(x * 100, 2)))
                
    else:
        df_optimal_mean_resume = df_optimal[["symbol", "time_series"]].sort_values(["symbol"])
        df_optimal_variance_resume = df_optimal[["symbol", "time_series"]].sort_values(["symbol"])
    
    df_optimal_mean_resume.sort_values(["symbol"], inplace = True)
    df_optimal_variance_resume.sort_values(["symbol"], inplace = True)
    
    df_optimal_mean_resume.to_csv(
        "{}/mean_summary_{}.csv".format(output_path, re.sub("-", "", input_generation_date)),
        index = False
    )
    
    df_optimal_variance_resume.to_csv(
        "{}/variance_summary_{}.csv".format(output_path, re.sub("-", "", input_generation_date)),
        index = False
    )
    
    return df_optimal_mean_resume, df_optimal_variance_resume, df_optimal