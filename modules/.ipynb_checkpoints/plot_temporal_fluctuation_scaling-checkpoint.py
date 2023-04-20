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
import plot_optimal_window as plt_ow
import estimate_optimal_window as eow
import matplotlib.transforms as mtransforms
import estimate_temporal_fluctuation_scaling as etfs

from scipy.optimize import curve_fit
from scipy.stats import percentileofscore

# Global options ----
warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None
pd.set_option('display.max_columns', None)

colors = ["blue", "green", "red", "purple", "orange", "brown", "pink", "olive", "gray", "cyan"]

# Estimate tfs parameters ----
def prepare_data(
    df_fts,
    df_optimal,
    df_tfs,
    interval
):
    """Preparation of data for plotting
    Join original data with optimal window size data:
        df_fts: Dataframe with multiple financial time series
        df_optimal: Dataframe with optimal window size per financial time series
        df_tfs: Dataframe with temporal fluctuation scaling parameters
        interval: Select transformation for estimation of time between minimum and maximum date
    """
    
    # Prepare data from optimal window size ----
    df_plot_data = plt_ow.prepare_data(df_fts = df_fts, df_optimal = df_optimal, interval = interval)
    df_tfs = df_tfs.dropna().rename(columns = {"p_norm" : "p_norm_tfs"})
    
    # Merge final data ----
    df_plot_data = df_plot_data.merge(df_tfs, left_on = ["symbol", "time_series", "step"], right_on = ["symbol", "time_series", "max_step"])
    
    del [df_plot_data["max_step"]]
        
    return df_plot_data

# Plot evolution of tfs parameters ----
def plot_tfs_evolution(
    df_fts,
    df_optimal,
    df_tfs,
    interval,
    threshold,
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
        df_tfs: Dataframe with temporal fluctuation scaling parameters
        interval: Select transformation for estimation of time between minimum and maximum date
        threshold: Threshold for estimation of inverse percentile function (percentage of data with R2 >= threshold)
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
    df_tfs = df_tfs[df_tfs["symbol"].isin(symbols)]
    df_graph = prepare_data(df_fts = df_fts, df_tfs = df_tfs, df_optimal = df_optimal, interval = interval)
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
            ave_tfs_j = df_aux["average_error_tfs"]
            rsquared_j = round(percentileofscore(df_aux["rsquared_tfs"], threshold), 2)
            window_mean_j = df_aux["window_size_mean"].unique()[0]
            window_variance_j = df_aux["window_size_variance"].unique()[0]
            dd = df_aux["drift_degree"].unique()[0]
            
            # Extract empirical data ----
            dates_tfs_j = pd.to_datetime(df_aux["date"], errors = "coerce")
            exponent_j = df_aux["exponent_tfs"]
            coefficient_j = df_aux["coefficient_tfs"]
            
            if (df_aux[df_aux["step"]%window_mean_j == 0].shape[0] > 0) & (df_aux[df_aux["step"]%window_variance_j == 0].shape[0] > 0):
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

                else:
                    # Theoretical mean ----
                    mean_prome = eow.mean_evolution(df_aux["step"], *popt_mean)
                    mean_lower = eow.mean_evolution(df_aux["step"], *lower_mean)
                    mean_upper = eow.mean_evolution(df_aux["step"], *upper_mean)

                    # Theoretical variance ----
                    variance_prome = eow.variance_evolution(df_aux["step"], *popt_variance)
                    variance_lower = eow.variance_evolution(df_aux["step"], *lower_variance)
                    variance_upper = eow.variance_evolution(df_aux["step"], *upper_variance)

                # Extract evolution of TFS exponent ----
                tfs_prome = (np.log(variance_prome) - np.log(coefficient_j)) / np.log(mean_prome)
                tfs_lower = (np.log(variance_lower) - np.log(coefficient_j)) / np.log(mean_upper)
                tfs_upper = (np.log(variance_upper) - np.log(coefficient_j)) / np.log(mean_lower)

                # Estimation of R squared (TFS) ----
                r2 = exponent_j
                r2 = max(0, 1- np.sum(np.power(r2.values - tfs_prome.values, 2)) / np.sum(np.power(r2.values - np.mean(r2), 2)))
                r2 = round(r2 * 100, 2)

                # Plot graphs ----
                if len(loop_index) == 1:
                    # Plot graph (Mean) ----
                    plot_1 = ax1[counter_i].plot(
                        dates_tfs_j,
                        exponent_j,
                        alpha = 1,
                        zorder = 2,
                        color = "black",
                        marker = "o",
                        linestyle = "",
                        label = "empirical data",
                        markersize = markersize
                    )
                    ax1[counter_i].plot(dates_j, tfs_prome, alpha = 1, zorder = 1, color = colors[counter_i], linewidth = 3, label = "fitting")
                    ax1[counter_i].fill_between(
                        dates_j,
                        tfs_lower,
                        tfs_upper,
                        where = ((tfs_upper >= tfs_lower) & (tfs_upper >= tfs_prome) & (tfs_prome >= tfs_lower)),
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
                    ax1[counter_i].set_ylabel("TFS exponent - {}".format(j.capitalize()), fontsize = fontsize_labels)
                    ax1[counter_i].legend(fancybox = fancy_legend, shadow = True, ncol = n_cols, fontsize = fontsize_legend)
                    ax1[counter_i].set_title(
                        r"({}) $MAE_p={}$, $q(R^2={})={}\%$, $R^2={}\%$".format(chr(counter_i + 65), round(ave_tfs_j.min(), 5), threshold, rsquared_j, r2),
                        loc = "left",
                        y = 1.005,
                        fontsize = fontsize_labels
                    )

                    # Plot graph (Variance) ----
                    plot_2 = ax2[counter_i].plot(
                        dates_tfs_j,
                        coefficient_j,
                        alpha = 1,
                        zorder = 2,
                        color = "black",
                        marker = "o",
                        linestyle = "",
                        label = "empirical data",
                        markersize = markersize
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
                    ax2[counter_i].set_ylabel("TFS coefficient - {}".format(j.capitalize()), fontsize = fontsize_labels)
                    ax2[counter_i].legend(fancybox = fancy_legend, shadow = True, ncol = n_cols, fontsize = fontsize_legend)
                    ax2[counter_i].set_title("({})".format(chr(counter_i + 65)), loc = "left", y = 1.005)
                    ax2[counter_i].set_title(
                        r"({}) $MAE_p={}$, $q(R^2={})={}\%$, $R^2={}\%$".format(chr(counter_i + 65), round(ave_tfs_j.min(), 5), threshold, rsquared_j, r2),
                        loc = "left",
                        y = 1.005,
                        fontsize = fontsize_labels
                    )
                else:
                    # Plot graph (Mean) ----
                    plot_1 = ax1[counter, counter_i].plot(
                        dates_tfs_j,
                        exponent_j,
                        alpha = 1,
                        zorder = 2,
                        color = "black",
                        marker = "o",
                        linestyle = "",
                        label = "empirical data",
                        markersize = markersize
                    )
                    ax1[counter, counter_i].plot(dates_j, tfs_prome, alpha = 1, zorder = 2, color = colors[counter_i], linewidth = 3, label = "fitting")
                    ax1[counter, counter_i].fill_between(
                        dates_j,
                        tfs_lower,
                        tfs_upper,
                        where = ((tfs_upper >= tfs_lower) & (tfs_upper >= tfs_prome) & (tfs_prome >= tfs_lower)),
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
                    ax1[counter, counter_i].set_ylabel("TFS exponent - {}".format(j.capitalize()), fontsize = fontsize_labels)
                    ax1[counter, counter_i].legend(fancybox = fancy_legend, shadow = True, ncol = n_cols, fontsize = fontsize_legend)
                    ax1[counter, counter_i].set_title(
                        r"({}) $MAE_p={}$, $q(R^2={})={}\%$, $R^2={}\%$".format(chr(counter_i + 65), round(ave_tfs_j.min(), 5), threshold, rsquared_j, r2),
                        loc = "left",
                        y = 1.005,
                        fontsize = fontsize_labels
                    )

                    # Plot graph (Variance) ----
                    plot_2 = ax2[counter, counter_i].plot(
                        dates_tfs_j,
                        coefficient_j,
                        alpha = 1,
                        zorder = 2,
                        color = "black",
                        marker = "o",
                        linestyle = "",
                        label = "empirical data",
                        markersize = markersize
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
                    ax2[counter, counter_i].set_ylabel("TFS coefficient - {}".format(j.capitalize()), fontsize = fontsize_labels)
                    ax2[counter, counter_i].legend(fancybox = fancy_legend, shadow = True, ncol = n_cols, fontsize = fontsize_legend)
                    ax2[counter, counter_i].set_title(
                        r"({}) $MAE_p={}$, $q(R^2={})={}\%$, $R^2={}\%$".format(chr(counter_i + 65), round(ave_tfs_j.min(), 5), threshold, rsquared_j, r2),
                        loc = "left",
                        y = 1.005,
                        fontsize = fontsize_labels
                    )
            else:
                # Remove vacuum plots ----
                if len(loop_index) == 1:
                    fig1.delaxes(ax1[counter_i])
                    fig2.delaxes(ax2[counter_i])
                else:
                    fig1.delaxes(ax1[counter, counter_i])
                    fig2.delaxes(ax2[counter, counter_i])
                    
            # Function development ----
            counter_i += 1
            print("Generated plot for {} and time series {}".format(i, j))
        
        counter += 1
    
    fig1.tight_layout()
    fig2.tight_layout()
    if save_figures:
        plt.show()
        fig1.savefig(
            "{}/{}_tfs_exponent_evolution_{}.png".format(output_path, information_name, re.sub("-", "", input_generation_date)),
            bbox_inches = "tight",
            facecolor = fig1.get_facecolor(),
            transparent = False,
            pad_inches = 0.03,
            dpi = dpi
        )
        plt.show()
        fig2.savefig(
            "{}/{}_tfs_coefficient_evolution_{}.png".format(output_path, information_name, re.sub("-", "", input_generation_date)),
            bbox_inches = "tight",
            facecolor = fig2.get_facecolor(),
            transparent = False,
            pad_inches = 0.03,
            dpi = dpi
        )
    plt.close()
    plt.close()
    
    return df_graph
