import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.offline import plot  # Allow to plot in browser plotly figures
from scipy import stats

import importlib  # importlib.reload(module)
import reduce_memory_m


def load_data(folder_path, debug=False, seed=2021):
    train = pd.read_csv(os.path.join(folder_path, "train.csv"))
    supple_train = pd.read_csv(os.path.join(folder_path, "supplemental_train.csv"))
    asset_details = pd.read_csv(os.path.join(folder_path, "asset_details.csv"))

    if debug:
        np.random.seed(seed)
        # test = test[test["breath_id"].isin(np.random.choice(["breath_id"].unique(), 1000))].reset_index(drop=True)

    print(f"Train shape: {train.shape}, supplemental_train shape: {supple_train.shape},"
          f" asset_details shape: {asset_details.shape}")
    return train, supple_train, asset_details


def compare_plt_col_dist(df1, df2, col_names, save_path=None, figsize_in=(16, 16)):
    fig, axes = plt.subplots(2, 5, figsize=figsize_in)
    axes = axes.flatten()

    col_pos = 0
    for idx, ax in enumerate(axes):
        values = df1[col_names[col_pos]].values
        sns.kdeplot(data=df1, x=col_names[col_pos], color='red', fill=True, ax=ax)
        sns.kdeplot(data=df2, x=col_names[col_pos], color='green', fill=True, ax=ax)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(f'skew:{round(stats.skew(values), 2)}, kurt:{round(stats.kurtosis(values), 2)}')
        ax.set_ylabel('')
        ax.spines['left'].set_visible(False)
        ax.set_title(col_names[col_pos], loc='right', weight='bold', fontsize=10)
        col_pos += 1

    fig.supxlabel('Average by class (by feature)', ha='center', fontweight='bold')
    fig.tight_layout()

    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_close_tack_day(df_in, day, save_path=None, figsize_in=(16, 16)):
    mask = df_in.Time.dt.day == day
    data = df_in[mask]
    fig, ax = plt.subplots(1, 1, figsize=figsize_in)
    ax2 = plt.twinx()
    for id in data.Asset_ID.unique():
        data_plt = data.query("Asset_ID == @id")
        if id == 1:
            sns.lineplot(y=data_plt.Close, x=data_plt.Time, ax=ax2)
        else:
            sns.lineplot(y=data_plt.Close, x=data_plt.Time, ax=ax, label=id)

    ax2.legend(["BITCOIN"], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    ax.set_title(f"Day {day} close value for every month.", loc='left', weight='bold', fontsize=10)

    if save_path:
        plt.savefig(save_path)
    plt.show()


def missing_statistics(df):
    statitics = pd.DataFrame(df.isnull().sum()).reset_index()
    statitics.columns = ['COLUMN NAME', "MISSING VALUES"]
    statitics['TOTAL ROWS'] = df.shape[0]
    statitics['% MISSING'] = round((statitics['MISSING VALUES'] / statitics['TOTAL ROWS']) * 100, 2)
    return statitics


if __name__ == "__main__":
    PATH = "../input/g-research-crypto-forecasting/"
    EXP = "../"
    EXP_MODEL = os.path.join(EXP, "model")
    EXP_FIG = os.path.join(EXP, "fig")
    EXP_PREDS = os.path.join(EXP, "preds")

    # make dirs
    for d in [EXP_MODEL, EXP_FIG, EXP_PREDS]:
        os.makedirs(d, exist_ok=True)
    del EXP_MODEL, EXP_FIG, EXP_PREDS, EXP

    """Load data"""
    train, supple_train, asset_details = load_data(PATH)
    train = reduce_memory_m.reduce_mem_usage(train, force_float=["Count"])
    supple_train = reduce_memory_m.reduce_mem_usage(supple_train, force_float=["Count"])

    """Compare distributions"""
    # compare_plt_col_dist(train, supple_train, supple_train.columns, figsize_in=(20, 8), save_path="../fig/compare_polt")

    """Check if in the data set there is missing data"""
    print(missing_statistics(train))
    print(missing_statistics(supple_train))

    """Add readable time in data frames"""
    train['Time'] = pd.to_datetime(train['timestamp'], unit='s')
    supple_train['Time'] = pd.to_datetime(supple_train['timestamp'], unit='s')
    print(f"Train data goes from {train.Time.min()} to {train.Time.max()}")
    print(f"Train supplement data goes from {supple_train.Time.min()} to {supple_train.Time.max()}")

    """Bitcoin plot dynamic graph"""
    # btc = train.query("Asset_ID == 1").set_index("Time")  # Asset_ID = 1 for Bitcoin
    # btc_mini = btc.iloc[-1000:]  # Select recent data rows
    # fig = go.Figure(data=[go.Candlestick(x=btc_mini.index, open=btc_mini['Open'], high=btc_mini['High'],
    #                                      low=btc_mini['Low'], close=btc_mini['Close'])])

    """Plot graph tracking day"""
    # plot_close_tack_day(train, 1, figsize_in=(20, 8), save_path="../fig/plot_close_crypto")