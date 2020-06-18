import bs4 as bs  # screen scraper
import datetime as dt
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import pandas_datareader.data as web
from pandas_datareader._utils import RemoteDataError
import pickle  # serializes python objects
import requests
import yfinance as yf
import numpy as np

style.use('ggplot')


def save_sp500_tickers():
    resp = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    soup = bs.BeautifulSoup(resp.text, 'lxml')
    table = soup.find('table', {'class': 'wikitable sortable'})
    tickers = []
    for row in table.findAll('tr')[1:]:  # tr = table rows
        ticker = row.findAll('td')[0].text  # table data, col
        tickers.append(ticker)

    with open("sp500tickers.pickle", "wb") as f:
        pickle.dump(tickers, f)  # dump tickers into f, is byte writable

    return tickers


def get_data_from_yahoo(reload_sp500=False):
    if (reload_sp500):
        tickers = save_sp500_tickers()
    else:
        with open("sp500tickers.pickle", "rb") as f:
            tickers = pickle.load(f)

    # save data to file (big file)
    if not os.path.exists('stocks_dfs'):
        os.makedirs("stocks_dfs")

    start = dt.datetime(2000, 1, 1)
    end = dt.datetime(2020, 1, 20)

    for ticker in tickers:
        str = 'stocks_dfs/{}.csv'.format(ticker).replace("\n", "")
        print(str)
        if not os.path.exists(str):
            # df = web.DataReader(ticker, 'google', start, end)
            df = yf.download(tickers=ticker,
                             period="ytd",  # year to date (not start to end)
                             auto_adjust=False,  # ohlc
                             threads=True)  # multithread
            df.to_csv(str)
        else:
            print("Already Have {}".format(ticker))


# get_data_from_yahoo(reload_sp500=True)

def compile_data():
    with open("sp500tickers.pickle", "rb") as f:
        tickers = pickle.load(f)

    main_df = pd.DataFrame()
    for count, ticker in enumerate(tickers):
        df = pd.read_csv('stocks_dfs/{}.csv'.format(ticker).replace("\n", ""))
        df.set_index('Date', inplace=True)
        df.rename(columns={'Adj Close': ticker.replace("\n", "")}, inplace=True)
        df.drop(['Open', 'High', 'Low', 'Close', 'Volume'], 1, inplace=True)

        if main_df.empty:
            main_df = df
        else:
            main_df = main_df.join(df, how='outer')

        if count % 10 == 0:
            print(count)

    print(main_df.tail())
    main_df.to_csv('sp500_joined_closes.csv')


# compile_data()

def visualize_data():
    df = pd.read_csv('sp500_joined_closes.csv')
    # df['AAPL'].plot()
    # plt.show()
    df_corr = df.corr()  # determines correlation between stonks
    print(df_corr.head())

    data = df_corr.values
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    heatmap = ax.pcolor(data, cmap=plt.cm.RdYlGn)  # heatmap color range
    fig.colorbar(heatmap)
    ax.set_xticks(np.arange(data.shape[0]) + 0.5, minor=False)  # tick marks at x.5
    ax.set_yticks(np.arange(data.shape[1]) + 0.5, minor=False)
    ax.invert_yaxis()  # flip y axis
    ax.xaxis.tick_top()  # move ticks to top of table

    column_labels = df_corr.columns
    row_labels = df_corr.index

    ax.set_xticklabels(column_labels)
    ax.set_yticklabels(row_labels)  # flip x and y labels
    plt.xticks(rotation=90)
    heatmap.set_clim(-1,1) #limit of colors for correlations
    plt.tight_layout()
    plt.show()

# visualize_data()
