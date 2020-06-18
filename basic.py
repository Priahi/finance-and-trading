import pandas as pd
import pandas_datareader.data as web
from pandas.plotting import register_matplotlib_converters
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import mplfinance as mpf  # use mpf.plt(df)
from mplfinance.original_flavor import candlestick_ohlc
from matplotlib import style
# import beautifulsoup4
# import scikit-learn as scikit
import datetime as dt

register_matplotlib_converters()
style.use('ggplot')

# start = dt.datetime(2000, 1, 1)
# end = dt.datetime(2020, 4, 20)
#
# df = web.DataReader('TSLA', 'yahoo', start, end)
# print(df.head())
# print(df.tail(6))
# df.to_csv('tsla.csv')
df = pd.read_csv('tsla.csv', parse_dates=True, index_col=0)

# 100 day rolling mean average of data
df['100ma'] = df['Adj Close'].rolling(window=100, min_periods=0).mean()  # .sum()
df.dropna(inplace=True)

# change sampling rate of data
df_ohlc = df['Adj Close'].resample('10D').ohlc()  # .sum(), .mean()
df_volume = df['Volume'].resample('10D').sum()

df_ohlc.reset_index(inplace=True)
df_ohlc['Date'] = df_ohlc['Date'].map(mdates.date2num)

#                    (rows,col),(xo,yo)
ax1 = plt.subplot2grid((6, 1), (0, 0), rowspan=5, colspan=1)
ax2 = plt.subplot2grid((6, 1), (5, 0), rowspan=1, colspan=1, sharex=ax1)
ax1.xaxis_date()  # converts mdates to regular dates

# mpf.plot(df, type='candle', mav=(3,6,9), volume=True)
candlestick_ohlc(ax1, df_ohlc.values, width=2, colordown='r', colorup='g')
# fill from 0 to y, first param x, second y, 3rd start val of y
ax2.fill_between(df_volume.index.map(mdates.date2num), df_volume.values, 0)

# ax1.plot(df.index, df['Adj Close'])
# ax1.plot(df.index, df['100ma'])
# ax2.bar(df.index, df['Volume'])

# print(df[['Open', 'High', 'Adj Close']].head())
# df[['Open', 'High', 'Adj Close']].plot()
plt.show()
