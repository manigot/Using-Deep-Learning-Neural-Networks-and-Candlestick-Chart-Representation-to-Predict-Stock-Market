import argparse
import arrow
import pandas as pd
from pandas.tseries.offsets import BDay
from pandas.tseries.offsets import Hour
from datetime import datetime
import numpy as np
import datetime as dt
from datetime import datetime
from pandas_datareader import data, wb
from keras.models import load_model
import traceback
import os
import yfinance as yf
import imageio
import time
import sys
import dataset
import scipy.misc
import subprocess
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
# https://github.com/matplotlib/mpl_finance
from mpl_finance import candlestick2_ochl, volume_overlay
# fixed pandas_datareader can't download from yahoo finance
yf.pdr_override()


def build_dataset(data_directory, img_width):
    X, y, tags = dataset.dataset(data_directory, int(img_width))
    feature = X
    return feature


def fetch_yahoo_data(dat, fname, max_attempt, check_exist):
    if (os.path.exists(fname) == True) and check_exist:
        print("file exist")
    else:
        # remove exis-t file
        if os.path.exists(fname):
            os.remove(fname)
        for attempt in range(max_attempt):
            time.sleep(2)
            try:
                
                dat.to_csv(fname)
            except Exception as e:
                if attempt < max_attempt - 1:
                    print('Attempt {}: {}'.format(attempt + 1, str(e)))
                else:
                    raise
            else:
                break


def ohlc2cs(fname, dimension):
    # python preprocess.py -m ohlc2cs -l 20 -i stockdatas/EWT_testing.csv -t testing
    print("Converting olhc to candlestick")
    inout = fname
    df = pd.read_csv(fname, parse_dates=True, index_col=0)
    df.fillna(0)
    plt.style.use('dark_background')
    df.reset_index(inplace=True)
    df['Date'] = df['Datetime'].map(mdates.date2num)
    my_dpi = 96
    fig = plt.figure(figsize=(dimension / my_dpi,
                              dimension / my_dpi), dpi=my_dpi)
    ax1 = fig.add_subplot(1, 1, 1)
    candlestick2_ochl(ax1, df['Open'], df['Close'], df['High'],
                      df['Low'], width=1,
                      colorup='#77d879', colordown='#db3f3f')
    ax1.grid(False)
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    ax1.xaxis.set_visible(False)
    ax1.yaxis.set_visible(False)
    ax1.axis('off')

    # create the second axis for the volume bar-plot
    # Add a seconds axis for the volume overlay
    """
    ax2 = ax1.twinx()
    # Plot the volume overlay
    bc = volume_overlay(ax2, df['Open'], df['Close'], df['Volume'],
                        colorup='#77d879', colordown='#db3f3f', alpha=0.5, width=1)
    ax2.add_collection(bc)
    ax2.grid(False)
    ax2.set_xticklabels([])
    ax2.set_yticklabels([])
    ax2.xaxis.set_visible(False)
    ax2.yaxis.set_visible(False)
    ax2.axis('off')
    """
    pngfile = "temp_class/{}.png".format(inout)
    fig.savefig(pngfile,  pad_inches=0, transparent=False)
    plt.close(fig)
    # normal length - end
    params = []
    params += ["-alpha", "off"]
    
    #subprocess.check_call(["convert", pngfile] + params + [pngfile])
    print("Converting olhc to candlestik finished.")


def main():

    ticker = sys.argv[1]
    end_date = sys.argv[2]
    dimension = sys.argv[3]
    model_name = sys.argv[4]
    period = sys.argv[5]
    fileparam = end_date.replace("-", "_")
    df = yf.download('BTC-USD',interval="1h", period="8d")
    def create_groups(df):
      groups = []
      for i, row in df.iterrows():
          end_time = i + pd.Timedelta(hours=9)
          group = df.loc[i:end_time]
          groups.append(group)
      return groups

    result = create_groups(df)
    passed = True
    result_df = pd.DataFrame()
    last_close = None
    last_predict = None
# Print each group
    for i, group_df in enumerate(result):
      if group_df.shape[0] == 10:
        close = group_df.tail(1).Close[0]
        date_time = group_df.head(1).index[0]
        if i > 0:
          pct = (close - last_close) / last_close
          result_df = result_df.append({"datetime":date_time,"real":pct,"pred":last_predict}, ignore_index=True)
        print(f"Group {i}:")
        #print(group_df.tail(1).Close)
        #print(group_df.tail(1).index)
        print("------------------------")
        last_close = group_df.tail(1).Close[0]
        try:
            # convert to candlestickchart      
            fetch_yahoo_data(group_df,
                            "{}_{}.csv".format(ticker, fileparam), 10, False)
            ohlc2cs("{}_{}.csv".format(ticker, fileparam), int(dimension))
            pass
        except Exception as e:
            tb = traceback.format_exc()
            os.remove("{}_{}.csv".format(ticker, fileparam))
            print("Error when download historical data, please re-run.")
            passed = False
            print(tb)
            pass
        if passed:
            # prepare dataset
            img = imageio.imread(
                "temp_class/{}_{}.csv.png".format(ticker, fileparam))
            img = img[:,:,:3]
            X_test = np.array([img]).astype(np.float32)
            # load model and predict
            model = load_model(model_name)
            last_predict = model.predict(X_test)
            print(last_predict)
            # cleaning
            os.remove("{}_{}.csv".format(ticker, fileparam))
            os.remove("temp_class/{}_{}.csv.png".format(ticker, fileparam))
    result_df.to_csv("test_results.csv",index=False)

if __name__ == '__main__':
    main()
