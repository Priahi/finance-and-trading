from collections import Counter
import numpy as np
import pandas as pd
import pickle
# svm for sport machine, model for testing on data unlike test data, compare with neighbors
from sklearn import svm, model_selection, neighbors
# voting for more than 1 classifier voting, forest for addtional classification
from sklearn.ensemble import VotingClassifier, RandomForestClassifier


def process_data_for_labels(ticker):
    num_days = 7
    df = pd.read_csv('sp500_joined_closes.csv', index_col=0)
    tickers = df.columns.values.tolist()
    df.fillna(0, inplace=True)

    for i in range(1, num_days + 1):
        df['{}_{}d'.format(ticker, i).replace("\n", "")] = \
            (df[ticker].shift(-i) - df[ticker]) / df[ticker]

    df.fillna(0, inplace=True)
    return tickers, df


# process_data_for_labels('XOM')

def buy_sell_hold(*args):
    cols = [c for c in args]
    # requirement = .045  # % change in stock
    for col in cols:
        if col > .01:
            return 1  # buy
        elif col < -.04:
            return -1  # sell
    return 0  # hold


def extract_featuresets(ticker):
    tickers, df = process_data_for_labels(ticker)

    df['{}_target'.format(ticker)] = list(map(buy_sell_hold,
                                              df['{}_1d'.format(ticker)],
                                              df['{}_2d'.format(ticker)],
                                              df['{}_3d'.format(ticker)],
                                              df['{}_4d'.format(ticker)],
                                              df['{}_5d'.format(ticker)],
                                              df['{}_6d'.format(ticker)],
                                              df['{}_7d'.format(ticker)],
                                              ))
    vals = df['{}_target'.format(ticker)].values.tolist()
    str_vals = [str(i) for i in vals]
    print('Data spread:', Counter(str_vals))  # counter deals with strings

    df.fillna(0, inplace=True)  # replace NaN with 0, keep inplace
    df = df.replace([np.inf, -np.inf], np.nan)  # replace inf vals with NaN
    df.dropna(inplace=True)  # drop NaNs

    # ensure that the values we learn from are not just the next 7 days value
    # else we get false positives, pct change normalizes prices per day
    df_vals = df[[ticker for ticker in tickers]].pct_change()
    df_vals = df_vals.replace([np.inf, -np.inf], 0)  # replace inf vals with 0
    df_vals.fillna(0, inplace=True)

    # X is the feature sets (whats changing), y is the labels
    X = df_vals.values
    y = df['{}_target'.format(ticker)].values

    return X, y, df


def do_ml(ticker):
    X, y, df = extract_featuresets(ticker)

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X,
                                                                        y,
                                                                        test_size=0.25)
    #lsvc = linear support vecotr classifier, knn = k near neighbors
    #rfor = random forest
    clf = VotingClassifier([('lsvc', svm.LinearSVC()),
                            ('knn', neighbors.KNeighborsClassifier()),
                            ('rfor', RandomForestClassifier())])


    clf.fit(X_train, y_train) # train model
    confidence = clf.score(X_test, y_test)
    print('Accuracy:', confidence)

    predictions = clf.predict(X_test) # can be called by itself with pickle to have all pred
    print('Predicted spread:', Counter(predictions))

    return confidence

do_ml('BAC')


# extract_featuresets('XOM')
