import os
import sys
import warnings
import fire

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from forecast import *
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_model import ARIMA

warnings.filterwarnings("ignore", "statsmodels.tsa.arima_model.ARMA", FutureWarning)
warnings.filterwarnings("ignore", "statsmodels.tsa.arima_model.ARIMA", FutureWarning)

DATE = "DATE"
HOUR = "HOUR"
TIMESTAMP = "TIMESTAMP"
TIME_DIFF = "TIME_DIFF"
GLUCOSE_FIELD_ORIG = "Glucose Value (mg/dL)"
GLUCOSE_FIELD_CORRECTED = "Glucose Value Corrected (mg/dL)"


def read_csv(
    infile,
    cols=[1, 7],
    skiprows=10,
    col_names=[TIMESTAMP, GLUCOSE_FIELD_ORIG],
):
    # TODO(josh): to add docstring about the kind of file that expected here

    df = pd.read_csv(infile, usecols=cols, skiprows=skiprows)
    df.columns = col_names
    df[TIMESTAMP] = pd.to_datetime(df[TIMESTAMP])
    df[GLUCOSE_FIELD_CORRECTED] = df[GLUCOSE_FIELD_ORIG]

    return df


def handle_lows(df):
    df[GLUCOSE_FIELD_CORRECTED] = df[GLUCOSE_FIELD_ORIG].replace({"Low": "40"})
    df[GLUCOSE_FIELD_CORRECTED] = df[GLUCOSE_FIELD_CORRECTED].astype(int)
    return df


def split_timestamp(df):

    df[DATE] = df[TIMESTAMP].dt.date
    df[HOUR] = df[TIMESTAMP].dt.time

    return df


def get_lows(df, low_threshold=70):

    df[TIME_DIFF] = (
        pd.to_timedelta(df[TIMESTAMP].astype(str)).diff(-1).dt.total_seconds().div(60)
    )

    print(df.head())


def create_labeled_data(df, num=10):

    labeled = pd.DataFrame(columns=range(0, num + 1))

    for i in range(df.shape[0] // num):

        if df[GLUCOSE_FIELD_CORRECTED].iloc[i * num] <= 70:
            label = 1

        else:
            label = 0

        if i * num < 6 + num:
            pass

        else:
            row = np.append(
                df[GLUCOSE_FIELD_CORRECTED]
                .iloc[i * num - (6 + num) : i * num - 6]
                .values,
                label,
            )
            entry = pd.DataFrame(data=[row])
            labeled = labeled.append(entry)

    return labeled


def create_labeled_data_two(df, num=10):

    names = ["Min", "Max", "Diff", "Label"]
    labeled = pd.DataFrame(columns=names)
    for i in range(df.shape[0]):

        if df[GLUCOSE_FIELD_CORRECTED].iloc[i] <= 70:
            label = 1

        else:
            label = 0

        if i < 6 + num:
            pass

        else:
            min_entry = df[GLUCOSE_FIELD_CORRECTED].iloc[i - (6 + num) : i - 6].min()
            max_entry = df[GLUCOSE_FIELD_CORRECTED].iloc[i - (6 + num) : i - 6].max()
            diff_entry = (
                df[GLUCOSE_FIELD_CORRECTED].iloc[i - 6]
                - df[GLUCOSE_FIELD_CORRECTED].iloc[i - (6 + num)]
            )
            row = [min_entry, max_entry, diff_entry, label]
            entry = pd.DataFrame(data=[row], columns=names)
            labeled = labeled.append(entry, ignore_index=True)

    return labeled


def get_min_aic(ts, nr=10, nd=2, nq=10, ic="aic", plotting=True):

    result = []
    params = []

    for p in range(1, nr):
        for d in range(1, nd):
            for q in range(nq):

                try:
                    print(p, d, q)
                    model = ARIMA(ts, order=(p, d, q))
                    result.append(getattr(model.fit(disp=-1), ic))
                    params.append([p, d, q])
                except ValueError:  ### coeffs not invertible
                    result.append(np.inf)
                    params.append([p, d, q])

    ind = np.argmin(result)

    print(result[ind])
    print(params[ind])


def hyperparam_tuning(train_x, train_y, test_x, test_y, param_min, param_max):

    ### max_depth = 6
    ### min_samples_split = unconstrained
    ### min_samples_leaf = 9
    ### n_estimators = >9

    param_vals = range(param_min, param_max + 1)
    result = []
    for i in param_vals:
        model = RandomForestClassifier(max_features=i).fit(train_x, train_y)
        report = classification_report(test_y, model.predict(test_x), output_dict=True)
        result.append(report["1"]["recall"])

    plt.plot(param_vals, result)
    plt.show()

    return


# global_std = df['Glucose Value (mg/dL)'].rolling(window = 30 // 5).std().median()

# fig, ax = plt.subplots()
# ax.plot_date(df['Timestamp'][-250:-75].values, df['Glucose Value (mg/dL)'][-250:-75].values, fmt='o-')
# ax.axhline(70, c='k', ls='--', label='Low Glucose Threshold')
# ax.set_ylabel('Glucose Value (mg/dL)', fontsize=14)
# ax.set_xlabel('Date & Hour', fontsize=14)
# ax.legend(fontsize=12)

# plt.gcf().autofmt_xdate()
# plt.rcParams.update({'figure.figsize':(9,7), 'figure.dpi':120})

# plt.show()


# ### ACF - DIFF TERM - 1

# fig, axes = plt.subplots(3, 2, sharex=True)
# axes[0, 0].plot(df['Glucose Value (mg/dL)']); axes[0, 0].set_title('Original Series')
# plot_acf(df['Glucose Value (mg/dL)'], ax=axes[0, 1])

# # 1st Differencing
# axes[1, 0].plot(df['Glucose Value (mg/dL)'].diff()); axes[1, 0].set_title('1st Order Differencing')
# plot_acf(df['Glucose Value (mg/dL)'].diff().dropna(), ax=axes[1, 1])

# # 2nd Differencing
# axes[2, 0].plot(df['Glucose Value (mg/dL)'].diff().diff()); axes[2, 0].set_title('2nd Order Differencing')
# plot_acf(df['Glucose Value (mg/dL)'].diff().diff().dropna(), ax=axes[2, 1])
# plt.show()

# ### PACF - AR term - 1

# fig, axes = plt.subplots(1, 2, sharex=True)
# axes[0].plot(df['Glucose Value (mg/dL)'].diff()); axes[0].set_title('1st Differencing')
# axes[1].set(ylim=(0,5))
# plot_pacf(df['Glucose Value (mg/dL)'].diff().dropna(), ax=axes[1])

# plt.show()


# fig, axes = plt.subplots(1, 2, sharex=True)
# axes[0].plot(df['Glucose Value (mg/dL)'].diff()); axes[0].set_title('1st Differencing')
# axes[1].set(ylim=(0,1.2))
# plot_acf(df['Glucose Value (mg/dL)'].diff().dropna(), ax=axes[1])

# plt.show()


# model = ARIMA(df['Glucose Value (mg/dL)'], order=(1,1,1))
# model_fit = model.fit(disp=0)
# print(model_fit.summary())

# # # Plot residual errors
# # residuals = pd.DataFrame(model_fit.resid)
# # fig, ax = plt.subplots(1,2)
# # residuals.plot(title="Residuals", ax=ax[0])
# # residuals.plot(kind='kde', title='Density', ax=ax[1])
# # plt.show()

# # Actual vs Fitted
# model_fit.plot_predict(dynamic=False)
# plt.show()

# ###############

# npred = 13
# train = df['Glucose Value (mg/dL)'][17500:17690]
# test = df['Glucose Value (mg/dL)'][17690:17690+npred]

# # get_min_aic(train)

# model = ARIMA(train, order=(1, 1, 1))
# fitted = model.fit(disp=0)

# fitted.plot_predict(dynamic=False)
# plt.show()


# print(fitted.summary())
# # Forecast
# print(fitted.params)

# fc, se, conf = fitted.forecast(npred, alpha=0.05)  # 95% conf

# # Make as pandas series
# fc_series = pd.Series(fc, index=test.index)
# lower_series = pd.Series(conf[:, 0], index=test.index)
# upper_series = pd.Series(conf[:, 1], index=test.index)

# # Plot
# plt.figure(figsize=(12,5), dpi=100)
# plt.plot(train, label='training')
# plt.plot(test, label='actual')
# plt.plot(fc_series, label='forecast')
# plt.fill_between(lower_series.index, lower_series, upper_series,
#                  color='k', alpha=.15)
# plt.title('Forecast vs Actuals')
# plt.legend(loc='upper left', fontsize=8)
# plt.show()


def main(
    infile="CLARITY_Export_2021-01-28_222148.csv",
    split_timestamp=False,
):

    df = read_csv(infile)
    df = handle_lows(df)

    if split_timestamp:
        df = split_timestamp(df)

    # calculate a 60 day rolling mean and plot

    num = 5
    labeled = create_labeled_data_two(df, num=num)

    # print(labeled.info())

    train, test = train_test_split(labeled, test_size=0.2)
    train_x = train.iloc[:, 0:3].astype(int)
    test_x = test.iloc[:, 0:3].astype(int)

    train_y = train.iloc[:, 3].astype(int)
    test_y = test.iloc[:, 3].astype(int)

    sm = SMOTE(random_state=42)

    x_res, y_res = sm.fit_resample(train_x, train_y)

    # hyperparam_tuning(x_res, y_res, test_x, test_y, 1, 100)

    # TODO(hallacy): These results are pretty variable.  We should probably find the source of randomness
    model = LogisticRegression().fit(x_res, y_res)

    print(confusion_matrix(test_y, model.predict(test_x)))

    print(classification_report(test_y, model.predict(test_x)))


if __name__ == "__main__":
    fire.Fire(main)
