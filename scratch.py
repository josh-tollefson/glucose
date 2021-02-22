import os
import sys

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import warnings
warnings.filterwarnings('ignore', 'statsmodels.tsa.arima_model.ARMA',
                        FutureWarning)
warnings.filterwarnings('ignore', 'statsmodels.tsa.arima_model.ARIMA',
                        FutureWarning)

def read_csv(infile, cols=[1, 7], skiprows=10, col_names=['Timestamp', 'Glucose Value (mg/dL)']):

	df = pd.read_csv(infile, usecols=cols, skiprows=skiprows)
	df.columns=col_names
	df['Timestamp'] = pd.to_datetime(df['Timestamp'])
	
	df['Glucose Value (mg/dL)'] = df['Glucose Value (mg/dL)'].replace({'Low': '40'})
	df['Glucose Value (mg/dL)'] = df['Glucose Value (mg/dL)'].astype(int)

	return df

def normalize(df):
	'''
	Normalize df Glucose Values col between 0 and 1
	INPUTS: df, dataframe
	OUTPUT: df, dataframe
	'''

	min_glucose = df['Glucose Value (mg/dL)'].min()
	max_glucose = df['Glucose Value (mg/dL)'].max()

	df['Glucose Value (mg/dL)'] = (df['Glucose Value (mg/dL)'] - min_glucose) / (max_glucose - min_glucose)

	return df

def get_processed_series(cur_timestamp, low_label, df, min_time=30, max_time=60, step=5, unit='m'):
	'''
	Given a timestamp cur_timestamp, return the rows of dataframe df
	with Timestamp between [cur_timestamp - max_time, cur_timestamp - min_time].
	If data are missing, interpolate by nearest value or return none if insufficient data
	INPUTS: cur_timestamp: datetime
					label: int, Low label
					df: dataframe [Timestamp, Glucose, Low]
					min_time, max_time: int
					unit: str, unit of time for timedelta (e.g., 's'=second, 'm'=minute) 
	OUTPUT: array containing glucose values within time interval and low label
	'''

	rows = df[ (df['Timestamp'] <= cur_timestamp - pd.to_timedelta(min_time, unit=unit)) & \
						 (df['Timestamp'] >= cur_timestamp - pd.to_timedelta(max_time, unit=unit)) ].to_numpy()

	# Reverse time order in order to do interpolation of missing values
	rows[::-1,0] = [ (cur_timestamp - t).total_seconds() // 60 for t, g, l in rows]

	if rows.shape[0] <= 1:
		return []

	else:
		from scipy import interpolate
		f = interpolate.interp1d(rows[:,0], rows[:,1], kind='nearest', fill_value="extrapolate") ### Linear interp gives divide by zero error
		times = np.arange(min_time, max_time + step, step=step)
		glucose_new = f(times)
		row_new = np.append(glucose_new, low_label)
		return row_new

def get_processed_stats(cur_timestamp, low_label, df, min_time=30, max_time=60, step=5, unit='m'):
	'''
	Given a timestamp cur_timestamp, return the stats of dataframe df
	with Timestamp between [cur_timestamp - max_time, cur_timestamp - min_time].
	If data are missing, interpolate by nearest value or return none if insufficient data
	INPUTS: cur_timestamp: datetime
					label: int, Low label
					df: dataframe [Timestamp, Glucose, Low]
					min_time, max_time: int
					unit: str, unit of time for timedelta (e.g., 's'=second, 'm'=minute) 
	OUTPUT: array containing stats of glucose values within time interval and low label
	'''

	rows = df[ (df['Timestamp'] <= cur_timestamp - pd.to_timedelta(min_time, unit=unit)) & \
						 (df['Timestamp'] >= cur_timestamp - pd.to_timedelta(max_time, unit=unit)) ]

	if rows.shape[0] <= 1:
		return []

	else:
		min_glucose = rows['Glucose Value (mg/dL)'].min()
		max_glucose = rows['Glucose Value (mg/dL)'].max()
		diff_glucose = rows['Glucose Value (mg/dL)'].iloc[-1] - rows['Glucose Value (mg/dL)'].iloc[0]
		diff_time = (rows['Timestamp'].iloc[-1] - rows['Timestamp'].iloc[0]).total_seconds() // 60

		row_new = [min_glucose, max_glucose, diff_glucose, diff_time, low_label]

		return row_new

def create_processed_df(df, method='series', min_time=30, max_time=60, step=5, unit='m', savefile=None):

	if method == 'series':

		times = np.arange(min_time, max_time + step, step=step)
		names = [str(t) for t in times] + ['Low']
		f = globals()['get_processed_series']

	elif method == 'stats':

		names = ['Min', 'Max', 'Diff Glucose', 'Diff Time', 'Low']
		f = globals()['get_processed_stats']

	else:
		raise ValueError("Only 'series' and 'stats' methods allowed")

	features = pd.DataFrame(columns=names)

	rows = df.apply(lambda row: f(row['Timestamp'], row['Low'], df, min_time=30, max_time=60, step=5, unit='m'), axis=1)

	for i, r in enumerate(rows.to_numpy()):
		if len(r) > 0:
			features.loc[i] = r

	if savefile != None:
		features.to_csv(savefile, index=False)

	return features

def label_lows(df, low_threshold=70):
	'''
	A df col based on low glucose values set by low_threshold
	with 1 (yes, low) or 0 (no, not low)
	INPUTS: df, dataframe
					low_threshold, int
	OUTPUT: df with binary 'Low' column added
	'''

	df['Low'] = np.where(df['Glucose Value (mg/dL)'] <= low_threshold, 1, 0)

	return df

#######################

def label_gaps(df):

	i=109
	print(df.iloc[i])
	print(get_rows(i, df))

def split_timestamp(df):

	df['Date'] = df['Timestamp'].dt.date
	df['Hour'] = df['Timestamp'].dt.time

	return df

def get_lows(df, low_threshold=70):

	df['Time Diff'] = pd.to_timedelta(df['Timestamp'].astype(str)).diff(-1).dt.total_seconds().div(60)

	print(df.head())

def create_labeled_data(df, num=10):

	labeled = pd.DataFrame(columns=range(0,num+1))

	for i in range(df.shape[0] // num):

		if df['Glucose Value (mg/dL)'].iloc[i * num] <= 70:
				label = 1

		else:
				label = 0

		if i * num < 6 + num:
			pass

		else:
			row = np.append(df['Glucose Value (mg/dL)'].iloc[i * num - (6+num) : i * num - 6].values, label)
			entry = pd.DataFrame(data=[row])
			labeled = labeled.append(entry)

	return labeled

def create_labeled_data_two(df, num=10):

	names =['Min', 'Max', 'Diff', 'Label']
	labeled = pd.DataFrame(columns=names)
	for i in range(df.shape[0]):

		if df['Glucose Value (mg/dL)'].iloc[i] <= 70:
				label = 1

		else:
				label = 0

		if i < 6 + num:
			pass

		else:
			min_entry = df['Glucose Value (mg/dL)'].iloc[i - (6+num) : i - 6].min()
			max_entry = df['Glucose Value (mg/dL)'].iloc[i - (6+num) : i - 6].max()
			diff_entry = df['Glucose Value (mg/dL)'].iloc[i - 6] - df['Glucose Value (mg/dL)'].iloc[i - (6+num)]
			row = [min_entry, max_entry, diff_entry, label]
			entry = pd.DataFrame(data=[row], columns=names)
			labeled = labeled.append(entry, ignore_index=True)

	return labeled


def get_min_aic(ts, nr=10, nd=2, nq=10, ic='aic', plotting=True):

		result = []
		params = []

		for p in range(1, nr):
			for d in range(1, nd):
				for q in range(nq):

					try:
						print(p, d, q)
						model = ARIMA(ts, order=(p, d, q))  
						result.append( getattr(model.fit(disp=-1), ic) )
						params.append([p,d,q])
					except ValueError: ### coeffs not invertible
						result.append(np.inf)
						params.append([p,d,q])						

		ind = np.argmin(result)

		print(result[ind])
		print(params[ind])

def hyperparam_tuning(train_x, train_y, test_x, test_y, param_min, param_max):

	### max_depth = 6
	### min_samples_split = unconstrained
	### min_samples_leaf = 9
	### n_estimators = >9

	param_vals = range(param_min, param_max+1)
	result = []
	for i in param_vals:
		model = RandomForestClassifier(max_features=i).fit(train_x, train_y)
		report = classification_report(test_y, model.predict(test_x), output_dict=True)
		result.append(report['1']['recall'])

	plt.plot(param_vals, result)
	plt.show()

	return


infile = 'CLARITY_Export_2021-01-28_222148.csv'

df = read_csv(infile)
df = label_lows(df)
df = normalize(df)

savefile='./processed/CLARITY_Export_2021-01-28_222148_processed_series_normalized.csv'
create_processed_df(df, method='series', savefile=savefile)

savefile='./processed/CLARITY_Export_2021-01-28_222148_processed_stats_normalized.csv'
create_processed_df(df, method='stats', savefile=savefile)

# calculate a 60 day rolling mean and plot

# num = 5
# labeled = create_labeled_data_two(df, num=num)
# labeled = normalize(labeled)



# print(labeled.info())

# from sklearn.model_selection import train_test_split

# train, test = train_test_split(labeled, test_size=0.2)
# train_x = train.iloc[:,0:3].astype(int)
# test_x = test.iloc[:,0:3].astype(int)

# train_y = train.iloc[:,3].astype(int)
# test_y = test.iloc[:,3].astype(int)

# from sklearn.linear_model import LogisticRegression
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.neighbors import KNeighborsClassifier

# from sklearn.metrics import classification_report, confusion_matrix
# from imblearn.over_sampling import SMOTE 

# sm = SMOTE(random_state=42)

# x_res, y_res = sm.fit_resample(train_x, train_y)

# #hyperparam_tuning(x_res, y_res, test_x, test_y, 1, 100)

# model = LogisticRegression().fit(x_res, y_res)

# print(confusion_matrix(test_y, model.predict(test_x)))

# print(classification_report(test_y, model.predict(test_x)))

# # global_std = df['Glucose Value (mg/dL)'].rolling(window = 30 // 5).std().median()

# # fig, ax = plt.subplots()
# # ax.plot_date(df['Timestamp'][-250:-75].values, df['Glucose Value (mg/dL)'][-250:-75].values, fmt='o-')
# # ax.axhline(70, c='k', ls='--', label='Low Glucose Threshold')
# # ax.set_ylabel('Glucose Value (mg/dL)', fontsize=14)
# # ax.set_xlabel('Date & Hour', fontsize=14)
# # ax.legend(fontsize=12)

# # plt.gcf().autofmt_xdate()
# # import numpy as np, pandas as pd
# # from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
# # import matplotlib.pyplot as plt
# # plt.rcParams.update({'figure.figsize':(9,7), 'figure.dpi':120})

# # plt.show()


# # ### ACF - DIFF TERM - 1

# # fig, axes = plt.subplots(3, 2, sharex=True)
# # axes[0, 0].plot(df['Glucose Value (mg/dL)']); axes[0, 0].set_title('Original Series')
# # plot_acf(df['Glucose Value (mg/dL)'], ax=axes[0, 1])

# # # 1st Differencing
# # axes[1, 0].plot(df['Glucose Value (mg/dL)'].diff()); axes[1, 0].set_title('1st Order Differencing')
# # plot_acf(df['Glucose Value (mg/dL)'].diff().dropna(), ax=axes[1, 1])

# # # 2nd Differencing
# # axes[2, 0].plot(df['Glucose Value (mg/dL)'].diff().diff()); axes[2, 0].set_title('2nd Order Differencing')
# # plot_acf(df['Glucose Value (mg/dL)'].diff().diff().dropna(), ax=axes[2, 1])
# # plt.show()

# # ### PACF - AR term - 1

# # fig, axes = plt.subplots(1, 2, sharex=True)
# # axes[0].plot(df['Glucose Value (mg/dL)'].diff()); axes[0].set_title('1st Differencing')
# # axes[1].set(ylim=(0,5))
# # plot_pacf(df['Glucose Value (mg/dL)'].diff().dropna(), ax=axes[1])

# # plt.show()


# # fig, axes = plt.subplots(1, 2, sharex=True)
# # axes[0].plot(df['Glucose Value (mg/dL)'].diff()); axes[0].set_title('1st Differencing')
# # axes[1].set(ylim=(0,1.2))
# # plot_acf(df['Glucose Value (mg/dL)'].diff().dropna(), ax=axes[1])

# # plt.show()

# from statsmodels.tsa.arima_model import ARIMA

# # model = ARIMA(df['Glucose Value (mg/dL)'], order=(1,1,1))
# # model_fit = model.fit(disp=0)
# # print(model_fit.summary())

# # # # Plot residual errors
# # # residuals = pd.DataFrame(model_fit.resid)
# # # fig, ax = plt.subplots(1,2)
# # # residuals.plot(title="Residuals", ax=ax[0])
# # # residuals.plot(kind='kde', title='Density', ax=ax[1])
# # # plt.show()

# # # Actual vs Fitted
# # model_fit.plot_predict(dynamic=False)
# # plt.show()

# # ###############

# # npred = 13
# # train = df['Glucose Value (mg/dL)'][17500:17690]
# # test = df['Glucose Value (mg/dL)'][17690:17690+npred]

# # # get_min_aic(train)

# # model = ARIMA(train, order=(1, 1, 1))  
# # fitted = model.fit(disp=0)  

# # fitted.plot_predict(dynamic=False)
# # plt.show()


# # print(fitted.summary())
# # # Forecast
# # print(fitted.params)

# # fc, se, conf = fitted.forecast(npred, alpha=0.05)  # 95% conf

# # # Make as pandas series
# # fc_series = pd.Series(fc, index=test.index)
# # lower_series = pd.Series(conf[:, 0], index=test.index)
# # upper_series = pd.Series(conf[:, 1], index=test.index)

# # # Plot
# # plt.figure(figsize=(12,5), dpi=100)
# # plt.plot(train, label='training')
# # plt.plot(test, label='actual')
# # plt.plot(fc_series, label='forecast')
# # plt.fill_between(lower_series.index, lower_series, upper_series, 
# #                  color='k', alpha=.15)
# # plt.title('Forecast vs Actuals')
# # plt.legend(loc='upper left', fontsize=8)
# # plt.show()

def main(
    infile="CLARITY_Export_2021-01-28_222148.csv",
    split_timestamp=False,
    seed=42,
    test_size=0.2,
):

    df = read_csv(infile)
    df = handle_lows(df)

    if split_timestamp:
        df = split_timestamp(df)

    # calculate a 60 day rolling mean and plot
    # TODO(hallacy): does this mean the mean over the last 60 days?  Did I miss where this was happening?
    # TODO(hallacy): I think this function takes longer than I would expect.  We should maybe do some light profiling of the function to see if we can get speedup
    labeled = create_min_max_labeled_data(df, width=5)

    # print(labeled.info())

    train, test = train_test_split(labeled, test_size=test_size)

    # TODO(hallacy): remove magic strings
    train_x = train[["Min", "Max", "Diff"]].astype(int)
    test_x = test[["Min", "Max", "Diff"]].astype(int)

    train_y = train[["Label"]].astype(int)
    test_y = test[["Label"]].astype(int)

    sm = SMOTE(random_state=seed)

    x_res, y_res = sm.fit_resample(train_x, train_y)

    # hyperparam_tuning(x_res, y_res, test_x, test_y, 1, 100)

    # TODO(hallacy): These results are pretty variable.  We should probably find the source of randomness
    model = LogisticRegression().fit(x_res, y_res)

    print(confusion_matrix(test_y, model.predict(test_x)))

    print(classification_report(test_y, model.predict(test_x)))


if __name__ == "__main__":
    fire.Fire(main)
