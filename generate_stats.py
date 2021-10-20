import fire
import random
import numpy as np
import pandas as pd
from scipy.stats import linregress

INFILE = './processed/glucose.csv'
OUTFILE = './processed/glucose_stats.csv'

TIME_DELTA = 5
MAX_NUM_POINTS = 6 # 30 minutes
MAX_MINUTES = TIME_DELTA * MAX_NUM_POINTS

MINUTES_TO_PREDICT_LOW = 15 # 15 minutes
MAX_POINTS_TO_LOW = 3 # 15 minutes

LOW_THRESHOLD = 70

TIMESTAMP_COL = 'Timestamp'
GLUCOSE_COL = 'Glucose'
TIME_DIFF_COL = 'Time_Diff'

COL_NAMES = [TIMESTAMP_COL, GLUCOSE_COL, TIME_DIFF_COL]
COL_OUTPUT_NAMES = ['INDEX',
                    'CURRENT_VALUE',
                    'LAST_VALUE',
                    'SLOPE',
                    'R_SQUARED',
                    'SLOPE_SECOND_HALF',
                    'C0',
                    'C1',
                    'C2',
                    'GLUCOSE_MEAN',
                    'GLUCOSE_MEDIAN',
                    'MEAN_TO_MEDIAN',
                    'GLUCOSE_STDDEV',
                    'GLUCOSE_CARDINALITY',
                    'TOTAL_DIFF',
                    'TOTAL_DIFF_SECOND_HALF',
                    'DIFF_LAST',
                    'DIFF_MEAN',
                    'DIFF_SLOPE',
                    'DIFF_R_SQUARED',
                    'DIFF_SLOPE_SECOND_HALF',
                    'GLUCOSE_MAX',
                    'GLUCOSE_MIN',
                    'GLUCOSE_MIN_MINUTE',
                    'GRADIENT_MIN',
                    'GRADIENT_MIN_GLUCOSE',
                    'GRADIENT_MIN_MINUTE',
                    'CURVATURE_MIN',
                    'CURVATURE_MIN_GLUCOSE',
                    'CURVATURE_MIN_MINUTE',
                    'EXTRAPOLATED_VALUE_FIRST_ORDER',
                    'EXTRAPOLATED_VALUE_SECOND_ORDER',
                    'IS_LOW',
                    'LOOK_AHEAD_VALUE']

def return_last_index(timings, start, end, minutes_threshold, index_threshold):
    '''
    helper function for get_valid_glucose_series
    '''

    cumulative_minutes = []

    for i in range(start, end):
        if i == start:
            minute = 0
        else:
            minute += timings[i-1]

        if minute >= minutes_threshold:
            last_index = i
            break

        if i == end-1:
            last_index = i

        cumulative_minutes.append(minute)

    return last_index, cumulative_minutes


def get_valid_glucose_series(timings, glucose_values):

    last_index, cumulative_minutes = return_last_index(timings, 0, len(timings), MAX_MINUTES, MAX_NUM_POINTS)
    low_index, _cumulative_low_minutes = return_last_index(timings, last_index+1, len(timings), MINUTES_TO_PREDICT_LOW, MAX_POINTS_TO_LOW)

    return cumulative_minutes, glucose_values[:last_index], glucose_values[low_index]


def is_valid_series(timings):

    # last timing diff not needed
    return np.all(timings[:-1] == 5)

def get_regression(x, y):

    result = linregress(x, y)

    return round(result.slope, 2), round(result.rvalue ** 2, 2)

def get_polyfit_coeffs(x, y, deg=2):

    coeffs = np.polyfit(x, y, deg)

    return coeffs


def get_extrapolated_value(x, y, deg=1, npts=1):

    coeffs = np.polyfit(x, y, deg)

    value = 0
    for i, c in enumerate(coeffs):

        value += c * (x[-1] + npts * TIME_DELTA) ** (deg - i)

    return value

def get_curvature_min(x, y):

    curvature = np.gradient(np.gradient(y))
    curvature_minima_index = np.argmin(curvature)
    curvature_minima = curvature[curvature_minima_index]
    curvature_minima_glucose_value = y[curvature_minima_index]
    curvature_minima_minute = x[curvature_minima_index]

    return curvature_minima, curvature_minima_glucose_value, curvature_minima_minute

def get_gradient_min(x, y):

    gradients = np.gradient(y)
    gradients_minima_index = np.argmin(gradients)
    gradients_minima = gradients[gradients_minima_index]
    gradients_minima_glucose_value = y[gradients_minima_index]
    gradients_minima_minute = x[gradients_minima_index]

    return gradients_minima, gradients_minima_glucose_value, gradients_minima_minute

def get_glucose_min(x, y):

    minimum_index = np.argmin(y)

    return y[minimum_index], x[minimum_index]

def get_statistics(timings, values, low):

    if is_valid_series(timings):

        minutes = np.concatenate((np.array([0]), np.cumsum(timings[:-1])))
        slope, r_squared = get_regression(minutes, values)
        glucose_min, glucose_min_minute = get_glucose_min(minutes, values)
        gradient_min, gradient_min_glucose, gradient_min_minute = get_gradient_min(minutes, values)
        curvature_min, curvature_min_glucose, curvature_min_minute = get_curvature_min(minutes, values)
        extrapolated_value_first_order = round(get_extrapolated_value(minutes, values, deg=1, npts=MAX_POINTS_TO_LOW), 2)
        extrapolated_value_second_order = round(get_extrapolated_value(minutes, values, deg=2, npts=MAX_POINTS_TO_LOW), 2)
        c0, c1, c2 = get_polyfit_coeffs(minutes, values, deg=2)
        glucose_mean = round(np.mean(values), 2)
        glucose_median = round(np.median(values), 2)
        mean_to_median = round(glucose_mean / glucose_median, 2)
        glucose_stddev = round(np.std(values), 2)
        glucose_max = np.max(values)
        glucose_cardinality = len(np.unique(values))
        total_diff = np.sum(np.diff(values))
        total_diff_second_half =  np.sum(np.diff(values[MAX_NUM_POINTS//2:]))
        diff_mean = np.mean(np.diff(values))
        diff_slope, diff_r_squared = get_regression(minutes[1:], np.diff(values))
        slope_second_half = get_regression(minutes[MAX_NUM_POINTS//2:], values[MAX_NUM_POINTS//2:])
        diff_slope_second_half = get_regression(minutes[MAX_NUM_POINTS//2+1:], np.diff(values)[MAX_NUM_POINTS//2:])
        diff_last = np.diff(values)[-1]


        is_low = 1 if low < LOW_THRESHOLD else 0

        return [slope,
                r_squared,
                slope_second_half,
                c0,
                c1,
                c2,
                glucose_mean,
                glucose_median,
                mean_to_median,
                glucose_stddev,
                glucose_cardinality,
                total_diff,
                total_diff_second_half,
                diff_last,
                diff_mean,
                diff_slope,
                diff_r_squared,
                diff_slope_second_half,
                glucose_max,
                glucose_min,
                glucose_min_minute,
                gradient_min,
                gradient_min_glucose,
                gradient_min_minute,
                curvature_min,
                curvature_min_glucose,
                curvature_min_minute,
                extrapolated_value_first_order,
                extrapolated_value_second_order,
                is_low]

    else:
        return [None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None]

def main():

    glucose_df = pd.read_csv(INFILE, usecols=COL_NAMES)

    glucose_stats = []
    skip_flag = True
    count = 0

    for index in glucose_df.index.values:

        if index+MAX_NUM_POINTS+MAX_POINTS_TO_LOW-1 > glucose_df.index.values[-1]:
            break

        # If last value is low, then do not use in testing
        if glucose_df[GLUCOSE_COL].iloc[index+MAX_NUM_POINTS-1] < LOW_THRESHOLD:
            continue

        if np.any(glucose_df[TIME_DIFF_COL].iloc[index:index+MAX_NUM_POINTS+MAX_POINTS_TO_LOW].values != TIME_DELTA):
            continue

        if count == 3:
            skip_flag = False
            count = 0

        if skip_flag:
            count += 1
            continue

        timings = glucose_df[TIME_DIFF_COL].iloc[index:index+MAX_NUM_POINTS].values
        values = glucose_df[GLUCOSE_COL].iloc[index:index+MAX_NUM_POINTS].values
        low = glucose_df[GLUCOSE_COL].iloc[index+MAX_NUM_POINTS+MAX_POINTS_TO_LOW-1]
        new_row = [index, glucose_df[GLUCOSE_COL].iloc[index],glucose_df[GLUCOSE_COL].iloc[index+MAX_NUM_POINTS]]+get_statistics(timings, values, low) + [low]
        glucose_stats.append(new_row)
        skip_flag = True

    glucose_stats_df = pd.DataFrame(glucose_stats, columns=COL_OUTPUT_NAMES)
    glucose_stats_df.dropna(inplace=True)
    glucose_stats_df.to_csv(OUTFILE, index=False)


if __name__ == "__main__":
    fire.Fire(main)
