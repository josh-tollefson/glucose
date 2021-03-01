import os
import sys
import fire
import numpy as np
import pandas as pd

DATE = "DATE"
HOUR = "HOUR"
TIMESTAMP = "Timestamp"
TIME_DIFF = "TIME_DIFF"
GLUCOSE_FIELD_ORIG = "Glucose Value (mg/dL)"
# GLUCOSE_FIELD_CORRECTED = "Glucose Value Corrected (mg/dL)" ### not sure why we want to add a new col instead of modifying df in place
LOW_FIELD = "Low"


def read_csv(
    infile,
    cols=[1, 7],
    skiprows=10,
    col_names=[TIMESTAMP, GLUCOSE_FIELD_ORIG],
):
    """
    Reads and processes Dexcom Clarity glucose data
    INPUTS: infile: str, path and file name to raw Clarity csv data
            cols: list of int, column indices to read
            skiprows: int, number of rows to skip (must be at least 10 to skip user, device, & alert info)
            col_names: list of str, column names corresponding to cols
    OUTPUT: df: dataframe, processed glucose data
    """

    if skiprows < 10:
        raise ValueError(
            "skiprows must be >=10 to not read in user, device, & alert info."
        )

    if len(cols) != len(col_names):
        raise ValueError(
            "number of columns read must be equal to number of defined column names."
        )

    df = pd.read_csv(infile, usecols=cols, skiprows=skiprows)
    df.columns = col_names
    df[TIMESTAMP] = pd.to_datetime(df[TIMESTAMP])

    df = handle_lows(df)

    return df


def handle_lows(df, low_val="40"):

    df[GLUCOSE_FIELD_ORIG] = df[GLUCOSE_FIELD_ORIG].replace({LOW_FIELD: low_val})
    df[GLUCOSE_FIELD_ORIG] = df[GLUCOSE_FIELD_ORIG].astype(int)

    return df


def normalize(df):
    """
    Normalize df Glucose Values col between 0 and 1
    INPUTS: df, dataframe
    OUTPUT: df, dataframe
    """

    min_glucose = df[GLUCOSE_FIELD_ORIG].min()
    max_glucose = df[GLUCOSE_FIELD_ORIG].max()

    df[GLUCOSE_FIELD_ORIG] = (df[GLUCOSE_FIELD_ORIG] - min_glucose) / (
        max_glucose - min_glucose
    )

    return df


def get_processed_series(
    cur_timestamp, low_label, df, min_time=30, max_time=60, step=5, unit="m"
):
    """
    Given a timestamp cur_timestamp, return the rows of dataframe df
    with Timestamp between [cur_timestamp - max_time, cur_timestamp - min_time].
    If data are missing, interpolate by nearest value or return none if insufficient data
    INPUTS: cur_timestamp: datetime
                    label: int, Low label
                    df: dataframe [Timestamp, Glucose, Low]
                    min_time, max_time: int
                    unit: str, unit of time for timedelta (e.g., 's'=second, 'm'=minute)
    OUTPUT: array containing glucose values within time interval and low label
    """

    rows = df[
        (df[TIMESTAMP] <= cur_timestamp - pd.to_timedelta(min_time, unit=unit))
        & (df[TIMESTAMP] >= cur_timestamp - pd.to_timedelta(max_time, unit=unit))
    ].to_numpy()

    # Reverse time order in order to do interpolation of missing values
    rows[::-1, 0] = [(cur_timestamp - t).total_seconds() // 60 for t, g, l in rows]

    if rows.shape[0] <= 1:
        return []

    else:
        from scipy import interpolate

        f = interpolate.interp1d(
            rows[:, 0], rows[:, 1], kind="nearest", fill_value="extrapolate"
        )  ### Linear interp gives divide by zero error
        times = np.arange(min_time, max_time + step, step=step)
        glucose_new = f(times)
        row_new = np.append(glucose_new, low_label)
        return row_new


def get_processed_stats(
    cur_timestamp, low_label, df, min_time=30, max_time=60, step=5, unit="m"
):
    """
    Given a timestamp cur_timestamp, return the stats of dataframe df
    with Timestamp between [cur_timestamp - max_time, cur_timestamp - min_time].
    If data are missing, interpolate by nearest value or return none if insufficient data
    INPUTS: cur_timestamp: datetime
                    label: int, Low label
                    df: dataframe [Timestamp, Glucose, Low]
                    min_time, max_time: int
                    unit: str, unit of time for timedelta (e.g., 's'=second, 'm'=minute)
    OUTPUT: array containing stats of glucose values within time interval and low label
    """

    rows = df[
        (df[TIMESTAMP] <= cur_timestamp - pd.to_timedelta(min_time, unit=unit))
        & (df[TIMESTAMP] >= cur_timestamp - pd.to_timedelta(max_time, unit=unit))
    ]

    if rows.shape[0] <= 1:
        return []

    else:
        min_glucose = rows[GLUCOSE_FIELD_ORIG].min()
        max_glucose = rows[GLUCOSE_FIELD_ORIG].max()
        diff_glucose = (
            rows[GLUCOSE_FIELD_ORIG].iloc[-1] - rows[GLUCOSE_FIELD_ORIG].iloc[0]
        )
        diff_time = (
            rows[TIMESTAMP].iloc[-1] - rows[TIMESTAMP].iloc[0]
        ).total_seconds() // 60

        row_new = [min_glucose, max_glucose, diff_glucose, diff_time, low_label]

        return row_new


def get_processed_df(
    df,
    method="series",
    min_time=30,
    max_time=60,
    step=5,
    unit="m",
    savefile=None,
):
    if method == "series":

        times = np.arange(min_time, max_time + step, step=step)
        names = [str(t) for t in times] + [LOW_FIELD]
        f = get_processed_series

    elif method == "stats":

        names = ["Min", "Max", "Diff Glucose", "Diff Time", LOW_FIELD]
        f = get_processed_stats

    else:
        raise ValueError("Only 'series' and 'stats' methods allowed")
    features = pd.DataFrame(columns=names)

    # TODO: this function currently takes 20 seconds which seems nuts.
    # I think a large part of this is because we're interpolating results for every calculation
    # Instead, can we interpolate once over the whole dataset?
    rows = df.apply(
        lambda row: f(
            row[TIMESTAMP],
            row[LOW_FIELD],
            df,
            min_time=min_time,
            max_time=max_time,
            step=step,
            unit=unit,
        ),
        axis=1,
    )

    # TODO: This function currently takes 25 seconds to complete which seems nuts
    # Seems weird we have to move to numpy to see if the row is empty
    # Also seems like adding rows in a panda dataframe one at a time might be expensive
    # for i, r in enumerate(rows.to_numpy()):
    #     if len(r) > 0:
    #         features.loc[i] = r

    # For the case of series I got this to work I think:
    # Set all rows to same length
    rows = rows[rows.apply(len) != 0]
    # Unpack the arrays
    rows = [x for x in rows.to_numpy()]
    features = pd.DataFrame(rows, columns=names)

    if savefile is not None:
        features.to_csv(savefile, index=False)

    return features


def label_lows(df, low_threshold=70):
    """
    A df col based on low glucose values set by low_threshold
    with 1 (yes, low) or 0 (no, not low)
    INPUTS: df, dataframe
                    low_threshold, int
    OUTPUT: df with binary 'Low' column added
    """

    df["Low"] = np.where(df[GLUCOSE_FIELD_ORIG] <= low_threshold, 1, 0)

    return df


def split_timestamp(df):

    df[DATE] = df[TIMESTAMP].dt.date
    df[HOUR] = df[TIMESTAMP].dt.time

    return df


def main(
    infile="./raw/CLARITY_Export_2021-01-28_222148.csv",
    outfile="./processed/CLARITY_Export_2021-01-28_222148_processed_series_normalized.csv",
    method="series",
    normalize_data=True,
    split_timestamp=False,
):
    df = read_csv(infile)
    df = label_lows(df)

    if normalize_data:
        df = normalize(df)

    if split_timestamp:
        df = split_timestamp(df)
    get_processed_df(df, method=method, savefile=outfile)


if __name__ == "__main__":
    fire.Fire(main)
