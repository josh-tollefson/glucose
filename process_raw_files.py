import os
import glob
import sys
import fire
import pandas as pd
import datetime as dt

# column name constants
TIMESTAMP_INPUT_COL = "Timestamp (YYYY-MM-DDThh:mm:ss)"
TIMESTAMP_COL = "Timestamp"
GLUCOSE_INPUT_COL = "Glucose Value (mg/dL)"
GLUCOSE_COL = "Glucose"
TIME_DELTA_COL = "Time_Diff"

# number of header rows to skip in Clarity csvs
HEADER_ROWS = 10

# Relevant files
INFILES = "./raw_files/CLARITY_*.csv"
OUTFILE = "./processed/glucose.csv"

# The Clarity entry uses the string 'Low' for very low blood sugars
# instead of a numeric value. This sets the numeric value to use instead.
LOW_VALUE = 40

def load_csv(infile):
    """
    Reads and processes Dexcom Clarity glucose data
    INPUTS: infile: str, path and file name to raw Clarity csv data
    OUTPUT: df: dataframe, processed glucose data
    """

    df = pd.read_csv(infile, usecols=[TIMESTAMP_INPUT_COL, GLUCOSE_INPUT_COL])
    df[TIMESTAMP_INPUT_COL] = pd.to_datetime(df[TIMESTAMP_INPUT_COL])
    df.rename(columns={TIMESTAMP_INPUT_COL: TIMESTAMP_COL, GLUCOSE_INPUT_COL: GLUCOSE_COL},
              inplace=True)

    return df

def merge_files(infiles):

    merged_df_list = []

    for f in infiles:
        df = load_csv(f)
        df = df.dropna()
        df.sort_values(by=TIMESTAMP_COL, inplace=True)
        merged_df_list.append(df)

    merged = pd.concat(merged_df_list, ignore_index=True)

    merged.drop_duplicates(inplace=True)
    merged.sort_values(by=TIMESTAMP_COL, inplace=True, ascending=True)
    merged.reset_index(drop=True, inplace=True)

    return merged

def get_time_deltas(df):
    '''
    Populate df with TIME_DELTA_COL
    TIME_DELTA_COL is the time difference between consecutive rows in minutes
    df is assumed to be sorted by TIMESTAMP_COL
    INPUTS: df, DataFrame
    OUTPUS: df, DataFrame
    '''
    # print(df.info())
    df[TIME_DELTA_COL] = round(df[TIMESTAMP_COL].diff(-1).dt.total_seconds().abs().div(60),0)

    return df

def delete_bad_time_deltas(df):
    '''
    The sensor returns data every 5 minutes.
    Erroneous time deltas may indicate: time between new sensor insertion,
    calibration error, or bad data transmission.
    The last time delta will be NaN but it is 5 minutes after the prior point, so it is valid.
    '''

    return df[[TIMESTAMP_COL, GLUCOSE_COL]].loc[(df[TIME_DELTA_COL].shift(1).fillna(False) == 5.0) | (df[TIME_DELTA_COL] == 5.0)]

def main():
    files_arr = glob.glob(INFILES)
    glucose_df = merge_files(files_arr)

    # get time deltas on original data and clean up rows with bad data
    glucose_df = get_time_deltas(glucose_df)
    glucose_df = delete_bad_time_deltas(glucose_df)

    # get time deltas on cleaned data
    glucose_df = get_time_deltas(glucose_df)

    # Replace 'Low' with chosen numeric value
    glucose_df.replace({'Low': LOW_VALUE}, inplace=True)

    glucose_df.to_csv(OUTFILE, index=False)

if __name__ == "__main__":
    fire.Fire(main)
