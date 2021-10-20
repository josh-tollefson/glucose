### Introduction
Continuous glucose monitoring devices are awesome because they allow diabetics to better manage their blood sugar levels. These data are extremely granular and easy to access. However, one service not part of these devices is hypoglycemia prediction. While monitoring apps can alert diabetics when their blood sugar is *currently* low, it does not indicate if it is *going* to be low in the near future. Even with immediate treatment, diabetics can be stuck with symptoms of hypoglycemia for 15-60 minutes. If incoming lows could be anticipated ahead of time, the amount of time spent with hypoglycemic symptoms could be reduced. 

This work processes data from glucose monitoring devices and uses ML classifiers and regressions to build models that predict hypoglycemia 15 minutes ahead of time.

The data shared in this repository is my own. I welcome other diabetics to clone this repo and use their own data to see whether this model works for them!

Read about this project in more detail [here](https://josh-tollefson.github.io/2021-10-19-glucose-monitoring/)

### Running the Model
Install requirements with 
```
pip install -r requirements.txt
```
Place exported csv files of dexcom data into `raw_files` then run:
```
python process_raw_files.py
```
This builds `glucose.csv` and places it into `processed`. This file merges all csvs in `raw_files`, sorts the data, and determines the time difference between successive data points. Next, run:
```
python generate_stats.py
```
This removes bad data, generates features, and writes them to `glucose_stats.csv` in `processed`. 

The data can be played with in the `classify-and-regression-scratch` jupyter notebook. In this notebook, classifiers and regression ML models are built to predict incoming low blood sugar levels. The coefficients and models generated here could be wrapped with glucose monitoring apps in order to forecast low blood sugars 15 minutes ahead of time.