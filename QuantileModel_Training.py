'''
A script for training a linear regression based runoff model.
This version will performs quantile regression.
'''

import pandas as pd
from sklearn.linear_model import QuantileRegressor
from typing import Tuple

# ------  Load the SNODAS and training datasets ------
snodas_data = pd.DataFrame()
for year in range(2005,2024):
    year_data = pd.read_csv(f'./data/snodas/snodas_swe_{year}.csv', index_col=0, parse_dates=True)
    snodas_data = pd.concat([snodas_data, year_data])

flow_data = pd.read_csv('./data/competition/train.csv')

# Simplify flow data to just years 2005 to 2023 since the SNODAS data is limited
flow_data = flow_data[flow_data['year'] >= 2005]

# Change the column names of the SNODAS data to match the flow data
snodas_data.columns = flow_data['site_id'].unique()

# ------  Define functions ------

# SNODAS Extraction Function
# Given a particular date and site, extract the SNODAS data so it can be used to train the linear regression model.
def extract_swe(site: str, day: int, month: int) -> pd.Series:
    # Create a daterange for the given day and month between 2005 and 2023
    input_dates = pd.date_range(start='1/1/2005', end='1/1/2024', freq='D',inclusive='left')
    input_dates = input_dates[(input_dates.day == day) & (input_dates.month == month)]

    # If date is April 1st, change input_dates 2017 value to April 8th since April 1st is missing that year
    if day == 1 and month == 4:
        input_dates = input_dates.drop(pd.to_datetime('2017-04-01'))
        input_dates = input_dates.append(pd.DatetimeIndex(['2017-04-08']))

    snodas_day = snodas_data.loc[input_dates, site]

    # Change the index to be the year
    snodas_day.index = snodas_day.index.year

    # Rename column to swe
    snodas_day.name = 'swe'

    return snodas_day


#  Runoff extraction function
# This is pretty simple, just extract the runoff data for a site and index by year
def extract_runoff(site: str) -> pd.Series:
    # Extract the runoff data for the given site
    runoff_site = flow_data[flow_data['site_id'] == site]

    # Change the index to be the year
    runoff_site.index = runoff_site['year']

    # Extract the runoff data
    runoff_site = runoff_site['volume']

    return runoff_site


# Generate a linear regression model
def generate_model(site: str, day: int, month: int) -> Tuple[QuantileRegressor, ...]:
    # Extract the swe and runoff data
    swe = extract_swe(site, day, month)
    runoff = extract_runoff(site)

    # Merge the dataframes
    data = pd.merge(swe, runoff, left_index=True, right_index=True, how='inner')

    # Change swe units to KAF (see RunoffAssessment for more info)
    data['swe_KAF'] = data['swe'] / 1233.48

    quantiles = [0.1, 0.5, 0.9]
    quantile_models = []
    for quantile in quantiles:
        qr = QuantileRegressor(quantile=quantile,alpha=0,solver='highs')
        X = data['swe_KAF'].to_numpy().reshape(-1,1)
        y = data['volume'].to_numpy()
        quantile_models.append(qr.fit(X, y))

    return tuple(quantile_models)


# ------  Generate the models using the training data ------

quantile_models = {}
sites = snodas_data.columns
model_months = [1,2,3,4]
model_days = [1,8,15,22]
model_dates = [(month, day) for month in model_months for day in model_days]

for site in sites:
    quantile_models[site] = {}
    for month, day in model_dates:
        quantile_models[site][(month, day)] = generate_model(site, day, month)


# ------  Save the models ------
import pickle
save_path = './model/quantile_models.pkl'
with open(save_path, 'wb') as f:
    pickle.dump(quantile_models, f)
