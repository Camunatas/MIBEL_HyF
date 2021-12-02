import statsmodels.api as sm
import numpy as np
import datetime
import pandas as pd
import warnings

# Disabling Statsmodels warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.simplefilter('ignore', ConvergenceWarning)
warnings.simplefilter('ignore', UserWarning)

# Generating direct day-ahead price prediction with SARIMA
def day_ahead_pred(day_str, prices_df):
    day= pd.Timestamp(day_str)  # Converting day string to timestamp
    daily_direct_forecast = []  # Array for storing direct forecast

    # SARIMA model parameters
    train_length = 100                          # Training set length (days)
    model_order = (2, 1, 3)                     # SARIMA order
    model_seasonal_order = (1, 0, 1, 24)        # SARIMA seasonal order

    # Generating training set
    train_end = day - pd.Timedelta('1h')
    train_start = train_end - pd.Timedelta('{}d 23h'.format(train_length))
    train_set = prices_df[train_start:train_end]

    # Generating SARIMA model from doi 10.1109/SSCI44817.2019.9002930
    model = sm.tsa.SARIMAX(train_set, order=model_order, seasonal_order=model_seasonal_order,
                           initialization='approximate_diffuse')
    model_fit = model.fit(disp=False)

    # Generating prediction & storing on daily array
    prediction = model_fit.forecast(24)
    for i in range(24):
        daily_direct_forecast.append(round(prediction[i],2))

    return daily_direct_forecast