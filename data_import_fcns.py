import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import datetime

# Day Ahead Spot Market Prices Import
def day_ahead_prices(day_str, prices_df):
    day_start = pd.Timestamp(day_str)
    daily_prices_real = []  # Array with real prices
    day_end = day_start + pd.Timedelta('23h')
    real_prices_df = prices_df[day_start:day_end]
    for i in range(24):
        daily_prices_real.append(real_prices_df["Price"][i])

    return daily_prices_real

def generation(day_str, generation_df, column_str):
    day_start = pd.Timestamp(day_str)
    generation_array = []   # Array with hourly generation
    day_end = day_start + pd.Timedelta('23h')
    generation_df = generation_df[day_start:day_end]
    for i in range(24):
        generation_array.append(generation_df[column_str][i])

    return generation_array