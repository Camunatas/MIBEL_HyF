import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import datetime
from operator import sub
# Making thight layout default on Matplotlib
plt.rcParams['figure.autolayout'] = True
plt.rcParams['axes.axisbelow'] = True
# Importing functions from other files
from data_import_fcns import day_ahead_prices, generation
from forecast_fcns import day_ahead_pred
from arbitrage_fcns import day_ahead_ESS, day_ahead_hybrid, day_ahead_hybrid_subsidied
from plot_fcns import plot_dm_hybrid_daily
from aux_fcns import dm_dev_cost
#%% Variables & Parameters
# ESS parameters
ESS_Enom = 50                       # [MWh] ESS nominal capacity
ESS_Pnom = ESS_Enom/4               # [MW] ESS nominal power
ESS_Cost= 0
ESS_Eff = 0.9                       # ESS efficiency
ESS_SOC_init = 0                    # Initial SOC
# Auxiliary variables
figcount = 0                        # Figures counter
dates_label = []                    # X axis dates label
for i in range(24):                 # Filling X axis dates label
    dates_label.append('{}:00'.format(i))

#%% Importing data
# Importing prices dataset
prices_df = pd.read_csv('Prices.csv', sep=';', usecols=["Price","Hour"], parse_dates=['Hour'], index_col="Hour")
prices_df = prices_df.asfreq('H')

# Importing generated power dataset
generation_df = pd.read_csv('Generation.csv', sep=';', usecols=["Hour","WTG10","WTG5","WTG35","WTG0"],
                            parse_dates=['Hour'], index_col="Hour")
# generation_df = generation_df.asfreq('H')

#%% Running simulations
# Hybrid plant daily arbitrage
day = '2019-03-02 00:00:00'
src_p_fore_day = generation(day, generation_df, 'WTG35')
src_p_real = generation(day, generation_df, 'WTG0')

dm_prices_real = day_ahead_prices(day, prices_df)
daily_prices_pred = day_ahead_pred(day, prices_df)
WTG_Pgen, WTG_Psold, ESS_C, ESS_D, ESS_P, SOC= day_ahead_hybrid(ESS_SOC_init, dm_prices_real, ESS_Enom, ESS_Pnom,
              ESS_Eff, ESS_Cost, src_p_fore_day)
WTG_Pgen, WTG_Psold, ESS_C, ESS_D, ESS_P, SOC= day_ahead_hybrid_subsidied(ESS_SOC_init, dm_prices_real, ESS_Enom, ESS_Pnom,
              ESS_Eff, ESS_Cost, src_p_fore_day)
# plot_dm_hybrid_daily(dm_prices_real, WTG_Pgen, WTG_Psold, ESS_C, ESS_D, ESS_P, SOC, figcount + 1)
WTG_Pchar = [a - b for a,b in zip(WTG_Pgen, WTG_Psold)]
# dm_dev_cost(dm_prices_real, src_p_real, WTG_Pchar, WTG_Psold,
#                 daily_prices_pred, src_p_fore_day, ESS_P, ESS_D)


#%% Hybrid plant yearly comparison
day = '2019-01-01 00:00:00'
PPC_e = []
WTG_e = []
WTGbatt_e = []
ESS_e = []
PPC_ben = []
WTG_ben = []
ESS_ben = []
# Importing data
src_p_fore_day = generation(day, generation_df, 'WTG35')
dm_prices_real = day_ahead_prices(day, prices_df)
for i in range(365):
    print('Day: {}'.format(day))
    dm_prices_real = day_ahead_prices(day, prices_df)
    src_p_fore_day = generation(day, generation_df, 'WTG35')
    # Calling arbitrage function
    WTG_Pgen, WTG_Psold, ESS_C, ESS_D, ESS_P, SOC = day_ahead_hybrid(ESS_SOC_init, dm_prices_real, ESS_Enom, ESS_Pnom,
                  ESS_Eff, ESS_Cost, src_p_fore_day)
    PCC_P = [g + s + z for g, s, z in zip(ESS_D, WTG_Psold, ESS_P)]
    WTGbatt_e.append(sum([g - s for g, s in zip(WTG_Pgen, WTG_Psold)]))
    # Computing circulated energy per component
    PPC_e.append(sum([g + s - z for g, s, z in zip(ESS_D, WTG_Psold, ESS_P)]))
    WTG_e.append(sum([g + s for g, s in zip(WTGbatt_e, WTG_Psold)]))
    ESS_e.append(sum([g - s  for g, s in zip(ESS_D, ESS_P)]))
    # Calculating benefits per component
    daily_WTG_ben = sum([a*b for a,b in zip(WTG_Psold, dm_prices_real)])
    daily_ESS_ben = sum([(a+b)*c for a,b,c in zip(ESS_D, ESS_P, dm_prices_real)])
    daily_PPC_ben = daily_WTG_ben + daily_ESS_ben
    WTG_ben.append(daily_WTG_ben)
    ESS_ben.append(daily_ESS_ben)
    PPC_ben.append(daily_PPC_ben)

    day = pd.Timestamp(day) + pd.Timedelta('1d')

#%% Plotting results
fig = plt.figure(figcount)
ben_plot = fig.add_subplot(3, 1, 1)
plt.plot(PPC_e, label='Total')
plt.plot(WTG_e, label='Turbine')
plt.plot(ESS_e, label='ESS')
plt.ylabel('Circulated energy (MWh)')
plt.xlabel('Day')
plt.grid()
plt.legend()
en_plot = fig.add_subplot(3, 1, 2)
plt.ylabel('Benefits (€)')
plt.xlabel('Day')
plt.plot(PPC_ben, label='Total')
plt.plot(WTG_ben, label='Turbine')
plt.plot(ESS_ben, label='ESS')
plt.legend()
plt.grid()
ESS_plot = fig.add_subplot(3, 1, 3)
plt.ylabel('Generated energy stored (MWh)')
plt.xlabel('Day')
plt.plot(WTGbatt_e)
plt.grid()
