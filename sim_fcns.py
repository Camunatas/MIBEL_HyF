from matplotlib import pyplot as plt
import numpy as np
#%% Auxiliary variables & parameters
# Making thight layout default on Matplotlib
plt.rcParams['figure.autolayout'] = True
plt.rcParams['axes.axisbelow'] = True
#%% Single day hybrid plant arbitrage simulation
def plot_dm_hybrid_daily(prices, WTG_Pgen, WTG_Psold, ESS_C, ESS_D, ESS_P, SOC, figurecount):
    WTG_Pbatt = [g - s for g, s in zip(WTG_Pgen, WTG_Psold)]
    dates_label = []  # X axis dates label
    for i in range(24):  # Filling X axis dates label
        dates_label.append('{}:00'.format(i))
    x = np.arange(24)
    # Plotting
    fig = plt.figure(figurecount)  # Creating the figure
    # Energy price
    price_plot = fig.add_subplot(5, 1, 1)  # Creating subplot
    ticks_x = np.arange(0, 24, 1)  # Vertical grid spacing
    plt.xticks(np.arange(0, 24, 1), dates_label, rotation=45)
    axes = plt.gca()
    axes.set_xlim([0, 24])  # X axis limits
    axes.set_ylim([min(prices) * 0.9, max(prices) * 1.1])  # X axis limits
    plt.bar(ticks_x, prices, align='edge', width=1, edgecolor='black', color='r')
    plt.ylabel('Price (€/MWh)')
    plt.grid()
    # Generation
    generation_plot = fig.add_subplot(5, 1, 2)  # Creating subplot
    ticks_x = np.arange(0, 24, 1)  # Vertical grid spacing
    plt.xticks(np.arange(0, 24, 1), dates_label, rotation=45)
    axes = plt.gca()
    axes.set_xlim([0, 24])  # X axis limits
    plt.bar(x + 0.00, WTG_Pgen, color='b', width=0.25, label='Generated', edgecolor='black')
    plt.bar(x + 0.25, WTG_Psold, color='g', width=0.25, label='Sent to the grid', edgecolor='black')
    plt.bar(x + 0.50, WTG_Pbatt, color='r', width=0.25, label='Sent to the battery', edgecolor='black')
    plt.legend(loc="upper center", ncol=3)
    plt.ylabel('WTG (MW)')
    plt.grid()
    # ESS
    ESS_plot = fig.add_subplot(5, 1, 3)  # Creating subplot
    ticks_x = np.arange(0, 24, 1)  # Vertical grid spacing
    plt.xticks(np.arange(0, 24, 1), dates_label, rotation=45)
    axes = plt.gca()
    axes.set_xlim([0, 24])  # X axis limits
    plt.bar(x + 0.00, ESS_C, color='b', width=0.25, label='Charge', edgecolor='black')
    plt.bar(x + 0.25, ESS_P, color='g', width=0.25, label='Purchased', edgecolor='black')
    plt.bar(x + 0.50, ESS_D, color='r', width=0.25, label='Discharged', edgecolor='black')
    plt.legend(loc="upper center", ncol=3)
    plt.ylabel('ESS (MW)')
    plt.grid()
    # SOC
    SOC_plot = fig.add_subplot(5, 1, 4)  # Creating subplot
    ticks_x = np.arange(0, 24, 1)  # Vertical grid spacing
    plt.xticks(np.arange(0, 24, 1), dates_label, rotation=45)
    axes = plt.gca()
    axes.set_xlim([0, 24])  # X axis limits
    plt.plot(SOC)
    plt.ylabel('SOC (%)')
    plt.grid()
    # PCC
    PCC_plot = fig.add_subplot(5, 1, 5)  # Creating subplot
    ticks_x = np.arange(0, 24, 1)  # Vertical grid spacing
    plt.xticks(np.arange(0, 24, 1), dates_label, rotation=45)
    axes = plt.gca()
    axes.set_xlim([0, 24])  # X axis limits
    plt.bar(x - 0.5 * 0.25, WTG_Psold, color='b', width=0.25, label='From WTG', edgecolor='black')
    plt.bar(x + 0.5 * 0.25, ESS_D, color='g', width=0.25, label='From ESS', edgecolor='black')
    plt.bar(x + 0.5 * 0.25, ESS_P, color='g', width=0.25, edgecolor='black')
    plt.legend(loc="upper center", ncol=2)
    plt.ylabel('PCC (MW)')
    plt.grid()
    # Launching the plot
    plt.tight_layout()
    plt.show()
    figurecount = figurecount + 1

