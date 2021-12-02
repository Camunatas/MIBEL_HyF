import numpy as np
#%% Circulated energy function
def energy(powers):
    circulated_energy = 0
    for P in powers:
        circulated_energy = circulated_energy + abs(P)
    return circulated_energy


#%% Net benefit function
def dm_benefits(powers, prices, SOC, cost, batt_capacity):
    # Degradation model
    DOD_index = [0., 5., 10., 20., 30., 40., 50., 60., 70., 80., 90., 100.]
    deg_cost_per_cycle = [0., cost / 15000., cost / 7000., cost / 3300., cost / 2050., cost / 1475., cost / 1150.,
                          cost / 950., cost / 760., cost / 675., cost / 580., cost / 500.]

    benefits = []

    en100 = energy(powers)/2/batt_capacity*100
    DOD = max(SOC) - min(SOC)

    for d in range(len(DOD_index) - 1):
        if DOD >= DOD_index[d] and DOD <= DOD_index[d + 1]:
            deg_cost = deg_cost_per_cycle[d] + (deg_cost_per_cycle[d + 1] - deg_cost_per_cycle[d]) * (
                        DOD - DOD_index[d]) / (DOD_index[d + 1] - DOD_index[d])
            break

    DOD1 = max(en100-DOD,0)
    if DOD1>100:
        deg_cost1 = deg_cost_per_cycle[-1]
    for d in range(len(DOD_index) - 1):
        if DOD1 >= DOD_index[d] and DOD1 <= DOD_index[d + 1]:
            deg_cost1 = deg_cost_per_cycle[d] + (deg_cost_per_cycle[d + 1] - deg_cost_per_cycle[d]) * (
                        DOD1 - DOD_index[d]) / (DOD_index[d + 1] - DOD_index[d])
            break


    # Obtaining benefits, sales and purchases
    for i in range(len(prices)):
        Benh = powers[i] * prices[i]
        benefits.append(Benh)

    return sum(benefits) - deg_cost - deg_cost1, 100*(deg_cost + deg_cost1)/cost

#%% Daily arbitrage deviation losses calculation
def dm_dev_cost(real_prices, real_generation, generation_charged, generation_sold,
                forecast_prices, forecast_generation, ESS_purchases, ESS_sold):
    dev_price = 0.3
    hourly_errors_p = []
    hourly_errors_g = []
    schedule_dev = []
    gen_surpluses_h = []
    gen_deficits_h = []
    gen_surpluses_sum = []
    ESS_sold_new = ESS_sold
    WTG_gendevs = []
    ESS_devs = []
    Dev_costs_h = []
    ESS_bendevs_h = []
    Plant_benreals_h = []
    # Extracting hourly errors and deviations
    for i in range(24):
        # Calculating price forecasting mean error
        hourly_error_p = 100 * (abs(real_prices[i] - forecast_prices[i])) / real_prices[i]
        hourly_errors_p.append(hourly_error_p)
        # Calculating generation forecasting mean error
        hourly_error_g = 100 * (abs(real_generation[i] - forecast_generation[i])) / real_generation[i]
        hourly_errors_g.append(hourly_error_g)
        # Obtaining hourly generation surplus which can be used to compensate deficits on charged energy
        gen_surpluses_h.append(-min(0, real_generation[i] - forecast_generation[i]))
        # Obtaining hourly generation defidicts which can be used to compensate deficits on charged energy
        gen_deficits_h.append(max(real_generation[i] - forecast_generation[i], 0))
    price_error = np.mean(hourly_errors_p)
    gen_error = np.mean(hourly_errors_g)
    print("Price forecast mean error is {}%".format(round(price_error, 2)))
    print("Generation mean error is {}%".format(round(gen_error, 2)))
    for i in range(24):
        gen_surpluses_sum.append(sum(gen_surpluses_h[i:24]))
    # Identifying and managing internally deviations in generated energy
    for i in range(24):
        # Breaking loop if no generation is being stored on the ESS
        if sum(generation_charged) == 0:
            break
            # print("No generation is scheduled to be derived into the ESS")
        # Identifying generation deviations
        gen_dev_h = real_generation[i] - forecast_generation[i]
        # Skipping hour if no generation is expected to be stored on the ESS
        if generation_charged[i] == 0:
            pass
        else:
            # Calculating hourly deviation in charged generation
            # In case hourly generation is partially expected to be use to charge, the hourly
            # scheduled energy is substracted from the real hourly generated energy
            gen_charge_dev = (real_generation[i] - generation_sold[i]) - generation_charged[i]
            # Managing charge deviations
            if gen_charge_dev > 0:
                pass  # In case of energy excess, it is expected to be curtailed
            if gen_charge_dev < 0:
                # Switching deviation sign to positive
                gen_charge_dev = gen_charge_dev * -1
                # print("Deficit of {}MWh for generation expected to be sent to the ESS "
                      # "at {}:00h".format(round(gen_charge_dev, 2), i))
                # In case of energy deficit there are two scenarios
                # Scenario A: The sum of hourly expected generation surplusses exceeds charge deficit
                # This surplus will be used to compensate the charge deficit and therefore no
                if gen_surpluses_sum[i] > gen_charge_dev:
                    for j in range(i + 1, 24):
                        # Checking if there is a generation surplus at this hour
                        if gen_surpluses_h[i] > 0:
                            # Checking if hourly generation future surplus is enough to relocate this deficit
                            if gen_charge_dev > gen_surpluses_h[i]:
                                # In case it is not enough, charged energy deficit  and summation of future
                                # generation susrpluss are updated
                                gen_charge_dev = gen_charge_dev - gen_surpluses_h[i]
                                gen_surpluses_h[i] = 0
                            else:
                                gen_surpluses_h[i] = gen_surpluses_h[i] - gen_charge_dev
                        gen_surpluses_sum = []
                        for i in range(24):
                            gen_surpluses_sum.append(sum(gen_surpluses_h[i:24]))
                    # print("A generation surplus is employed to compensate such deficit")
                # In case there is not enough expected generation surplus, energy sold by the ESS must be curtailed
                ESS_curtailed = sum(ESS_sold) - gen_charge_dev
                ESS_sold_hours = 0
                for j in range(24):
                    if ESS_sold[j] > 0:
                        ESS_sold_hours = ESS_sold_hours + 1
                hourly_curtailment = ESS_curtailed / ESS_sold_hours
                for j in range(24):
                    if ESS_sold[j] > 0:
                        ESS_sold_new[j] = ESS_sold[j] - hourly_curtailment
                        ESS_sold[j] = ESS_sold_new[j]  # Workaround
    # Calculating deviations on scheduled WTG power and ESS arbitrage benefits caused by forecast errorsn
    for i in range(24):
        # Calculating power deviations caused by a deficit in generation
        if gen_deficits_h[i] and generation_sold[i] != 0:
            WTG_gendevs.append(gen_deficits_h[i])
            Dev_cost_h = gen_deficits_h[i] * real_prices[i] * dev_price
            Dev_costs_h.append(Dev_cost_h)
            # This print is disabled but saved in case a mor
            # print("Negative deviation of {}MWh at {}:00h with a cost of {}€".format(round(WTG_gendev,2), i,
            #                                                                        round(Dev_cost_h,2)))
        else:
            WTG_gendevs.append(0)
        # Calculating ESS schedule deviations caused by a deficit on charged energy
        if ESS_sold_new[i] != ESS_sold[i]:
            ESS_dev = ESS_sold[i] - ESS_sold_new[i]
            Dev_cost_h = ESS_dev * real_prices[i] * dev_price
            Dev_costs_h[i] = Dev_costs_h[i] + Dev_cost_h
        # Calculating deviations on expected revenues by price forecast errors
        ESS_bendevs_h.append(ESS_sold_new[i] * (real_prices[i] - forecast_prices[i]))
        # Calculating real benefits
        Plant_benreals_h.append(real_prices[i] * (ESS_sold[i] + generation_sold[i] - gen_deficits_h[i]))
    print("{}MWh of negative deviation of scheduled generated energy with a total cost of {}€".
          format(round(sum(WTG_gendevs), 2), round(sum(Dev_costs_h), 2)))
    print("Real benefits are {}€".format(round(sum(Plant_benreals_h) + sum(Dev_costs_h), 2)))