import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import numpy as np
from aux_fcns import dm_benefits

# Disabling pyomo warnings
import logging
logging.getLogger('pyomo.core').setLevel(logging.ERROR)
#%% Day ahead arbitrage of standalone ESS
def day_ahead_ESS(initial_SOC, energy_price, batt_capacity, batt_maxpower,
              batt_efficiency, cost):
    # Model initialization
    model = pyo.ConcreteModel()
    model.time = pyo.RangeSet(0, 23)
    model.SOC = pyo.Var(model.time, bounds=(0, batt_capacity),
                        initialize=0)  # Battery SOC at the end of period. in energy units
    model.charging = pyo.Var(model.time, domain=pyo.Binary)  # Charge verifier
    model.discharging = pyo.Var(model.time, domain=pyo.Binary)  # Discharge verifier
    model.ESS_C = pyo.Var(model.time, bounds=(0, batt_maxpower), initialize=0)  # Energy being charged during period
    model.ESS_D = pyo.Var(model.time, bounds=(0, batt_maxpower))  # Energy being discharged during period
    model.DOD = pyo.Var(bounds=(0, 100))
    model.deg_cost = pyo.Var()
    model.max_SOC = pyo.Var(bounds=(initial_SOC, 100))
    model.min_SOC = pyo.Var(bounds=(0, initial_SOC))

    # Defining the optimization constraints
    def c1_rule(model, t):  # Forces limit on charging power
        return (batt_maxpower * model.charging[t]) >= model.ESS_C[t]

    model.c1 = pyo.Constraint(model.time, rule=c1_rule)

    def c2_rule(model, t):  # Forces limit on discharging power
        return (batt_maxpower * model.discharging[t]) >= model.ESS_D[t]

    model.c2 = pyo.Constraint(model.time, rule=c2_rule)

    def c3_rule(model, t):  # Prevents orders of charge and discharge simultaneously
        return (model.charging[t] + model.discharging[t]) <= 1

    model.c3 = pyo.Constraint(model.time, rule=c3_rule)

    def c4_rule(model, t):  # The SOC must be the result of (SOC + charge*eff - discharge/eff)
        if t == 0:
            soc_prev = initial_SOC
        else:
            soc_prev = model.SOC[t - 1]
        return model.SOC[t] == soc_prev + model.ESS_C[t] * batt_efficiency - model.ESS_D[t] / batt_efficiency

    model.c4 = pyo.Constraint(model.time, rule=c4_rule)

    def c5_rule(model):
        return model.SOC[23] == 0.0

    model.c5 = pyo.Constraint(rule=c5_rule)

    def c6_rule(model, t):
        return model.max_SOC >= model.SOC[t] * (100 // batt_capacity)

    model.c6 = pyo.Constraint(model.time, rule=c6_rule)

    def c7_rule(model, t):
        return model.min_SOC <= model.SOC[t] * (100 // batt_capacity)

    model.c7 = pyo.Constraint(model.time, rule=c7_rule)

    def c8_rule(model):
        return model.DOD == model.max_SOC - model.min_SOC

    model.c8 = pyo.Constraint(rule=c8_rule)

    # Degradation model
    DOD_index = [0., 5., 10., 20., 30., 40., 50., 60., 70., 80., 90., 100]
    deg_cost_per_cycle = [0., cost / 15000., cost / 7000., cost / 3300., cost / 2050., cost / 1475., cost / 1150.,
                          cost / 950., cost / 760., cost / 675., cost / 580., cost / 500]
    model.deg = pyo.Piecewise(model.deg_cost, model.DOD,  # range and domain variables
                              pw_pts=DOD_index,
                              pw_constr_type='EQ',
                              f_rule=deg_cost_per_cycle,
                              pw_repn='INC')

    def EN_rule(model):
        return sum((model.ESS_D[t1] + model.ESS_C[t1]) / 2. * (100 // batt_capacity)
                   for t1 in model.time)

    model.EN = pyo.Expression(rule=EN_rule)  # Half of the total energy throughput in %

    model.DOD1 = pyo.Var(bounds=(0, 100))

    def DOD1_rule(model):
        return model.DOD1 >= model.EN - model.DOD

    model.DOD1_con = pyo.Constraint(rule=DOD1_rule)
    model.deg_cost1 = pyo.Var(domain=pyo.NonNegativeReals)
    model.deg1 = pyo.Piecewise(model.deg_cost1, model.DOD1,  # range and domain variables
                               pw_pts=DOD_index,
                               pw_constr_type='EQ',
                               f_rule=deg_cost_per_cycle,
                               pw_repn='INC')

    # Objective Function: Maximize profitability
    model.obj = pyo.Objective(
        expr=sum((energy_price[t] * (model.ESS_D[t] - model.ESS_C[t]))
                 for t in model.time) - model.deg_cost - model.deg_cost1, sense=pyo.maximize)

    # Applying the solver
    opt = SolverFactory('cbc')
    opt.solve(model)
    # model.pprint()

    # Extracting data from model
    _SOC_E = [model.SOC[t1]() for t1 in model.time]
    _SOC_E.insert(0, initial_SOC)
    _SOC = [i * (100 // batt_capacity) for i in _SOC_E]
    _P_output = [-model.ESS_D[t1]() + model.ESS_C[t1]() for t1 in model.time]

    return _P_output, _SOC

#%% Day ahead arbitrage of hybrid plant participating in the free market
def day_ahead_hybrid(initial_SOC, energy_price, batt_capacity, batt_maxpower,
              batt_efficiency, cost, WTG_Pgen):
    # Model initialization
    model = pyo.ConcreteModel()
    model.time = pyo.RangeSet(0, 23)
    model.SOC = pyo.Var(model.time, bounds=(0, batt_capacity),
                        initialize=0)  # Battery SOC at the end of period. in energy units
    model.charging = pyo.Var(model.time, domain=pyo.Binary)  # Charge verifier
    model.discharging = pyo.Var(model.time, domain=pyo.Binary)  # Discharge verifier
    model.ESS_C = pyo.Var(model.time, bounds=(0, batt_maxpower))  # Energy being charged during period
    model.ESS_D = pyo.Var(model.time, bounds=(0, batt_maxpower))  # Energy being discharged during period
    model.ESS_Purch = pyo.Var(model.time, bounds=(0, batt_maxpower))  # Energy being purchased during period
    model.DOD = pyo.Var(bounds=(0, 100))
    model.deg_cost = pyo.Var()
    model.max_SOC = pyo.Var(bounds=(initial_SOC, 100))
    model.min_SOC = pyo.Var(bounds=(0, initial_SOC))
    model.WTG_Psold = pyo.Var(model.time, bounds=(0, 20), initialize=0) # Generated power directly sold
    model.WTG_Pgen = pyo.Var(model.time, bounds=(0, 20), initialize=0) # Generated power
    WTG_Pgen = np.array(WTG_Pgen)
    # Defining the optimization constraints
    def c1_rule(model, t):  # Forces limit on charging power
        return (batt_maxpower * model.charging[t]) >= model.ESS_C[t]
    model.c1 = pyo.Constraint(model.time, rule=c1_rule)

    def c2_rule(model, t):  # Forces limit on discharging power
        return (batt_maxpower * model.discharging[t]) >= model.ESS_D[t]
    model.c2 = pyo.Constraint(model.time, rule=c2_rule)

    def c3_rule(model, t):  # Prevents orders of charge and discharge simultaneously
        return (model.charging[t] + model.discharging[t]) <= 1
    model.c3 = pyo.Constraint(model.time, rule=c3_rule)

    def c4_rule(model, t):  # The SOC must be the result of (SOC + charge*eff - discharge/eff)
        if t == 0:
            soc_prev = initial_SOC
        else:
            soc_prev = model.SOC[t - 1]
        return model.SOC[t] == soc_prev + model.ESS_C[t] * batt_efficiency - model.ESS_D[t] / batt_efficiency
    model.c4 = pyo.Constraint(model.time, rule=c4_rule)

    def c5_rule(model):
        return model.SOC[23] == 0.0
    model.c5 = pyo.Constraint(rule=c5_rule)

    def c6_rule(model, t):
        return model.max_SOC >= model.SOC[t] * (100 // batt_capacity)
    model.c6 = pyo.Constraint(model.time, rule=c6_rule)

    def c7_rule(model, t):
        return model.min_SOC <= model.SOC[t] * (100 // batt_capacity)
    model.c7 = pyo.Constraint(model.time, rule=c7_rule)

    def c8_rule(model):
        return model.DOD == model.max_SOC - model.min_SOC
    model.c8 = pyo.Constraint(rule=c8_rule)

    def c9_rule(model, t):         # ESS charge power can't be lower than purchase power
        return model.ESS_C[t] >= model.ESS_Purch[t]
    model.c9 = pyo.Constraint(model.time, rule=c9_rule)

    def c10_rule(model, t):        # Plant power balance constraint
        return model.WTG_Pgen[t] >= model.WTG_Psold[t] + (model.ESS_C[t] - model.ESS_Purch[t])
    model.c10 = pyo.Constraint(model.time, rule=c10_rule)

    def c11_rule(model, t):        # Plant power balance constraint
        return model.WTG_Pgen[t] == WTG_Pgen[t]
    model.c11 = pyo.Constraint(model.time, rule=c11_rule)

    # Degradation model
    DOD_index = [0., 5., 10., 20., 30., 40., 50., 60., 70., 80., 90., 100]
    deg_cost_per_cycle = [0., cost / 15000., cost / 7000., cost / 3300., cost / 2050., cost / 1475., cost / 1150.,
                          cost / 950., cost / 760., cost / 675., cost / 580., cost / 500]
    model.deg = pyo.Piecewise(model.deg_cost, model.DOD,  # range and domain variables
                              pw_pts=DOD_index,
                              pw_constr_type='EQ',
                              f_rule=deg_cost_per_cycle,
                              pw_repn='INC')

    def EN_rule(model):
        return sum((model.ESS_D[t1] + model.ESS_C[t1]) / 2. * (100 // batt_capacity)
                   for t1 in model.time)

    model.EN = pyo.Expression(rule=EN_rule)  # Half of the total energy throughput in %

    model.DOD1 = pyo.Var(bounds=(0, 100))

    def DOD1_rule(model):
        return model.DOD1 >= model.EN - model.DOD

    model.DOD1_con = pyo.Constraint(rule=DOD1_rule)
    model.deg_cost1 = pyo.Var(domain=pyo.NonNegativeReals)
    model.deg1 = pyo.Piecewise(model.deg_cost1, model.DOD1,  # range and domain variables
                               pw_pts=DOD_index,
                               pw_constr_type='EQ',
                               f_rule=deg_cost_per_cycle,
                               pw_repn='INC')

    # Objective Function: Maximize profitability
    model.obj = pyo.Objective(
        expr=sum((energy_price[t] * (model.WTG_Psold[t] + model.ESS_D[t] - model.ESS_Purch[t]))
                 for t in model.time) - model.deg_cost - model.deg_cost1, sense=pyo.maximize)

    # Applying the solver
    opt = SolverFactory('cbc')
    opt.solve(model)
    # model.pprint()

    # Extracting data from model
    WTG_Pgen = [model.WTG_Pgen[t]() for t in model.time]
    WTG_Psold = [model.WTG_Psold[t]() for t in model.time]
    ESS_C = [model.ESS_C[t]() for t in model.time]
    ESS_D = [model.ESS_D[t]() for t in model.time]
    ESS_P = [model.ESS_Purch[t]() for t in model.time]
    SOC_E = [model.SOC[t]() for t in model.time]
    SOC_E.insert(0, initial_SOC)
    SOC = [i * (100 // batt_capacity) for i in SOC_E]
    # Clearing Nonetypes and switching sings of charging powers
    for i in range(len(energy_price)):
        if ESS_D[i] is None:
            ESS_D[i] = 0
        if ESS_C[i] is None:
            ESS_C[i] = 0
        else:
            ESS_C[i] = - ESS_C[i]
        if ESS_P[i] is None:
            ESS_P[i] = 0
        else:
            ESS_P[i] = - ESS_P[i]
    # Extracting degradation costs avoiding Nonetype error
    if sum(ESS_C) == 0:
        deg_cost = 0
    else:
        deg_cost = model.deg_cost1() + model.deg_cost()
    # Calculating and printing expected benefits
    PCC_P = [g + s + z for g, s, z in zip(ESS_D, WTG_Psold, ESS_P)]
    benefits_h = []
    for i in range(len(energy_price)):
        benefit_h = energy_price[i]*PCC_P[i]
        benefits_h.append(benefit_h)

    print("Daily Market Expected Benefits: {}€".format(round(sum(benefits_h),2)))

    return WTG_Pgen, WTG_Psold, ESS_C, ESS_D, ESS_P, SOC
#%% Day ahead arbitrage of hybrid plant subsidied with the RD 960/2020 renewable economic regime
def day_ahead_hybrid_subsidied(initial_SOC, energy_price, batt_capacity, batt_maxpower,
              batt_efficiency, cost, WTG_Pgen):
    # Model initialization
    model = pyo.ConcreteModel()
    model.time = pyo.RangeSet(0, 23)
    model.SOC = pyo.Var(model.time, bounds=(0, batt_capacity),
                        initialize=0)  # Battery SOC at the end of period. in energy units
    model.charging = pyo.Var(model.time, domain=pyo.Binary)  # Charge verifier
    model.discharging = pyo.Var(model.time, domain=pyo.Binary)  # Discharge verifier
    model.ESS_C = pyo.Var(model.time, bounds=(0, batt_maxpower))  # Energy being charged during period
    model.ESS_D = pyo.Var(model.time, bounds=(0, batt_maxpower))  # Energy being discharged during period
    model.ESS_Purch = pyo.Var(model.time, bounds=(0, batt_maxpower))  # Energy being purchased during period
    model.DOD = pyo.Var(bounds=(0, 100))
    model.deg_cost = pyo.Var()
    model.max_SOC = pyo.Var(bounds=(initial_SOC, 100))
    model.min_SOC = pyo.Var(bounds=(0, initial_SOC))
    model.WTG_Psold = pyo.Var(model.time, bounds=(0, 20), initialize=0) # Generated power directly sold
    model.WTG_Pgen = pyo.Var(model.time, bounds=(0, 20), initialize=0) # Generated power
    WTG_Pgen = np.array(WTG_Pgen)
    # Defining the optimization constraints
    def c1_rule(model, t):  # Forces limit on charging power
        return (batt_maxpower * model.charging[t]) >= model.ESS_C[t]
    model.c1 = pyo.Constraint(model.time, rule=c1_rule)

    def c2_rule(model, t):  # Forces limit on discharging power
        return (batt_maxpower * model.discharging[t]) >= model.ESS_D[t]
    model.c2 = pyo.Constraint(model.time, rule=c2_rule)

    def c3_rule(model, t):  # Prevents orders of charge and discharge simultaneously
        return (model.charging[t] + model.discharging[t]) <= 1
    model.c3 = pyo.Constraint(model.time, rule=c3_rule)

    def c4_rule(model, t):  # The SOC must be the result of (SOC + charge*eff - discharge/eff)
        if t == 0:
            soc_prev = initial_SOC
        else:
            soc_prev = model.SOC[t - 1]
        return model.SOC[t] == soc_prev + model.ESS_C[t] * batt_efficiency - model.ESS_D[t] / batt_efficiency
    model.c4 = pyo.Constraint(model.time, rule=c4_rule)

    def c5_rule(model):
        return model.SOC[23] == 0.0
    model.c5 = pyo.Constraint(rule=c5_rule)

    def c6_rule(model, t):
        return model.max_SOC >= model.SOC[t] * (100 // batt_capacity)
    model.c6 = pyo.Constraint(model.time, rule=c6_rule)

    def c7_rule(model, t):
        return model.min_SOC <= model.SOC[t] * (100 // batt_capacity)
    model.c7 = pyo.Constraint(model.time, rule=c7_rule)

    def c8_rule(model):
        return model.DOD == model.max_SOC - model.min_SOC
    model.c8 = pyo.Constraint(rule=c8_rule)

    def c9_rule(model, t):         # ESS charge power can't be lower than purchase power
        return model.ESS_C[t] >= model.ESS_Purch[t]
    model.c9 = pyo.Constraint(model.time, rule=c9_rule)

    def c10_rule(model, t):        # Plant power balance constraint
        return model.WTG_Pgen[t] >= model.WTG_Psold[t] + (model.ESS_C[t] - model.ESS_Purch[t])
    model.c10 = pyo.Constraint(model.time, rule=c10_rule)

    def c11_rule(model, t):        # Plant power balance constraint
        return model.WTG_Pgen[t] == WTG_Pgen[t]
    model.c11 = pyo.Constraint(model.time, rule=c11_rule)

    # Degradation model
    DOD_index = [0., 5., 10., 20., 30., 40., 50., 60., 70., 80., 90., 100]
    deg_cost_per_cycle = [0., cost / 15000., cost / 7000., cost / 3300., cost / 2050., cost / 1475., cost / 1150.,
                          cost / 950., cost / 760., cost / 675., cost / 580., cost / 500]
    model.deg = pyo.Piecewise(model.deg_cost, model.DOD,  # range and domain variables
                              pw_pts=DOD_index,
                              pw_constr_type='EQ',
                              f_rule=deg_cost_per_cycle,
                              pw_repn='INC')

    def EN_rule(model):
        return sum((model.ESS_D[t1] + model.ESS_C[t1]) / 2. * (100 // batt_capacity)
                   for t1 in model.time)

    model.EN = pyo.Expression(rule=EN_rule)  # Half of the total energy throughput in %

    model.DOD1 = pyo.Var(bounds=(0, 100))

    def DOD1_rule(model):
        return model.DOD1 >= model.EN - model.DOD

    model.DOD1_con = pyo.Constraint(rule=DOD1_rule)
    model.deg_cost1 = pyo.Var(domain=pyo.NonNegativeReals)
    model.deg1 = pyo.Piecewise(model.deg_cost1, model.DOD1,  # range and domain variables
                               pw_pts=DOD_index,
                               pw_constr_type='EQ',
                               f_rule=deg_cost_per_cycle,
                               pw_repn='INC')

    # Objective Function: Maximize profitability
    model.obj = pyo.Objective(
        expr=sum(((24.99+0.25*(energy_price[t]-24.99)) * (model.WTG_Psold[t] + model.ESS_D[t])
                  - energy_price[t]*model.ESS_Purch[t])for t in model.time)
             - model.deg_cost - model.deg_cost1, sense=pyo.maximize)

    # Applying the solver
    opt = SolverFactory('cbc')
    opt.solve(model)
    # model.pprint()

    # Extracting data from model
    WTG_Pgen = [model.WTG_Pgen[t]() for t in model.time]
    WTG_Psold = [model.WTG_Psold[t]() for t in model.time]
    ESS_C = [model.ESS_C[t]() for t in model.time]
    ESS_D = [model.ESS_D[t]() for t in model.time]
    ESS_P = [model.ESS_Purch[t]() for t in model.time]
    SOC_E = [model.SOC[t]() for t in model.time]
    SOC_E.insert(0, initial_SOC)
    SOC = [i * (100 // batt_capacity) for i in SOC_E]
    # Clearing Nonetypes and switching sings of charging powers
    for i in range(len(energy_price)):
        if ESS_D[i] is None:
            ESS_D[i] = 0
        if ESS_C[i] is None:
            ESS_C[i] = 0
        else:
            ESS_C[i] = - ESS_C[i]
        if ESS_P[i] is None:
            ESS_P[i] = 0
        else:
            ESS_P[i] = - ESS_P[i]
    # Extracting degradation costs avoiding Nonetype error
    if sum(ESS_C) == 0:
        deg_cost = 0
    else:
        deg_cost = model.deg_cost1() + model.deg_cost()
    # Calculating and printing expected benefits according to subsidies and purchases
    benefits_h = []
    for i in range(len(energy_price)):
        benefit_h = (24.99+0.25*(energy_price[i]-24.99)) * (WTG_Psold[i] + ESS_D[i]) - energy_price[i]*ESS_P[i]
        benefits_h.append(benefit_h)

    print("Expected Daily Market benefits with subsidies: {}€".format(round(sum(benefits_h)-deg_cost,2)))

    return WTG_Pgen, WTG_Psold, ESS_C, ESS_D, ESS_P, SOC