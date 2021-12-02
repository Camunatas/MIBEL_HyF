from pyomo.environ import *
from pyomo.opt import SolverFactory
import pandas as pd
import numpy as np

def Arbitrage(Tfore,Tfore_resol,Src_Pfore, HyF_Pmax, HyF_Pmin, HyF_Pcost, Gen_Pprice, Es_Plimdis, Es_Plimcha, Es_Enom, Es_SOC, Es_SOCmax, Es_SOCmin, Arb_SOCnsample, Arb_SOCendmax, Arb_SOCendmin, Es_Peffch, Es_Peffdh, Es_Pcost, Time_h, Arb_start, Arb_end, Schedule_nsample):
    model = ConcreteModel()

    # Pass energy storage state of charge from % to MWh as the energy storage capacity value.
    Es_E=Es_SOC*Es_Enom/100
    Es_Emax=Es_SOCmax*Es_Enom/100
    Es_Emin=Es_SOCmin*Es_Enom/100
    Arb_SOCendmax=Arb_SOCendmax*Es_Enom/100
    Arb_SOCendmin=Arb_SOCendmin*Es_Enom/100

    # Arbitrage optimization problem time horizon.
    model.Varper = range(int(Arb_start-1),int(Arb_end-1))  # Optimization problem period for all the decision variables.
    model.Vardim = range(0,int(Schedule_nsample))          # Dimension of all the decision variables.
    model.Es_Edim = range(0,int(Schedule_nsample+1))       # Dimension of the energy storage state of charge.
   
    # Optimization problem variables of decision.
    model.Es_E = Var(model.Es_Edim, bounds = (Es_Emin, Es_Emax))
    model.NC = Var(model.Vardim, domain=Binary)
    model.ND = Var(model.Vardim, domain=Binary)
    model.Es_Pcha = Var(model.Vardim, bounds = (0, Es_Plimcha), initialize =0)
    model.Es_Pdis = Var(model.Vardim, bounds = (0, Es_Plimdis), initialize =0)
    model.Src_Pgf = Var(model.Vardim, bounds = (0, 1), initialize =1)
    
     # Constraints
    def c1_rule(model, t1):     # Maximum energy storage power charge in MW.
        return Es_Plimcha*model.NC[t1] >= model.Es_Pcha[t1]
    model.c1 = Constraint( model.Varper, rule=c1_rule )

    def c2_rule(model, t1):     # Minimum energy storage power discharge in MW.
        return Es_Plimdis*model.ND[t1] >= model.Es_Pdis[t1]
    model.c2 = Constraint( model.Varper, rule=c2_rule )
    
    def c30_rule(model, t1):     # No charge and discharge at same time.
        return model.NC[t1] + model.ND[t1] <= 1
    model.c30 = Constraint( model.Varper, rule=c30_rule )
    
    def c3_rule(model, t1):     # Upper ESS charging constraint in MW.
        return Es_Emax >= model.Es_E[t1] + (model.Es_Pcha[t1]
         - model.Es_Pdis[t1])*(Tfore_resol/3600)
    model.c3 = Constraint( model.Varper, rule=c3_rule )

    def c4_rule(model, t1):     # Lower ESS discharging constraint in MW.
        return Es_Emin <= model.Es_E[t1] + (model.Es_Pcha[t1]
         - model.Es_Pdis[t1])*(Tfore_resol/3600)
    model.c4 = Constraint( model.Varper, rule=c4_rule )

    def c5_rule(model, t1):     # Maximum allowable generation factor.
        return model.Src_Pgf[t1] <= 1
    model.c5 = Constraint( model.Varper, rule=c5_rule )

    def c6_rule(model, t1):     # Minimum allowable generation factor.
        return model.Src_Pgf[t1] >= 0
    model.c6 = Constraint( model.Varper, rule=c6_rule )

    def c7_rule(model, t1):     # Minimum energy storage state of charge in MWh.
        return model.Es_E[t1] >= Es_Emin
    model.c7 = Constraint( model.Varper, rule=c7_rule )

    def c8_rule(model, t1):     # Maximum energy storage state of charge in MWh.
        return model.Es_E[t1] <= Es_Emax
    model.c8 = Constraint( model.Varper, rule=c8_rule )

    def c9_rule(model, t1):     # Energy storage charge rate in MW.
        return model.Es_E[t1+1] <= model.Es_E[t1] + Es_Plimcha*(Tfore_resol/3600)
    model.c9 = Constraint( model.Varper, rule=c9_rule )

    def c10_rule(model, t1):     # Energy storage discharge rate in MW.
        return model.Es_E[t1+1] >= model.Es_E[t1] - Es_Plimdis*(Tfore_resol/3600)
    model.c10 = Constraint( model.Varper, rule=c10_rule )

    def c11_rule(model, t1):     # Limit of maximun hybrid farm power in MW.
        return HyF_Pmax - Src_Pfore[t1]*model.Src_Pgf[t1] >=  model.Es_Pdis[t1] - model.Es_Pcha[t1]
    model.c11 = Constraint( model.Varper, rule=c11_rule )

    def c12_rule(model, t1):     # Limit of minimun hybrid farm power in MW.
        return HyF_Pmin - Src_Pfore[t1]*model.Src_Pgf[t1] <= model.Es_Pdis[t1] - model.Es_Pcha[t1]
    model.c12 = Constraint( model.Varper, rule=c12_rule )
    
    def c13_rule(model):        # Initial state of charge in MWh.
        return model.Es_E[Arb_start-1] == Es_E
    model.c13 = Constraint( rule=c13_rule )

    def c14_rule(model):        # Minimum final value of state of charge in MWh.
        return model.Es_E[Arb_SOCnsample-1] <= Arb_SOCendmax
    model.c14 = Constraint( rule=c14_rule )

    def c15_rule(model):        # Maximum final value of state of charge in MWh.
        return model.Es_E[Arb_SOCnsample-1] >= Arb_SOCendmin
    model.c15 = Constraint( rule=c15_rule )

    def c16_rule(model, t1):     # Current value of energy storage state of charge in MWh.
        return model.Es_E[t1+1] == model.Es_E[t1] + (model.Es_Pcha[t1]
         - model.Es_Pdis[t1])*(Tfore_resol/3600)
    model.c16 = Constraint( model.Varper, rule=c16_rule )

    model.obj = Objective(        
        expr = sum((Gen_Pprice[t1]*(Src_Pfore[t1]*model.Src_Pgf[t1]+model.Es_Pdis[t1]-model.Es_Pcha[t1])-HyF_Pcost*(Src_Pfore[t1]*model.Src_Pgf[t1]+model.Es_Pdis[t1]-model.Es_Pcha[t1])) for t1 in model.Varper),
        sense = maximize)

    # Objective function - Maximize benefit based on resolution of solver of Python (Pyomo).
    import pyutilib.subprocess.GlobalData
    pyutilib.subprocess.GlobalData.DEFINE_SIGNAL_HANDLERS_DEFAULT = False
    opt = SolverFactory('ipopt')
    opt.solve(model)
        
    Es_PRefdismod=[model.Es_Pdis[t1]() for t1 in model.Vardim] 
    Es_PRefdis=np.asarray(Es_PRefdismod)
    Es_PRefchamod=[model.Es_Pcha[t1]() for t1 in model.Vardim] 
    Es_PRefcha=np.asarray(Es_PRefchamod)   
    HyF_PRefmod=[Src_Pfore[t1]*model.Src_Pgf[t1]()+model.Es_Pdis[t1]()-model.Es_Pcha[t1]() for t1 in model.Vardim] 
    HyF_PRef=np.asarray(HyF_PRefmod)  
    return (Es_PRefdis,Es_PRefcha,HyF_PRef)
   