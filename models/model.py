from pyomo.environ import * #optimización
import numpy as np
import pandas as pd

def create_dict(df):
    d = {}
    for i in df.index:
        for j in df.columns:
            d[i,j] = df.loc[i,j]
    return d

def T_dict(T, array):

    data = pd.DataFrame(index = T, data={"data":array})["data"].to_dict()

    return data

def create_model(data_model):

    model = ConcreteModel()

    t_s = 1

    ## -- COEFICIENTES VALOR PRESENTE NETO -- ## 
    VPN_F = [round(1/np.power(1+data_model["interest"],i),3) for i in np.arange(1,data_model["lifeyears"]+1)]
    VPN_FS = round(np.sum(VPN_F),3)   

    ## -- VIDA UTIL DEL PROYECTO -- ## 
    model.lifeyears = Param(initialize = data_model["lifeyears"])

    ## -- SETS FIJOS -- ## 
    T = range(1,data_model["load"]["len"]+1)  
    model.T= Set(initialize=T) #LLamando los indices del dataframe
    model.PVT = Set(initialize=data_model["pv_modules"]["type"].columns.tolist())
    model.BATT= Set(initialize=data_model["batteries"]["type"].columns.tolist())
    model.CH = Set(initialize=data_model["inverters"]["type"].columns.tolist())
    
    ## -- PARÁMETROS DE PANELES, BATERÍAS E INVERSORES -- ## 
    model.PVtype = Param(data_model["pv_modules"]["type"].index.to_list(), model.PVT, initialize = create_dict(data_model["pv_modules"]["type"]), domain = Any)
    model.Battype = Param(data_model["batteries"]["type"].index.to_list(), model.BATT, initialize = create_dict(data_model["batteries"]["type"]), domain = Any)
    model.ConH = Param(data_model["inverters"]["type"].index.to_list(), model.CH, initialize = create_dict(data_model["inverters"]["type"]), domain = Any)

    ## -- GENERACIÓN DE POTENCIA DURANTE CADA PASO DE TIEMPO POR MÓDULO FOTOVOLTAICO -- ## 
    data_model["pv_modules"]["Pmpp"].index = T
    model.P_mpp = Param(model.T, model.PVT, initialize = create_dict(data_model["pv_modules"]["Pmpp"]))

    ## -- PARÁMETRO DE CARGA -- ## 
    model.Carga = Param(model.T, initialize= T_dict(T, data_model["load"]["value"]))


    ## -- ENERGÍA NO SUMINISTRADA -- ## 
    model.ENS = Var(model.T, domain=NonNegativeReals) # Variable energía no suministrada
    
    if data_model["ENS"]["active"]:        
        if data_model["ENS"]["type"] == "fixed":            
            model.Price_ENS = Param(model.T, initialize= T_dict(T, np.repeat(data_model["ENS"]["value"], len(T))))
        elif data_model["ENS"]["type"] == "variable":
            model.Price_ENS = Param(model.T, initialize= T_dict(T, data_model["ENS"]["value"]))
    else:
        model.Price_ENS = Param(model.T, initialize= T_dict(T, np.repeat(0, len(T))))
        def None_ENS(m,t):
            return m.ENS[t] == 0
        model.None_ENS = Constraint(model.T, rule=None_ENS)    

    

    ## -- PARÁMETROS Y VARIABLES DE RED PRINCIPAL -- ## 
    model.PGL = Var(model.T, domain=NonNegativeReals)                              # Potencia de la red electrovichada a la carga    
    model.PpvG = Var(model.CH, model.T, domain=NonNegativeReals)                        # Potencia del panel dirigida a la red   
    model.PGB = Var(model.CH, model.BATT, model.T, domain=NonNegativeReals)                        # Potencia de la red electrovichada a la bateria
    model.PTG = Var(model.T, domain=NonNegativeReals)                        # Potencia de la turbina a la red
    if data_model["grid"]["active"]:

        model.MaxPGrid = Param(initialize = data_model["grid"]["pmax_buy"])
        model.MaxPpvG = Param(initialize = data_model["grid"]["pmax_sell"])

        if data_model["grid"]["av"]["active"]:
            model.GridAv = Param(model.T, initialize = T_dict(T, data_model["grid"]["av"]["value"])) 
        else:
            model.GridAv = Param(model.T, initialize = T_dict(T, np.repeat(1, len(T)))) 

        def PG_lim_rule(m,t):
            return m.PGL[t] + sum(m.PGB[tch,tb,t] for tb in m.BATT for tch in m.CH) <= m.GridAv[t]*m.MaxPGrid
        model.PG_lim=Constraint(model.T,rule=PG_lim_rule)
        
        def PpvG_lim_rule(m,t):
            return sum(m.PpvG[tch,t] for tch in m.CH) + m.PTG[t] <= m.GridAv[t]*m.MaxPpvG
        model.PpvG_lim=Constraint(model.T,rule=PpvG_lim_rule)

        if data_model["grid"]["buy_price"]["type"] == "fixed":
            model.Price_Grid = Param(model.T, initialize = T_dict(T, np.repeat(data_model["grid"]["buy_price"]["value"], len(T)))) 
        elif data_model["grid"]["buy_price"]["type"] == "variable":
            model.Price_Grid = Param(model.T, initialize = T_dict(T, data_model["grid"]["buy_price"]["value"])) 
        
        
        if data_model["grid"]["sell_price"]["type"] == "fixed":
            model.Ppvusd = Param(model.T, initialize= T_dict(T, np.repeat(data_model["grid"]["sell_price"]["value"], len(T)))) 
        elif data_model["grid"]["sell_price"]["type"] == "variable":
            model.Ppvusd  = Param(model.T, initialize = T_dict(T, data_model["grid"]["sell_price"]["value"])) 

    else:
        model.Price_Grid = Param(model.T, initialize = T_dict(T, np.repeat(0, len(T)))) 
        model.Ppvusd = Param(model.T, initialize= T_dict(T, np.repeat(0, len(T)))) 

        def None_grid(m,t):
            return m.PGL[t] + sum(m.PGB[tch,tb,t] for tb in m.BATT for tch in m.CH) + sum(m.PpvG[tch,t] for tch in m.CH) + m.PTG[t] == 0
        model.None_grid=Constraint(model.T,rule=None_grid)


    ## -- PARÁMETROS Y VARIABLES DE TURBINAS EÓLICAS -- ## 
    if data_model["windgen"]["active"]:
        model.WT = Set(initialize=data_model["windgen"]["type"].columns.tolist())
        model.WT_gen = Param(model.T, model.WT, initialize = create_dict(data_model["windgen"]["generation"]))
        model.Windtype = Param(data_model["windgen"]["type"].index.to_list(), model.WT, initialize = create_dict(data_model["windgen"]["type"]), domain = Any)

        model.XT = Var(model.WT, domain=NonNegativeIntegers)
        model.PTL = Var(model.T, domain=NonNegativeReals)                        # Potencia de la turbina a la carga
        model.PTB = Var(model.CH, model.BATT, model.T, domain=NonNegativeReals)                        # Potencia de la turbina a la batería
        model.PTCur = Var(model.T, domain=NonNegativeReals)                      # Potencia de la turbina no utilizada

        def WT_balance_rule(m,t):
            return m.PTL[t] + m.PTG[t] + sum(m.PTB[tch,tb,t] for tch in model.CH for tb in m.BATT) + model.PTCur[t] == sum(model.XT[tt]*m.WT_gen[t,tt] for tt in model.WT)
        model.WT_balance_rule=Constraint(model.T,rule=WT_balance_rule)

    else:
        model.WT = Set(initialize=["None"])
        WT_none_df = pd.DataFrame(index=["C_inst", "C_OM_y"], data={"None":[0,0]})
        model.Windtype = Param(WT_none_df.index.to_list(), model.WT, initialize = create_dict(WT_none_df), domain = Any)

        model.XT = Var(model.WT, domain=NonNegativeIntegers)
        model.PTL = Var(model.T, domain=NonNegativeReals)                        # Potencia de la turbina a la carga
        model.PTB = Var(model.CH, model.BATT, model.T, domain=NonNegativeReals)                        # Potencia de la turbina a la batería
        model.PTCur = Var(model.T, domain=NonNegativeReals)                      # Potencia de la turbina no utilizada
        
        def None_WT(m,t):
            return m.PTL[t] + m.PTG[t] + sum(m.PTB[tch,tb,t] for tch in model.CH for tb in m.BATT) + model.PTCur[t] == 0
        model.None_WT = Constraint(model.T, rule=None_WT)
        
        def None_WT_num(m,tt):
            return m.XT[tt] == 0
        model.None_WT_num = Constraint(model.WT, rule=None_WT_num)

        
    ## -- PARÁMETROS Y VARIABLES DEL GENERADOR DE RESPALDO -- ## 
    if data_model["generator"]["active"]:

        model.FuelCost = Param(initialize = data_model["generator"]["fuel_cost"])
        model.GenCost = Param(initialize = data_model["generator"]["gen_cost"])
        model.GenPmax = Param(initialize = data_model["generator"]["pmax"])
        model.GenFmin = Param(initialize = data_model["generator"]["fmin"])
        model.GenFmax = Param(initialize = data_model["generator"]["fmax"])
        model.GenFm = Param(initialize = data_model["generator"]["fm"])
        model.GenOMCost = Param(initialize = data_model["generator"]["gen_OM_cost"])
        model.GenMinLoad = Param(initialize = data_model["generator"]["min_p_load"])
                            
        model.PD = Var(model.T, domain=NonNegativeReals)                              
        model.GenOn = Var(model.T, within=Binary)        

        if data_model["generator"]["av"]["active"]:
            model.DieselAv = Param(model.T, initialize = T_dict(T, data_model["generator"]["av"]["value"]))
        else:
            model.DieselAv = Param(model.T, initialize = T_dict(T, np.repeat(1, len(T)))) 
        
        # Restricción Potencia diesel
        def PD_lim_rule1(m,t):#
            return m.PD[t] <= m.GenOn[t]*m.GenPmax
        model.PD_lim1=Constraint(model.T,rule=PD_lim_rule1)

        def PD_lim_rule2(m,t):#
            return m.PD[t] >= m.GenOn[t]*(m.GenMinLoad/100)*m.GenPmax
        model.PD_lim2=Constraint(model.T,rule=PD_lim_rule2)

        def PD_lim_rule3(m,t):#
            return m.GenOn[t] <= m.DieselAv[t]
        model.PD_lim3=Constraint(model.T,rule=PD_lim_rule3)

    else:
        
        model.FuelCost = Param(initialize = 0)
        model.GenCost = Param(initialize = 0)
        model.GenPmax = Param(initialize = 0)
        model.GenFmin = Param(initialize = 0)
        model.GenFmax = Param(initialize = 0)
        model.GenFm = Param(initialize = 0)
        model.GenOMCost = Param(initialize = 0)
        model.GenMinLoad = Param(initialize = 0)

        model.PD = Var(model.T, domain=NonNegativeReals)                              
        model.GenOn = Var(model.T, within=Binary)  
        
        def None_PD(m,t):#
            return m.PD[t] == 0
        model.None_PD=Constraint(model.T,rule=None_PD)   

        def None_GenOn(m,t):#
            return m.GenOn[t] == 0
        model.None_GenOn=Constraint(model.T,rule=None_GenOn)   


    # Variables discretas
    model.Xpvs  = Var(model.PVT, model.CH, domain=NonNegativeIntegers)      # Número de strings de paneles solares (1)
    model.Xpv = Var(model.PVT, model.CH, domain=NonNegativeIntegers)        # Número de paneles solares (1)
    model.XBs  = Var(model.BATT, model.CH, domain=NonNegativeIntegers)      # Número de strings de Baterías (1)
    model.XB  = Var(model.BATT, model.CH, domain=NonNegativeIntegers)       # Número de Baterías (1)
    model.XCh  = Var(model.PVT,model.BATT,model.CH, domain=NonNegativeIntegers)   # Número de inversores híbridos (1)


    #Variables binarias/Toma de decisiones logicas (0,1)
    model.Bceff = Var(model.CH, model.T, within=Binary)                                # Carga efectiva de baterías (1 = Carga) (0 = Descarga/stand-by) (2)
    model.Bdeff = Var(model.CH, model.T, within=Binary)                                # Descarga efectiva de baterías (1 = Descarga) (0 = Carga/stand-by) (2)
    model.Bxch = Var(model.PVT, model.BATT, model.CH, within=Binary)                           # Sólo un tipo de tecnología de baterías y paneles por inversor (1)

    # Variables continuas
    
    model.PpvL = Var(model.CH, model.T, domain=NonNegativeReals)                        # Potencia del panel dirigida a la carga [kW] (2)
    model.PpvCur = Var(model.CH, model.T, domain=NonNegativeReals)                      # Potencia recortada PV [kW] (2)    
    model.PpvB = Var(model.CH, model.BATT, model.T, domain=NonNegativeReals)                  # Potencia del panel dirigida a los almacenadores [kW] (2)
    model.PBL = Var(model.CH, model.BATT, model.T, domain=NonNegativeReals)                   # Potencia de los almacenadores dirigida a la carga [kW] (2)
    model.SoC = Var(model.BATT, model.T, domain=NonNegativeReals)                       # Estado de carga de las baterías tipo [kWh] (2)
    model.Bcap = Var(model.BATT, model.T, domain=NonNegativeReals)                      # Capacidad de las baterías tipo [kWh] (2)


    ## -- PARÁMETROS Y VARIABLES DE REACTIVOS -- ## 
    model.QGn = Var(model.T, domain=NonNegativeReals) 
    model.QGe = Var(model.T, domain=NonNegativeReals)
    model.QIL = Var(model.T, domain=NonNegativeReals)

    if data_model["load"]["reactive"]:
        if data_model["grid"]["q_price"]["type"] == "fixed":            
            model.Price_Q = Param(model.T, initialize= T_dict(T, np.repeat(data_model["grid"]["q_price"]["value"], len(T))))
        elif data_model["grid"]["q_price"]["type"] == "variable":
            model.Price_Q = Param(model.T, initialize= T_dict(T, data_model["grid"]["q_price"]["value"]))

        model.Lim_Q = Param(initialize = data_model["grid"]["lim_q"])
        model.fp_I = Param(initialize = np.tan(np.arccos(data_model["inverters"]["fp_set"])))
        model.QL = Param(model.T, initialize = T_dict(T, data_model["load"]["reactive_value"]))
        ## -- EXCEDENTES DE COMPRA DE REACTIVA -- ##
        def exc_q(m, t):
            return m.Lim_Q*(m.PGL[t] + sum(m.PGB[tch,tb,t] for tb in m.BATT for tch in m.CH)) >= m.QGn[t] - m.QGe[t]
        model.exc_q = Constraint(model.T, rule=exc_q) 

        def balance_QI(m, t):
            return m.QIL[t] <= sum(sum(m.Xpv[tpv,tch]*m.P_mpp[t,tpv] for tpv in m.PVT) - m.PpvCur[tch,t] for tch in m.CH)*m.fp_I
        model.balance_QI = Constraint(model.T, rule=balance_QI)   

        def balance_QL(m, t):
            return m.QIL[t] + m.QGn[t] + m.QGe[t] == m.QL[t]
        model.balance_QL = Constraint(model.T, rule=balance_QL)  

    else: 
        model.Price_Q = Param(model.T, initialize= T_dict(T, np.repeat(0, len(T))))

        def balance_QL(m, t):
            return m.QIL[t] + m.QGn[t] + m.QGe[t] == 0
        model.balance_QL = Constraint(model.T, rule=balance_QL)  
    #----------------------------------------------------------------------#
    ## -- ELEGIR UN SOLO INVERSOR -- ##
    if data_model["inverters"]["flex"]:
        def PV_BATT_onetype_CH(model,tch):
            return sum(model.Bxch[tpv,tb,tch] for tpv in model.PVT for tb in model.BATT) == 1
        model.PV_onetype_CH=Constraint(model.CH, rule=PV_BATT_onetype_CH)
    ## -- ELEGIR MÁS DE UN INVERSOR -- ##
    else:
        def PV_BATT_onetype_CH(model):
            return sum(model.Bxch[tpv,tb,tch] for tpv in model.PVT for tb in model.BATT for tch in model.CH) == 1
        model.PV_onetype_CH=Constraint(rule=PV_BATT_onetype_CH)

    def Bxch_rule(model,tpv,tb,tch):#
        return model.XCh[tpv,tb,tch] <= 10e3*model.Bxch[tpv,tb,tch]
    model.Bxch_rule=Constraint(model.PVT, model.BATT, model.CH,rule=Bxch_rule)
    #----------------------------------------------------------------------#

    #----------------------------------------------------------------------#
    # Número de strings por tipo de panel e inversor
    def PV_string_rule(m,tpv,tch):#,t para todo t en T
        return m.Xpvs[tpv, tch] <= sum(m.XCh[tpv,tb,tch] for tb in model.BATT)*int(m.ConH['Num_mpp',tch]*m.ConH['Num_in_mpp',tch]*np.floor(m.ConH['Idc_max_in',tch]/m.PVtype['Isc_STC',tpv]))
    model.PV_string=Constraint(model.PVT, model.CH, rule=PV_string_rule)

    #Numero de paneles por tipo de panel e inversor
    def PV_num_rule1(m,tpv, tch):
        return m.Xpv[tpv,tch] >= m.Xpvs[tpv, tch]*np.ceil(m.ConH['V_mpp_inf',tch]/m.PVtype['Vmp_STC',tpv])
    model.PV_num_rule1=Constraint(model.PVT, model.CH, rule=PV_num_rule1)

    def PV_num_rule2(m,tpv, tch):
        return m.Xpv[tpv,tch] <= m.Xpvs[tpv, tch]*np.floor(m.ConH['Vdc_max_in',tch]/m.PVtype['Voc_max',tpv])
    model.PV_num_rule2=Constraint(model.PVT, model.CH, rule=PV_num_rule2)
    #----------------------------------------------------------------------#

    #----------------------------------------------------------------------#
    # Número de baterías por tipo de batería e inversor
    def Batt_string_rule(m,tb,tch):
        return m.XBs[tb,tch] <= 10e3*sum(m.XCh[tpv,tb,tch] for tpv in model.PVT)
    model.Batt_string_rule=Constraint(model.BATT, model.CH,rule=Batt_string_rule)

    def Batt_num_rule(m, tb, tch):
        return m.XB[tb,tch] == m.XBs[tb,tch]*np.floor(m.ConH['V_n_batt',tch]/m.Battype['V_nom',tb])
    model.Batt_num_rule=Constraint(model.BATT, model.CH,rule=Batt_num_rule)
    #----------------------------------------------------------------------#

    #----------------------------------------------------------------------#
    #Balance de Potencia PV por inversor
    def PV_balance_rule(m,tch,t):
        return m.PpvL[tch,t]+ m.PpvG[tch,t] + sum(m.PpvB[tch,tb,t] for tb in m.BATT) + m.PpvCur[tch,t] == sum(m.Xpv[tpv,tch]*m.P_mpp[t,tpv] for tpv in m.PVT)
    model.PV_balance=Constraint(model.CH, model.T,rule=PV_balance_rule)

    #Balance de Potencia  
    def P_balance_rule(m,t):
        return m.PGL[t]+ sum(m.PBL[tch,tb,t] for tb in m.BATT for tch in m.CH) + sum(m.ConH['n_dcac',tch]*m.PpvL[tch,t] for tch in m.CH) + m.PD[t] + m.PTL[t] + m.ENS[t] == m.Carga[t]
    model.P_balance=Constraint(model.T,rule=P_balance_rule)
    #----------------------------------------------------------------------#

    #----------------------------------------------------------------------#
    # Cálculo de estado de carga en baterías cada hora
    def Batt_ts_rule(m,tb,t):#
        if t > m.T.first():
            return (m.SoC[tb,t] == m.SoC[tb,t-1]*(1-m.Battype['Auto_des',tb]) + sum(m.Battype['n',tb]*t_s*(m.PpvB[tch,tb,t] + m.ConH['n_acdc',tch]*(m.PGB[tch,tb,t] + m.PTB[tch,tb,t])) -
                    m.PBL[tch,tb,t]/(m.Battype['n',tb]*m.ConH['n_dcac',tch]) for tch in m.CH))
        else:
            return m.SoC[tb,t] == m.Battype['Cap_nom',tb]*sum(m.XB[tb,tch] for tch in m.CH)
    model.Batt_ts=Constraint(model.BATT, model.T,rule=Batt_ts_rule)

    # Capacidad mínima de baterías 
    def Batt_socmin_rule(m,tb,t):#
        return m.SoC[tb,t] >= sum(m.XB[tb,tch] for tch in m.CH)*m.Battype['Cap_inf',tb]
    model.Batt_socmin=Constraint(model.BATT, model.T,rule=Batt_socmin_rule)

    # Capacidad máxima de baterías 
    def Batt_socmax_rule(m,tb,t):#
        return m.SoC[tb,t] <= m.Bcap[tb,t] 
    model.Batt_socmax=Constraint(model.BATT, model.T,rule=Batt_socmax_rule)

    # Degradación de la capacidad de las baterías
    def Bcap_rule1(m,tb,t):#
        if t > m.T.first():
            return m.Bcap[tb,t] == m.Bcap[tb,t-1] - (m.Battype['Deg_kwh',tb]/m.Battype['n',tb])*sum(m.PBL[tch,tb,t]/m.ConH['n_dcac',tch] for tch in m.CH)
        else:
            return m.Bcap[tb,t] == m.Battype['Cap_nom',tb]*sum(m.XB[tb,tch] for tch in m.CH)
    model.Bcap_rule1=Constraint(model.BATT, model.T,rule=Bcap_rule1)

    # Degradación anual de la capacidad de las baterías
    def Bcap_rule2(m,tb):#
        return m.Bcap[tb,m.T.first()]-m.Bcap[tb,m.T.last()] <= 0.2*m.Bcap[tb,m.T.first()]/m.Battype['ty',tb]
    model.Bcap_rule2=Constraint(model.BATT,rule=Bcap_rule2)
    #----------------------------------------------------------------------#

    #----------------------------------------------------------------------#
    # Restricción salida AC del inversor
    def ac_out_ch(m,tch,t):
        return m.PpvG[tch,t] + sum(m.PBL[tch,tb,t] for tb in m.BATT) + m.PpvL[tch,t] <= m.ConH['Pac_max_out',tch]*sum(m.XCh[tpv,tb,tch] for tb in m.BATT for tpv in m.PVT)
    model.ac_out_ch = Constraint(model.CH, model.T,rule=ac_out_ch)

    # Restricción entrada AC del inversor
    def ac_in_ch(m,tch,tb,t):
        return m.PGB[tch,tb,t] + m.PTB[tch,tb,t] <= m.ConH['Pac_max_in',tch]*sum(m.XCh[tpv,tb,tch] for tpv in m.PVT)
    model.ac_in_ch = Constraint(model.CH, model.BATT,model.T,rule=ac_in_ch)
    #----------------------------------------------------------------------#

    #----------------------------------------------------------------------#
    def PpvB_lim_rule1(m,tch,tb,t):
        return m.PpvB[tch,tb,t] + m.PTB[tch,tb,t] + m.PGB[tch,tb,t] <= m.XB[tb, tch]*m.Battype['P_ch',tb]
    model.PpvB_lim_rule1=Constraint(model.CH, model.BATT, model.T,rule=PpvB_lim_rule1)
    
    def PpvB_lim_rule3(m,tch,tb,t):
        return m.PpvB[tch,tb,t] + m.PTB[tch,tb,t] + m.PGB[tch,tb,t] <= (m.ConH['V_n_batt',tch]*m.ConH['I_max_ch_pv',tch]/1000)*sum(m.XCh[tpv,tb,tch] for tpv in m.PVT)
    model.PpvB_lim_rule3=Constraint(model.CH,model.BATT,model.T,rule=PpvB_lim_rule3)

    def PBL_lims_rule1(m,tch,tb,t):
        return m.PBL[tch,tb,t] <= m.Battype['P_des',tb]*m.XB[tb, tch]
    model.PBL_lims_rule1=Constraint(model.CH, model.BATT, model.T,rule=PBL_lims_rule1)

    def PBL_lims_rule2(m,tch,tb,t):
        return m.PBL[tch,tb,t] <= (m.ConH['V_n_batt',tch]*m.ConH['I_max_des',tch]/1000)*sum(m.XCh[tpv,tb,tch] for tpv in m.PVT)
    model.PBL_lims_rule2=Constraint(model.CH, model.BATT, model.T,rule=PBL_lims_rule2)
    #----------------------------------------------------------------------#

    #----------------------------------------------------------------------#
    # Batería cargando con PV
    def Ceff_rule1(m,tch,t):#
        return sum(m.PpvB[tch,tb,t] + m.PTB[tch,tb,t] + m.PGB[tch,tb,t] for tb in m.BATT) <= 100e6*m.Bceff[tch,t]
    model.Ceff_rule1=Constraint(model.CH,model.T,rule=Ceff_rule1)

    # Batería descargando
    def Deff_rule1(model,tch,t):#
        return sum(model.PBL[tch,tb,t] for tb in model.BATT) <= 100e6*model.Bdeff[tch,t]
    model.Deff_rule1=Constraint(model.CH,model.T,rule=Deff_rule1)

    # Estado único de batería
    def Bstate_rule(model,tch,t):#
        return model.Bceff[tch,t] + model.Bdeff[tch,t] <= 1
    model.Bstate_rule=Constraint(model.CH,model.T,rule=Bstate_rule)
    #----------------------------------------------------------------------#
   
    
    if data_model["area"]["active"]:
        model.Area = Param(initialize = data_model["area"]["value"])
        def PV_number_rule(m):#
            return sum(m.Xpv[tpv,tch]*m.PVtype['A',tpv]  for tch in m.CH for tpv in m.PVT) + sum(m.XT[tt]*m.Windtype['A',tt] for tt in m.WT) <= m.Area
        model.PV_number=Constraint(rule=PV_number_rule)


    if data_model["max_invest"]["active"]:
        model.MaxBudget = Param(initialize = data_model["max_invest"]["value"])
        
        def Budget_rule(m):#
            return sum(m.Xpv[tpv,tch]*(m.PVtype['C_inst',tpv]) for tch in m.CH for tpv in m.PVT) + sum(m.XB[tb,tch]*m.Battype['C_inst',tb] for tch in m.CH for tb in m.BATT) \
                   + sum(m.XCh[tpv,tb,tch]*m.ConH['C_inst',tch] for tpv in m.PVT for tb in m.BATT for tch in m.CH) + sum(m.XT[tt]*m.Windtype['C_inst',tt] for tt in m.WT) + model.GenCost <= m.MaxBudget
        model.Budget=Constraint(rule=Budget_rule)

    if data_model["environment"]["active"]:
        model.EnvC = Param(initialize = data_model["environment"]["mu"]*data_model["environment"]["Cbono"]/1e6)
    else:
        model.EnvC = Param(initialize = 0)
    

    


    #Función objetivo 
    def obj_rule(m):#regla(Función python)
        return  sum(m.Xpv[tpv,tch]*(m.PVtype['C_inst',tpv] + VPN_FS*m.PVtype['C_OM_y',tpv]) for tch in m.CH for tpv in m.PVT) \
                + sum(m.XB[tb,tch]*(m.Battype['C_inst',tb] + VPN_FS*m.Battype['C_OM_y',tb]) for tch in m.CH for tb in m.BATT) \
                + sum(sum(m.XCh[tpv,tb,tch] for tb in m.BATT for tpv in m.PVT)*(m.ConH['C_inst',tch]  + VPN_FS*m.ConH['C_OM_y',tch]) for tch in m.CH) \
                + sum(m.XT[tt]*(m.Windtype['C_inst',tt] + VPN_FS*m.Windtype['C_OM_y',tt]) for tt in m.WT) + m.GenCost \
                + sum(sum(VPN_F[ii-1]*m.ConH['C_inst',tch]*sum(m.XCh[tpv,tb,tch] for tb in m.BATT for tpv in m.PVT) for ii in np.arange(int(m.ConH['ty',tch]),m.lifeyears,int(m.ConH['ty',tch]))) for tch in m.CH) \
                + sum(sum(VPN_F[ii-1]*m.Battype['C_inst',tb]*sum(m.XB[tb,tch] for tch in m.CH) for ii in np.arange(int(m.Battype['ty',tb]),m.lifeyears,int(m.Battype['ty',tb]))) for tb in m.BATT) \
                + sum(sum(VPN_F[ii-1]*m.Windtype['C_inst',tt]*m.XT[tt] for ii in np.arange(int(m.Windtype['ty',tt]),m.lifeyears,int(m.Windtype['ty',tt]))) for tt in m.WT) \
                + VPN_FS*sum(m.Price_Grid[t]*(m.PGL[t] + sum(m.PGB[tch,tb,t] for tb in m.BATT for tch in m.CH))  +
                             m.FuelCost*(m.GenFmin*m.GenOn[t] + m.GenFm*m.PD[t]) + m.GenOMCost*m.GenOn[t] +
                             m.Price_ENS[t]*m.ENS[t] - m.Ppvusd[t]*(sum(m.PpvG[tch,t] for tch in m.CH) + m.PTG[t]) -
                             m.EnvC*sum(sum(m.Xpv[tpv,tch]*m.P_mpp[t,tpv] for tpv in m.PVT) - m.PpvCur[tch,t] for tch in m.CH) -
                             m.EnvC*(sum(m.XT[tt]*m.WT_gen[t,tt] for tt in m.WT) - m.PTCur[t]) + m.Price_Q[t]*(m.QGe[t]) for t in m.T)
                
                

    model.Obj=Objective(rule=obj_rule,sense=minimize)                  #Objetive=Objetive, maximizar Valor presente neto


    return model
    # Cbono*mu_co2*sum(sum(model.Xpv[tpv,tch]*P_mpp[tpv].iloc[t-1] for tpv in PVT) - model.PpvCur[tch,t] for tch in CH)

