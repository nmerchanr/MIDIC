import streamlit as st
import pandas as pd
import ssl
import numpy as np
from numpy import radians as r
import json, requests
import matplotlib.pyplot as plt
import altair as alt
from numpy import cos
from numpy import sin
import collections
import cloudpickle
from st_aggrid import GridOptionsBuilder

@st.cache
def load_cat():
    # Cargar datos de módulos PV
    PVtype = pd.read_excel('Catalogo.xlsx',sheet_name='PVModules',header=0,index_col=0)
    # Cargar datos de módulos de Batt
    Battype = pd.read_excel('Catalogo.xlsx',sheet_name='BattModuleS',header=0,index_col=0)
    # Cargar datos de generadores
    Gens = pd.read_excel('Catalogo.xlsx',sheet_name='Generator',header=0,index_col=0)
    # Cargar datos de Convertidores híbridos
    ConH = pd.read_excel('Catalogo.xlsx',sheet_name='Hybrid OnGrid',header=0,index_col=0)
    
    WindGens = pd.read_excel('Catalogo.xlsx',sheet_name='WindTurbines',header=0,index_col=0)

    return PVtype, Battype, Gens, ConH, WindGens

@st.cache(allow_output_mutation=True)
def get_data_fromNSRDB(lat, lon, year):
    
    ssl._create_default_https_context = ssl._create_unverified_context
    # You must request an NSRDB api key from the link above
    api_key = 'cTc2xIqsUEZws0YRXLH2wgfu4HL6ifazGnQJFp50'
    # Set the attributes to extract (e.g., dhi, ghi, etc.), separated by commas.
    attributes = 'ghi,dhi,dni,wind_speed,air_temperature,solar_zenith_angle'
    # Choose year of data
    #year = '2019'
    # Set leap year to true or false. True will return leap day data if present, false will not.
    leap_year = 'false'
    # Set time interval in minutes, i.e., '30' is half hour intervals. Valid intervals are 30 & 60.
    interval = '30'
    # Specify Coordinated Universal Time (UTC), 'true' will use UTC, 'false' will use the local time zone of the data.
    # NOTE: In order to use the NSRDB data in SAM, you must specify UTC as 'false'. SAM requires the data to be in the
    # local time zone.
    utc = 'false'
    # Your full name, use '+' instead of spaces.
    your_name = 'Nicolas+Merchan'
    # Your reason for using the NSRDB.
    reason_for_use = 'beta+testing'
    # Your affiliation
    your_affiliation = 'universidad+nacional'
    # Your email address
    your_email = 'nmerchanr@unal.edu.co'
    # Please join our mailing list so we can keep you up-to-date on new developments.
    mailing_list = 'false'

    # Declare url string
    url = 'https://developer.nrel.gov/api/solar/nsrdb_psm3_download.csv?wkt=POINT({lon}%20{lat})&names={year}&leap_day={leap}&interval={interval}&utc={utc}&full_name={name}&email={email}&affiliation={affiliation}&mailing_list={mailing_list}&reason={reason}&api_key={api}&attributes={attr}'.format(year=year, lat=lat, lon=lon, leap=leap_year, interval=interval, utc=utc, name=your_name, email=your_email, mailing_list=mailing_list, affiliation=your_affiliation, reason=reason_for_use, api=api_key, attr=attributes)

    df = pd.read_csv(url, header=2)
    df = df.loc[df['Minute'] == 0, :]
    info = pd.read_csv(url, nrows=1)
    
    return df, info


@st.cache(allow_output_mutation=True)
def power_PV_calculation(df_meteo, PVtype, azimut, inc_panel, lat):

    df = df_meteo.copy()

    azimut, inc_panel = r(azimut), r(inc_panel)

    df["Day of year"] = df.index.day_of_year
    df['decl'] = 23.45*np.sin(2*np.pi*(284+df['Day of year'].to_numpy())/365)
    df['omega'] = 15*(df['Hour'].to_numpy() - 12)

    decl = r(df['decl'].to_numpy())
    omega = r(df['omega'].to_numpy())
    lat_r = r(lat)

    df['tetha'] = np.arccos(np.sin(decl)*np.sin(lat_r)*(np.cos(inc_panel)-np.cos(azimut)) +
                            np.cos(decl)*np.cos(lat_r)*np.cos(inc_panel)*np.cos(omega) +
                            np.cos(decl)*np.sin(lat_r)*np.sin(inc_panel)*np.cos(azimut)*np.cos(omega) +
                            np.cos(decl)*np.sin(inc_panel)*np.sin(azimut)*np.sin(omega))

    df['zenit'] = np.arccos(np.cos(lat_r)*np.cos(decl)*np.cos(omega) +
                        np.sin(lat_r)*np.sin(decl))

    Rb = np.cos(df['tetha'].to_numpy())/np.cos(df['zenit'].to_numpy())

    df['IRR'] = df['GHI'].to_numpy()*Rb
    
    df['IRR'] = df['IRR'].where(df['IRR']>0, 0)
    
    Tm = df['IRR'].values*np.exp(-3.47-0.0594*df['Wind Speed'].values)+df['Temperature'].values

    T_panel = Tm+(df['IRR'].values/1000)*3

    P_mpp = pd.DataFrame(index = df.index, columns = PVtype.columns)

    for k in list(PVtype.columns):
        P_mpp[k] = (PVtype.loc['P_stc',k]*(1+(PVtype.loc['Tc_Pmax',k]/100)*(T_panel-25))*(df['IRR'].values/1000))/1000
                
    return P_mpp, df

def extract_tem_min(lat,lon):

    base_url = r"https://power.larc.nasa.gov/api/temporal/climatology/point?parameters=T2M,T2M_MAX,T2M_MIN&community=SB&longitude={longitude}&latitude={latitude}&format=JSON"
    api_request_url = base_url.format(longitude=lon, latitude=lat)
    response = requests.get(url=api_request_url, verify=True, timeout=30.00)
    temp_hist = json.loads(response.content.decode('utf-8'))
        
    Temp_min = temp_hist['properties']['parameter']['T2M_MIN']['ANN']

    return Temp_min

@st.cache
def get_symbols():

    api_key = '22019e0efd32e7810ff8ea67724ecbb5'
    url = 'http://api.exchangeratesapi.io/v1/symbols?access_key={api_key}'.format(api_key=api_key)

    response = requests.get(url=url, verify=True, timeout=30.00)
    symbols = json.loads(response.content.decode('utf-8'))
    return symbols

@st.cache
def get_exchangerate(data_s,vis_s):

    api_key = '75280a87f0c3ef64dafd'    
    url = 'https://free.currconv.com/api/v7/convert?q={data_s}_{vis_s},{vis_s}_{data_s}&compact=ultra&apiKey={api_key}'.format(api_key=api_key, data_s=data_s, vis_s=vis_s)
    
    response = requests.get(url=url, verify=True, timeout=30.00)
    exchange = json.loads(response.content.decode('utf-8'))
    return exchange

def Grafica_panel(azimut_p,ele_p):
    
    an=1/2
    la=1.5/2
    al=0.07
    caja_x = np.array([an,an ,-an,-an,an,an,an ,an ,an ,-an,-an,-an,-an,-an,-an,an])
    caja_y = np.array([la,-la,-la,la ,la,la,-la,-la,-la,-la,-la,-la,la ,la ,la ,la])
    caja_z = np.array([0 ,0  ,0  ,0  ,0 ,al,al ,0  ,al ,al ,0  ,al ,al ,0  ,al ,al])
    supe_x = np.array([an,-an,0 ,-an,an ])
    supe_y = np.array([la,-la,0 ,la ,-la])
    supe_z = np.array([al,al ,al,al ,al ])
    vec_dir_x=np.array([0,0])
    vec_dir_y=np.array([0,0])
    vec_dir_z=np.array([al,al*15])    
    cz=cos(np.radians(azimut_p))
    sz=sin(np.radians(azimut_p))
    
    cy=cos(np.radians(ele_p))
    sy=sin(np.radians(ele_p))
    
    M1 = np.array([[1,0,0],[0,cy,sy],[0,-sy,cy]])
                    
    M2 = np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]])
    
    MR = np.matmul(M2, M1)
    
    caja_x2 = MR[0,0]*caja_x +MR[0,1]*caja_y +MR[0,2]*caja_z 
    caja_y2 = MR[1,0]*caja_x +MR[1,1]*caja_y +MR[1,2]*caja_z 
    caja_z2 = MR[2,0]*caja_x +MR[2,1]*caja_y +MR[2,2]*caja_z
    
    supe_x2 = MR[0,0]*supe_x +MR[0,1]*supe_y +MR[0,2]*supe_z 
    supe_y2 = MR[1,0]*supe_x +MR[1,1]*supe_y +MR[1,2]*supe_z 
    supe_z2 = MR[2,0]*supe_x +MR[2,1]*supe_y +MR[2,2]*supe_z
    
    vec_dir_x2= MR[0,0]*vec_dir_x +MR[0,1]*vec_dir_y +MR[0,2]*vec_dir_z 
    vec_dir_y2= MR[1,0]*vec_dir_x +MR[1,1]*vec_dir_y +MR[1,2]*vec_dir_z 
    vec_dir_z2= MR[2,0]*vec_dir_x +MR[2,1]*vec_dir_y +MR[2,2]*vec_dir_z
    
    
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(projection='3d')
    ax.set_title("Dirección panel")    
   
    ax.plot(caja_x2,caja_y2,caja_z2+la*1.2,c="grey")
    
    ax.plot(supe_x2,supe_y2,supe_z2+la*1.2,c="deeppink")
    ax.plot(vec_dir_x2,vec_dir_y2,vec_dir_z2+la*1.2,c="deeppink")
    

    z = np.linspace(0, la*1.2, 50)
    theta = np.linspace(0, 2*np.pi, 50)
    theta_grid, z_grid=np.meshgrid(theta, z)
    x_grid = al*np.cos(theta_grid)/3 
    y_grid = al*np.sin(theta_grid)/3 

    ax.plot_surface(x_grid, y_grid, z_grid, alpha=0.3,color="grey")
    
    
    ax.set_xlabel('Sur')
    ax.set_ylabel('Occidente')
    ax.set_zlabel('')
    ax.set_xlim3d([-1.2,1.2])
    ax.set_ylim3d([-1.2,1.2])
    ax.set_zlim3d([0, 2.4])
    ax.view_init(elev=20, azim=45)
    
    return fig

def createfig_meanhour(df, col, fechas, color, ylabel):
    df_fig = df.copy()
    df_fig['Hora'] = fechas.hour
    fig = alt.Chart(df_fig.groupby(['Hora'], as_index=False).mean()).mark_bar(size=10).encode(x=alt.X('Hora:Q', scale=alt.Scale(domain=(-1, 24))), y=alt.Y(col + ':Q', axis=alt.Axis(title=ylabel)), color=alt.value(color))
    
    return fig

def createfig_heatmap(df, col, fechas, binary, units):

    df_fig = df.copy()
    df_fig['Hora'] = fechas.hour
    df_fig['day_of_year'] = fechas.dayofyear
    df_fig['tooltip'] = "D: " + df_fig["day_of_year"].map(str) + ", H: " + df_fig["Hora"].map(str) + ", Val: " + df_fig[col].map(str)

    if binary:
        color = alt.Color(col + ':Q', title=units, scale=alt.Scale(domain=[0, 1],range=['#030303', '#239B56'],type='linear'))
    else:
        color = alt.Color(col + ':Q', title=units)

    fig = alt.Chart(df_fig).mark_rect().encode(x=alt.X('day_of_year:O', title='Día del año', axis=alt.Axis(values=np.arange(0,366,25), labelAngle=0)), y=alt.Y('Hora:O', title='Hora'), color=color, tooltip='tooltip:N')
    
    return fig


@st.cache
def generate_metrics(df, col, units):
    metrics = pd.DataFrame(data={"Valor":[df[col].mean(),df[col].min(),df[col].max()],
                                         "Unidad": [units,units,units]}, index = ["Media anual","Mínimo anual","Máximo anual"])
    metrics.index.name = "Métrica"

    return metrics


@st.cache
def generate_metrics_av(df, col):    
    concurrent =  collections.Counter(df[col].to_numpy())
    len_df = len(df)
    metrics = pd.DataFrame(index = ["Disponibilidad", "Indisponibilidad"], data = {"Cantidad de franjas":[concurrent[1],concurrent[0]],
                                                                                   "Porcentaje de ocurrencia (%)":[np.round(concurrent[1]*100/len_df),np.round(concurrent[0]*100/len_df)]})

    return metrics

@st.cache
def calculate_WT_power(df, WindGens, z0, height, elevation):
    Turbine_profile = pd.DataFrame(columns = WindGens.columns, index = range(1, len(df)+1))
    Profiles = {}
    
    coef_den = ((1-(0.0065*elevation)/288.16)**(9.81/(287*0.0065)))*(288.16/(288.16-0.0065*elevation))    
    
    for i in Turbine_profile.columns:
        Profiles[i] = pd.read_excel('Catalogo.xlsx',sheet_name = i,header=0,index_col=0)
        #height = WindGens.loc["h", i]
        min_vel = WindGens.loc["v_st",i]
        max_vel = Profiles[i].index[-1]
        break_vel = WindGens.loc["v_max",i]

        for k, ws_value_st in enumerate(df["Wind Speed"].values, 1):

            #ws_value = ws_value_st*((height/10)**alpha_wind)
            

            ws_value = ws_value_st*(np.log(height/z0)/np.log(2/z0))
            if (ws_value <= min_vel) or (ws_value > break_vel):
                Turbine_profile.loc[k, i] = 0
            elif ws_value > max_vel:
                Turbine_profile.loc[k, i] = Profiles[i]["power"].iloc[-1]*coef_den
            else:
                new_index = np.sort(np.append(Profiles[i].index.values, ws_value))        
                Turbine_profile.loc[k, i] = Profiles[i].reindex(new_index).interpolate(method='linear').loc[ws_value,"power"]*coef_den

    return Profiles, Turbine_profile

@st.cache(allow_output_mutation=True)
def read_model(file):   
    #with open('test.pkl', mode='rb') as file:
    #st.spinner("Cargando resultados...")
    model_dict = cloudpickle.load(file)
    return model_dict

def results_num_equipment(model, data_model):
    data = {}
    summary_num = pd.DataFrame(columns = ["Nombre del inversor","Número de inversores","Nombre del panel solar", "Número de strings de paneles","Número total de paneles", "Nombre de la batería", 
                                "Número de strings de baterías", "Número total de baterías"])
    for k, c in enumerate(model.CH):

        if sum(model.XCh[tpv,tb,c].value for tpv in model.PVT for tb in model.BATT) != 0:
            for i in model.PVT:
                if model.Xpv[i,c].value != 0:
                    name_PV = i
                    num_string_PV = model.Xpvs[i,c].value
                    num_PV = model.Xpv[i,c].value
                    break
                else:
                    name_PV = ""
                    num_string_PV = 0
                    num_PV = 0
            for j in model.BATT:
                
                if model.XB[j,c].value != 0:
                    name_B = j
                    num_string_B = model.XBs[j,c].value
                    num_B = model.XB[j,c].value
                    break
                else:
                    name_B = ""
                    num_string_B = 0
                    num_B = 0
            summary_num.loc[k] = [c,sum(model.XCh[tpv,tb,c].value for tpv in model.PVT for tb in model.BATT), name_PV, num_string_PV, num_PV, name_B, num_string_B, num_B]    
    
    summary_num = summary_num.set_index(["Nombre del inversor"])
    summary_num.columns.name = summary_num.index.name
    summary_num.index.name = None
    data["summary"] = summary_num

    data["BATT"] = pd.DataFrame(columns = ["Valor unitario: " + data_model["currency_results"], "Número a instalar"])

    for tb in model.BATT:
        
        data["BATT"].loc[tb,["Valor unitario: " + data_model["currency_results"], "Número a instalar"]] = [data_model["batteries"]["type"].loc["C_inst",tb]*data_model["usd_to_results"], sum(model.XB[tb,tch].value for tch in model.CH)]
        data["BATT"].loc[tb,"Total: " + data_model["currency_results"]] = np.prod(data["BATT"].loc[tb, ["Valor unitario: " + data_model["currency_results"], "Número a instalar"]])
        data["BATT"].loc[tb,"Energía total [kWh]"] = data["BATT"].loc[tb,"Número a instalar"]*data_model["batteries"]["type"].loc["Cap_nom",tb]
    data["BATT"].index.name = "Tecnología"
    data["BATT"].columns.name = data["BATT"].index.name
    data["BATT"].index.name = None

    data["PVT"] = pd.DataFrame(columns = ["Valor unitario: " + data_model["currency_results"], "Número a instalar"])

    for tpv in model.PVT:
        
        data["PVT"].loc[tpv,["Valor unitario: " + data_model["currency_results"], "Número a instalar"]] = [data_model["pv_modules"]["type"].loc["C_inst",tpv]*data_model["usd_to_results"], sum(model.Xpv[tpv,tch].value for tch in model.CH)]
        data["PVT"].loc[tpv,"Total: " + data_model["currency_results"]] = np.prod(data["PVT"].loc[tpv, ["Valor unitario: " + data_model["currency_results"], "Número a instalar"]])
        data["PVT"].loc[tpv,"Potencia total [kW]"] = data["PVT"].loc[tpv,"Número a instalar"]*data_model["pv_modules"]["type"].loc["P_stc",tpv]/1000
    data["PVT"].index.name = "Tecnología"
    data["PVT"].columns.name = data["PVT"].index.name
    data["PVT"].index.name = None

    data["CH"] = pd.DataFrame(columns = ["Valor unitario: " + data_model["currency_results"], "Número a instalar"])

    for tch in model.CH:
        
        data["CH"].loc[tch,["Valor unitario: " + data_model["currency_results"], "Número a instalar"]] = [data_model["inverters"]["type"].loc["C_inst",tch]*data_model["usd_to_results"], sum(model.XCh[tpv,tb,tch].value for tb in model.BATT for tpv in model.PVT)]
        data["CH"].loc[tch,"Total: " + data_model["currency_results"]] = np.prod(data["CH"].loc[tch, ["Valor unitario: " + data_model["currency_results"], "Número a instalar"]])
        data["CH"].loc[tch,"Potencia total [kW]"] = data["CH"].loc[tch,"Número a instalar"]*data_model["inverters"]["type"].loc["Pac_max_out",tch]
    data["CH"].index.name = "Tecnología"
    data["CH"].columns.name = data["CH"].index.name
    data["CH"].index.name = None

    if data_model["windgen"]["active"]:
        data["WT"] = pd.DataFrame(columns = ["Valor unitario: " + data_model["currency_results"], "Número a instalar"])

        for tt in model.WT:
            
            data["WT"].loc[tt,["Valor unitario: " + data_model["currency_results"], "Número a instalar"]] = [data_model["windgen"]["type"].loc["C_inst",tt]*data_model["usd_to_results"], model.XT[tt].value]
            data["WT"].loc[tt,"Total: " + data_model["currency_results"]]  = np.prod(data["WT"].loc[tt, ["Valor unitario: " + data_model["currency_results"], "Número a instalar"]])
            data["WT"].loc[tt,"Potencia total [kW]"] = data["WT"].loc[tt,"Número a instalar"]*data_model["windgen"]["type"].loc["P_nom",tt]
        data["WT"].index.name = "Tecnología"
        data["WT"].columns.name = data["WT"].index.name
        data["WT"].index.name = None
    
    return data

def results_economic(m, data_model):
    data = {}
    VPN_F = [round(1/np.power(1+data_model["interest"],i),3) for i in np.arange(1,data_model["lifeyears"]+1)]
    
    data["Capin"] = (sum(m.Xpv[tpv,tch].value*(m.PVtype['C_inst',tpv]) for tch in m.CH for tpv in m.PVT) \
                    + sum(m.XB[tb,tch].value*(m.Battype['C_inst',tb]) for tch in m.CH for tb in m.BATT) \
                    + sum(sum(m.XCh[tpv,tb,tch].value for tb in m.BATT for tpv in m.PVT)*(m.ConH['C_inst',tch]) for tch in m.CH) \
                    + sum(m.XT[tt].value*(m.Windtype['C_inst',tt]) for tt in m.WT) + m.GenCost)*data_model["usd_to_results"]

    data["COM"] = (sum(m.Xpv[tpv,tch].value*(m.PVtype['C_OM_y',tpv]) for tch in m.CH for tpv in m.PVT) \
                + sum(m.XB[tb,tch].value*(m.Battype['C_OM_y',tb]) for tch in m.CH for tb in m.BATT) \
                + sum(sum(m.XCh[tpv,tb,tch].value for tb in m.BATT for tpv in m.PVT)*(m.ConH['C_OM_y',tch]) for tch in m.CH) \
                + sum(m.XT[tt].value*(m.Windtype['C_OM_y',tt]) for tt in m.WT) + sum(m.GenOMCost*m.GenOn[t].value for t in m.T))*data_model["usd_to_results"]

    data["Renv"] = sum(m.EnvC*sum(sum(m.Xpv[tpv,tch].value*m.P_mpp[t,tpv] for tpv in m.PVT) - m.PpvCur[tch,t].value for tch in m.CH) +
                       m.EnvC*(sum(m.XT[tt].value*m.WT_gen[t,tt] for tt in m.WT) - m.PTCur[t].value) for t in m.T)*data_model["usd_to_results"]
    
    data["Re"] = (sum(m.Ppvusd[t]*(sum(m.PpvG[tch,t].value for tch in m.CH) + m.PTG[t].value) for t in m.T))*data_model["usd_to_results"] 

    data["Ry"] = data["Re"] + data["Renv"]

    data["Ay"] = (sum(m.Price_Grid[t]*(sum(m.PBL[tch,tb,t].value for tb in m.BATT for tch in m.CH) + sum(m.ConH['n_dcac',tch]*m.PpvL[tch,t].value for tch in m.CH)) + m.PTL[t].value for t in m.T))*data_model["usd_to_results"]

    data["Ce"] = (sum(m.Price_Grid[t]*(m.PGL[t].value + sum(m.PGB[tch,tb,t].value for tb in m.BATT for tch in m.CH)) + m.FuelCost*(m.GenFmin*m.GenOn[t].value + m.GenFm*m.PD[t].value) for t in m.T))*data_model["usd_to_results"]

    data["Cens"] = (sum(m.Price_ENS[t]*m.ENS[t].value for t in m.T))*data_model["usd_to_results"]

    data["Cy"] = data["Ce"] + data["Cens"]

    

    VPN__F = np.array(VPN_F)

    VPN_cash_flow = pd.DataFrame(index = np.arange(0,data_model["lifeyears"]+1), columns = ["Cap. inicial", "COM", "Recaudos anuales", "Ahorros anuales", "Costos anuales", "Remplazo baterías","Remplazo inversores", "Reemplazo WT"]).fillna(0)
    VPN_cash_flow.index.name = 'Año'

    VPN_cash_flow.loc[0,"Cap. inicial"] = -np.round(data["Capin"], 1)
    VPN_cash_flow.loc[1:data_model["lifeyears"]+1,"COM"] = -np.round(VPN__F*data["COM"], 1)
    VPN_cash_flow.loc[1:data_model["lifeyears"]+1,"Recaudos anuales"] = np.round(VPN__F*data["Ry"], 1)
    VPN_cash_flow.loc[1:data_model["lifeyears"]+1,"Ahorros anuales"] = np.round(VPN__F*data["Ay"], 1)    
    VPN_cash_flow.loc[1:data_model["lifeyears"]+1,"Costos anuales"] = -np.round(VPN__F*data["Cy"], 1)
    

    for tch in m.CH:
        for i in np.arange(int(m.ConH['ty',tch]),data_model["lifeyears"],int(m.ConH['ty',tch])):
            VPN_cash_flow.loc[i,"Remplazo inversores"] -= round(VPN_F[i-1]*m.ConH['C_inst',tch]*sum(m.XCh[tpv,tb,tch].value for tpv in m.PVT for tb in m.BATT)*data_model["usd_to_results"], 1)

    for tb in m.BATT:
        for i in np.arange(int(m.Battype['ty',tb]),data_model["lifeyears"],int(m.Battype['ty',tb])):
            VPN_cash_flow.loc[i,"Remplazo baterías"] -= round(VPN_F[i-1]*m.Battype['C_inst',tb]*sum(m.XB[tb,tch].value for tch in m.CH)*data_model["usd_to_results"], 1)

    for tt in m.WT:
        for i in np.arange(int(m.Windtype['ty',tt]),data_model["lifeyears"],int(m.Windtype['ty',tt])):
            VPN_cash_flow.loc[i,"Reemplazo WT"] -= round(VPN_F[i-1]*m.Windtype['C_inst',tt]*m.XT[tt].value*data_model["usd_to_results"], 1)

    Nom_flow = VPN_cash_flow.copy()
    Nom_flow.iloc[1:,:] = Nom_flow.iloc[1:,:].apply(lambda x: x/VPN_F)
    Nom_flow["Total"] = Nom_flow.sum(axis = 1).to_numpy()
    Nom_flow.loc["Total"] = Nom_flow.sum(axis = 0).to_numpy()

    VPN_cash_flow["Total"] = VPN_cash_flow.sum(axis = 1).to_numpy()
    VPN_cash_flow.loc["Total"] = VPN_cash_flow.sum(axis = 0).to_numpy()

    VPN_cash_flow.columns.name = VPN_cash_flow.index.name
    VPN_cash_flow.index.name = None

    return data, VPN_cash_flow, Nom_flow

def createline_echart(df, x_col, y_col, y_name, xlabel, ylabel, color, x_date = False):
    
    if x_date:
        x_type = "category"
        x_values = df[x_col].dt.strftime('%d-%m-%Y %H:%M').tolist()
    else:
        x_type = "value"
        x_values = list(df[x_col])

    series = []

    for k, col in enumerate(y_col):
        series.append({'name': y_name[k], "data": list(df[col]), "type": "line", "emphasis": {"focus": "series"}})

    option = {                
                "tooltip": {
                    "trigger": "axis",
                    "axisPointer": {"type": "cross", "label": {"backgroundColor": "#6a7985"}},
                },
                'legend': {
                    'data': y_name
                },
                "xAxis": {
                    "type": x_type,
                    'name': xlabel,
                    "data": x_values
                },
                'dataZoom': [{'type': 'inside', 'start': 0, 'end': 20},
                            {
                                'start': 0,
                                'end': 20
                            }],
                "yAxis": {"type": "value", "name": ylabel},
                "series": series,
            }

    if len(color) > 0:
        option["color"] = color

    return option

def interactive_table(df):
    df = df.T
    df.index.name = "Referencia"
    df.reset_index(inplace=True)
    df.drop([0] ,axis=0, inplace = True)

    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_default_column(groupable=True, value=True, enableRowGroup=True, aggFunc='sum', editable=True)
    gb.configure_selection("multiple", use_checkbox=True, rowMultiSelectWithClick=True, suppressRowDeselection=True)
    gb.configure_grid_options(domLayout='normal')
    gridOptions = gb.build()

    return df, gridOptions

def extract_table(df, grid_response):
    selected = grid_response['selected_rows']
    selected_df = pd.DataFrame(selected).apply(pd.to_numeric, errors='coerce')
    selected_df.drop(["Referencia"], axis=1, inplace = True)
    selected_df.columns = df["ID"]
    selected_df = selected_df.T
    selected_df.dropna(inplace= True)

    cols = []
    for i in selected:
        cols.append(list(i.values())[0])

    selected_df.columns = cols

    return selected_df