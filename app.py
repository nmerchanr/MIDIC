import streamlit as st

# Importación de librerias

from PIL import Image
import re
import cloudpickle
from streamlit_echarts import st_echarts
import random
import openpyxl
from  pyomo import environ as pym
import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 
import plotly.graph_objects as go
import os
from datetime import datetime
import altair as alt
import plotly.express as px
from streamlit_option_menu import option_menu
from st_aggrid import AgGrid, GridUpdateMode

from models.model import create_model
from funciones import load_cat, get_data_fromNSRDB, power_PV_calculation, extract_tem_min, get_symbols, get_exchangerate, Grafica_panel, createfig_meanhour, createfig_heatmap
from funciones import generate_metrics, generate_metrics_av, calculate_WT_power, read_model, results_num_equipment, results_economic, createline_echart, interactive_table, extract_table

# Título y configuración de página
st.set_page_config(page_title="MIDOTIC", page_icon=":battery:", layout="wide", initial_sidebar_state="auto", menu_items=None)
st.title("MIDOTIC - Microgrid Design Optimization Tool with Inverter Constraints")
st.markdown("""<hr style="height:7px;border-radius:5px;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)

data_model = {}


symbols = get_symbols()
option_symbols = []
for i,j in zip(symbols['symbols'].keys(), symbols['symbols'].values()):
    option_symbols.append(i + ' : ' + j)

fechas = pd.date_range(start = '01/01/2019 00:00', end='31/12/2019 23:00', freq='1H')

menu_options = ["Crear modelo","Visualizar resultados"]

menu_select = option_menu(menu_title = "Menú de opciones", options = menu_options, 
                          icons = ["window-plus", "graph-up"], orientation= "horizontal")

if menu_select == menu_options[0]:

    st.sidebar.title("Crear modelo :wrench:")  
    st.header("Parámetros generales del proyecto :black_nib:")

    cols_gen = st.columns(2)
    with cols_gen[0]:

        

        st.subheader("Ubicación :triangular_flag_on_post:")

        data_model["lat"] = st.number_input("Latitud: ", value= 6.1849)
        data_model["lon"] = st.number_input("Longitud: ", value = -67.4894)
              
        try:
            lista=[data_model.get("lat"),data_model.get("lon")]
            from geopy.geocoders import Nominatim 
            geolocator = Nominatim(user_agent="ubicacion") 
            location = geolocator.reverse(lista) 
            components=location.raw   
            if 'city' in components['address'].keys():
                st.write("La ubicación del proyecto es en: " + components['address']['city'] + " , " + components['address']['country']) 
            elif 'town' in components['address'].keys():
                st.write("La ubicación del proyecto es en: " + components['address']['town'] + " , " + components['address']['country'])      
        except:
            components = {}
            components['address'] = {}
            components['address']['country'] = "indefinido"
        

        st.map(data=pd.DataFrame(data = {'lat':[data_model["lat"]],'lon':[data_model["lon"]]}), zoom=12, use_container_width=True)

       

    with st.expander("Visualizar datos meteorológicos"):
        
        data_model["year"] = st.selectbox("Seleccione el año de consulta de los datos meteorológicos", [2015,2016,2017,2018,2019,2020], 4)

        try:
            df, info = get_data_fromNSRDB(data_model["lat"],data_model["lon"], data_model["year"])
        except:
            st.error("No es posible descargar los datos meteorológicos de la ubicacion especificada")
            st.stop()
        
            

        date_vec = np.vectorize(datetime)
        df_index = date_vec(df.Year.values,df.Month.values,df.Day.values, df.Hour.values, df.Minute.values, tzinfo=None)
        df.index = df_index

        df_meteo = pd.DataFrame(data={'Irra_year':df.GHI.values, 'Temperatura':df.Temperature.values, 'vel (m/s)':df['Wind Speed'].values}, index=df_index)
        df_meteo.index.name = 'Fecha'
        
        st.metric("Zona horaria", "GTM" + str(info['Time Zone'].iloc[0]))

        st.metric("Elevación", str(info['Elevation'].iloc[0]) + " msnm")

        st.markdown("""<hr style="height:5px;border-radius:5px;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)

        Graf_irr = df_meteo.loc[:,['Irra_year']]
        Graf_irr = Graf_irr.reset_index()
        st.subheader("Perfil de irradiancia en el año " + str(data_model["year"]) + ":sunny:")
        st_echarts(options=createline_echart(Graf_irr, "Fecha", ["Irra_year"], ["Irradiancia"], "Fecha", "W/m^2", "#E6B020", x_date = True) , height="400px")
    

        Irr_metric = generate_metrics(df_meteo, 'Irra_year', "W/m^2")
        st.table(Irr_metric.iloc[[0,2],:])


        st.markdown("""<hr style="border:2px dashed IndianRed;border-radius:5px;" /> """, unsafe_allow_html=True)

        Graf_tem = df_meteo.loc[:,['Temperatura']]
        Graf_tem = Graf_tem.reset_index()
        st.subheader("Perfil de Temperatura en el año " + str(data_model["year"]) + ":hotsprings:")
        st_echarts(options=createline_echart(Graf_tem, "Fecha", ["Temperatura"], ["Temperatura"], "Fecha", "°C", "#EF8268", x_date = True) , height="400px")

        Tem_metric = generate_metrics(df_meteo, 'Temperatura', "°C")
        st.table(Tem_metric)

        st.markdown("""<hr style="border:2px dashed IndianRed;border-radius:5px;" /> """, unsafe_allow_html=True)

        Graf_vel = df_meteo.loc[:,['vel (m/s)']]
        Graf_vel = Graf_vel.reset_index()
        st.subheader("Perfil de velocidad del viento en el año " + str(data_model["year"]) + ":cyclone:")
        st_echarts(options=createline_echart(Graf_vel, "Fecha", ["vel (m/s)"], ["Velocidad"], "Fecha", "m/s", "#21AEF0", x_date = True) , height="400px")

        wind_metric = generate_metrics(df_meteo, 'vel (m/s)', "m/s")
        st.table(wind_metric)

            
            
            
            



    with cols_gen[1]:

        st.subheader("Variables económicas :moneybag:")
        
        data_model["interest"] = st.number_input("Tasa de descuento anual (%)", min_value = 0, max_value = 100, value = 5)/100
        data_model["lifeyears"] = st.number_input("Vida útil (años)", min_value = 2, max_value = 50, value = 25)

        st.subheader("Divisas :currency_exchange:")       

        
        try:      
            currency_data = st.selectbox("Divisa de datos de entrada", option_symbols, int(np.where(np.array(option_symbols)=='USD : United States Dollar')[0][0]))[0:3]
            exchange_data = get_exchangerate(currency_data, "USD")
            data_model["in_data_to_usd"] = exchange_data[currency_data+"_USD"]

            cop_data = get_exchangerate("USD", "COP")
            data_model["cop_to_usd"] = cop_data["COP_" + currency_data]
            data_model["usd_to_cop"] = cop_data[currency_data + "_COP"]

            if currency_data != "USD": 
                for i,j in zip(exchange_data.keys(),exchange_data.values()):    
                    st.metric(i[0:3], str(round(j,5)) + " " + i[4:])
        except:
            st.error("Error con el servidor de divisas. Solo es posible crear el modelo en dolares")
            currency_data = "USD"
            data_model["in_data_to_usd"] = 1
            

        
    st.markdown("""<hr style="height:7px;border-radius:5px;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)

    st.header("Demanda :electric_plug:")
    cols_gen = st.columns(2)
    with cols_gen[0]:    
        
        st.subheader("Perfil de demanda :bulb:")

        load_profile_csv = st.file_uploader("Cargar perfil de carga en csv", type = ['csv'])
        if load_profile_csv is not None:
            # Can be used wherever a "file-like" object is accepted:
            
            data_model["load"] = {} 
            
            load_profile = pd.read_csv(load_profile_csv)

            lp_col = st.selectbox("Columna que contiene los datos", options=load_profile.columns.to_numpy())
            lp_scale = st.number_input("Escala",min_value=0.0, value=1.0)
            load_profile  = load_profile.loc[:,[lp_col]]*lp_scale
            
            data_model["load"]["value"] = load_profile[lp_col].to_numpy()         
            data_model["load"]["len"] = len(load_profile)

            if st.checkbox("¿Considerar consumo de energía reactiva?"):
                data_model["load"]["reactive"] = True
                data_model["load"]["fp"] = st.number_input("Factor de potencia de la carga: ", min_value= 0.00, max_value = 1.00, value = 0.95)
                data_model["load"]["reactive_value"] = np.round(data_model["load"]["value"]*np.tan(np.arccos(data_model["load"]["fp"])), 5)
            else:
                data_model["load"]["reactive"] = False


            with st.expander("Visualizar gráficos y métricas de la carga"):
                
                st.subheader("Métricas de demanda de potencia activa")
                load_metric = generate_metrics(load_profile, lp_col, "kW")
                st.dataframe(load_metric)

                st.subheader("Media horaria. Demanda de potencia activa")
                lp_color = st.color_picker('Color gráfica', '#00f900')
                st.altair_chart(createfig_meanhour(load_profile, lp_col, fechas, lp_color, "kW"),use_container_width=True)

                st.subheader("Mapa anual del perfil de demanda de potencia activa")
                st.altair_chart(createfig_heatmap(load_profile, lp_col, fechas, False, "kW"),use_container_width=True)

                if data_model["load"]["reactive"]:
                    st.subheader("Perfil de demanda de potencia activa y reactiva")
                    st_echarts(options=createline_echart(pd.DataFrame({'Fecha':fechas, 'Activa':data_model["load"]["value"],'Reactiva': data_model["load"]["reactive_value"]}), 
                                                        "Fecha", ['Activa','Reactiva'], ['Activa','Reactiva'], "Fecha", "kW - kVAr", [], x_date = True) , height="400px")

    with cols_gen[1]:

        st.subheader("Energía no suministrada :x:")

        data_model["ENS"] = {}

        ENS_choice = st.radio("Precio de penalización por ENS", ["Sin penalización", "Fijo","Variable - Serie temporal"])
        
        if (ENS_choice == "Variable - Serie temporal"):
            #st.subheader("Carga archivo csv con la serie termporal con precios por penalización ENS")
            ENS_profile_csv = st.file_uploader("Cargar csv con precios ENS en " + currency_data + "/kWh", type = ['csv'], help = "Link de ayuda: " + r"https://plotly.com/python/heatmaps/")
            if ENS_profile_csv is not None:
                ENS_price_profile = pd.read_csv(ENS_profile_csv)

        if ENS_choice == "Sin penalización":

            data_model["ENS"]["type"] = None
            data_model["ENS"]["active"] = False
            data_model["ENS"]["value"] = None
            data_model["ENS"]["len"] = None

        elif ENS_choice == "Fijo":
            Price_ENS = st.number_input("Precio ENS: " + currency_data + "/kWh")
            
            data_model["ENS"]["type"] = "fixed"
            data_model["ENS"]["active"] = True
            data_model["ENS"]["value"] = Price_ENS*data_model["in_data_to_usd"]
            data_model["ENS"]["len"] = 0

        elif ENS_choice == "Variable - Serie temporal":
            if ENS_profile_csv is not None:
                ENS_price_col = st.selectbox("Columna que contiene los precios de ENS", options = ENS_price_profile.columns.to_numpy())
                Price_ENS = ENS_price_profile.loc[:,[ENS_price_col]]

                data_model["ENS"]["type"] = "variable"
                data_model["ENS"]["active"] = True
                data_model["ENS"]["value"] = Price_ENS[ENS_price_col].to_numpy()*data_model["in_data_to_usd"]
                data_model["ENS"]["len"] = len(Price_ENS)

                with st.expander("Visualizar gráficos. Precios de penalización ENS"):
                    st.subheader("Métricas")
                                    
                    ENS_price_metric = generate_metrics(Price_ENS, ENS_price_col, currency_data + "/kWh")
                    st.dataframe(ENS_price_metric)

                    st.subheader("Promedio por hora")

                    ENS_price_color = st.color_picker('Color gráfica precio ENS', '#00f900')
                    st.altair_chart(createfig_meanhour(Price_ENS, ENS_price_col, fechas, ENS_price_color, currency_data + "/kWh").interactive(),use_container_width=True)

                    st.subheader("Precio ENS por hora y día del año")
                    st.altair_chart(createfig_heatmap(Price_ENS, ENS_price_col, fechas, False, currency_data + "/kWh").interactive(), use_container_width=True)


    st.markdown("""<hr style="height:7px;border-radius:5px;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)

    st.header("Topología de la microrred ⚡")
    
    options = st.multiselect(
        '¿Qué activos tiene la microrred?',
        ["Módulos fotovoltaicos","Red",'Generador', 'Sistema de almacenamiento (ESS)',"Aerogenerador"],["Módulos fotovoltaicos",'Sistema de almacenamiento (ESS)'])

        #GENERAR diagrama
        
    fondo=Image.open("imagenes/"+"fondo.PNG").convert("RGBA")

    img12=Image.open("imagenes/"+"Carga.PNG").convert("RGBA").resize((145, 250))
    fondo.paste(img12,(635,0),img12)

    if 'Red'in options:
        img=Image.open("imagenes/"+"Red.PNG").convert("RGBA").resize((170, 280))
        fondo.paste(img,(630,490),img)
        
    if 'Aerogenerador'in options:
        img2=Image.open("imagenes/"+"Aerogenerador.PNG").convert("RGBA").resize((220, 200))
        fondo.paste(img2,(350,75),img2)
        
        
    if 'Módulos fotovoltaicos'in options:
        img1=Image.open("imagenes/"+"panel.PNG").convert("RGBA").resize((290, 188))
        fondo.paste(img1,(260,257),img1)

    if 'Generador'in options:
        img3=Image.open("imagenes/"+"generador.PNG").convert("RGBA").resize((230, 190))
        fondo.paste(img3,(795,472),img3)

    if 'Sistema de almacenamiento (ESS)'in options:
        img4=Image.open("imagenes/"+"almacenamiento.PNG").convert("RGBA").resize((300, 150))
        fondo.paste(img4,(900,310),img4)
        
    st.image(fondo,use_column_width="auto")

    data_model["grid"] = {}

    if 'Red' in options:

        st.markdown("""<hr style="border:2px dashed Salmon;border-radius:5px;" /> """, unsafe_allow_html=True)

        st.header("Red principal :zap:")
        data_model["grid"]["active"] = True        

        st.subheader("Precios de la red principal :heavy_dollar_sign:")

        if components['address']['country'] == "Colombia":
            price_input = st.selectbox("Origen de los datos", ["Históricos dados por el usuario", "Históricos de operador de red colombiano"])
        else:
            price_input = "Históricos dados por el usuario"

        if price_input == "Históricos dados por el usuario":

            cols = st.columns(2)

            cols[0].subheader("Precio de compra :inbox_tray:")
            cols[1].subheader("Precio de venta :outbox_tray:")

            purchase_rate_choice = cols[0].radio("Precio de compra a la red", ["Valor constante", "Variable - Serie temporal"])
            sell_rate_choice = cols[1].radio("Precio de venta a la red", ["Valor constante",  "Variable - Serie temporal"])

                    

            if (purchase_rate_choice == "Variable - Serie temporal") or (sell_rate_choice == "Variable - Serie temporal"):
                #st.subheader("Carga archivo csv con series temporales de los precios")
                price_profile_csv = st.file_uploader("Cargar csv con precios en " + currency_data + "/kWh", type = ['csv'], help = "Link de ayuda: " + r"https://plotly.com/python/heatmaps/")
                if price_profile_csv is not None:
                    price_profile = pd.read_csv(price_profile_csv)

            

            with cols[0]:               
                
                if purchase_rate_choice == "Valor constante":
                    
                    Price_Grid = st.number_input("Precio de compra: " + currency_data + "/kWh")

                    data_model["grid"]["buy_price"] = {}
                    data_model["grid"]["buy_price"]["type"] = "fixed"
                    data_model["grid"]["buy_price"]["value"] = Price_Grid*data_model["in_data_to_usd"]
                    data_model["grid"]["buy_price"]["len"] = 0

                elif purchase_rate_choice == "Variable - Serie temporal":
                    if price_profile_csv is not None:
                        purchase_rate_col = st.selectbox("Columna que contiene los precios de compra", options=price_profile.columns.to_numpy())
                        Price_Grid = price_profile.loc[:,[purchase_rate_col]]

                        data_model["grid"]["buy_price"] = {}
                        data_model["grid"]["buy_price"]["type"] = "variable"
                        data_model["grid"]["buy_price"]["value"] = Price_Grid[purchase_rate_col].to_numpy()*data_model["in_data_to_usd"]
                        data_model["grid"]["buy_price"]["len"] = len(Price_Grid)


                        with st.expander("Visualizar gráficos. Precio de compra"):
                            st.subheader("Métricas")                    

                            purchase_rate_metric = generate_metrics(price_profile, purchase_rate_col, currency_data + "/kWh")
                            st.dataframe(purchase_rate_metric)
                            
                            st.subheader("Promedio por hora")
                            purchase_rate_color = st.color_picker('Color gráfica precio compra', '#00f900')
                            st.altair_chart(createfig_meanhour(Price_Grid, purchase_rate_col, fechas, purchase_rate_color, currency_data + "/kWh").interactive(),use_container_width=True)

                            st.subheader("Precio de compra por hora y día del año")                    
                            st.altair_chart(createfig_heatmap(Price_Grid, purchase_rate_col, fechas, False, currency_data + "/kWh").interactive(), use_container_width=True)

                    else: 
                        st.info("Cargue el archivo un archivo csv con la serie temporal de precios de compra en una de sus columnas.")

            with cols[1]:

                
                if sell_rate_choice == "Valor constante":
                    Ppvusd = st.number_input("Precio de venta: " + currency_data + "/kWh")
                    data_model["grid"]["sell_price"] = {}
                    data_model["grid"]["sell_price"]["type"] = "fixed"
                    data_model["grid"]["sell_price"]["value"] = Ppvusd*data_model["in_data_to_usd"]
                    data_model["grid"]["sell_price"]["len"] = 0

                elif sell_rate_choice == "Variable - Serie temporal":
                    if price_profile_csv is not None:
                        sell_rate_col = st.selectbox("Columna que contiene los precios de venta", options=price_profile.columns.to_numpy())
                        Ppvusd = price_profile.loc[:,[sell_rate_col]]

                        data_model["grid"]["sell_price"] = {}
                        data_model["grid"]["sell_price"]["type"] = "variable"
                        data_model["grid"]["sell_price"]["value"] = Ppvusd[sell_rate_col].to_numpy()*data_model["in_data_to_usd"]
                        data_model["grid"]["sell_price"]["len"] = len(Ppvusd)
                        
                        with st.expander("Visualizar gráficos. Precio de venta"):
                            st.subheader("Métricas")                     
                                            
                            sell_rate_metric = generate_metrics(price_profile, sell_rate_col, currency_data + "/kWh")
                            st.dataframe(sell_rate_metric)

                    
                            st.subheader("Promedio por hora")
                            sell_rate_color = st.color_picker('Color gráfica precio venta', '#00f900')
                            st.altair_chart(createfig_meanhour(Ppvusd, sell_rate_col, fechas, sell_rate_color, currency_data + "/kWh").interactive(),use_container_width=True)

                            st.subheader("Precio de venta por hora y día del año")
                            st.altair_chart(createfig_heatmap(Ppvusd, sell_rate_col, fechas, False, currency_data + "/kWh").interactive(), use_container_width=True)

                    else: 
                        st.info("Cargue el archivo un archivo csv con la serie temporal de precios de venta en una de sus columnas.")

        elif price_input == "Históricos de operador de red colombiano":

            OR_df = openpyxl.load_workbook('OR_price.xlsx')
            OR_indice = pd.read_excel("OR_price.xlsx", header=12, usecols="B:E", sheet_name= "INDICE")

            OR_grid, gridOptions_OR = interactive_table(OR_indice, selection = "single", cat = False, editable = False)
            update_mode_value = GridUpdateMode.MODEL_CHANGED
            st.subheader("Operadores de red")
            st.write("A continuación encontrará el listado de las empresas sobre las cuales puede consultar el costo unitario y la tarifa de prestación del servicio.")
            grid_response_OR = AgGrid(OR_grid, gridOptions=gridOptions_OR, update_mode=update_mode_value, allow_unsafe_jscode=True, theme= "streamlit", fit_columns_on_grid_load=True)
            
            selected_OR = grid_response_OR['selected_rows']
            if len(selected_OR) > 0:
                OR_sheet = ""
                OR_num = list(selected_OR[0].values())[0]
                for i in OR_df.sheetnames:
                    n_temp =  re.findall(r'\d+', i)
                    if len(n_temp) > 0:                    
                        if OR_num == n_temp[0]:
                            OR_sheet = i

                price_comp = pd.read_excel("OR_price.xlsx", sheet_name=OR_sheet, usecols="A:J", header=5).dropna()
                price_user = pd.read_excel("OR_price.xlsx", sheet_name=OR_sheet, usecols="N:U", header=5).dropna()
                
                price_user.columns = ["Mes", "Año", "Periodo", "Estrato 1", "Estrato 2", "Estrato 3", "Estrato 4", "Estrato 5 y 6. Ind y Com"] 
                price_comp.columns= ["Mes" ,"Año", "Periodo", "GM", "TM", "PR", "D", "CV", "RM", "CU"]

                cols = st.columns(2)
                
                with cols[0]:
                    st.subheader("Precio de compra :inbox_tray:")
                    purchase_type = st.radio("¿Cómo se constituirá el precio de compra a la red?",["Promedio de los ultimos 12 meses", "Seleccionar precio de un mes específico", "Serie temporal mensual"])
                    level_price = st.selectbox("Seleccione el estrato o sector",price_user.columns[3:])

                    if purchase_type == "Promedio de los ultimos 12 meses":
                        AgGrid(price_user, theme='streamlit')
                        purchase_price = price_user[level_price].mean()
                        data_model["grid"]["buy_price"] = {}
                        data_model["grid"]["buy_price"]["type"] = "fixed"
                        data_model["grid"]["buy_price"]["value"] = purchase_price*data_model["cop_to_usd"]
                        data_model["grid"]["buy_price"]["len"] = 0
                                               
                        st.info(f'Precio promedio ultimos 12 meses para  {level_price}: **{purchase_price:.5f} COP/kWh** o **{data_model["grid"]["buy_price"]["value"]:.5f} USD/kWh**')
                        
                    elif purchase_type == "Seleccionar precio de un mes específico":

                        price_grid, gridOptions_price = interactive_table(price_user, selection = "single", cat = False, editable = False)
                        update_mode_value = GridUpdateMode.MODEL_CHANGED                        
                        st.write("Seleccione el mes del cual se desea extraer el precio para " + level_price)
                        grid_response_price = AgGrid(price_grid, gridOptions= gridOptions_price, update_mode=update_mode_value, allow_unsafe_jscode=True, theme= "streamlit")
                        
                        if len(grid_response_price['selected_rows']) > 0:
                            purchase_price = float(grid_response_price['selected_rows'][0][level_price])
                            
                            data_model["grid"]["buy_price"] = {}
                            data_model["grid"]["buy_price"]["type"] = "fixed"
                            data_model["grid"]["buy_price"]["value"] = purchase_price*data_model["cop_to_usd"]
                            data_model["grid"]["buy_price"]["len"] = 0

                            st.info(f'Precio seleccionado para {level_price}: **{purchase_price:.5f} COP/kWh** o **{data_model["grid"]["buy_price"]["value"]:.5f} USD/kWh**')
                            

                    elif purchase_type == "Serie temporal mensual":
                        
                        mes = fechas.month
                        purchase_price = np.zeros_like(mes)
                        AgGrid(price_user, theme='streamlit')
                        for i in mes:
                            purchase_price = np.where(mes == i, price_user[level_price].iloc[i-1], purchase_price)

                        data_model["grid"]["buy_price"] = {}
                        data_model["grid"]["buy_price"]["type"] = "variable"
                        data_model["grid"]["buy_price"]["value"] = purchase_price*data_model["cop_to_usd"]
                        data_model["grid"]["buy_price"]["len"] = len(purchase_price)

                        st.subheader("Precio de compra por hora y día del año")                    
                        st.altair_chart(createfig_heatmap(pd.DataFrame(data = {'Price': purchase_price}), 'Price', fechas, False, "COP/kWh").interactive(), use_container_width=True)
                        
                with cols[1]:
                    st.subheader("Precio de venta :outbox_tray:")
                    st.info("El precio de venta se define como: Ps = CU - CV")

                    sell_type = st.radio("¿Cómo se constituirá el precio de venta a la red?",["Promedio de los ultimos 12 meses", "Seleccionar precio de un mes específico", "Serie temporal mensual"])
                    
                    if sell_type == "Promedio de los ultimos 12 meses":
                        AgGrid(price_comp, theme='streamlit')
                        
                        sell_price = (price_comp["CU"] - price_comp["CV"]).mean()
                        data_model["grid"]["sell_price"] = {}
                        data_model["grid"]["sell_price"]["type"] = "fixed"
                        data_model["grid"]["sell_price"]["value"] = sell_price*data_model["cop_to_usd"]
                        data_model["grid"]["sell_price"]["len"] = 0
                        
                        st.info(f'Precio promedio ultimos 12 meses: **{sell_price:.5f} COP/kWh** o **{data_model["grid"]["sell_price"]["value"]:.5f} USD/kWh**')
                    
                    elif sell_type == "Seleccionar precio de un mes específico":

                        price_grid_sell, gridOptions_price_sell = interactive_table(price_comp, selection = "single", cat = False, editable = False)
                        update_mode_value = GridUpdateMode.MODEL_CHANGED                        
                        st.write("Seleccione el mes del cual se desea extraer el precio de venta  de las componentes CU y CV")
                        grid_response_price_sell = AgGrid(price_grid_sell, gridOptions= gridOptions_price_sell, update_mode=update_mode_value, allow_unsafe_jscode=True, theme= "streamlit")
                        
                        if len(grid_response_price_sell['selected_rows']) > 0:
                            sell_price = float(grid_response_price_sell['selected_rows'][0]["CU"]) - float(grid_response_price_sell['selected_rows'][0]["CV"])
                            
                            data_model["grid"]["sell_price"] = {}
                            data_model["grid"]["sell_price"]["type"] = "fixed"
                            data_model["grid"]["sell_price"]["value"] = sell_price*data_model["cop_to_usd"]
                            data_model["grid"]["sell_price"]["len"] = 0

                            st.info(f'Precio seleccionado: **{sell_price:.5f} COP/kWh** o **{data_model["grid"]["sell_price"]["value"]:.5f} USD/kWh**')

                    elif sell_type == "Serie temporal mensual":
                        
                        mes = fechas.month
                        sell_price = np.zeros_like(mes)
                        AgGrid(price_comp, theme='streamlit')
                        for i in mes:
                            sell_price = np.where(mes == i, price_comp["CU"].iloc[i-1] - price_comp["CV"].iloc[i-1], sell_price)

                        data_model["grid"]["sell_price"] = {}
                        data_model["grid"]["sell_price"]["type"] = "variable"
                        data_model["grid"]["sell_price"]["value"] = sell_price*data_model["cop_to_usd"]
                        data_model["grid"]["sell_price"]["len"] = len(sell_price)

                        st.subheader("Precio de venta por hora y día del año")                    
                        st.altair_chart(createfig_heatmap(pd.DataFrame(data = {'Price': sell_price}), 'Price', fechas, False, "COP/kWh").interactive(), use_container_width=True)
                        
         
    


        if load_profile_csv is not None:
            if data_model["load"]["reactive"]:
                
                st.markdown("""<hr style=" border-top: 2px double #5D7837;" /> """, unsafe_allow_html=True)
                st.header("Penalización por exceso de reactivos")
                if price_input == "Históricos de operador de red colombiano":
                    rec_sel = st.radio("Aplicar penalización por exceso de consumo de reactivos según: ", ["Resolución CREG 015 del 2018", "Parámetros del usuario"])
                else:
                    rec_sel = "Parámetros del usuario"

                if rec_sel == "Resolución CREG 015 del 2018":
                    if len(selected_OR) > 0:
                        st.subheader("Componente D (distribución) de " + selected_OR[0]["Empresa"] + " durante los últimos 12 meses")
                        st_echarts(options=createline_echart(price_comp, "Mes", "D", ["Componente D"], "Periodo", "COP/kWh", [], x_date = False, x_type = "category", data_zoom = False) , height="400px")
                        st.info("Esta serie será el costo utilizado mes a mes para penalizar los excedentes consumo de reactiva horarios mayores al 50% del consumo de potencia activa demandada de la red. Así mismo, será la penalización para las exportaciones de energía capacitiva por parte de la microrred a la red principal (Resolución CREG 015 del 2018. Capítulo 12).")
                        
                        mes = fechas.month
                        q_price = np.zeros_like(mes)
                        for i in mes:
                            q_price = np.where(mes == i, price_comp["D"].iloc[i-1], q_price)
                        
                        data_model["grid"]["q_price"] = {}
                        data_model["grid"]["q_price"]["type"] = "variable"
                        data_model["grid"]["q_price"]["value"] = q_price*data_model["cop_to_usd"]
                        data_model["grid"]["q_price"]["len"] = len(q_price)
                        data_model["grid"]["lim_q"] = 0.5 
                        st.subheader("Costo de penalización por exceso de reactivos. Gráfico por hora y día del año")                    
                        st.altair_chart(createfig_heatmap(pd.DataFrame(data = {'Price': q_price}), 'Price', fechas, False, "COP/kWh").interactive(), use_container_width=True)

                    else:
                        st.warning("Seleccione un operador de red en la sección de precios para continuar")
                elif rec_sel == "Parámetros del usuario":
                    cols = st.columns(2)
                    with cols[0]:
                        q_price = st.number_input("Costo de penalización por exceso de reactivos: " + currency_data + "/kWh")
                        data_model["grid"]["q_price"] = {}
                        data_model["grid"]["q_price"]["type"] = "fixed"
                        data_model["grid"]["q_price"]["value"] = q_price*data_model["in_data_to_usd"]
                        data_model["grid"]["q_price"]["len"] = 0
                    with cols[1]:
                        data_model["grid"]["lim_q"] = st.number_input("Porcentaje máximo de potencia activa horaria para uso de reactivos horarios sin penalización (%)", min_value=0.0, max_value=100.0, value = 50.0)/100
                    

        st.markdown("""<hr style=" border-top: 2px double #5D7837;" /> """, unsafe_allow_html=True)

        st.subheader("Límites de la red :no_entry:")

        cols = st.columns(2)
        with cols[0]:
            st.subheader("Potencia máxima de demanda de la red")
            data_model["grid"]["pmax_buy"] = st.number_input("Potencia máxima del transformador o acometida (kW)", min_value = 0, value = 50)            
        with cols[1]:  
            st.subheader("Potencia máxima de inyección a la red")      
            if components["address"]["country"] == "Colombia":
                ms_sel = st.radio("Potencia máxima de inyección a la red dada por: ", ["Resolución CREG 174 de 2021", "Usuario"])
            else: 
                ms_sel = "Usuario"
            
            if ms_sel == "Resolución CREG 174 de 2021":
                sys_type = st.selectbox("El sistema está conectado al transformador de distribucion : ", ["Independiente", "Con otros usuarios"])
                if sys_type == "Independiente":                
                    if load_profile_csv is not None:
                        min_load_mean = np.round(np.mean([np.min(data_model["load"]["value"][i:i+24]) for i in np.arange(0,len(data_model["load"]["value"]), 24)]), 4)

                        if min_load_mean > data_model["grid"]["pmax_buy"]*0.5:
                            st.info("Según el perfil de demanda de potencia activa ingresado, el promedio de potencia de las horas con minima demanda diaria es de: **" + str(min_load_mean) + "kW** el cual es mayor al 50% de la potencia maxima de compra de la red. Por tanto, el límite de inyecciones de potencia a la red será **el 50% de la potencia maxima de compra de la red**")
                            data_model["grid"]["pmax_sell"] = data_model["grid"]["pmax_buy"]*0.5
                        else:
                            st.info("Según el perfil de demanda de potencia activa ingresado, el promedio de potencia de las horas con minima demanda diaria es de: **" + str(min_load_mean) + "kW** el cual es menor al 50% de la potencia maxima de compra de la red. Por tanto, el límite de inyecciones de potencia a la red será **el promedio de potencia de las horas con minima demanda diaria**")
                            data_model["grid"]["pmax_sell"] = min_load_mean
                elif sys_type == "Con otros usuarios": 
                    data_model["grid"]["pmax_sell"] = st.number_input("Potencia máxima de inyección a la red según estudio de conexión (kW)", value = 0.5*data_model["grid"]["pmax_buy"])
            elif ms_sel == "Usuario":
                data_model["grid"]["pmax_sell"] = st.number_input("Potencia máxima de inyección a la red según estudio de conexión (kW)", value = 0.5*data_model["grid"]["pmax_buy"])

        #data_model["grid"]["pmax_sell"] = MaxPpvG


        st.markdown("""<hr style=" border-top: 2px double #5D7837;" /> """, unsafe_allow_html=True)

        st.subheader("Disponibilidad de la red :white_check_mark:")

        data_model["grid"]["av"] = {}

        av_choice = st.radio("Ingresar perfil de disponobilidad:", ["No", "Si"])
        
        if av_choice == "Si":        

            av_profile_csv = st.file_uploader("Cargar csv binario con (1: red disponible, 0: red no disponible)", type = ['csv'])
            if av_profile_csv is not None:

                data_model["grid"]["av"]["active"] = True

                av_profile = pd.read_csv(av_profile_csv)
                av_col = st.selectbox("Columna que contiene la disponibilidad de la red", options=av_profile.columns.to_numpy())
                av_grid = av_profile.loc[:,[av_col]]

                data_model["grid"]["av"]["value"] = av_grid[av_col].to_numpy()

                with st.expander("Visualizar gráficos. Disponibilidad de la red"):

                    st.subheader("Métricas")                     

                    av_grid_metric = generate_metrics_av(av_grid, av_col)

                    st.dataframe(av_grid_metric)          

                    st.subheader("Disponibilidad de la red por hora y día del año")
                    st.altair_chart(createfig_heatmap(av_grid, av_col, fechas, True, "[1,0]").interactive(), use_container_width=True)

            else: 
                data_model["grid"]["av"]["active"] = False

        elif av_choice == "No":

            data_model["grid"]["av"]["active"] = False

    else:
        data_model["grid"]["active"] = False


    
    PVtype, Battype, Gens, ConH, WindGens = load_cat()

    if ('Módulos fotovoltaicos' in options) or ('Sistema de almacenamiento (ESS)' in options):

        st.markdown("""<hr style="border:2px dashed Salmon;border-radius:5px;" /> """, unsafe_allow_html=True)

        st.header("Selección de inversores híbridos :electric_plug:")        
        
        st.write("En esta sección puedes seleccionar un inversor híbrido de nuestro catálogo o crear uno propio.")
        
        st.info("Si vas a crear uno debes conocer todas las especificaciones del producto.")
        
        data_model["inverters"] = {}

        df_grid_CH, gridOptions_CH = interactive_table(ConH)
        update_mode_value = GridUpdateMode.MODEL_CHANGED
        st.subheader("Catálogo de Inversores Híbridos")
        grid_response_CH = AgGrid(df_grid_CH, gridOptions=gridOptions_CH, update_mode=update_mode_value, allow_unsafe_jscode=True, theme= "blue")

        if len(grid_response_CH['selected_rows']) > 0:
            data_model["inverters"]["type"] = extract_table(ConH, grid_response_CH)
        if load_profile_csv is not None:
            if data_model["load"]["reactive"]:
                data_model["inverters"]["fp_set"] = st.number_input("Factor de potencia minimo del inversor", min_value=0.0, max_value=1.0, value = 0.85)

        
        inv_flex = st.radio("La microrred se puede componer por:", ["Una sola tecnología de inversor", "Varias tecnolgías de inversor"])
        if inv_flex == "Varias tecnolgías de inversor":
            data_model["inverters"]["flex"] = True
        else: 
            data_model["inverters"]["flex"] = False


    
    if 'Módulos fotovoltaicos' in options:

        st.markdown("""<hr style="border:2px dashed Salmon;border-radius:5px;" /> """, unsafe_allow_html=True)

        st.header("Selección de los módulos fotovoltaicos :large_blue_diamond:")
        
        st.write("En esta sección puedes seleccionar un módulo fotovoltaico de nuestro catálogo o crear uno propio.")
        
        st.info("Si vas a crear uno debes conocer todas las especificaciones del producto.")
        
        image_panel = Image.open('Imagenes/Panel.jpeg')

        st.image(image_panel, width = 400)
        
        data_model["pv_modules"] = {}

        df_grid_PV, gridOptions_PV = interactive_table(PVtype)
        update_mode_value = GridUpdateMode.MODEL_CHANGED
        st.subheader("Catálogo de módulos fotovoltaicos")
        grid_response_PV = AgGrid(df_grid_PV, gridOptions=gridOptions_PV, update_mode=update_mode_value, allow_unsafe_jscode=True, theme= "blue")

        cols = st.columns(2)

        with cols[0]:

            st.subheader("Orientación de los módulos")
            azimut = st.number_input("Azimut", 0, 359)
            inclinacion = st.number_input("Inclinación", 0, 90)
            st.pyplot(Grafica_panel(azimut, inclinacion))

        with cols[1]:

            if len(grid_response_PV['selected_rows']) > 0:
                data_model["pv_modules"]["type"] = extract_table(PVtype, grid_response_PV)
                PV_names = list(data_model["pv_modules"]["type"].columns)
                Temp_min = extract_tem_min(data_model["lat"], data_model["lon"])

                for k in PV_names:
                    data_model["pv_modules"]["type"].loc['Voc_max', k] = np.round(data_model["pv_modules"]["type"].loc['Voc_STC',k]*(1+(data_model["pv_modules"]["type"].loc['Tc_Voc',k]/100)*(Temp_min-25)),2)

                data_model["pv_modules"]["Pmpp"], df_irr = power_PV_calculation(df, data_model["pv_modules"]["type"], azimut, inclinacion,data_model["lat"])    
                data_model["pv_modules"]["Pmpp"].index.name = "Fecha"
                with st.expander("Desplegar cálculos de los módulos"):
                    
                    st.write("En esta sección podrás observar la potencia generada por el módulo fotovoltaico seleccionado y su respectiva eficiencia a lo largo del año bajo estudio.")
                    
                    #color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(len(PV_names))]
                    Pmpp_graf = data_model["pv_modules"]["Pmpp"].copy()
                    Pmpp_graf["Fecha"] = fechas 
                    st_echarts(options=createline_echart(Pmpp_graf.copy(), "Fecha", PV_names, PV_names, "Fecha", "kW", [], x_date = True) , height="400px")

                    n =  data_model["pv_modules"]["Pmpp"].copy()         
                    n = n.reset_index()
                    for k in PV_names:
                        
                        a = data_model["pv_modules"]["Pmpp"][k].to_numpy()
                        b = data_model["pv_modules"]["type"].loc["A",k]*df_irr["IRR"].to_numpy()/1000
                        n[k] = np.divide(a, b, out=np.zeros_like(a), where=b!=0)*100 
                        
                    #st_echarts(options=createline_echart(n.copy(), "Fecha", PV_names, PV_names, "Fecha", "%", [], x_date = True) , height="400px")
                    
    if ('Sistema de almacenamiento (ESS)' in options):

        st.markdown("""<hr style="border:2px dashed Salmon;border-radius:5px;" /> """, unsafe_allow_html=True)
        
        st.header("Selección de los almacenadores :battery:")
        
        st.write("En esta sección puedes seleccionar un almacenador de nuestro catálogo o crear uno propio.")
        
        st.info("Si vas a crear uno debes conocer todas las especificaciones del producto.")
        
        data_model["batteries"] = {}

        df_grid_BT, gridOptions_BT = interactive_table(Battype)
        update_mode_value = GridUpdateMode.MODEL_CHANGED
        
        st.subheader("Catálogo de almacenadores")
        grid_response_BT = AgGrid(df_grid_BT, gridOptions=gridOptions_BT, update_mode=update_mode_value, allow_unsafe_jscode=True, theme= "blue")

        if len(grid_response_BT['selected_rows']) > 0:
            data_model["batteries"]["type"] = extract_table(Battype, grid_response_BT)
    

    if 'Aerogenerador'in options:

        st.markdown("""<hr style="border:2px dashed Salmon;border-radius:5px;" /> """, unsafe_allow_html=True)    
        st.header("Selección de generadores eólicos :wind_chime:")    
        
        st.write("En esta sección puedes seleccionar un aerogenerador de nuestro catálogo o crear uno propio.")
        
        st.info("Si vas a crear uno debes conocer todas las especificaciones del producto.")
        
        st.subheader("Catálogo de generadores eólicos")   
    
        
        data_model["windgen"] = {}        
        data_model["windgen"]["active"] = True

        df_grid_WT, gridOptions_WT = interactive_table(WindGens)
        update_mode_value = GridUpdateMode.MODEL_CHANGED
        
        st.subheader("Catálogo de almacenadores")
        grid_response_WT = AgGrid(df_grid_WT, gridOptions=gridOptions_WT, update_mode=update_mode_value, allow_unsafe_jscode=True, theme= "blue")
                
        height_wt = st.number_input("Altura de instalación de las turbinas eólicas (m).", min_value= 0.0, max_value=None, value = 20.0)
        z0 = float(st.text_input("Longitud de rugosidad superficial (m)", value = "0.001"))
        
        if len(grid_response_WT['selected_rows']) > 0: 
            data_model["windgen"]["type"] = extract_table(WindGens, grid_response_WT)    
            Profiles, Wind_generation = calculate_WT_power(df, data_model["windgen"]["type"], z0, height_wt, info['Elevation'].iloc[0])               
            data_model["windgen"]["generation"] = Wind_generation 

            with st.expander("Curva de generación y producción anual"):
                fig_choice_wind = st.selectbox("Generador eólico a visualizar", data_model["windgen"]["type"].columns)
                st.subheader("Producción por hora del aerogenerador " + fig_choice_wind)
                st.altair_chart(createfig_heatmap(Wind_generation, fig_choice_wind, fechas, False, "kW").interactive(), use_container_width=True)
                
                Wind_Graf = Wind_generation.copy()
                Wind_Graf["Fecha"] = fechas
                
                fig_prod = alt.Chart(Wind_Graf).mark_line().encode(alt.X('Fecha:T'),alt.Y(fig_choice_wind+':Q', axis=alt.Axis(title="kW")), color=alt.value("#4389F6"))
                st.altair_chart(fig_prod.interactive(),use_container_width=True)
                
                st.subheader("Curva de generación: " + fig_choice_wind)                                    
                fig_windturbine = alt.Chart(Profiles[fig_choice_wind].reset_index()).mark_line().encode(alt.X('wind_speed:Q',axis=alt.Axis(title="Velocidad del viento (m/s)")),alt.Y('power:Q', axis=alt.Axis(title="Potencia (kW)")), color=alt.value("#E8AD0E"))
                st.altair_chart(fig_windturbine.interactive(),use_container_width=True)
    else: 
        data_model["windgen"] = {}
        data_model["windgen"]["active"] = False
    

    if 'Generador' in options:
        st.markdown("""<hr style="border:2px dashed Salmon;border-radius:5px;" /> """, unsafe_allow_html=True)    
        st.header("Generador de respaldo :red_circle:")
        
        
        data_model["generator"] = {}
        data_model["generator"]["active"] = True
        #data_model["generator"]["type"] = Gens.set_index('ID', drop = True)[[generadores]]
        data_model["generator"]["pmax"] = st.number_input("Potencia maxima: kW", min_value=0.0, max_value=None, value = 10.0)
        data_model["generator"]["fmin"] = st.number_input("Consumo de combustible a potencia 0: L/h", min_value=0.0, max_value=None, value = 1.0)
        data_model["generator"]["fmax"] = st.number_input("Consumo de combustible a máxima potencia: L/h", min_value=0.0, max_value=None, value = 18.0)
        data_model["generator"]["fuel_cost"] = st.number_input("Costo del litro de combustible: " + currency_data + "/L", min_value=0.0, max_value=None, step = 0.1, value = 0.5934)*data_model["in_data_to_usd"]
        data_model["generator"]["gen_cost"] = st.number_input("Costo de instalación: " + currency_data, min_value=0.0, max_value=None, step = 1.0, value = 0.0)*data_model["in_data_to_usd"]
        data_model["generator"]["gen_OM_cost"] = st.number_input("Costo de OM: " + currency_data + "/h", min_value=0.0, max_value=None, step = 1.0, value = 1.0)*data_model["in_data_to_usd"]          
        data_model["generator"]["min_p_load"] = st.number_input("Porcentaje de carga mínimo para funcionar: (%)", min_value=0, max_value=None, value = 10)
        data_model["generator"]["fm"] = (data_model["generator"]["fmax"] - data_model["generator"]["fmin"]) / data_model["generator"]["pmax"]

        st.subheader("Disponibilidad del generador :white_check_mark:")

        data_model["generator"]["av"] = {}

        av_gen_choice = st.radio("Ingresar perfil de disponobilidad del generador:", ["No", "Si"])
        
        if av_gen_choice == "Si":        

            av_gen_profile_csv = st.file_uploader("Cargar csv binario con (1: generador disponible, 0: generador no disponible)", type = ['csv'])
            if av_gen_profile_csv is not None:

                data_model["generator"]["av"]["active"] = True

                av_gen_profile = pd.read_csv(av_gen_profile_csv)
                av_gen_col = st.selectbox("Columna que contiene la disponibilidad del generador", options=av_gen_profile.columns.to_numpy())
                av_gen = av_gen_profile.loc[:,[av_gen_col]]

                data_model["generator"]["av"]["value"] = av_gen[av_gen_col].to_numpy()

                with st.expander("Visualizar gráficos. Disponibilidad del generador"):

                    st.subheader("Métricas")                     

                    av_gen_metric = generate_metrics_av(av_gen, av_gen_col)

                    st.dataframe(av_gen_metric)

                    st.subheader("Disponibilidad de la red por hora y día del año")
                    st.altair_chart(createfig_heatmap(av_gen, av_gen_col, fechas, True, "[1,0]").interactive(), use_container_width=True)

            else: 
                data_model["generator"]["av"]["active"] = False

        elif av_gen_choice == "No":
            data_model["generator"]["av"]["active"] = False

    else: 
        data_model["generator"] = {}
        data_model["generator"]["active"] = False

    st.markdown("""<hr style="height:7px;border-radius:5px;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)

    st.header("Restricciones y condiciones adicionales del modelo :lock:")

    cols = st.columns(2)

    with cols[0]:
        max_invest_bool = st.checkbox("Restricción de máxima inversión inicial")
        if max_invest_bool:
            max_invest = st.number_input("Ingrese la máxima inversión inicial en " + currency_data, min_value=0.0, max_value=None, step = 1.0, value = 1000.0)

            data_model["max_invest"] = {}
            data_model["max_invest"]["active"] = True
            data_model["max_invest"]["value"] = max_invest*data_model["in_data_to_usd"]

        else:

            data_model["max_invest"] = {}
            data_model["max_invest"]["active"] = False




        st.markdown("""<hr style="border:2px dashed Salmon;border-radius:5px;" /> """, unsafe_allow_html=True)

        bono_bool = st.checkbox("Incentivo por bono de disminuciones de dioxido de carbono")
        if bono_bool:
            data_model["environment"] = {}
            data_model["environment"]["active"] = True

            st.info("Ingrese la información del ahorro específico de gramos CO2 por kWh que se da al no utilizar la red principal de servicios públicos y el incentivo en " + currency_data + " por Tonelada de CO2 que se obtendría.")
            data_model["environment"]["mu"] = st.number_input("Ahorro específico gCO2/kWh", min_value = 0.0, max_value=None, step = 0.1, value = 370.00)
            data_model["environment"]["Cbono"] = st.number_input("Incentivo "+ currency_data + "/TmCO2", min_value = 0.0, max_value=None, step = 0.1, value = 4.647)*data_model["in_data_to_usd"]
        else:
            data_model["environment"] = {}
            data_model["environment"]["active"] = False

    with cols[1]:
        Av_area_bool = st.checkbox("Restricción del área disponible para instalación de paneles")
        if Av_area_bool:
            st.subheader("Dimensiones del área disponible")
            Area = st.number_input("Área disponible (m^2)", min_value = 2.0, max_value=None, step = 1.0, value = 175.0)
            

            data_model["area"] = {}
            data_model["area"]["active"] = True
            data_model["area"]["value"] = Area

        else:

            data_model["area"] = {}
            data_model["area"]["active"] = False

    st.markdown("""<hr style="height:7px;border-radius:5px;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)

    colsrun = st.columns(2)

    with colsrun[0]:

        st.subheader("Crear modelo de optimización")

        if st.button("Crear modelo"):
            model_dict = {'data_model':data_model, 'model': create_model(data_model)}
            with open('test.pkl', mode='wb') as file:
                cloudpickle.dump(model_dict, file)

    with colsrun[1]:
        st.subheader("Optimizar modelo")
        
        solver = st.selectbox("Solver: ", ["gurobi","cplex","glpk"])
        use_neos = st.checkbox("Utilizar servidor NEOS")
        if use_neos:
            email_neos = st.text_input("Ingrese su correo electrónico para recibir los resultados.")

        if st.button("Optimizar"):
            
            if use_neos:
                if email_neos.find("@") != -1:
                    os.environ['NEOS_EMAIL'] = email_neos
                    with open('test.pkl', mode='rb') as file:
                        model_dict = cloudpickle.load(file)
                        model = model_dict["model"]
                        results = pym.SolverManagerFactory('neos').solve(model, opt = solver)
                        model_dict["model"] = model
                        
                    with open('test.pkl', mode='wb') as file:
                        cloudpickle.dump(model_dict, file)

                else:
                    st.error("Email inválido")
            else:
                if solver == "gurobi":
                    solver = pym.SolverFactory(solver)
                    solver.options['mipgap'] = 1e-2
                elif solver == "cplex":
                    solver = pym.SolverFactory(solver)
                    solver.options['mip_tolerances_mipgap'] = 1e-2
                with open('test.pkl', mode='rb') as file:
                        model_dict = cloudpickle.load(file)
                        model = model_dict["model"]
                        results = solver.solve(model, tee=True, keepfiles=True) 
                        model_dict["model"] = model
                        
                with open('test.pkl', mode='wb') as file:
                    cloudpickle.dump(model_dict, file)

                
    #st.json(data_model)

    #st.json(data_model)

elif menu_select == menu_options[1]:

    st.sidebar.title("Visualizar Resultados")

    #st.markdown("""<hr style="height:7px;border-radius:5px;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)


    st.sidebar.subheader("Carga del archivo de resultados")
    results_model = st.sidebar.file_uploader("Cargue el archivo .pkl generado con los resultados")
    if results_model is not None:
        model_dict = read_model(results_model)

        model = model_dict["model"]        
        
        data_model = model_dict["data_model"]

        st.sidebar.header("Divisas :currency_exchange:")   

        currency_results = st.sidebar.selectbox("Divisa de visualización de resultados", option_symbols, int(np.where(np.array(option_symbols)=='USD : United States Dollar')[0][0]))[0:3]
        
        try:      
        
            exchange_data = get_exchangerate("USD", currency_results)
            data_model["usd_to_results"] = exchange_data["USD_" + currency_results]
            data_model["currency_results"] = currency_results

            if currency_results != "USD": 
                for i,j in zip(exchange_data.keys(),exchange_data.values()):    
                    st.metric(i[0:3], str(round(j,5)) + " " + i[4:])
        except:
            st.sidebar.error("Error con el servidor de divisas. Solo es posible crear el modelo en dolares")
            data_model["usd_to_results"] = 1
            data_model["currency_results"] = "USD"

        

        st.sidebar.subheader("Ubicación del proyecto")

        st.sidebar.map(data=pd.DataFrame(data = {'lat':[data_model["lat"]],'lon':[data_model["lon"]]}), zoom=12, use_container_width=True)

        st.header("Visualizar resultados de:")

        options_vis = ["Análisis económico","Número de equipos", "Alimentación de la carga"]

        if data_model["generator"]["active"]:
            options_vis.append("Generador diesel")

        visualizar_op = st.selectbox("Elija una opción",options_vis)

        st.markdown("""<hr style="height:5px;border-radius:5px;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)

        if visualizar_op == "Análisis económico":
            res_ec, vpn_tab, nom_tab = results_economic(model, data_model)

            cols = st.columns(3)

            vpn = vpn_tab.iloc[-1,-1]            

            cols[0].metric("Valor Presente Neto (VPN)", f'{vpn:.2f} {currency_results}', "Proyecto inviable" if vpn < 0 else "Proyecto viable", "inverse" if vpn < 0 else "normal")
            cols[1].metric("Inversión incial", f'{res_ec["Capin"]:.2f} {currency_results}')
            cols[2].metric("Costos de OM anuales", f'{res_ec["COM"]:.2f} {currency_results}')

            cols[0].metric("Costos anuales de compra de energía", f'{res_ec["Ce"]:.2f} {currency_results}')
            cols[1].metric("Costos anuales por energía no suministrada", f'{res_ec["Cens"]:.2f} {currency_results}')
            cols[2].metric("Costos anuales totales", f'{res_ec["Cy"]:.2f} {currency_results}')
            
            cols[0].metric("Recaudos anuales por bonos ambientales", f'{res_ec["Renv"]:.2f} {currency_results}')
            cols[1].metric("Recaudos anuales por venta de energía", f'{res_ec["Re"]:.2f} {currency_results}')
            cols[2].metric("Recaudos anuales totales", f'{res_ec["Ry"]:.2f} {currency_results}')

            cols[1].metric("Ahorros anuales en energía", f'{res_ec["Ay"]:.2f} {currency_results}')

            

            st.markdown("""<hr style="border:2px dashed Salmon;border-radius:5px;" /> """, unsafe_allow_html=True)
            
            cols = st.columns(2)

            cols[0].metric("Tasa de descuento", f'{data_model["interest"]*100} %')
            cols[1].metric("Vida útil del proyecto", f'{data_model["lifeyears"]} años')

            Cash_flow_red = [nom_tab["Total"].iloc[0],]

            for i in np.arange(1,data_model["lifeyears"]+1):
                Cash_flow_red.append(np.round(Cash_flow_red[i-1] + nom_tab["Total"].iloc[i]))

            st.header("Flujo de caja nominal")

            fig = px.bar(x = np.arange(0,data_model["lifeyears"]+1), y = Cash_flow_red)
            fig.update_layout(yaxis=dict(title=currency_results, titlefont_size=16, tickfont_size=14), xaxis=dict(title='Año', titlefont_size=16, tickfont_size=14, tickmode = 'linear', tick0 = 0, dtick = 1))

            fig.update_traces(marker_color='rgb(14,110,232)', marker_line_color='rgb(8,48,107)',
                            marker_line_width=1.5, opacity=0.6)

            import numpy_financial as npf
            st.subheader("Tasa interna de retorno: " + str(np.round(npf.irr(nom_tab["Total"].iloc[:-1].values)*100,2)) + "%")
            
            st.plotly_chart(fig, use_container_width=True)

            st.header("Tabla de amortización VPN")
            

            st.write(vpn_tab.to_html(justify = "center"), unsafe_allow_html=True)
            
           


        elif visualizar_op == "Número de equipos":
            
            
            num_eq = results_num_equipment(model, data_model)
            
            st.subheader("Agrupación de equipos por inversor")            
            st.write(num_eq["summary"].to_html(justify = "center"), unsafe_allow_html=True)

            #st.markdown("""<hr style="height:5px;border-radius:5px;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)
            
            st.markdown("""<hr style="border:2px dashed Salmon;border-radius:5px;" /> """, unsafe_allow_html=True)

            cols = st.columns(2)

            

            cost_list_eq = {}

            with cols[0]:
                
                cost_list_eq["Inversores"] = num_eq["CH"]["Total: " + data_model["currency_results"]].sum()
                st.subheader("Número de inversores híbridos")
                st.write(num_eq["CH"].to_html(justify = "center"), unsafe_allow_html=True)
                st.metric("Costo total de inversores híbridos", f'{cost_list_eq["Inversores"]:.3f} {currency_results}')  

                if data_model["windgen"]["active"]:

                    cost_list_eq["Turbinas eólicas"] = num_eq["WT"]["Total: " + data_model["currency_results"]].sum()
                    st.subheader("Número de turbinas eólicas")
                    st.write(num_eq["WT"].to_html(justify = "center"), unsafe_allow_html=True)
                    st.metric("Costo total de turbinas eólicas", f'{cost_list_eq["Turbinas eólicas"]:.3f} {currency_results}')  
            
            with cols[1]:
                
                cost_list_eq["PV"] = num_eq["PVT"]["Total: " + data_model["currency_results"]].sum()
                st.subheader("Número de paneles solares")
                st.write(num_eq["PVT"].to_html(justify = "center"), unsafe_allow_html=True)
                st.metric("Costo total de paneles solares", f'{cost_list_eq["PV"]:.3f} {currency_results}')  

                cost_list_eq["Baterías"] = num_eq["BATT"]["Total: " + data_model["currency_results"]].sum()
                st.subheader("Número de baterías")
                st.write(num_eq["BATT"].to_html(justify = "center"), unsafe_allow_html=True)
                st.metric("Costo total de baterías", f'{cost_list_eq["Baterías"]:.3f} {currency_results}')    

            pie_fig_df = pd.DataFrame(data = {"Equipo":list(cost_list_eq.keys()), "Costo": list(cost_list_eq.values())})
            
            st.markdown("""<hr style="border:2px dashed Salmon;border-radius:5px;" /> """, unsafe_allow_html=True)
            
            cols = st.columns(2)

            with cols[0]:

                st.subheader("Porcentaje de inversión")

                fig1, ax1 = plt.subplots()
                ax1.pie(pie_fig_df["Costo"], labels=pie_fig_df["Equipo"], autopct='%1.1f%%',
                        shadow=True, startangle=90)
                ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

                st.pyplot(fig1)

            with cols[1]:

                st.subheader("Inversión en cada activo")

                fig = plt.figure()
 

                plt.bar(pie_fig_df["Equipo"], pie_fig_df["Costo"], color=['orange', 'red', 'green', 'blue', 'cyan'], width = 0.4)
                
                #plt.xlabel("Courses offered")
                plt.ylabel(data_model["currency_results"])
                
                plt.grid()
                st.pyplot(fig)

        elif visualizar_op == "Generador diesel":

            cols = st.columns(3)
            
            fuel_con_gen = np.round(sum(model.FuelCost*(model.GenFmin*model.GenOn[t].value + model.GenFm*model.PD[t].value) for t in model.T))
            cost_fuel_gen = np.round(model.FuelCost*fuel_con_gen*data_model["usd_to_results"], 3)
            cost_OM_gen = np.round(sum(model.GenOMCost*model.GenOn[t].value for t in model.T)*data_model["usd_to_results"], 3)
            
            GenOn = pd.DataFrame(data = {"GenOn":[model.GenOn[t].value for t in model.T]})
            PD = pd.DataFrame(data = {"PD":[model.PD[t].value for t in model.T]})
            with cols[0]:
                st.metric("Consumo de litros de combustible al año: ", f'{fuel_con_gen} L')
                st.metric("Costo de instalación: ", f'{model.GenCost.value} {currency_results}')

            with cols[1]:
                st.metric("Costo anual por uso de combustible", f'{cost_fuel_gen} {currency_results}')
                st.metric("Número de horas de operación anual: ", f'{np.sum(GenOn["GenOn"])} h')
            with cols[2]:
                st.metric("Costo OM anual", f'{cost_OM_gen} {currency_results}')
                st.metric("Cantidad de energía entregada: ", f'{np.round(np.sum(PD["PD"]))} kWh')
            
            st.subheader("Franjas de uso del generador diesel")
            st.altair_chart(createfig_heatmap(GenOn, "GenOn", fechas, True, "[1,0]").interactive(), use_container_width=True)

            st.subheader("Potencia generada en cada franja por el generador diesel")
            st.altair_chart(createfig_heatmap(PD, "PD", fechas, False, "kW").interactive(), use_container_width=True)


        elif visualizar_op == "Alimentación de la carga":    

            x = model.T
            fig = go.Figure(go.Bar(x=x, y=[model.PD[t].value for t in model.T], name='Diesel'))
            fig.add_trace(go.Bar(x=x, y=[model.PTL[t].value for t in model.T], name='Wind Turbine'))
            fig.add_trace(go.Bar(x=x, y=[model.PGL[t].value for t in model.T], name='Grid'))
            fig.add_trace(go.Bar(x=x, y=[model.ENS[t].value for t in model.T], name='ENS'))
            fig.add_trace(go.Bar(x=x, y=[sum(model.ConH['n_dcac',tch]*model.PpvL[tch,t].value for tch in model.CH) for t in model.T], name='PV'))
            fig.add_trace(go.Bar(x=x, y=[sum(model.PBL[tch,tb,t].value for tb in model.BATT for tch in model.CH) for t in model.T], name='Batteries'))
            fig.update_layout(barmode='stack', xaxis={'categoryorder':'category ascending'})
            

            st.subheader("Aporte de cada fuente de energía a la carga")
            
            
            st.plotly_chart(fig, use_container_width=True)

    
            #fig_res_carga = alt.Chart(data_fig).mark_bar().encode(x='Hora:Q',  y='sum(kW):Q', color='Aporte:N')    
            
            #st.altair_chart((fig_res_carga).interactive(), use_container_width=True)


st.sidebar.image(Image.open('Imagenes/UNAL.png'))