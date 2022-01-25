import streamlit as st
import plotly.express as px
from datetime import datetime
import pandas as pd
import requests
from streamlit import caching
from dotenv import load_dotenv
import cufflinks as cf 
import os

load_dotenv()
API_KEY = os.getenv('API_KEY')

BASE_URL = 'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=IBM&apikey={}'.format(API_KEY)
QUOTE_URL = "https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol=IBM&apikey={}".format(API_KEY)


st.set_page_config(
    page_title = "My First Web App XD",
    page_icon = "🔌 ",
    layout = "wide",
    initial_sidebar_state = "auto"
)


@st.cache()
def get_data(url):
    r = requests.get(url)
    data = r.json()
    
    return data

st.empty()

st.title("Milestone Project: My Stock Ticker")
st.markdown("**Created By: Trey W.**")
st.sidebar.title("Navigation")
search = st.sidebar.text_input("Search")

if search == '':
    data = get_data(BASE_URL)
    quote = get_data(QUOTE_URL)

    last_rfsh = data['Meta Data']['3. Last Refreshed']
    symbl = data['Meta Data']['2. Symbol']
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.write('Last Refresh: {}'.format(last_rfsh))
    
    with c2:
        st.write('Current Symbol: {}'.format(symbl))
        
    df = pd.DataFrame.from_dict(data['Time Series (Daily)'],orient='index')
    df = df.sort_index()
    
    df['date'] = pd.to_datetime(df.index)
    df[['Year', 'Week', 'Day']] = df['date'].dt.isocalendar()
    df['Month'] = df['date'].dt.month
    df['Month Name'] = df['date'].dt.month_name()
    
    st.sidebar.markdown('### Filter Shelf')
    
    Year = st.sidebar.selectbox("Year filter", df['Year'].unique())
    SMA = st.sidebar.slider("Moving Average Filter", min_value=0, max_value=100, value=14, step = 1)
    RSI = st.sidebar.slider("Relative Strength Indicator Filter", min_value=0, max_value=100, value=14, step = 1)
    BOLLB = st.sidebar.slider("Bollinger Bands Filter", min_value=0, max_value=100, value=14, step = 1)
#     Month = st.sidebar.selectbox("Month filter", df['Month Name'].unique())
    with st.container():
        
        temp_df = df[(df['Year'] == Year)]
        temp_df['1. open'] = temp_df['1. open'].astype(float).astype(int)
        temp_df['2. high'] = temp_df['2. high'].astype(float).astype(int)
        temp_df['3. low'] = temp_df['3. low'].astype(float).astype(int)
        temp_df['4. close'] = temp_df['4. close'].astype(float).astype(int)
        temp_df['5. volume'] = temp_df['5. volume'].astype(float).astype(int)

        qf = cf.QuantFig(temp_df, name=symbl)
        qf.add_sma(periods=SMA, column='2. high', color='red')
        qf.add_rsi(periods=RSI, color='green')
        qf.add_bollinger_bands(periods=BOLLB
                               ,boll_std=2 
                               ,colors=['orange','grey']
                               , fill=True)
        qf.add_volume()
        qf.add_macd()

        fig = qf.iplot(asFigure=True)

    #     fig = px.line(temp_df, x=temp_df.index, y='2. high')
    #     fig.update_xaxes(rangeslider_visible=True)

        st.plotly_chart(fig, use_container_width=True)
        
        tmp_open = round(int(float(quote['Global Quote']['02. open'])),2)
        tmp_high = round(int(float(quote['Global Quote']['03. high'])),2)
        tmp_low = round(int(float(quote['Global Quote']['04. low'])),2)
        tmp_price = round(int(float(quote['Global Quote']['05. price'])),2)
        tmp_vol = quote['Global Quote']['06. volume']
        tmp_prev_cls = round(int(float(quote['Global Quote']['08. previous close'])),2)
        tmp_change = quote['Global Quote']['09. change']
        tmp_chngpct = quote['Global Quote']['10. change percent']
        tmp_lst_trd_day = quote['Global Quote']['07. latest trading day']

        c1, c2, c3, c4, c5, c6 = st.columns(6)

        c1.metric("Price", tmp_price)
        c2.metric("Open", tmp_open)
        c3.metric("High", tmp_high)
        c4.metric("Low", tmp_low)
        c5.metric("Volume", tmp_vol)
        c6.metric("Previous Close", tmp_prev_cls, tmp_change)
        

    
else:
    url = 'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={}&apikey={}'.format(search,API_KEY)
    q_url = "https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={}&apikey={}".format(search,API_KEY)
    r = requests.get(url)
    ur = requests.get(q_url)
    
    data = r.json()
    quote = ur.json()
    
    try:
        last_rfshed = data['Meta Data']['3. Last Refreshed']
        symbl = data['Meta Data']['2. Symbol']
        
        c11, c22, c33 = st.columns(3)
    
        with c11:
            st.write('Last Refresh: {}'.format(last_rfshed))

        with c22:
            st.write('Current Symbol: {}'.format(symbl))
        
        
        df = pd.DataFrame.from_dict(data['Time Series (Daily)'],orient='index')
        df = df.sort_index()

        df['date'] = pd.to_datetime(df.index)
        df[['Year', 'Week', 'Day']] = df['date'].dt.isocalendar()
        df['Month'] = df['date'].dt.month
        

        st.sidebar.markdown('### Filter Shelf')

        Year = st.sidebar.selectbox("Year filter", df['Year'].unique())
        SMA = st.sidebar.slider("Moving Average Filter", min_value=0, max_value=100, value=14, step = 1)
        RSI = st.sidebar.slider("Relative Strength Indicator Filter", min_value=0, max_value=100, value=14, step = 1)
        BOLLB = st.sidebar.slider("Bollinger Bands Filter", min_value=0, max_value=100, value=14, step = 1)
#     Month = st.sidebar.selectbox("Month filter", df['Month Name'].unique())
        with st.container():
            temp_df = df[(df['Year'] == Year)]
            temp_df['1. open'] = temp_df['1. open'].astype(float).astype(int)
            temp_df['2. high'] = temp_df['2. high'].astype(float).astype(int)
            temp_df['3. low'] = temp_df['3. low'].astype(float).astype(int)
            temp_df['4. close'] = temp_df['4. close'].astype(float).astype(int)
            temp_df['5. volume'] = temp_df['5. volume'].astype(float).astype(int)

            qf = cf.QuantFig(temp_df, name=symbl)
            qf.add_sma(periods=SMA, column='2. high', color='red')
            qf.add_rsi(periods=RSI, color='green')
            qf.add_bollinger_bands(periods=BOLLB
                                   ,boll_std=2 
                                   ,colors=['orange','grey']
                                   , fill=True)
            qf.add_volume()
            qf.add_macd()

            fig = qf.iplot(asFigure=True)
            st.plotly_chart(fig, use_container_width=True)

    #     fig = px.line(temp_df, x=temp_df.index, y='2. high')
    #     fig.update_xaxes(rangeslider_visible=True)
            tmp_open = round(int(float(quote['Global Quote']['02. open'])),2)
            tmp_high = round(int(float(quote['Global Quote']['03. high'])),2)
            tmp_low = round(int(float(quote['Global Quote']['04. low'])),2)
            tmp_price = round(int(float(quote['Global Quote']['05. price'])),2)
            tmp_vol = quote['Global Quote']['06. volume']
            tmp_prev_cls = round(int(float(quote['Global Quote']['08. previous close'])),2)
            tmp_change = quote['Global Quote']['09. change']
            tmp_chngpct = quote['Global Quote']['10. change percent']
            tmp_lst_trd_day = quote['Global Quote']['07. latest trading day']

            c1, c2, c3, c4, c5, c6 = st.columns(6)

            c1.metric("Price", tmp_price)
            c2.metric("Open", tmp_open)
            c3.metric("High", tmp_high)
            c4.metric("Low", tmp_low)
            c5.metric("Volume", tmp_vol)
            c6.metric("Previous Close", tmp_prev_cls, tmp_change)
        

        
    except (KeyError,NameError):
        st.error("Symbol does not exisit")
        

