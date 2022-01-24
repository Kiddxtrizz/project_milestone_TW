import streamlit as st
import plotly.express as px
from datetime import datetime
import pandas as pd
import requests
from streamlit import caching
from dotenv import load_dotenv
import os

load_dotenv()
API_KEY = os.getenv('API_KEY')

BASE_URL = 'https://www.alphavantage.co/query?function=TIME_SERIES_WEEKLY&symbol=IBM&apikey={}'.format(API_KEY)



st.set_page_config(
    page_title = "My First Web App XD",
    page_icon = "random",
    layout = "wide",
    initial_sidebar_state = "expanded"
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

    last_rfsh = data['Meta Data']['3. Last Refreshed']
    symbl = data['Meta Data']['2. Symbol']
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.write('Last Refresh: {}'.format(last_rfsh))
    
    with c2:
        st.write('Current Symbol: {}'.format(symbl))
        
    df = pd.DataFrame.from_dict(data['Weekly Time Series'],orient='index')
    
    df['date'] = pd.to_datetime(df.index)
    df[['Year', 'Week', 'Day']] = df['date'].dt.isocalendar()
    
    st.sidebar.markdown('### Filter Shelf')
    
    Year = st.sidebar.selectbox("Year filter", df['Year'].unique()) 
    
    temp_df = df[df['Year'] == Year]
    
    fig = px.line(temp_df, x=temp_df.index, y='1. open')
    st.plotly_chart(fig)
else:
    url = 'https://www.alphavantage.co/query?function=TIME_SERIES_WEEKLY&symbol={}&apikey={}'.format(search,API_KEY)
    r = requests.get(url)
    
    data = r.json()
    
    try:
        last_rfshed = data['Meta Data']['3. Last Refreshed']
        symbl = data['Meta Data']['2. Symbol']
        
        c11, c22, c33 = st.columns(3)
    
        with c11:
            st.write('Last Refresh: {}'.format(last_rfshed))

        with c22:
            st.write('Current Symbol: {}'.format(symbl))
        
        
        df = pd.DataFrame.from_dict(data['Weekly Time Series'],orient='index')

        df['date'] = pd.to_datetime(df.index)
        df[['Year', 'Week', 'Day']] = df['date'].dt.isocalendar()

        st.sidebar.markdown('### Filter Shelf')

        Year = st.sidebar.selectbox("Year filter", df['Year'].unique()) 

        temp_df = df[df['Year'] == Year]

        fig = px.line(temp_df, x=temp_df.index, y='1. open')
        st.plotly_chart(fig)
    except (KeyError,NameError):
        st.error("Symbol does not exisit")
        

