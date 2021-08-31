import streamlit as st
import plotly.express as px
from datetime import datetime
import pandas as pd
import numpy as np
import requests


st.set_page_config(
    page_icon = "random",
    layout = "wide"
)



base_url = 'https://www.alphavantage.co/query?function=TIME_SERIES_WEEKLY&symbol=IBM&apikey=235NZQZTP0UKSBEQ'
new_url = 'https://www.alphavantage.co/query?function=TIME_SERIES_WEEKLY&symbol={}&apikey=235NZQZTP0UKSBEQ'

def get_data(url):
    r = requests.get(url)
    data = r.json()
    df = pd.DataFrame.from_dict(data['Weekly Time Series'], orient='index')
    df.columns = ['Open', 'High', 'low', 'close', 'volume']
    df['Date'] = df.index
    df = df.reset_index()
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] =  pd.DatetimeIndex(df['Date']).year
    df['Month'] =  pd.DatetimeIndex(df['Date']).month
    df['volume'] = df['volume'].apply(int)
    df['Open'] = df['Open'].apply(float)
    df['High'] = df['High'].apply(float)
    df['low'] = df['low'].apply(float)
    df['close'] = df['close'].apply(float)
    return df

st.title("Milestone Project: My Stock Ticker")
st.markdown("**Created By: Trey W.**")


st.sidebar.markdown(" **Visual Filter Selection** ")

results_df = get_data(base_url)

with st.sidebar.form(key='my_form'):
    text_input = st.text_input(label='Enter Stock Symbol (e.g. AMZN)')
    submit_button = st.form_submit_button(label='Submit')

if submit_button:
    try:
        search_url = new_url.format(text_input)
        results_df = get_data(search_url)
    except:
        st.warning('You either left the input blank or the Symbol you entered does not exist.\nTry again.')
        st.stop()

with st.container():
    select = st.sidebar.selectbox(label='Year', options=results_df['Year'].unique())
    select1 = st.sidebar.selectbox(label='Month', options= sorted(results_df['Month'].unique()))

yr_data = results_df[results_df['Year'] == select]
mnth_data = results_df[results_df['Month'] == select1]

with st.container():
    col1,col2,col3,col4 = st.columns(4)
    col1.metric("High", results_df['High'][0], round((results_df['High'][1] - results_df['High'][0])/100,2))
    col2.metric("Low", results_df['low'][0], round((results_df['low'][1] - results_df['low'][0])/100,2))
    col3.metric("Open", results_df['Open'][0], round((results_df['Open'][1] - results_df['Open'][0])/100,2))
    col4.metric("Close", results_df['close'][0], round((results_df['close'][1] - results_df['close'][0])/100,2))
    st.markdown("*Current Stats as of {}*".format(pd.to_datetime(results_df['Date'][0]).date()))

    if select == 2021:
        if select1 == 1:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 1], y=yr_data.close[yr_data.Month == 1], text = yr_data.close[yr_data.Month == 1],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 2:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 2], y=yr_data.close[yr_data.Month == 2], text = yr_data.close[yr_data.Month == 2],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 3:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 3], y=yr_data.close[yr_data.Month == 3], text = yr_data.close[yr_data.Month == 3],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 4:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 4], y=yr_data.close[yr_data.Month == 4], text = yr_data.close[yr_data.Month == 4],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 5:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 5], y=yr_data.close[yr_data.Month == 5], text = yr_data.close[yr_data.Month == 5],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 6:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 6], y=yr_data.close[yr_data.Month == 6], text = yr_data.close[yr_data.Month == 6],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 7:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 7], y=yr_data.close[yr_data.Month == 7], text = yr_data.close[yr_data.Month == 7],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 8:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 8], y=yr_data.close[yr_data.Month == 8], text = yr_data.close[yr_data.Month == 8],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 9:
            try:
                fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 9], y=yr_data.close[yr_data.Month == 9], text = yr_data.close[yr_data.Month == 9],
                              width = 1200, height = 575, markers = True)
                fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_layout(xaxis_title='Date',
                      yaxis_title='Close', plot_bgcolor="white")
                fig.update_traces(textposition="bottom right")
                st.plotly_chart(fig, use_container_width=False)
            except:
                st.warning('This data does not exist yet. Please use the current or past month(s)')
                st.stop()
        if select1 == 10:
            try:
                fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 10], y=yr_data.close[yr_data.Month == 10], text = yr_data.close[yr_data.Month == 10],
                              width = 1200, height = 575, markers = True)
                fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_layout(xaxis_title='Date',
                      yaxis_title='Close', plot_bgcolor="white")
                fig.update_traces(textposition="bottom right")
                st.plotly_chart(fig, use_container_width=False)
            except:
                st.warning('This data does not exist yet. Please use the current or past month(s)')
                st.stop()
        if select1 == 11:
            try:
                fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 11], y=yr_data.close[yr_data.Month == 11], text = yr_data.close[yr_data.Month == 11],
                              width = 1200, height = 575, markers = True)
                fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_layout(xaxis_title='Date',
                      yaxis_title='Close', plot_bgcolor="white")
                fig.update_traces(textposition="bottom right")
                st.plotly_chart(fig, use_container_width=False)
            except:
                st.warning('This data does not exist yet. Please use the current or past month(s)')
                st.stop()
        if select1 == 12:
            try:
                fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 12], y=yr_data.close[yr_data.Month == 12], text = yr_data.close[yr_data.Month == 12],
                              width = 1200, height = 575, markers = True)
                fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_layout(xaxis_title='Date',
                      yaxis_title='Close', plot_bgcolor="white")
                fig.update_traces(textposition="bottom right")
                st.plotly_chart(fig, use_container_width=False)
            except:
                st.warning('This data does not exist yet. Please use the current or past month(s)')
                st.stop()

    elif select == 2020:
        if select1 == 1:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 1], y=yr_data.close[yr_data.Month == 1], text = yr_data.close[yr_data.Month == 1],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 2:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 2], y=yr_data.close[yr_data.Month == 2], text = yr_data.close[yr_data.Month == 2],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 3:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 3], y=yr_data.close[yr_data.Month == 3], text = yr_data.close[yr_data.Month == 3],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 4:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 4], y=yr_data.close[yr_data.Month == 4], text = yr_data.close[yr_data.Month == 4],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 5:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 5], y=yr_data.close[yr_data.Month == 5], text = yr_data.close[yr_data.Month == 5],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 6:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 6], y=yr_data.close[yr_data.Month == 6], text = yr_data.close[yr_data.Month == 6],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 7:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 7], y=yr_data.close[yr_data.Month == 7], text = yr_data.close[yr_data.Month == 7],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 8:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 8], y=yr_data.close[yr_data.Month == 8], text = yr_data.close[yr_data.Month == 8],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 9:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 9], y=yr_data.close[yr_data.Month == 9], text = yr_data.close[yr_data.Month == 9],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 10:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 10], y=yr_data.close[yr_data.Month == 10], text = yr_data.close[yr_data.Month == 10],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 11:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 11], y=yr_data.close[yr_data.Month == 11], text = yr_data.close[yr_data.Month == 11],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 12:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 12], y=yr_data.close[yr_data.Month == 12], text = yr_data.close[yr_data.Month == 12],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)

    elif select == 2019:
        if select1 == 1:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 1], y=yr_data.close[yr_data.Month == 1], text = yr_data.close[yr_data.Month == 1],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 2:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 2], y=yr_data.close[yr_data.Month == 2], text = yr_data.close[yr_data.Month == 2],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 3:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 3], y=yr_data.close[yr_data.Month == 3], text = yr_data.close[yr_data.Month == 3],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 4:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 4], y=yr_data.close[yr_data.Month == 4], text = yr_data.close[yr_data.Month == 4],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 5:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 5], y=yr_data.close[yr_data.Month == 5], text = yr_data.close[yr_data.Month == 5],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 6:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 6], y=yr_data.close[yr_data.Month == 6], text = yr_data.close[yr_data.Month == 6],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 7:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 7], y=yr_data.close[yr_data.Month == 7], text = yr_data.close[yr_data.Month == 7],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 8:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 8], y=yr_data.close[yr_data.Month == 8], text = yr_data.close[yr_data.Month == 8],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 9:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 9], y=yr_data.close[yr_data.Month == 9], text = yr_data.close[yr_data.Month == 9],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 10:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 10], y=yr_data.close[yr_data.Month == 10], text = yr_data.close[yr_data.Month == 10],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 11:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 11], y=yr_data.close[yr_data.Month == 11], text = yr_data.close[yr_data.Month == 11],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 12:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 12], y=yr_data.close[yr_data.Month == 12], text = yr_data.close[yr_data.Month == 12],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)

    elif select == 2018:
        if select1 == 1:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 1], y=yr_data.close[yr_data.Month == 1], text = yr_data.close[yr_data.Month == 1],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 2:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 2], y=yr_data.close[yr_data.Month == 2], text = yr_data.close[yr_data.Month == 2],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 3:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 3], y=yr_data.close[yr_data.Month == 3], text = yr_data.close[yr_data.Month == 3],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 4:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 4], y=yr_data.close[yr_data.Month == 4], text = yr_data.close[yr_data.Month == 4],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 5:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 5], y=yr_data.close[yr_data.Month == 5], text = yr_data.close[yr_data.Month == 5],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 6:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 6], y=yr_data.close[yr_data.Month == 6], text = yr_data.close[yr_data.Month == 6],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 7:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 7], y=yr_data.close[yr_data.Month == 7], text = yr_data.close[yr_data.Month == 7],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 8:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 8], y=yr_data.close[yr_data.Month == 8], text = yr_data.close[yr_data.Month == 8],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 9:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 9], y=yr_data.close[yr_data.Month == 9], text = yr_data.close[yr_data.Month == 9],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 10:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 10], y=yr_data.close[yr_data.Month == 10], text = yr_data.close[yr_data.Month == 10],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 11:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 11], y=yr_data.close[yr_data.Month == 11], text = yr_data.close[yr_data.Month == 11],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 12:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 12], y=yr_data.close[yr_data.Month == 12], text = yr_data.close[yr_data.Month == 12],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)


    elif select == 2017:
        if select1 == 1:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 1], y=yr_data.close[yr_data.Month == 1], text = yr_data.close[yr_data.Month == 1],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 2:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 2], y=yr_data.close[yr_data.Month == 2], text = yr_data.close[yr_data.Month == 2],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 3:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 3], y=yr_data.close[yr_data.Month == 3], text = yr_data.close[yr_data.Month == 3],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 4:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 4], y=yr_data.close[yr_data.Month == 4], text = yr_data.close[yr_data.Month == 4],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 5:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 5], y=yr_data.close[yr_data.Month == 5], text = yr_data.close[yr_data.Month == 5],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 6:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 6], y=yr_data.close[yr_data.Month == 6], text = yr_data.close[yr_data.Month == 6],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 7:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 7], y=yr_data.close[yr_data.Month == 7], text = yr_data.close[yr_data.Month == 7],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 8:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 8], y=yr_data.close[yr_data.Month == 8], text = yr_data.close[yr_data.Month == 8],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 9:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 9], y=yr_data.close[yr_data.Month == 9], text = yr_data.close[yr_data.Month == 9],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 10:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 10], y=yr_data.close[yr_data.Month == 10], text = yr_data.close[yr_data.Month == 10],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 11:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 11], y=yr_data.close[yr_data.Month == 11], text = yr_data.close[yr_data.Month == 11],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 12:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 12], y=yr_data.close[yr_data.Month == 12], text = yr_data.close[yr_data.Month == 12],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)

    elif select == 2016:
        if select1 == 1:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 1], y=yr_data.close[yr_data.Month == 1], text = yr_data.close[yr_data.Month == 1],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 2:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 2], y=yr_data.close[yr_data.Month == 2], text = yr_data.close[yr_data.Month == 2],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 3:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 3], y=yr_data.close[yr_data.Month == 3], text = yr_data.close[yr_data.Month == 3],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 4:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 4], y=yr_data.close[yr_data.Month == 4], text = yr_data.close[yr_data.Month == 4],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 5:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 5], y=yr_data.close[yr_data.Month == 5], text = yr_data.close[yr_data.Month == 5],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 6:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 6], y=yr_data.close[yr_data.Month == 6], text = yr_data.close[yr_data.Month == 6],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 7:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 7], y=yr_data.close[yr_data.Month == 7], text = yr_data.close[yr_data.Month == 7],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 8:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 8], y=yr_data.close[yr_data.Month == 8], text = yr_data.close[yr_data.Month == 8],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 9:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 9], y=yr_data.close[yr_data.Month == 9], text = yr_data.close[yr_data.Month == 9],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 10:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 10], y=yr_data.close[yr_data.Month == 10], text = yr_data.close[yr_data.Month == 10],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 11:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 11], y=yr_data.close[yr_data.Month == 11], text = yr_data.close[yr_data.Month == 11],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 12:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 12], y=yr_data.close[yr_data.Month == 12], text = yr_data.close[yr_data.Month == 12],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)

    elif select == 2015:
        if select1 == 1:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 1], y=yr_data.close[yr_data.Month == 1], text = yr_data.close[yr_data.Month == 1],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 2:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 2], y=yr_data.close[yr_data.Month == 2], text = yr_data.close[yr_data.Month == 2],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 3:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 3], y=yr_data.close[yr_data.Month == 3], text = yr_data.close[yr_data.Month == 3],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 4:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 4], y=yr_data.close[yr_data.Month == 4], text = yr_data.close[yr_data.Month == 4],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 5:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 5], y=yr_data.close[yr_data.Month == 5], text = yr_data.close[yr_data.Month == 5],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 6:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 6], y=yr_data.close[yr_data.Month == 6], text = yr_data.close[yr_data.Month == 6],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 7:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 7], y=yr_data.close[yr_data.Month == 7], text = yr_data.close[yr_data.Month == 7],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 8:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 8], y=yr_data.close[yr_data.Month == 8], text = yr_data.close[yr_data.Month == 8],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 9:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 9], y=yr_data.close[yr_data.Month == 9], text = yr_data.close[yr_data.Month == 9],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 10:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 10], y=yr_data.close[yr_data.Month == 10], text = yr_data.close[yr_data.Month == 10],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 11:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 11], y=yr_data.close[yr_data.Month == 11], text = yr_data.close[yr_data.Month == 11],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 12:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 12], y=yr_data.close[yr_data.Month == 12], text = yr_data.close[yr_data.Month == 12],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)

    elif select == 2014:
        if select1 == 1:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 1], y=yr_data.close[yr_data.Month == 1], text = yr_data.close[yr_data.Month == 1],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 2:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 2], y=yr_data.close[yr_data.Month == 2], text = yr_data.close[yr_data.Month == 2],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 3:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 3], y=yr_data.close[yr_data.Month == 3], text = yr_data.close[yr_data.Month == 3],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 4:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 4], y=yr_data.close[yr_data.Month == 4], text = yr_data.close[yr_data.Month == 4],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 5:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 5], y=yr_data.close[yr_data.Month == 5], text = yr_data.close[yr_data.Month == 5],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 6:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 6], y=yr_data.close[yr_data.Month == 6], text = yr_data.close[yr_data.Month == 6],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 7:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 7], y=yr_data.close[yr_data.Month == 7], text = yr_data.close[yr_data.Month == 7],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 8:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 8], y=yr_data.close[yr_data.Month == 8], text = yr_data.close[yr_data.Month == 8],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 9:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 9], y=yr_data.close[yr_data.Month == 9], text = yr_data.close[yr_data.Month == 9],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 10:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 10], y=yr_data.close[yr_data.Month == 10], text = yr_data.close[yr_data.Month == 10],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 11:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 11], y=yr_data.close[yr_data.Month == 11], text = yr_data.close[yr_data.Month == 11],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 12:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 12], y=yr_data.close[yr_data.Month == 12], text = yr_data.close[yr_data.Month == 12],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)

    elif select == 2013:
        if select1 == 1:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 1], y=yr_data.close[yr_data.Month == 1], text = yr_data.close[yr_data.Month == 1],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 2:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 2], y=yr_data.close[yr_data.Month == 2], text = yr_data.close[yr_data.Month == 2],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 3:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 3], y=yr_data.close[yr_data.Month == 3], text = yr_data.close[yr_data.Month == 3],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 4:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 4], y=yr_data.close[yr_data.Month == 4], text = yr_data.close[yr_data.Month == 4],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 5:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 5], y=yr_data.close[yr_data.Month == 5], text = yr_data.close[yr_data.Month == 5],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 6:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 6], y=yr_data.close[yr_data.Month == 6], text = yr_data.close[yr_data.Month == 6],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 7:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 7], y=yr_data.close[yr_data.Month == 7], text = yr_data.close[yr_data.Month == 7],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 8:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 8], y=yr_data.close[yr_data.Month == 8], text = yr_data.close[yr_data.Month == 8],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 9:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 9], y=yr_data.close[yr_data.Month == 9], text = yr_data.close[yr_data.Month == 9],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 10:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 10], y=yr_data.close[yr_data.Month == 10], text = yr_data.close[yr_data.Month == 10],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 11:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 11], y=yr_data.close[yr_data.Month == 11], text = yr_data.close[yr_data.Month == 11],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 12:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 12], y=yr_data.close[yr_data.Month == 12], text = yr_data.close[yr_data.Month == 12],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)

    elif select == 2012:
        if select1 == 1:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 1], y=yr_data.close[yr_data.Month == 1], text = yr_data.close[yr_data.Month == 1],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 2:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 2], y=yr_data.close[yr_data.Month == 2], text = yr_data.close[yr_data.Month == 2],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 3:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 3], y=yr_data.close[yr_data.Month == 3], text = yr_data.close[yr_data.Month == 3],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 4:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 4], y=yr_data.close[yr_data.Month == 4], text = yr_data.close[yr_data.Month == 4],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 5:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 5], y=yr_data.close[yr_data.Month == 5], text = yr_data.close[yr_data.Month == 5],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 6:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 6], y=yr_data.close[yr_data.Month == 6], text = yr_data.close[yr_data.Month == 6],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 7:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 7], y=yr_data.close[yr_data.Month == 7], text = yr_data.close[yr_data.Month == 7],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 8:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 8], y=yr_data.close[yr_data.Month == 8], text = yr_data.close[yr_data.Month == 8],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 9:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 9], y=yr_data.close[yr_data.Month == 9], text = yr_data.close[yr_data.Month == 9],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 10:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 10], y=yr_data.close[yr_data.Month == 10], text = yr_data.close[yr_data.Month == 10],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 11:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 11], y=yr_data.close[yr_data.Month == 11], text = yr_data.close[yr_data.Month == 11],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 12:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 12], y=yr_data.close[yr_data.Month == 12], text = yr_data.close[yr_data.Month == 12],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)

    elif select == 2011:
        if select1 == 1:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 1], y=yr_data.close[yr_data.Month == 1], text = yr_data.close[yr_data.Month == 1],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 2:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 2], y=yr_data.close[yr_data.Month == 2], text = yr_data.close[yr_data.Month == 2],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 3:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 3], y=yr_data.close[yr_data.Month == 3], text = yr_data.close[yr_data.Month == 3],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 4:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 4], y=yr_data.close[yr_data.Month == 4], text = yr_data.close[yr_data.Month == 4],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 5:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 5], y=yr_data.close[yr_data.Month == 5], text = yr_data.close[yr_data.Month == 5],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 6:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 6], y=yr_data.close[yr_data.Month == 6], text = yr_data.close[yr_data.Month == 6],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 7:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 7], y=yr_data.close[yr_data.Month == 7], text = yr_data.close[yr_data.Month == 7],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 8:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 8], y=yr_data.close[yr_data.Month == 8], text = yr_data.close[yr_data.Month == 8],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 9:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 9], y=yr_data.close[yr_data.Month == 9], text = yr_data.close[yr_data.Month == 9],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 10:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 10], y=yr_data.close[yr_data.Month == 10], text = yr_data.close[yr_data.Month == 10],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 11:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 11], y=yr_data.close[yr_data.Month == 11], text = yr_data.close[yr_data.Month == 11],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 12:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 12], y=yr_data.close[yr_data.Month == 12], text = yr_data.close[yr_data.Month == 12],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)

    elif select == 2010:
        if select1 == 1:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 1], y=yr_data.close[yr_data.Month == 1], text = yr_data.close[yr_data.Month == 1],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 2:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 2], y=yr_data.close[yr_data.Month == 2], text = yr_data.close[yr_data.Month == 2],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 3:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 3], y=yr_data.close[yr_data.Month == 3], text = yr_data.close[yr_data.Month == 3],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 4:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 4], y=yr_data.close[yr_data.Month == 4], text = yr_data.close[yr_data.Month == 4],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 5:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 5], y=yr_data.close[yr_data.Month == 5], text = yr_data.close[yr_data.Month == 5],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 6:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 6], y=yr_data.close[yr_data.Month == 6], text = yr_data.close[yr_data.Month == 6],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 7:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 7], y=yr_data.close[yr_data.Month == 7], text = yr_data.close[yr_data.Month == 7],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 8:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 8], y=yr_data.close[yr_data.Month == 8], text = yr_data.close[yr_data.Month == 8],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 9:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 9], y=yr_data.close[yr_data.Month == 9], text = yr_data.close[yr_data.Month == 9],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 10:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 10], y=yr_data.close[yr_data.Month == 10], text = yr_data.close[yr_data.Month == 10],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 11:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 11], y=yr_data.close[yr_data.Month == 11], text = yr_data.close[yr_data.Month == 11],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 12:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 12], y=yr_data.close[yr_data.Month == 12], text = yr_data.close[yr_data.Month == 12],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)

    elif select == 2009:
        if select1 == 1:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 1], y=yr_data.close[yr_data.Month == 1], text = yr_data.close[yr_data.Month == 1],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 2:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 2], y=yr_data.close[yr_data.Month == 2], text = yr_data.close[yr_data.Month == 2],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 3:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 3], y=yr_data.close[yr_data.Month == 3], text = yr_data.close[yr_data.Month == 3],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 4:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 4], y=yr_data.close[yr_data.Month == 4], text = yr_data.close[yr_data.Month == 4],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 5:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 5], y=yr_data.close[yr_data.Month == 5], text = yr_data.close[yr_data.Month == 5],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 6:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 6], y=yr_data.close[yr_data.Month == 6], text = yr_data.close[yr_data.Month == 6],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 7:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 7], y=yr_data.close[yr_data.Month == 7], text = yr_data.close[yr_data.Month == 7],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 8:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 8], y=yr_data.close[yr_data.Month == 8], text = yr_data.close[yr_data.Month == 8],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 9:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 9], y=yr_data.close[yr_data.Month == 9], text = yr_data.close[yr_data.Month == 9],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 10:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 10], y=yr_data.close[yr_data.Month == 10], text = yr_data.close[yr_data.Month == 10],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 11:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 11], y=yr_data.close[yr_data.Month == 11], text = yr_data.close[yr_data.Month == 11],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 12:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 12], y=yr_data.close[yr_data.Month == 12], text = yr_data.close[yr_data.Month == 12],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)

    elif select == 2008:
        if select1 == 1:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 1], y=yr_data.close[yr_data.Month == 1], text = yr_data.close[yr_data.Month == 1],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 2:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 2], y=yr_data.close[yr_data.Month == 2], text = yr_data.close[yr_data.Month == 2],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 3:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 3], y=yr_data.close[yr_data.Month == 3], text = yr_data.close[yr_data.Month == 3],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 4:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 4], y=yr_data.close[yr_data.Month == 4], text = yr_data.close[yr_data.Month == 4],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 5:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 5], y=yr_data.close[yr_data.Month == 5], text = yr_data.close[yr_data.Month == 5],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 6:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 6], y=yr_data.close[yr_data.Month == 6], text = yr_data.close[yr_data.Month == 6],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 7:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 7], y=yr_data.close[yr_data.Month == 7], text = yr_data.close[yr_data.Month == 7],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 8:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 8], y=yr_data.close[yr_data.Month == 8], text = yr_data.close[yr_data.Month == 8],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 9:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 9], y=yr_data.close[yr_data.Month == 9], text = yr_data.close[yr_data.Month == 9],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 10:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 10], y=yr_data.close[yr_data.Month == 10], text = yr_data.close[yr_data.Month == 10],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 11:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 11], y=yr_data.close[yr_data.Month == 11], text = yr_data.close[yr_data.Month == 11],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 12:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 12], y=yr_data.close[yr_data.Month == 12], text = yr_data.close[yr_data.Month == 12],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)

    elif select == 2007:
        if select1 == 1:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 1], y=yr_data.close[yr_data.Month == 1], text = yr_data.close[yr_data.Month == 1],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 2:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 2], y=yr_data.close[yr_data.Month == 2], text = yr_data.close[yr_data.Month == 2],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 3:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 3], y=yr_data.close[yr_data.Month == 3], text = yr_data.close[yr_data.Month == 3],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 4:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 4], y=yr_data.close[yr_data.Month == 4], text = yr_data.close[yr_data.Month == 4],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 5:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 5], y=yr_data.close[yr_data.Month == 5], text = yr_data.close[yr_data.Month == 5],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 6:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 6], y=yr_data.close[yr_data.Month == 6], text = yr_data.close[yr_data.Month == 6],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 7:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 7], y=yr_data.close[yr_data.Month == 7], text = yr_data.close[yr_data.Month == 7],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 8:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 8], y=yr_data.close[yr_data.Month == 8], text = yr_data.close[yr_data.Month == 8],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 9:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 9], y=yr_data.close[yr_data.Month == 9], text = yr_data.close[yr_data.Month == 9],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 10:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 10], y=yr_data.close[yr_data.Month == 10], text = yr_data.close[yr_data.Month == 10],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 11:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 11], y=yr_data.close[yr_data.Month == 11], text = yr_data.close[yr_data.Month == 11],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 12:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 12], y=yr_data.close[yr_data.Month == 12], text = yr_data.close[yr_data.Month == 12],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)

    elif select == 2006:
        if select1 == 1:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 1], y=yr_data.close[yr_data.Month == 1], text = yr_data.close[yr_data.Month == 1],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 2:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 2], y=yr_data.close[yr_data.Month == 2], text = yr_data.close[yr_data.Month == 2],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 3:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 3], y=yr_data.close[yr_data.Month == 3], text = yr_data.close[yr_data.Month == 3],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 4:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 4], y=yr_data.close[yr_data.Month == 4], text = yr_data.close[yr_data.Month == 4],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 5:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 5], y=yr_data.close[yr_data.Month == 5], text = yr_data.close[yr_data.Month == 5],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 6:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 6], y=yr_data.close[yr_data.Month == 6], text = yr_data.close[yr_data.Month == 6],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 7:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 7], y=yr_data.close[yr_data.Month == 7], text = yr_data.close[yr_data.Month == 7],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 8:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 8], y=yr_data.close[yr_data.Month == 8], text = yr_data.close[yr_data.Month == 8],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 9:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 9], y=yr_data.close[yr_data.Month == 9], text = yr_data.close[yr_data.Month == 9],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 10:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 10], y=yr_data.close[yr_data.Month == 10], text = yr_data.close[yr_data.Month == 10],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 11:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 11], y=yr_data.close[yr_data.Month == 11], text = yr_data.close[yr_data.Month == 11],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 12:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 12], y=yr_data.close[yr_data.Month == 12], text = yr_data.close[yr_data.Month == 12],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)

    elif select == 2005:
        if select1 == 1:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 1], y=yr_data.close[yr_data.Month == 1], text = yr_data.close[yr_data.Month == 1],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 2:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 2], y=yr_data.close[yr_data.Month == 2], text = yr_data.close[yr_data.Month == 2],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 3:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 3], y=yr_data.close[yr_data.Month == 3], text = yr_data.close[yr_data.Month == 3],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 4:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 4], y=yr_data.close[yr_data.Month == 4], text = yr_data.close[yr_data.Month == 4],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 5:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 5], y=yr_data.close[yr_data.Month == 5], text = yr_data.close[yr_data.Month == 5],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 6:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 6], y=yr_data.close[yr_data.Month == 6], text = yr_data.close[yr_data.Month == 6],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 7:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 7], y=yr_data.close[yr_data.Month == 7], text = yr_data.close[yr_data.Month == 7],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 8:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 8], y=yr_data.close[yr_data.Month == 8], text = yr_data.close[yr_data.Month == 8],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 9:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 9], y=yr_data.close[yr_data.Month == 9], text = yr_data.close[yr_data.Month == 9],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 10:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 10], y=yr_data.close[yr_data.Month == 10], text = yr_data.close[yr_data.Month == 10],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 11:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 11], y=yr_data.close[yr_data.Month == 11], text = yr_data.close[yr_data.Month == 11],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 12:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 12], y=yr_data.close[yr_data.Month == 12], text = yr_data.close[yr_data.Month == 12],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)

    elif select == 2004:
        if select1 == 1:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 1], y=yr_data.close[yr_data.Month == 1], text = yr_data.close[yr_data.Month == 1],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 2:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 2], y=yr_data.close[yr_data.Month == 2], text = yr_data.close[yr_data.Month == 2],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 3:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 3], y=yr_data.close[yr_data.Month == 3], text = yr_data.close[yr_data.Month == 3],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 4:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 4], y=yr_data.close[yr_data.Month == 4], text = yr_data.close[yr_data.Month == 4],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 5:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 5], y=yr_data.close[yr_data.Month == 5], text = yr_data.close[yr_data.Month == 5],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 6:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 6], y=yr_data.close[yr_data.Month == 6], text = yr_data.close[yr_data.Month == 6],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 7:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 7], y=yr_data.close[yr_data.Month == 7], text = yr_data.close[yr_data.Month == 7],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 8:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 8], y=yr_data.close[yr_data.Month == 8], text = yr_data.close[yr_data.Month == 8],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 9:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 9], y=yr_data.close[yr_data.Month == 9], text = yr_data.close[yr_data.Month == 9],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 10:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 10], y=yr_data.close[yr_data.Month == 10], text = yr_data.close[yr_data.Month == 10],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 11:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 11], y=yr_data.close[yr_data.Month == 11], text = yr_data.close[yr_data.Month == 11],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 12:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 12], y=yr_data.close[yr_data.Month == 12], text = yr_data.close[yr_data.Month == 12],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)

    elif select == 2003:
        if select1 == 1:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 1], y=yr_data.close[yr_data.Month == 1], text = yr_data.close[yr_data.Month == 1],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 2:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 2], y=yr_data.close[yr_data.Month == 2], text = yr_data.close[yr_data.Month == 2],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 3:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 3], y=yr_data.close[yr_data.Month == 3], text = yr_data.close[yr_data.Month == 3],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 4:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 4], y=yr_data.close[yr_data.Month == 4], text = yr_data.close[yr_data.Month == 4],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 5:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 5], y=yr_data.close[yr_data.Month == 5], text = yr_data.close[yr_data.Month == 5],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 6:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 6], y=yr_data.close[yr_data.Month == 6], text = yr_data.close[yr_data.Month == 6],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 7:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 7], y=yr_data.close[yr_data.Month == 7], text = yr_data.close[yr_data.Month == 7],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 8:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 8], y=yr_data.close[yr_data.Month == 8], text = yr_data.close[yr_data.Month == 8],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 9:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 9], y=yr_data.close[yr_data.Month == 9], text = yr_data.close[yr_data.Month == 9],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 10:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 10], y=yr_data.close[yr_data.Month == 10], text = yr_data.close[yr_data.Month == 10],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 11:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 11], y=yr_data.close[yr_data.Month == 11], text = yr_data.close[yr_data.Month == 11],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 12:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 12], y=yr_data.close[yr_data.Month == 12], text = yr_data.close[yr_data.Month == 12],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)

    elif select == 2002:
        if select1 == 1:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 1], y=yr_data.close[yr_data.Month == 1], text = yr_data.close[yr_data.Month == 1],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 2:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 2], y=yr_data.close[yr_data.Month == 2], text = yr_data.close[yr_data.Month == 2],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 3:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 3], y=yr_data.close[yr_data.Month == 3], text = yr_data.close[yr_data.Month == 3],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 4:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 4], y=yr_data.close[yr_data.Month == 4], text = yr_data.close[yr_data.Month == 4],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 5:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 5], y=yr_data.close[yr_data.Month == 5], text = yr_data.close[yr_data.Month == 5],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 6:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 6], y=yr_data.close[yr_data.Month == 6], text = yr_data.close[yr_data.Month == 6],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 7:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 7], y=yr_data.close[yr_data.Month == 7], text = yr_data.close[yr_data.Month == 7],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 8:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 8], y=yr_data.close[yr_data.Month == 8], text = yr_data.close[yr_data.Month == 8],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 9:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 9], y=yr_data.close[yr_data.Month == 9], text = yr_data.close[yr_data.Month == 9],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 10:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 10], y=yr_data.close[yr_data.Month == 10], text = yr_data.close[yr_data.Month == 10],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 11:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 11], y=yr_data.close[yr_data.Month == 11], text = yr_data.close[yr_data.Month == 11],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 12:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 12], y=yr_data.close[yr_data.Month == 12], text = yr_data.close[yr_data.Month == 12],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)

    elif select == 2001:
        if select1 == 1:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 1], y=yr_data.close[yr_data.Month == 1], text = yr_data.close[yr_data.Month == 1],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 2:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 2], y=yr_data.close[yr_data.Month == 2], text = yr_data.close[yr_data.Month == 2],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 3:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 3], y=yr_data.close[yr_data.Month == 3], text = yr_data.close[yr_data.Month == 3],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 4:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 4], y=yr_data.close[yr_data.Month == 4], text = yr_data.close[yr_data.Month == 4],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 5:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 5], y=yr_data.close[yr_data.Month == 5], text = yr_data.close[yr_data.Month == 5],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 6:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 6], y=yr_data.close[yr_data.Month == 6], text = yr_data.close[yr_data.Month == 6],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 7:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 7], y=yr_data.close[yr_data.Month == 7], text = yr_data.close[yr_data.Month == 7],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 8:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 8], y=yr_data.close[yr_data.Month == 8], text = yr_data.close[yr_data.Month == 8],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 9:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 9], y=yr_data.close[yr_data.Month == 9], text = yr_data.close[yr_data.Month == 9],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 10:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 10], y=yr_data.close[yr_data.Month == 10], text = yr_data.close[yr_data.Month == 10],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 11:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 11], y=yr_data.close[yr_data.Month == 11], text = yr_data.close[yr_data.Month == 11],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 12:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 12], y=yr_data.close[yr_data.Month == 12], text = yr_data.close[yr_data.Month == 12],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)

    elif select == 2000:
        if select1 == 1:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 1], y=yr_data.close[yr_data.Month == 1], text = yr_data.close[yr_data.Month == 1],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 2:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 2], y=yr_data.close[yr_data.Month == 2], text = yr_data.close[yr_data.Month == 2],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 3:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 3], y=yr_data.close[yr_data.Month == 3], text = yr_data.close[yr_data.Month == 3],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 4:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 4], y=yr_data.close[yr_data.Month == 4], text = yr_data.close[yr_data.Month == 4],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 5:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 5], y=yr_data.close[yr_data.Month == 5], text = yr_data.close[yr_data.Month == 5],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 6:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 6], y=yr_data.close[yr_data.Month == 6], text = yr_data.close[yr_data.Month == 6],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 7:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 7], y=yr_data.close[yr_data.Month == 7], text = yr_data.close[yr_data.Month == 7],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 8:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 8], y=yr_data.close[yr_data.Month == 8], text = yr_data.close[yr_data.Month == 8],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 9:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 9], y=yr_data.close[yr_data.Month == 9], text = yr_data.close[yr_data.Month == 9],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 10:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 10], y=yr_data.close[yr_data.Month == 10], text = yr_data.close[yr_data.Month == 10],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 11:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 11], y=yr_data.close[yr_data.Month == 11], text = yr_data.close[yr_data.Month == 11],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)
        if select1 == 12:
            fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 12], y=yr_data.close[yr_data.Month == 12], text = yr_data.close[yr_data.Month == 12],
                          width = 1200, height = 575, markers = True)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
            fig.update_layout(xaxis_title='Date',
                  yaxis_title='Close', plot_bgcolor="white")
            fig.update_traces(textposition="bottom right")
            st.plotly_chart(fig, use_container_width=False)

    elif select == 1999:
        if select1 == 1:
            try:
                fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 1], y=yr_data.close[yr_data.Month == 1], text = yr_data.close[yr_data.Month == 1],
                              width = 1200, height = 575, markers = True)
                fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_layout(xaxis_title='Date',
                      yaxis_title='Close', plot_bgcolor="white")
                fig.update_traces(textposition="bottom right")
                st.plotly_chart(fig, use_container_width=False)
            except:
                st.warning('This data does not exist yet. Please use the current or past month(s)')
                st.stop()
        if select1 == 2:
            try:
                fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 2], y=yr_data.close[yr_data.Month == 2], text = yr_data.close[yr_data.Month == 2],
                              width = 1200, height = 575, markers = True)
                fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_layout(xaxis_title='Date',
                      yaxis_title='Close', plot_bgcolor="white")
                fig.update_traces(textposition="bottom right")
                st.plotly_chart(fig, use_container_width=False)
            except:
                st.warning('This data does not exist yet. Please use the current or past month(s)')
                st.stop()
        if select1 == 3:
            try:
                fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 3], y=yr_data.close[yr_data.Month == 3], text = yr_data.close[yr_data.Month == 3],
                              width = 1200, height = 575, markers = True)
                fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_layout(xaxis_title='Date',
                      yaxis_title='Close', plot_bgcolor="white")
                fig.update_traces(textposition="bottom right")
                st.plotly_chart(fig, use_container_width=False)
            except:
                st.warning('This data does not exist yet. Please use the current or past month(s)')
                st.stop()
        if select1 == 4:
            try:
                fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 4], y=yr_data.close[yr_data.Month == 4], text = yr_data.close[yr_data.Month == 4],
                              width = 1200, height = 575, markers = True)
                fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_layout(xaxis_title='Date',
                      yaxis_title='Close', plot_bgcolor="white")
                fig.update_traces(textposition="bottom right")
                st.plotly_chart(fig, use_container_width=False)
            except:
                st.warning('This data does not exist yet. Please use the current or past month(s)')
                st.stop()
        if select1 == 5:
            try:
                fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 5], y=yr_data.close[yr_data.Month == 5], text = yr_data.close[yr_data.Month == 5],
                              width = 1200, height = 575, markers = True)
                fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_layout(xaxis_title='Date',
                      yaxis_title='Close', plot_bgcolor="white")
                fig.update_traces(textposition="bottom right")
                st.plotly_chart(fig, use_container_width=False)
            except:
                st.warning('This data does not exist yet. Please use the current or past month(s)')
                st.stop()
        if select1 == 6:
            try:
                fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 6], y=yr_data.close[yr_data.Month == 6], text = yr_data.close[yr_data.Month == 6],
                              width = 1200, height = 575, markers = True)
                fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_layout(xaxis_title='Date',
                      yaxis_title='Close', plot_bgcolor="white")
                fig.update_traces(textposition="bottom right")
                st.plotly_chart(fig, use_container_width=False)
            except:
                st.warning('This data does not exist yet. Please use the current or past month(s)')
                st.stop()
        if select1 == 7:
            try:
                fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 7], y=yr_data.close[yr_data.Month == 7], text = yr_data.close[yr_data.Month == 7],
                              width = 1200, height = 575, markers = True)
                fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_layout(xaxis_title='Date',
                      yaxis_title='Close', plot_bgcolor="white")
                fig.update_traces(textposition="bottom right")
                st.plotly_chart(fig, use_container_width=False)
            except:
                st.warning('This data does not exist yet. Please use the current or past month(s)')
                st.stop()
        if select1 == 8:
            try:
                fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 8], y=yr_data.close[yr_data.Month == 8], text = yr_data.close[yr_data.Month == 8],
                              width = 1200, height = 575, markers = True)
                fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_layout(xaxis_title='Date',
                      yaxis_title='Close', plot_bgcolor="white")
                fig.update_traces(textposition="bottom right")
                st.plotly_chart(fig, use_container_width=False)
            except:
                st.warning('This data does not exist yet. Please use the current or past month(s)')
                st.stop()
        if select1 == 9:
            try:
                fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 9], y=yr_data.close[yr_data.Month == 9], text = yr_data.close[yr_data.Month == 9],
                              width = 1200, height = 575, markers = True)
                fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_layout(xaxis_title='Date',
                      yaxis_title='Close', plot_bgcolor="white")
                fig.update_traces(textposition="bottom right")
                st.plotly_chart(fig, use_container_width=False)
            except:
                st.warning('This data does not exist yet. Please use the current or past month(s)')
                st.stop()
        if select1 == 10:
            try:
                fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 10], y=yr_data.close[yr_data.Month == 10], text = yr_data.close[yr_data.Month == 10],
                              width = 1200, height = 575, markers = True)
                fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_layout(xaxis_title='Date',
                      yaxis_title='Close', plot_bgcolor="white")
                fig.update_traces(textposition="bottom right")
                st.plotly_chart(fig, use_container_width=False)
            except:
                st.warning('This data does not exist yet. Please use the current or past month(s)')
                st.stop()
        if select1 == 11:
            try:
                fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 11], y=yr_data.close[yr_data.Month == 11], text = yr_data.close[yr_data.Month == 11],
                              width = 1200, height = 575, markers = True)
                fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_layout(xaxis_title='Date',
                      yaxis_title='Close', plot_bgcolor="white")
                fig.update_traces(textposition="bottom right")
                st.plotly_chart(fig, use_container_width=False)
            except:
                st.warning('This data does not exist yet. Please use the current or past month(s)')
                st.stop()
        if select1 == 12:
            try:
                fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 12], y=yr_data.close[yr_data.Month == 12], text = yr_data.close[yr_data.Month == 12],
                              width = 1200, height = 575, markers = True)
                fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_layout(xaxis_title='Date',
                      yaxis_title='Close', plot_bgcolor="white")
                fig.update_traces(textposition="bottom right")
                st.plotly_chart(fig, use_container_width=False)
            except:
                st.warning('This data does not exist yet. Please use the current or past month(s)')
                st.stop()

    elif select == 1998:
        if select1 == 1:
            try:
                fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 1], y=yr_data.close[yr_data.Month == 1], text = yr_data.close[yr_data.Month == 1],
                              width = 1200, height = 575, markers = True)
                fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_layout(xaxis_title='Date',
                      yaxis_title='Close', plot_bgcolor="white")
                fig.update_traces(textposition="bottom right")
                st.plotly_chart(fig, use_container_width=False)
            except:
                st.warning('This data does not exist yet. Please use the current or past month(s)')
                st.stop()
        if select1 == 2:
            try:
                fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 2], y=yr_data.close[yr_data.Month == 2], text = yr_data.close[yr_data.Month == 2],
                              width = 1200, height = 575, markers = True)
                fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_layout(xaxis_title='Date',
                      yaxis_title='Close', plot_bgcolor="white")
                fig.update_traces(textposition="bottom right")
                st.plotly_chart(fig, use_container_width=False)
            except:
                st.warning('This data does not exist yet. Please use the current or past month(s)')
                st.stop()
        if select1 == 3:
            try:
                fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 3], y=yr_data.close[yr_data.Month == 3], text = yr_data.close[yr_data.Month == 3],
                              width = 1200, height = 575, markers = True)
                fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_layout(xaxis_title='Date',
                      yaxis_title='Close', plot_bgcolor="white")
                fig.update_traces(textposition="bottom right")
                st.plotly_chart(fig, use_container_width=False)
            except:
                st.warning('This data does not exist yet. Please use the current or past month(s)')
                st.stop()
        if select1 == 4:
            try:
                fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 4], y=yr_data.close[yr_data.Month == 4], text = yr_data.close[yr_data.Month == 4],
                              width = 1200, height = 575, markers = True)
                fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_layout(xaxis_title='Date',
                      yaxis_title='Close', plot_bgcolor="white")
                fig.update_traces(textposition="bottom right")
                st.plotly_chart(fig, use_container_width=False)
            except:
                st.warning('This data does not exist yet. Please use the current or past month(s)')
                st.stop()
        if select1 == 5:
            try:
                fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 5], y=yr_data.close[yr_data.Month == 5], text = yr_data.close[yr_data.Month == 5],
                              width = 1200, height = 575, markers = True)
                fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_layout(xaxis_title='Date',
                      yaxis_title='Close', plot_bgcolor="white")
                fig.update_traces(textposition="bottom right")
                st.plotly_chart(fig, use_container_width=False)
            except:
                st.warning('This data does not exist yet. Please use the current or past month(s)')
                st.stop()
        if select1 == 6:
            try:
                fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 6], y=yr_data.close[yr_data.Month == 6], text = yr_data.close[yr_data.Month == 6],
                              width = 1200, height = 575, markers = True)
                fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_layout(xaxis_title='Date',
                      yaxis_title='Close', plot_bgcolor="white")
                fig.update_traces(textposition="bottom right")
                st.plotly_chart(fig, use_container_width=False)
            except:
                st.warning('This data does not exist yet. Please use the current or past month(s)')
                st.stop()
        if select1 == 7:
            try:
                fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 7], y=yr_data.close[yr_data.Month == 7], text = yr_data.close[yr_data.Month == 7],
                              width = 1200, height = 575, markers = True)
                fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_layout(xaxis_title='Date',
                      yaxis_title='Close', plot_bgcolor="white")
                fig.update_traces(textposition="bottom right")
                st.plotly_chart(fig, use_container_width=False)
            except:
                st.warning('This data does not exist yet. Please use the current or past month(s)')
                st.stop()
        if select1 == 8:
            try:
                fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 8], y=yr_data.close[yr_data.Month == 8], text = yr_data.close[yr_data.Month == 8],
                              width = 1200, height = 575, markers = True)
                fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_layout(xaxis_title='Date',
                      yaxis_title='Close', plot_bgcolor="white")
                fig.update_traces(textposition="bottom right")
                st.plotly_chart(fig, use_container_width=False)
            except:
                st.warning('This data does not exist yet. Please use the current or past month(s)')
                st.stop()
        if select1 == 9:
            try:
                fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 9], y=yr_data.close[yr_data.Month == 9], text = yr_data.close[yr_data.Month == 9],
                              width = 1200, height = 575, markers = True)
                fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_layout(xaxis_title='Date',
                      yaxis_title='Close', plot_bgcolor="white")
                fig.update_traces(textposition="bottom right")
                st.plotly_chart(fig, use_container_width=False)
            except:
                st.warning('This data does not exist yet. Please use the current or past month(s)')
                st.stop()
        if select1 == 10:
            try:
                fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 10], y=yr_data.close[yr_data.Month == 10], text = yr_data.close[yr_data.Month == 10],
                              width = 1200, height = 575, markers = True)
                fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_layout(xaxis_title='Date',
                      yaxis_title='Close', plot_bgcolor="white")
                fig.update_traces(textposition="bottom right")
                st.plotly_chart(fig, use_container_width=False)
            except:
                st.warning('This data does not exist yet. Please use the current or past month(s)')
                st.stop()
        if select1 == 11:
            try:
                fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 11], y=yr_data.close[yr_data.Month == 11], text = yr_data.close[yr_data.Month == 11],
                              width = 1200, height = 575, markers = True)
                fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_layout(xaxis_title='Date',
                      yaxis_title='Close', plot_bgcolor="white")
                fig.update_traces(textposition="bottom right")
                st.plotly_chart(fig, use_container_width=False)
            except:
                st.warning('This data does not exist yet. Please use the current or past month(s)')
                st.stop()
        if select1 == 12:
            try:
                fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 12], y=yr_data.close[yr_data.Month == 12], text = yr_data.close[yr_data.Month == 12],
                              width = 1200, height = 575, markers = True)
                fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_layout(xaxis_title='Date',
                      yaxis_title='Close', plot_bgcolor="white")
                fig.update_traces(textposition="bottom right")
                st.plotly_chart(fig, use_container_width=False)
            except:
                st.warning('This data does not exist yet. Please use the current or past month(s)')
                st.stop()

    elif select == 1997:
        if select1 == 1:
            try:
                fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 1], y=yr_data.close[yr_data.Month == 1], text = yr_data.close[yr_data.Month == 1],
                              width = 1200, height = 575, markers = True)
                fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_layout(xaxis_title='Date',
                      yaxis_title='Close', plot_bgcolor="white")
                fig.update_traces(textposition="bottom right")
                st.plotly_chart(fig, use_container_width=False)
            except:
                st.warning('This data does not exist yet. Please use the current or past month(s)')
                st.stop()
        if select1 == 2:
            try:
                fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 2], y=yr_data.close[yr_data.Month == 2], text = yr_data.close[yr_data.Month == 2],
                              width = 1200, height = 575, markers = True)
                fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_layout(xaxis_title='Date',
                      yaxis_title='Close', plot_bgcolor="white")
                fig.update_traces(textposition="bottom right")
                st.plotly_chart(fig, use_container_width=False)
            except:
                st.warning('This data does not exist yet. Please use the current or past month(s)')
                st.stop()
        if select1 == 3:
            try:
                fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 3], y=yr_data.close[yr_data.Month == 3], text = yr_data.close[yr_data.Month == 3],
                              width = 1200, height = 575, markers = True)
                fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_layout(xaxis_title='Date',
                      yaxis_title='Close', plot_bgcolor="white")
                fig.update_traces(textposition="bottom right")
                st.plotly_chart(fig, use_container_width=False)
            except:
                st.warning('This data does not exist yet. Please use the current or past month(s)')
                st.stop()
        if select1 == 4:
            try:
                fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 4], y=yr_data.close[yr_data.Month == 4], text = yr_data.close[yr_data.Month == 4],
                              width = 1200, height = 575, markers = True)
                fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_layout(xaxis_title='Date',
                      yaxis_title='Close', plot_bgcolor="white")
                fig.update_traces(textposition="bottom right")
                st.plotly_chart(fig, use_container_width=False)
            except:
                st.warning('This data does not exist yet. Please use the current or past month(s)')
                st.stop()
        if select1 == 5:
            try:
                fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 5], y=yr_data.close[yr_data.Month == 5], text = yr_data.close[yr_data.Month == 5],
                              width = 1200, height = 575, markers = True)
                fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_layout(xaxis_title='Date',
                      yaxis_title='Close', plot_bgcolor="white")
                fig.update_traces(textposition="bottom right")
                st.plotly_chart(fig, use_container_width=False)
            except:
                st.warning('This data does not exist yet. Please use the current or past month(s)')
                st.stop()
        if select1 == 6:
            try:
                fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 6], y=yr_data.close[yr_data.Month == 6], text = yr_data.close[yr_data.Month == 6],
                              width = 1200, height = 575, markers = True)
                fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_layout(xaxis_title='Date',
                      yaxis_title='Close', plot_bgcolor="white")
                fig.update_traces(textposition="bottom right")
                st.plotly_chart(fig, use_container_width=False)
            except:
                st.warning('This data does not exist yet. Please use the current or past month(s)')
                st.stop()
        if select1 == 7:
            try:
                fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 7], y=yr_data.close[yr_data.Month == 7], text = yr_data.close[yr_data.Month == 7],
                              width = 1200, height = 575, markers = True)
                fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_layout(xaxis_title='Date',
                      yaxis_title='Close', plot_bgcolor="white")
                fig.update_traces(textposition="bottom right")
                st.plotly_chart(fig, use_container_width=False)
            except:
                st.warning('This data does not exist yet. Please use the current or past month(s)')
                st.stop()
        if select1 == 8:
            try:
                fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 8], y=yr_data.close[yr_data.Month == 8], text = yr_data.close[yr_data.Month == 8],
                              width = 1200, height = 575, markers = True)
                fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_layout(xaxis_title='Date',
                      yaxis_title='Close', plot_bgcolor="white")
                fig.update_traces(textposition="bottom right")
                st.plotly_chart(fig, use_container_width=False)
            except:
                st.warning('This data does not exist yet. Please use the current or past month(s)')
                st.stop()
        if select1 == 9:
            try:
                fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 9], y=yr_data.close[yr_data.Month == 9], text = yr_data.close[yr_data.Month == 9],
                              width = 1200, height = 575, markers = True)
                fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_layout(xaxis_title='Date',
                      yaxis_title='Close', plot_bgcolor="white")
                fig.update_traces(textposition="bottom right")
                st.plotly_chart(fig, use_container_width=False)
            except:
                st.warning('This data does not exist yet. Please use the current or past month(s)')
                st.stop()
        if select1 == 10:
            try:
                fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 10], y=yr_data.close[yr_data.Month == 10], text = yr_data.close[yr_data.Month == 10],
                              width = 1200, height = 575, markers = True)
                fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_layout(xaxis_title='Date',
                      yaxis_title='Close', plot_bgcolor="white")
                fig.update_traces(textposition="bottom right")
                st.plotly_chart(fig, use_container_width=False)
            except:
                st.warning('This data does not exist yet. Please use the current or past month(s)')
                st.stop()
        if select1 == 11:
            try:
                fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 11], y=yr_data.close[yr_data.Month == 11], text = yr_data.close[yr_data.Month == 11],
                              width = 1200, height = 575, markers = True)
                fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_layout(xaxis_title='Date',
                      yaxis_title='Close', plot_bgcolor="white")
                fig.update_traces(textposition="bottom right")
                st.plotly_chart(fig, use_container_width=False)
            except:
                st.warning('This data does not exist yet. Please use the current or past month(s)')
                st.stop()
        if select1 == 12:
            try:
                fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 12], y=yr_data.close[yr_data.Month == 12], text = yr_data.close[yr_data.Month == 12],
                              width = 1200, height = 575, markers = True)
                fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_layout(xaxis_title='Date',
                      yaxis_title='Close', plot_bgcolor="white")
                fig.update_traces(textposition="bottom right")
                st.plotly_chart(fig, use_container_width=False)
            except:
                st.warning('This data does not exist yet. Please use the current or past month(s)')
                st.stop()

    elif select == 1996:
        if select1 == 1:
            try:
                fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 1], y=yr_data.close[yr_data.Month == 1], text = yr_data.close[yr_data.Month == 1],
                              width = 1200, height = 575, markers = True)
                fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_layout(xaxis_title='Date',
                      yaxis_title='Close', plot_bgcolor="white")
                fig.update_traces(textposition="bottom right")
                st.plotly_chart(fig, use_container_width=False)
            except:
                st.warning('This data does not exist yet. Please use the current or past month(s)')
                st.stop()
        if select1 == 2:
            try:
                fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 2], y=yr_data.close[yr_data.Month == 2], text = yr_data.close[yr_data.Month == 2],
                              width = 1200, height = 575, markers = True)
                fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_layout(xaxis_title='Date',
                      yaxis_title='Close', plot_bgcolor="white")
                fig.update_traces(textposition="bottom right")
                st.plotly_chart(fig, use_container_width=False)
            except:
                st.warning('This data does not exist yet. Please use the current or past month(s)')
                st.stop()
        if select1 == 3:
            try:
                fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 3], y=yr_data.close[yr_data.Month == 3], text = yr_data.close[yr_data.Month == 3],
                              width = 1200, height = 575, markers = True)
                fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_layout(xaxis_title='Date',
                      yaxis_title='Close', plot_bgcolor="white")
                fig.update_traces(textposition="bottom right")
                st.plotly_chart(fig, use_container_width=False)
            except:
                st.warning('This data does not exist yet. Please use the current or past month(s)')
                st.stop()
        if select1 == 4:
            try:
                fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 4], y=yr_data.close[yr_data.Month == 4], text = yr_data.close[yr_data.Month == 4],
                              width = 1200, height = 575, markers = True)
                fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_layout(xaxis_title='Date',
                      yaxis_title='Close', plot_bgcolor="white")
                fig.update_traces(textposition="bottom right")
                st.plotly_chart(fig, use_container_width=False)
            except:
                st.warning('This data does not exist yet. Please use the current or past month(s)')
                st.stop()
        if select1 == 5:
            try:
                fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 5], y=yr_data.close[yr_data.Month == 5], text = yr_data.close[yr_data.Month == 5],
                              width = 1200, height = 575, markers = True)
                fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_layout(xaxis_title='Date',
                      yaxis_title='Close', plot_bgcolor="white")
                fig.update_traces(textposition="bottom right")
                st.plotly_chart(fig, use_container_width=False)
            except:
                st.warning('This data does not exist yet. Please use the current or past month(s)')
                st.stop()
        if select1 == 6:
            try:
                fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 6], y=yr_data.close[yr_data.Month == 6], text = yr_data.close[yr_data.Month == 6],
                              width = 1200, height = 575, markers = True)
                fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_layout(xaxis_title='Date',
                      yaxis_title='Close', plot_bgcolor="white")
                fig.update_traces(textposition="bottom right")
                st.plotly_chart(fig, use_container_width=False)
            except:
                st.warning('This data does not exist yet. Please use the current or past month(s)')
                st.stop()
        if select1 == 7:
            try:
                fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 7], y=yr_data.close[yr_data.Month == 7], text = yr_data.close[yr_data.Month == 7],
                              width = 1200, height = 575, markers = True)
                fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_layout(xaxis_title='Date',
                      yaxis_title='Close', plot_bgcolor="white")
                fig.update_traces(textposition="bottom right")
                st.plotly_chart(fig, use_container_width=False)
            except:
                st.warning('This data does not exist yet. Please use the current or past month(s)')
                st.stop()
        if select1 == 8:
            try:
                fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 8], y=yr_data.close[yr_data.Month == 8], text = yr_data.close[yr_data.Month == 8],
                              width = 1200, height = 575, markers = True)
                fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_layout(xaxis_title='Date',
                      yaxis_title='Close', plot_bgcolor="white")
                fig.update_traces(textposition="bottom right")
                st.plotly_chart(fig, use_container_width=False)
            except:
                st.warning('This data does not exist yet. Please use the current or past month(s)')
                st.stop()
        if select1 == 9:
            try:
                fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 9], y=yr_data.close[yr_data.Month == 9], text = yr_data.close[yr_data.Month == 9],
                              width = 1200, height = 575, markers = True)
                fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_layout(xaxis_title='Date',
                      yaxis_title='Close', plot_bgcolor="white")
                fig.update_traces(textposition="bottom right")
                st.plotly_chart(fig, use_container_width=False)
            except:
                st.warning('This data does not exist yet. Please use the current or past month(s)')
                st.stop()
        if select1 == 10:
            try:
                fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 10], y=yr_data.close[yr_data.Month == 10], text = yr_data.close[yr_data.Month == 10],
                              width = 1200, height = 575, markers = True)
                fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_layout(xaxis_title='Date',
                      yaxis_title='Close', plot_bgcolor="white")
                fig.update_traces(textposition="bottom right")
                st.plotly_chart(fig, use_container_width=False)
            except:
                st.warning('This data does not exist yet. Please use the current or past month(s)')
                st.stop()
        if select1 == 11:
            try:
                fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 11], y=yr_data.close[yr_data.Month == 11], text = yr_data.close[yr_data.Month == 11],
                              width = 1200, height = 575, markers = True)
                fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_layout(xaxis_title='Date',
                      yaxis_title='Close', plot_bgcolor="white")
                fig.update_traces(textposition="bottom right")
                st.plotly_chart(fig, use_container_width=False)
            except:
                st.warning('This data does not exist yet. Please use the current or past month(s)')
                st.stop()
        if select1 == 12:
            try:
                fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 12], y=yr_data.close[yr_data.Month == 12], text = yr_data.close[yr_data.Month == 12],
                              width = 1200, height = 575, markers = True)
                fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_layout(xaxis_title='Date',
                      yaxis_title='Close', plot_bgcolor="white")
                fig.update_traces(textposition="bottom right")
                st.plotly_chart(fig, use_container_width=False)
            except:
                st.warning('This data does not exist yet. Please use the current or past month(s)')
                st.stop()

    elif select == 1995:
        if select1 == 1:
            try:
                fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 1], y=yr_data.close[yr_data.Month == 1], text = yr_data.close[yr_data.Month == 1],
                              width = 1200, height = 575, markers = True)
                fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_layout(xaxis_title='Date',
                      yaxis_title='Close', plot_bgcolor="white")
                fig.update_traces(textposition="bottom right")
                st.plotly_chart(fig, use_container_width=False)
            except:
                st.warning('This data does not exist yet. Please use the current or past month(s)')
                st.stop()
        if select1 == 2:
            try:
                fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 2], y=yr_data.close[yr_data.Month == 2], text = yr_data.close[yr_data.Month == 2],
                              width = 1200, height = 575, markers = True)
                fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_layout(xaxis_title='Date',
                      yaxis_title='Close', plot_bgcolor="white")
                fig.update_traces(textposition="bottom right")
                st.plotly_chart(fig, use_container_width=False)
            except:
                st.warning('This data does not exist yet. Please use the current or past month(s)')
                st.stop()
        if select1 == 3:
            try:
                fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 3], y=yr_data.close[yr_data.Month == 3], text = yr_data.close[yr_data.Month == 3],
                              width = 1200, height = 575, markers = True)
                fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_layout(xaxis_title='Date',
                      yaxis_title='Close', plot_bgcolor="white")
                fig.update_traces(textposition="bottom right")
                st.plotly_chart(fig, use_container_width=False)
            except:
                st.warning('This data does not exist yet. Please use the current or past month(s)')
                st.stop()
        if select1 == 4:
            try:
                fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 4], y=yr_data.close[yr_data.Month == 4], text = yr_data.close[yr_data.Month == 4],
                              width = 1200, height = 575, markers = True)
                fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_layout(xaxis_title='Date',
                      yaxis_title='Close', plot_bgcolor="white")
                fig.update_traces(textposition="bottom right")
                st.plotly_chart(fig, use_container_width=False)
            except:
                st.warning('This data does not exist yet. Please use the current or past month(s)')
                st.stop()
        if select1 == 5:
            try:
                fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 5], y=yr_data.close[yr_data.Month == 5], text = yr_data.close[yr_data.Month == 5],
                              width = 1200, height = 575, markers = True)
                fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_layout(xaxis_title='Date',
                      yaxis_title='Close', plot_bgcolor="white")
                fig.update_traces(textposition="bottom right")
                st.plotly_chart(fig, use_container_width=False)
            except:
                st.warning('This data does not exist yet. Please use the current or past month(s)')
                st.stop()
        if select1 == 6:
            try:
                fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 6], y=yr_data.close[yr_data.Month == 6], text = yr_data.close[yr_data.Month == 6],
                              width = 1200, height = 575, markers = True)
                fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_layout(xaxis_title='Date',
                      yaxis_title='Close', plot_bgcolor="white")
                fig.update_traces(textposition="bottom right")
                st.plotly_chart(fig, use_container_width=False)
            except:
                st.warning('This data does not exist yet. Please use the current or past month(s)')
                st.stop()
        if select1 == 7:
            try:
                fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 7], y=yr_data.close[yr_data.Month == 7], text = yr_data.close[yr_data.Month == 7],
                              width = 1200, height = 575, markers = True)
                fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_layout(xaxis_title='Date',
                      yaxis_title='Close', plot_bgcolor="white")
                fig.update_traces(textposition="bottom right")
                st.plotly_chart(fig, use_container_width=False)
            except:
                st.warning('This data does not exist yet. Please use the current or past month(s)')
                st.stop()
        if select1 == 8:
            try:
                fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 8], y=yr_data.close[yr_data.Month == 8], text = yr_data.close[yr_data.Month == 8],
                              width = 1200, height = 575, markers = True)
                fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_layout(xaxis_title='Date',
                      yaxis_title='Close', plot_bgcolor="white")
                fig.update_traces(textposition="bottom right")
                st.plotly_chart(fig, use_container_width=False)
            except:
                st.warning('This data does not exist yet. Please use the current or past month(s)')
                st.stop()
        if select1 == 9:
            try:
                fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 9], y=yr_data.close[yr_data.Month == 9], text = yr_data.close[yr_data.Month == 9],
                              width = 1200, height = 575, markers = True)
                fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_layout(xaxis_title='Date',
                      yaxis_title='Close', plot_bgcolor="white")
                fig.update_traces(textposition="bottom right")
                st.plotly_chart(fig, use_container_width=False)
            except:
                st.warning('This data does not exist yet. Please use the current or past month(s)')
                st.stop()
        if select1 == 10:
            try:
                fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 10], y=yr_data.close[yr_data.Month == 10], text = yr_data.close[yr_data.Month == 10],
                              width = 1200, height = 575, markers = True)
                fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_layout(xaxis_title='Date',
                      yaxis_title='Close', plot_bgcolor="white")
                fig.update_traces(textposition="bottom right")
                st.plotly_chart(fig, use_container_width=False)
            except:
                st.warning('This data does not exist yet. Please use the current or past month(s)')
                st.stop()
        if select1 == 11:
            try:
                fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 11], y=yr_data.close[yr_data.Month == 11], text = yr_data.close[yr_data.Month == 11],
                              width = 1200, height = 575, markers = True)
                fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_layout(xaxis_title='Date',
                      yaxis_title='Close', plot_bgcolor="white")
                fig.update_traces(textposition="bottom right")
                st.plotly_chart(fig, use_container_width=False)
            except:
                st.warning('This data does not exist yet. Please use the current or past month(s)')
                st.stop()
        if select1 == 12:
            try:
                fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 12], y=yr_data.close[yr_data.Month == 12], text = yr_data.close[yr_data.Month == 12],
                              width = 1200, height = 575, markers = True)
                fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_layout(xaxis_title='Date',
                      yaxis_title='Close', plot_bgcolor="white")
                fig.update_traces(textposition="bottom right")
                st.plotly_chart(fig, use_container_width=False)
            except:
                st.warning('This data does not exist yet. Please use the current or past month(s)')
                st.stop()
                
    elif select == 1994:
        if select1 == 1:
            try:
                fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 1], y=yr_data.close[yr_data.Month == 1], text = yr_data.close[yr_data.Month == 1],
                              width = 1200, height = 575, markers = True)
                fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_layout(xaxis_title='Date',
                      yaxis_title='Close', plot_bgcolor="white")
                fig.update_traces(textposition="bottom right")
                st.plotly_chart(fig, use_container_width=False)
            except:
                st.warning('This data does not exist yet. Please use the current or past month(s)')
                st.stop()
        if select1 == 2:
            try:
                fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 2], y=yr_data.close[yr_data.Month == 2], text = yr_data.close[yr_data.Month == 2],
                              width = 1200, height = 575, markers = True)
                fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_layout(xaxis_title='Date',
                      yaxis_title='Close', plot_bgcolor="white")
                fig.update_traces(textposition="bottom right")
                st.plotly_chart(fig, use_container_width=False)
            except:
                st.warning('This data does not exist yet. Please use the current or past month(s)')
                st.stop()
        if select1 == 3:
            try:
                fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 3], y=yr_data.close[yr_data.Month == 3], text = yr_data.close[yr_data.Month == 3],
                              width = 1200, height = 575, markers = True)
                fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_layout(xaxis_title='Date',
                      yaxis_title='Close', plot_bgcolor="white")
                fig.update_traces(textposition="bottom right")
                st.plotly_chart(fig, use_container_width=False)
            except:
                st.warning('This data does not exist yet. Please use the current or past month(s)')
                st.stop()
        if select1 == 4:
            try:
                fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 4], y=yr_data.close[yr_data.Month == 4], text = yr_data.close[yr_data.Month == 4],
                              width = 1200, height = 575, markers = True)
                fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_layout(xaxis_title='Date',
                      yaxis_title='Close', plot_bgcolor="white")
                fig.update_traces(textposition="bottom right")
                st.plotly_chart(fig, use_container_width=False)
            except:
                st.warning('This data does not exist yet. Please use the current or past month(s)')
                st.stop()
        if select1 == 5:
            try:
                fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 5], y=yr_data.close[yr_data.Month == 5], text = yr_data.close[yr_data.Month == 5],
                              width = 1200, height = 575, markers = True)
                fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_layout(xaxis_title='Date',
                      yaxis_title='Close', plot_bgcolor="white")
                fig.update_traces(textposition="bottom right")
                st.plotly_chart(fig, use_container_width=False)
            except:
                st.warning('This data does not exist yet. Please use the current or past month(s)')
                st.stop()
        if select1 == 6:
            try:
                fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 6], y=yr_data.close[yr_data.Month == 6], text = yr_data.close[yr_data.Month == 6],
                              width = 1200, height = 575, markers = True)
                fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_layout(xaxis_title='Date',
                      yaxis_title='Close', plot_bgcolor="white")
                fig.update_traces(textposition="bottom right")
                st.plotly_chart(fig, use_container_width=False)
            except:
                st.warning('This data does not exist yet. Please use the current or past month(s)')
                st.stop()
        if select1 == 7:
            try:
                fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 7], y=yr_data.close[yr_data.Month == 7], text = yr_data.close[yr_data.Month == 7],
                              width = 1200, height = 575, markers = True)
                fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_layout(xaxis_title='Date',
                      yaxis_title='Close', plot_bgcolor="white")
                fig.update_traces(textposition="bottom right")
                st.plotly_chart(fig, use_container_width=False)
            except:
                st.warning('This data does not exist yet. Please use the current or past month(s)')
                st.stop()
        if select1 == 8:
            try:
                fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 8], y=yr_data.close[yr_data.Month == 8], text = yr_data.close[yr_data.Month == 8],
                              width = 1200, height = 575, markers = True)
                fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_layout(xaxis_title='Date',
                      yaxis_title='Close', plot_bgcolor="white")
                fig.update_traces(textposition="bottom right")
                st.plotly_chart(fig, use_container_width=False)
            except:
                st.warning('This data does not exist yet. Please use the current or past month(s)')
                st.stop()
        if select1 == 9:
            try:
                fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 9], y=yr_data.close[yr_data.Month == 9], text = yr_data.close[yr_data.Month == 9],
                              width = 1200, height = 575, markers = True)
                fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_layout(xaxis_title='Date',
                      yaxis_title='Close', plot_bgcolor="white")
                fig.update_traces(textposition="bottom right")
                st.plotly_chart(fig, use_container_width=False)
            except:
                st.warning('This data does not exist yet. Please use the current or past month(s)')
                st.stop()
        if select1 == 10:
            try:
                fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 10], y=yr_data.close[yr_data.Month == 10], text = yr_data.close[yr_data.Month == 10],
                              width = 1200, height = 575, markers = True)
                fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_layout(xaxis_title='Date',
                      yaxis_title='Close', plot_bgcolor="white")
                fig.update_traces(textposition="bottom right")
                st.plotly_chart(fig, use_container_width=False)
            except:
                st.warning('This data does not exist yet. Please use the current or past month(s)')
                st.stop()
        if select1 == 11:
            try:
                fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 11], y=yr_data.close[yr_data.Month == 11], text = yr_data.close[yr_data.Month == 11],
                              width = 1200, height = 575, markers = True)
                fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_layout(xaxis_title='Date',
                      yaxis_title='Close', plot_bgcolor="white")
                fig.update_traces(textposition="bottom right")
                st.plotly_chart(fig, use_container_width=False)
            except:
                st.warning('This data does not exist yet. Please use the current or past month(s)')
                st.stop()
        if select1 == 12:
            try:
                fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 12], y=yr_data.close[yr_data.Month == 12], text = yr_data.close[yr_data.Month == 12],
                              width = 1200, height = 575, markers = True)
                fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_layout(xaxis_title='Date',
                      yaxis_title='Close', plot_bgcolor="white")
                fig.update_traces(textposition="bottom right")
                st.plotly_chart(fig, use_container_width=False)
            except:
                st.warning('This data does not exist yet. Please use the current or past month(s)')
                st.stop()

    elif select == 1993:
        if select1 == 1:
            try:
                fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 1], y=yr_data.close[yr_data.Month == 1], text = yr_data.close[yr_data.Month == 1],
                              width = 1200, height = 575, markers = True)
                fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_layout(xaxis_title='Date',
                      yaxis_title='Close', plot_bgcolor="white")
                fig.update_traces(textposition="bottom right")
                st.plotly_chart(fig, use_container_width=False)
            except:
                st.warning('This data does not exist yet. Please use the current or past month(s)')
                st.stop()
        if select1 == 2:
            try:
                fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 2], y=yr_data.close[yr_data.Month == 2], text = yr_data.close[yr_data.Month == 2],
                              width = 1200, height = 575, markers = True)
                fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_layout(xaxis_title='Date',
                      yaxis_title='Close', plot_bgcolor="white")
                fig.update_traces(textposition="bottom right")
                st.plotly_chart(fig, use_container_width=False)
            except:
                st.warning('This data does not exist yet. Please use the current or past month(s)')
                st.stop()
        if select1 == 3:
            try:
                fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 3], y=yr_data.close[yr_data.Month == 3], text = yr_data.close[yr_data.Month == 3],
                              width = 1200, height = 575, markers = True)
                fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_layout(xaxis_title='Date',
                      yaxis_title='Close', plot_bgcolor="white")
                fig.update_traces(textposition="bottom right")
                st.plotly_chart(fig, use_container_width=False)
            except:
                st.warning('This data does not exist yet. Please use the current or past month(s)')
                st.stop()
        if select1 == 4:
            try:
                fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 4], y=yr_data.close[yr_data.Month == 4], text = yr_data.close[yr_data.Month == 4],
                              width = 1200, height = 575, markers = True)
                fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_layout(xaxis_title='Date',
                      yaxis_title='Close', plot_bgcolor="white")
                fig.update_traces(textposition="bottom right")
                st.plotly_chart(fig, use_container_width=False)
            except:
                st.warning('This data does not exist yet. Please use the current or past month(s)')
                st.stop()
        if select1 == 5:
            try:
                fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 5], y=yr_data.close[yr_data.Month == 5], text = yr_data.close[yr_data.Month == 5],
                              width = 1200, height = 575, markers = True)
                fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_layout(xaxis_title='Date',
                      yaxis_title='Close', plot_bgcolor="white")
                fig.update_traces(textposition="bottom right")
                st.plotly_chart(fig, use_container_width=False)
            except:
                st.warning('This data does not exist yet. Please use the current or past month(s)')
                st.stop()
        if select1 == 6:
            try:
                fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 6], y=yr_data.close[yr_data.Month == 6], text = yr_data.close[yr_data.Month == 6],
                              width = 1200, height = 575, markers = True)
                fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_layout(xaxis_title='Date',
                      yaxis_title='Close', plot_bgcolor="white")
                fig.update_traces(textposition="bottom right")
                st.plotly_chart(fig, use_container_width=False)
            except:
                st.warning('This data does not exist yet. Please use the current or past month(s)')
                st.stop()
        if select1 == 7:
            try:
                fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 7], y=yr_data.close[yr_data.Month == 7], text = yr_data.close[yr_data.Month == 7],
                              width = 1200, height = 575, markers = True)
                fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_layout(xaxis_title='Date',
                      yaxis_title='Close', plot_bgcolor="white")
                fig.update_traces(textposition="bottom right")
                st.plotly_chart(fig, use_container_width=False)
            except:
                st.warning('This data does not exist yet. Please use the current or past month(s)')
                st.stop()
        if select1 == 8:
            try:
                fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 8], y=yr_data.close[yr_data.Month == 8], text = yr_data.close[yr_data.Month == 8],
                              width = 1200, height = 575, markers = True)
                fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_layout(xaxis_title='Date',
                      yaxis_title='Close', plot_bgcolor="white")
                fig.update_traces(textposition="bottom right")
                st.plotly_chart(fig, use_container_width=False)
            except:
                st.warning('This data does not exist yet. Please use the current or past month(s)')
                st.stop()
        if select1 == 9:
            try:
                fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 9], y=yr_data.close[yr_data.Month == 9], text = yr_data.close[yr_data.Month == 9],
                              width = 1200, height = 575, markers = True)
                fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_layout(xaxis_title='Date',
                      yaxis_title='Close', plot_bgcolor="white")
                fig.update_traces(textposition="bottom right")
                st.plotly_chart(fig, use_container_width=False)
            except:
                st.warning('This data does not exist yet. Please use the current or past month(s)')
                st.stop()
        if select1 == 10:
            try:
                fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 10], y=yr_data.close[yr_data.Month == 10], text = yr_data.close[yr_data.Month == 10],
                              width = 1200, height = 575, markers = True)
                fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_layout(xaxis_title='Date',
                      yaxis_title='Close', plot_bgcolor="white")
                fig.update_traces(textposition="bottom right")
                st.plotly_chart(fig, use_container_width=False)
            except:
                st.warning('This data does not exist yet. Please use the current or past month(s)')
                st.stop()
        if select1 == 11:
            try:
                fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 11], y=yr_data.close[yr_data.Month == 11], text = yr_data.close[yr_data.Month == 11],
                              width = 1200, height = 575, markers = True)
                fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_layout(xaxis_title='Date',
                      yaxis_title='Close', plot_bgcolor="white")
                fig.update_traces(textposition="bottom right")
                st.plotly_chart(fig, use_container_width=False)
            except:
                st.warning('This data does not exist yet. Please use the current or past month(s)')
                st.stop()
        if select1 == 12:
            try:
                fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 12], y=yr_data.close[yr_data.Month == 12], text = yr_data.close[yr_data.Month == 12],
                              width = 1200, height = 575, markers = True)
                fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_layout(xaxis_title='Date',
                      yaxis_title='Close', plot_bgcolor="white")
                fig.update_traces(textposition="bottom right")
                st.plotly_chart(fig, use_container_width=False)
            except:
                st.warning('This data does not exist yet. Please use the current or past month(s)')
                st.stop()

    elif select == 1992:
        if select1 == 1:
            try:
                fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 1], y=yr_data.close[yr_data.Month == 1], text = yr_data.close[yr_data.Month == 1],
                              width = 1200, height = 575, markers = True)
                fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_layout(xaxis_title='Date',
                      yaxis_title='Close', plot_bgcolor="white")
                fig.update_traces(textposition="bottom right")
                st.plotly_chart(fig, use_container_width=False)
            except:
                st.warning('This data does not exist yet. Please use the current or past month(s)')
                st.stop()
        if select1 == 2:
            try:
                fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 2], y=yr_data.close[yr_data.Month == 2], text = yr_data.close[yr_data.Month == 2],
                              width = 1200, height = 575, markers = True)
                fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_layout(xaxis_title='Date',
                      yaxis_title='Close', plot_bgcolor="white")
                fig.update_traces(textposition="bottom right")
                st.plotly_chart(fig, use_container_width=False)
            except:
                st.warning('This data does not exist yet. Please use the current or past month(s)')
                st.stop()
        if select1 == 3:
            try:
                fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 3], y=yr_data.close[yr_data.Month == 3], text = yr_data.close[yr_data.Month == 3],
                              width = 1200, height = 575, markers = True)
                fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_layout(xaxis_title='Date',
                      yaxis_title='Close', plot_bgcolor="white")
                fig.update_traces(textposition="bottom right")
                st.plotly_chart(fig, use_container_width=False)
            except:
                st.warning('This data does not exist yet. Please use the current or past month(s)')
                st.stop()
        if select1 == 4:
            try:
                fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 4], y=yr_data.close[yr_data.Month == 4], text = yr_data.close[yr_data.Month == 4],
                              width = 1200, height = 575, markers = True)
                fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_layout(xaxis_title='Date',
                      yaxis_title='Close', plot_bgcolor="white")
                fig.update_traces(textposition="bottom right")
                st.plotly_chart(fig, use_container_width=False)
            except:
                st.warning('This data does not exist yet. Please use the current or past month(s)')
                st.stop()
        if select1 == 5:
            try:
                fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 5], y=yr_data.close[yr_data.Month == 5], text = yr_data.close[yr_data.Month == 5],
                              width = 1200, height = 575, markers = True)
                fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_layout(xaxis_title='Date',
                      yaxis_title='Close', plot_bgcolor="white")
                fig.update_traces(textposition="bottom right")
                st.plotly_chart(fig, use_container_width=False)
            except:
                st.warning('This data does not exist yet. Please use the current or past month(s)')
                st.stop()
        if select1 == 6:
            try:
                fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 6], y=yr_data.close[yr_data.Month == 6], text = yr_data.close[yr_data.Month == 6],
                              width = 1200, height = 575, markers = True)
                fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_layout(xaxis_title='Date',
                      yaxis_title='Close', plot_bgcolor="white")
                fig.update_traces(textposition="bottom right")
                st.plotly_chart(fig, use_container_width=False)
            except:
                st.warning('This data does not exist yet. Please use the current or past month(s)')
                st.stop()
        if select1 == 7:
            try:
                fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 7], y=yr_data.close[yr_data.Month == 7], text = yr_data.close[yr_data.Month == 7],
                              width = 1200, height = 575, markers = True)
                fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_layout(xaxis_title='Date',
                      yaxis_title='Close', plot_bgcolor="white")
                fig.update_traces(textposition="bottom right")
                st.plotly_chart(fig, use_container_width=False)
            except:
                st.warning('This data does not exist yet. Please use the current or past month(s)')
                st.stop()
        if select1 == 8:
            try:
                fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 8], y=yr_data.close[yr_data.Month == 8], text = yr_data.close[yr_data.Month == 8],
                              width = 1200, height = 575, markers = True)
                fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_layout(xaxis_title='Date',
                      yaxis_title='Close', plot_bgcolor="white")
                fig.update_traces(textposition="bottom right")
                st.plotly_chart(fig, use_container_width=False)
            except:
                st.warning('This data does not exist yet. Please use the current or past month(s)')
                st.stop()
        if select1 == 9:
            try:
                fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 9], y=yr_data.close[yr_data.Month == 9], text = yr_data.close[yr_data.Month == 9],
                              width = 1200, height = 575, markers = True)
                fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_layout(xaxis_title='Date',
                      yaxis_title='Close', plot_bgcolor="white")
                fig.update_traces(textposition="bottom right")
                st.plotly_chart(fig, use_container_width=False)
            except:
                st.warning('This data does not exist yet. Please use the current or past month(s)')
                st.stop()
        if select1 == 10:
            try:
                fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 10], y=yr_data.close[yr_data.Month == 10], text = yr_data.close[yr_data.Month == 10],
                              width = 1200, height = 575, markers = True)
                fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_layout(xaxis_title='Date',
                      yaxis_title='Close', plot_bgcolor="white")
                fig.update_traces(textposition="bottom right")
                st.plotly_chart(fig, use_container_width=False)
            except:
                st.warning('This data does not exist yet. Please use the current or past month(s)')
                st.stop()
        if select1 == 11:
            try:
                fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 11], y=yr_data.close[yr_data.Month == 11], text = yr_data.close[yr_data.Month == 11],
                              width = 1200, height = 575, markers = True)
                fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_layout(xaxis_title='Date',
                      yaxis_title='Close', plot_bgcolor="white")
                fig.update_traces(textposition="bottom right")
                st.plotly_chart(fig, use_container_width=False)
            except:
                st.warning('This data does not exist yet. Please use the current or past month(s)')
                st.stop()
        if select1 == 12:
            try:
                fig = px.line(yr_data, x= yr_data.Date[yr_data.Month == 12], y=yr_data.close[yr_data.Month == 12], text = yr_data.close[yr_data.Month == 12],
                              width = 1200, height = 575, markers = True)
                fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_yaxes(tickprefix="$", showline=True, linewidth=2, linecolor='black', showgrid=False)
                fig.update_layout(xaxis_title='Date',
                      yaxis_title='Close', plot_bgcolor="white")
                fig.update_traces(textposition="bottom right")
                st.plotly_chart(fig, use_container_width=False)
            except:
                st.warning('This data does not exist yet. Please use the current or past month(s)')
                st.stop()
