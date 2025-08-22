import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px 
import numpy as np
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import streamlit as st


def aggregated_timeseries_EDA(df):
    # Preprocess df to get unique date and value
    # 1. Aggregating across different ATMs
    df_aggregated = df.groupby('Date')[['Total_Withdrawals', 'Total_Deposits',
                                        'Cash_Demand_Next_Day', 'Previous_Day_Cash_Level']].sum().reset_index()
    correlation_matrix = df_aggregated.corr()

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_aggregated['Date'],
        y=df_aggregated['Total_Withdrawals'],
        mode='lines',
        name=f'Total Withdrawls'
    ))

    fig.add_trace(go.Scatter(
        x=df_aggregated['Date'],
        y=df_aggregated['Total_Deposits'],
        mode='lines',
        name=f'Total Deposits'
    ))

    fig.add_trace(go.Scatter(
        x=df_aggregated['Date'],
        y=df_aggregated['Cash_Demand_Next_Day'],
        mode='lines',
        name=f'Cash Demand Next Day',
        line=dict(dash='dash')
    ))

    fig.add_trace(go.Scatter(
        x=df_aggregated['Date'],
        y=df_aggregated['Previous_Day_Cash_Level'],
        mode='lines',
        name=f'Previous Day Cash Level',
        line=dict(dash='dash')
    ))

    fig.update_layout(
        title='ARIMA Forecast per ATM',
        xaxis_title='Date',
        yaxis_title='Total_Withdrawals',
        template='plotly_white'
    )

    fig.show()


def aggregated_timeseries_pred(df):
    # Preprocess df to get unique date and value
    # 1. Aggregating across different ATMs
    df_aggregated = df.groupby('Date')[['Total_Withdrawals', 'Previous_Day_Cash_Level']].sum().reset_index()
    # Train-test split
    n = len(df_aggregated)
    train_size = int(n * 0.7)
    train = df_aggregated.iloc[:train_size]
    test = df_aggregated.iloc[train_size:]

    # Fit SARIMAX with exogenous variable
    # Use previous day cash level to predict total withdrawals. 
    y = train['Total_Withdrawals']
    X = train['Previous_Day_Cash_Level']
    model = SARIMAX(y, exog=X, order=(1,1,1))
    model_fit = model.fit()

    # Forecast for test period
    forecast = model_fit.forecast(steps=len(test), exog=test[['Previous_Day_Cash_Level']])
    # R² calculation
    r2 = r2_score(test['Total_Withdrawals'], forecast)

    fig = go.Figure()
    # Plot actual
    fig.add_trace(go.Scatter(
        x=df_aggregated['Date'],
        y=df_aggregated['Total_Withdrawals'],
        mode='lines',
        name=f'Actual'
    ))

    # Plot forecast
    fig.add_trace(go.Scatter(
        x=test['Date'],
        y=forecast,
        mode='lines',
        name=f'Forecast <br> R²: {r2:.4f}',
        line=dict(dash='dash')
    ))

    fig.update_layout(
        title='SARIMAX Forecast of Aggregated ATM Cash Demand',
        xaxis_title='Date',
        yaxis_title='Total Withdrawals',
        template='plotly_white',
        height=700,
        legend=dict(
            font=dict(size=16)   # increase legend text size
        ),
        xaxis=dict(
            title=dict(font=dict(size=18)),  # axis title font
            tickfont=dict(size=14)  # tick label font
        ),
        yaxis=dict(
            title=dict(font=dict(size=18)),
            tickfont=dict(size=14)
        )
    )

    st.plotly_chart(fig, use_container_width=True)


def ATM_timeseries_pred(df):
    # 2. Groupby ATM IDs 
    col1, col2 = st.columns([1, 4])  # 1/5 of width for dropdown

    with col1:
        unique_groups = df['ATM_ID'].unique() # 50 ATMs
        selected_atm = st.selectbox("Select ATM ID:", sorted(unique_groups))

    # Filter df to only include 1 ATM's timeseries
    group_data = df[df['ATM_ID'] == selected_atm]
    
    # # use aggrgation after group by (SQL) 
    df_indiviual = group_data.groupby('Date')[['Previous_Day_Cash_Level', 'Total_Withdrawals']].sum().reset_index()

    # Univariate ARIMA using only total withdrawls didn't perform well, try multivariate version.
    # Train-test split
    n = len(df_indiviual)
    train_size = int(n * 0.7)
    train = df_indiviual.iloc[:train_size] # index slicing - going up to the train size (70%).
    test = df_indiviual.iloc[train_size:]
    
    # Fit SARIMAX with exogenous variable
    # Use previous day cash level to predict total withdrawals. 
    y = train['Total_Withdrawals']
    X = train['Previous_Day_Cash_Level']
    model = SARIMAX(y, exog=X, order=(1,1,1))
    model_fit = model.fit()

    # Forecast for test period
    forecast = model_fit.forecast(steps=len(test), exog=test[['Previous_Day_Cash_Level']])
    

    # R² calculation
    r2 = r2_score(test['Total_Withdrawals'], forecast)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_indiviual['Date'],
        y=df_indiviual['Total_Withdrawals'],
        mode='lines',
        name=f'Total Withdrawls'
    ))

    fig.add_trace(go.Scatter(
        x=test['Date'],
        y=forecast,
        mode='lines',
        name=f'Forecast'
    ))

    fig.update_layout(
        title='ARIMA Forecast per ATM',
        xaxis_title='Date',
        yaxis_title='Total Withdrawals',
        template='plotly_white', 
        legend=dict(
            font=dict(size=16)   # increase legend text size
        ),
        xaxis=dict(
            title=dict(font=dict(size=18)),  # axis title font
            tickfont=dict(size=14)  # tick label font
        ),
        yaxis=dict(
            title=dict(font=dict(size=18)),
            tickfont=dict(size=14)
        )
    )
    st.plotly_chart(fig, use_container_width=True)
    
def main():
    df = pd.read_csv('atm_cash_management_dataset.csv')
    df['Date'] = pd.to_datetime(df['Date']) 
    df.sort_values(by='Date')
    aggregated_timeseries_pred(df)
    # aggregated_timeseries_EDA(df)
    # ATM_timeseries_pred(df)


if __name__ == "__main__":
    main()