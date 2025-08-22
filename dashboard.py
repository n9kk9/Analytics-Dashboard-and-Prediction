import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
import pandas as pd
import pydeck as pdk
from atm_bank import aggregated_timeseries_pred, ATM_timeseries_pred 
import fraud_detection_model_compare as fd
from streamlit_option_menu import option_menu

st.set_page_config(layout="wide")

# Sidebar Navigation 
with st.sidebar:
    st.markdown("# Navigation")  # Sidebar title
    selection = option_menu(
        menu_title=None,  # No title here to avoid duplication
        options=["Home    ", "Fraud Detection   ", "ATM Cash Demand   "],
        icons=["house", "shield-exclamation	", "graph-up", "file-text"],
        default_index=0,
        styles={
            "container": {"padding": "15px", 
                          "background-color": "#transparent",
                          "border-radius": "12px",
                          "box-shadow": "2px 2px 12px rgba(0, 0, 0, 0.1)",
                          },
            "icon": {"color": "#bbb", "font-size": "28px"},
            "nav-link": {
                "font-size": "20px",
                "text-align": "left",
                "padding": "12px 25px",
                "color": "#bbb",
                "--hover-color": "#eee",
                "border-radius": "10px",
                "transition": "all 0.3s ease",
            },
            "nav-link-selected": {
                "background-color": "#e0e0e0",
                "font-weight": "bold",
                "color": "#000",
                "border-left": "4px solid #4285F4",
                "border-radius": "10px",
            },
        }
    )

# Home Tab 
if selection == "Home    ":
    st.title("📊 Multi-Tab Streamlit Dashboard")
    st.write("### Welcome to this interactive dashboard built using **Streamlit**. This dashboard displays the following projects:")

    st.write("## 🔍 Fraud Detection")
    st.write("""
    #### This project focuses on identifying fraudulent transactions using machine learning models. By analysing transaction patterns, merchant types, and payment sources, the system can highlight unusual activity and provide insights into potential fraud.
    """)

    st.write("## 🏧 ATM Cash Demand Prediction")
    st.write("#### Efficient cash management is critical for ATMs. This project uses time-series forecasting in two ways:")
    st.write("""
    - #### **ATM-level Prediction** : Forecasting withdrawals for individual ATM IDs to optimize schedules.

    - #### **Aggregated Prediction** : Combining multiple ATMs to predict overall demand, helping banks reduce costs while ensuring availability.
    #### Use the sidebar to navigate through the different tabs.
    """)

# Plotly Tab
elif selection == "ATM Cash Demand   ":
    st.title("ATM Cash Demand Prediction")
    st.markdown("""
### 🏧 ATM Cash Demand Prediction

#### This section visualises forecasts of cash withdrawals for ATMs, helping banks optimize cash replenishment and reduce operational costs. It includes two key perspectives:

#### **1. Aggregated ATM Forecast (SARIMAX)**  
- #### Displays total cash demand across all ATMs over time.  
- #### The line chart compares actual withdrawals with forecasted values, highlighting trends and seasonality.  
- #### Helps management plan overall cash logistics and ensure ATMs are sufficiently funded without overstocking.

#### **2. ATM-Level Forecast (ARIMA)**  
- #### Allows selection of an individual ATM to view its predicted withdrawals.  
- #### Compares historical withdrawals with model forecasts for that specific machine.  
- #### Enables branch managers and operations teams to optimize cash allocation per ATM, minimizing shortages and idle cash.

#### These visualizations provide actionable insights for **efficient cash management, cost reduction, and improved service availability**.
""")
    
    df = pd.read_csv('atm_cash_management_dataset.csv')
    df['Date'] = pd.to_datetime(df['Date']) 
    df.sort_values(by='Date')
    aggregated_timeseries_pred(df)
    
    ATM_timeseries_pred(df)

# Fraud Detection Tab 
elif selection == "Fraud Detection   ":
    st.title("Fraud Detection Model")
    st.markdown('''
                #### This section includes an interactive map, displaying transaction locations, where red points indicate cases of fraud. 
                ''')

    
    df = pd.read_csv('cleaned_anz3.csv')
    
    # Map transaction status to RBG colour. 
    status_color_map = {
    'authorized': [0, 255, 0],  # Green
    'posted': [255, 0, 0]       # Red
    }

    df['color'] = df['status'].map(status_color_map)

    # Create map layer. 
    layer = pdk.Layer(
        'ScatterplotLayer',
        data=df,
        get_position='[longitude, latitude]',
        get_fill_color='color',
        get_radius=110,
        pickable=True,  # required for tooltip to work
    )
    tooltip = {
        "html": "<b>Datetime:</b> {extraction}<br/>"
                "<b>Status:</b> {status}<br/>"
                "<b>Account Balence:</b> ${balance}<br/>"
                "<b>Transaction Amount:</b> ${amount}<br/>"
                "<b>Transaction Type:</b> {movement}<br/>"
                "<b>Description:</b> {txn_description}<br/>",
        "style": {
            "backgroundColor": "steelblue",
            "color": "white"
        }
    }

    # Set view state: start in Sydney
    view_state = pdk.ViewState(
        latitude=df['latitude'][2],
        longitude=df['longitude'][2],
        zoom=10
    )

    # Render map
    st.pydeck_chart(pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip=tooltip), 
        use_container_width=True,
        height=800
    )

    st.markdown("""
### 📊 Model Insights & Visualizations

#### After training the Random Forest model to detect fraudulent transactions, we provide two key visualizations to help interpret its performance and feature importance:

#### **1. Actual vs Predicted Fraud (Transaction Type)**  
- #### This grouped bar chart compares the **true fraud cases** against the **model’s predictions**. 
- #### It helps users and businesses quickly identify which transaction types are most prone to fraud, enabling better risk management and more informed decision-making.


#### **2. Feature Importances**  
- #### This horizontal bar chart shows the **relative importance of each feature** in the model’s decision-making.  
- #### Features with higher importance contribute more to predicting fraud, helping identify which transaction characteristics are most influential.
""")
    
    X, y, txn_description = fd.preprocess_dataset()
    fd.train_model_display_results(X, y, txn_description)
