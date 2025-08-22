import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
import streamlit as st


def preprocess_dataset():
    # Load dataset.
    df = pd.read_csv('anz 3.csv')

    # Convert the 'date' column to proper datetime format.
    df['extraction'] = pd.to_datetime(df['extraction']).dt.strftime('%Y-%m-%d %H:%M:%S')
    # Clean long_lat columns: split into separate longtitude, latitude
    df[['longitude', 'latitude']] = df['long_lat'].str.split(' ', expand=True).astype(float)
    df_clean = df[['status', 'extraction', 'currency', 'longitude', 'latitude', 'balance', 'gender', 'age', 'amount', 'movement', 'txn_description']]
    df_clean.to_csv('cleaned_anz3.csv', index=False)

    # Store txn_description for later use
    txn_description = df['txn_description']

    # Tranform/encode 'status' into categories. 
    # 'status' is your y-value - we decided to use 'status' as an indication of fraud. We equate posted as fraud. 
    df['status'] = df['status'].map({'authorized': 0, 'posted': 1})  # 1 = fraud
    y = df['status']

    # Keep relevant X feature columns
    X = df[['currency', 'longitude', 'latitude', 'balance', 'gender', 'age', 'amount', 'movement']]
    
    # One-hot encode categorical features.
    X = pd.get_dummies(X, drop_first=True)

    # Select numeric features to standardize
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns

    # Standardize numeric features (some models need standardizing)
    scaler = StandardScaler()
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    return X, y, txn_description

def train_model_display_results(X, y, txn_description):
    X_train, X_test, y_train, y_test, desc_train, desc_test = train_test_split(
        X, y, txn_description, stratify = y, test_size = 0.3, random_state = 42
    )

    # Random Forest 
    rf = RandomForestClassifier(class_weight='balanced', n_estimators=100, random_state=42)  #n_estimators specifies the number of decision trees in a Random Forest.
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    importances = rf.feature_importances_
    feature_names = X.columns
    feat_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feat_df = feat_df.sort_values(by='Importance', ascending=True)

    feat_fig = go.Figure(go.Bar(
            x=feat_df['Importance'],
            y=feat_df['Feature'],
            orientation='h',
            marker=dict(color='indianred')
        ))
    feat_fig.update_layout(
            title='Top Feature Importances (Random Forest)',
            xaxis_title='Importance',
            yaxis_title='Feature',
            template='plotly_white'
        )
    
    # Actual vs Predicted Fraud
    comparison_df = pd.DataFrame({
        'txn_description': desc_test,      # Transaction type
        'Actual': y_test.values,           # True fraud label
        'Predicted': y_pred                # Model's prediction
    })

    melted = comparison_df.melt(id_vars='txn_description', 
                                value_vars=['Actual', 'Predicted'], 
                                var_name='Source', 
                                value_name='Fraud_Label')

    # Keep only fraud cases (label == 1)
    fraud_only = melted[melted['Fraud_Label'] == 1]

    # Plot side-by-side bars
    fraud_fig = px.histogram(
        fraud_only,
        y='txn_description',
        color='Source',
        barmode='group',
        title="Actual vs Predicted Fraud (Only Fraud Cases) by Transaction Type",
        color_discrete_sequence=["#4A90E2", "#003f5c"]
    )

    fraud_fig.update_layout(
        xaxis_title="Count",
        yaxis_title="Transaction Type",
        legend_title='Label Source',
        template='plotly_white',
        yaxis=dict(
            categoryorder="total ascending"
        )
    )
    col1, col2 = st.columns([1, 1]) 

    with col1:
        st.plotly_chart(fraud_fig, use_container_width=True)
    with col2:
        st.plotly_chart(feat_fig, use_container_width=True)

    # plotly_figures['Fraud Comparison'] = fraud_fig

    # return plotly_figures

def main():
    X, y, txn_description = preprocess_dataset()
    train_model_display_results(X, y, txn_description) 


if __name__ == '__main__':
    main()