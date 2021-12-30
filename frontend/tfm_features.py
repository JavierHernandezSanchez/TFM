import streamlit as st
import joblib
from sklearn.dummy import DummyClassifier
from sklearn import metrics
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_roc_curve
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import numpy as np
import shap

def load_data(data_path):
    df = pd.read_excel(data_path, header=0, index_col=0, skiprows=1)
    df = df.rename(columns={'PAY_0': 'PAY_1'})
    return df
    
def clean_data(data):
    df = data.copy()
    # I set all invalid values to 'other'
    df.loc[~(df['EDUCATION'].isin([1, 2, 3, 4])), 'EDUCATION'] = 4
    df.loc[~(df['MARRIAGE'].isin([1, 2, 3])), 'MARRIAGE'] = 3

    # PAY_* fuera de rango
    paux = ~(df['PAY_1'].isin([-1,0,1,2,3,4,5,6,7,8,9]))
    for i in range(2, 7):
        paux = paux | ~(df['PAY_' + str(i)].isin([-1,0,1,2,3,4,5,6,7,8,9]))
        
    df = df[~paux]

    categorical_cols = ['SEX', 'EDUCATION', 'MARRIAGE', 'PAY_1', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']

    df[categorical_cols] = df[categorical_cols].astype('category')
    return df
    
def split_data(df):
    train, test = train_test_split(df, test_size=0.2, random_state=42)
    X_train = train.drop(columns=("default payment next month"))
    y_train = train[["default payment next month"]]
    X_test = test.drop(columns=("default payment next month"))
    y_test = test[["default payment next month"]]
    
    return X_train, y_train, X_test, y_test

df = load_data('../data/uci_data.xls')
df = clean_data(df)
X_train, y_train, X_test, y_test = split_data(df)

numeric_features = X_train.select_dtypes(include=np.number).columns.tolist()
categorical_features = X_train.select_dtypes(include='category').columns.tolist()
    
scaler = StandardScaler()
X_train[numeric_features] = scaler.fit_transform(X_train[numeric_features])
X_test[numeric_features] = scaler.transform(X_test[numeric_features])

# Apply one-hot encoder to each column with categorical data
ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)
ohe_train = pd.DataFrame(ohe.fit_transform(X_train[categorical_features]))
ohe_test = pd.DataFrame(ohe.transform(X_test[categorical_features]))

# One-hot encoding removed index; put it back
ohe_train.index = X_train.index
ohe_test.index = X_test.index
ohe_train.columns = ohe.get_feature_names(categorical_features)
ohe_test.columns = ohe.get_feature_names(categorical_features)

# Remove categorical columns (will replace with one-hot encoding)
num_X_train = X_train.drop(categorical_features, axis=1)
num_X_test = X_test.drop(categorical_features, axis=1)

# Add one-hot encoded columns to numerical features
ohe_X_train = pd.concat([num_X_train, ohe_train], axis=1)
ohe_X_test = pd.concat([num_X_test, ohe_test], axis=1)

modelos = ['LR', 'AdaBoost', 'RC', 'RF']

def app():
    st.title('Exploración de las características')

    option = st.selectbox('modelo', modelos)
    if option == 'LR':
        shap_values = joblib.load('lr_shapbars.joblib')
        shap.summary_plot(shap_values, ohe_X_test, plot_type='bar')
        fig1 = plt.gcf()
        st.pyplot(fig1)
        
        shap_values = joblib.load('lr_shapbee.joblib')
        shap.plots.beeswarm(shap_values)
        fig2 = plt.gcf()
        st.pyplot(fig2)
    elif option == 'RC':
        shap_values = joblib.load('rc_shapbars.joblib')
        shap.summary_plot(shap_values, ohe_X_test, plot_type='bar')
        fig1 = plt.gcf()
        st.pyplot(fig1)
        
        shap_values = joblib.load('rc_shapbee.joblib')
        shap.plots.beeswarm(shap_values)
        fig2 = plt.gcf()
        st.pyplot(fig2)
    elif option == 'RF':
        shap_values = joblib.load('rf_shapbars.joblib')
        shap.summary_plot(shap_values, ohe_X_test, plot_type='bar')
        fig1 = plt.gcf()
        st.pyplot(fig1)
        
        shap_values = joblib.load('rf_shapbee.joblib')
        shap.plots.beeswarm(shap_values[:,:,1])        
        fig2 = plt.gcf()
        st.pyplot(fig2)
    elif option == 'AdaBoost':
        shap_values = joblib.load('ab_shapbars.joblib')
        shap.summary_plot(shap_values, ohe_X_test, plot_type='bar')
        fig1 = plt.gcf()
        st.pyplot(fig1)
        
        shap_values = joblib.load('ab_shapbee.joblib')        
        shap.plots.beeswarm(shap_values)
        fig2 = plt.gcf()
        st.pyplot(fig2)
    
    
if __name__ == '__main__':
    app()