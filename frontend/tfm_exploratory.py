import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns

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
    
def process_data(X_train, X_test):
    scaler = StandardScaler()    
    numeric_features = X_train.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X_train.select_dtypes(include='category').columns.tolist()
    
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
    return ohe_X_train, ohe_X_test

def app():
    st.title('Exploración de los datos')
    df = load_data('../data/uci_data.xls')
    df = clean_data(df)

    X_train, y_train, X_test, y_test = split_data(df)
    st.subheader('Datos originales')
    st.write(X_train.head(10))

    ohe_X_train, ohe_X_test = process_data(X_train, X_test)
    st.subheader('Datos escalados y codificados')
    st.write(ohe_X_train.head(10))

    option = st.selectbox('columna', list(X_train.columns))
    numeric_features = X_train.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X_train.select_dtypes(include='category').columns.tolist()

    if option in categorical_features:
        fig, ax = plt.subplots()
        sns.countplot(x=option, data=X_train, ax=ax)
        st.pyplot(fig)
    else:
        fig, ax = plt.subplots(2, 1)
        sns.histplot(x=option, data=X_train, ax=ax[0])
        sns.boxplot(x=option, data=X_train, ax=ax[1])
        st.pyplot(fig)
        
    corr_opt = st.selectbox('datos', ['originales', 'escalados'])

    if corr_opt == 'originales':
        corrMatrix = X_train[numeric_features].corr()
        corr_fig, corr_ax = plt.subplots(figsize=(10,10))
        sns.heatmap(corrMatrix, annot=True, ax=corr_ax)
        st.pyplot(corr_fig)
    else:
        nf = ohe_X_train.select_dtypes(include=np.number).columns.tolist()
        corrMatrix = ohe_X_train[nf].corr()
        corrohe_fig, corrohe_ax = plt.subplots(figsize=(10,10))
        sns.heatmap(corrMatrix, annot=False, ax=corrohe_ax)
        st.pyplot(corrohe_fig)

if __name__ == '__main__':
    app()
        
