import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import data_preparation

def app():
    X_train, X_test, ohe_X_train, y_train, ohe_X_test, y_test = data_preparation.create_df('../data/uci_data.xls')
    
    st.title('Exploración de los datos')    
    st.subheader('Datos originales')
    st.write(X_train.head(10))
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
        
