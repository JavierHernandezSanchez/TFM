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
import data_preparation

modelos = ['LR', 'AdaBoost', 'RC', 'RF', 'LDA']

def app():
    X_train, X_test, ohe_X_train, y_train, ohe_X_test, y_test = data_preparation.create_df('../data/uci_data.xls')
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
    elif option == 'LDA':
        shap_values = joblib.load('lda_shapbars.joblib')
        shap.summary_plot(shap_values, ohe_X_test, plot_type='bar')
        fig1 = plt.gcf()
        st.pyplot(fig1)
        
        shap_values = joblib.load('lda_shapbee.joblib')        
        shap.plots.beeswarm(shap_values)
        fig2 = plt.gcf()
        st.pyplot(fig2)
    
    
if __name__ == '__main__':
    app()