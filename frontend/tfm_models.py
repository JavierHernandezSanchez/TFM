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
import data_preparation

modelos = ['Dummy', 'LR', 'AdaBoost', 'RC', 'RF', 'LDA']


def model_summary(model, X_train, X_test, y_train, y_test):
    train_accuracy = metrics.accuracy_score(y_train, model.predict(X_train))
    train_precision = metrics.precision_score(y_train, model.predict(X_train), zero_division=1)
    train_recall = metrics.recall_score(y_train, model.predict(X_train))
    train_f1 = metrics.f1_score(y_train, model.predict(X_train))
    test_accuracy = metrics.accuracy_score(y_test, model.predict(X_test))
    test_precision = metrics.precision_score(y_test, model.predict(X_test), zero_division=1)
    test_recall = metrics.recall_score(y_test, model.predict(X_test))
    test_f1 = metrics.f1_score(y_test, model.predict(X_test))
    
    indicators_df = pd.DataFrame({'accuracy': [train_accuracy, test_accuracy],
                                                        'precision': [train_precision, test_precision],
                                                        'recall': [train_recall, test_recall],
                                                        'f1': [train_f1, test_f1]},
                                                      index=['Train', 'Test'])
    return indicators_df
    
def show_model(model_file, X_train, X_test, y_train, y_test):
    model = joblib.load(model_file)
    indicators = model_summary(model, X_train, X_test, y_train, y_test)                                                  
    
    st.write(indicators)
    
    cmtr = metrics.confusion_matrix(y_train, model.predict(X_train))
    st.write('Matriz de confusión train', cmtr)
    cmte = metrics.confusion_matrix(y_test, model.predict(X_test))
    st.write('Matriz de confusión test', cmte)
    
    fig, ax = plt.subplots()
    plot_roc_curve(model, X_test, y_test, ax=ax)
    st.pyplot(fig)

def app():
    X_train, X_test, ohe_X_train, y_train, ohe_X_test, y_test = data_preparation.create_df('../data/uci_data.xls')
    st.title('Exploración de los modelos')

    option = st.selectbox('modelo', modelos)
    if option == 'Dummy':
        show_model('dummy.joblib', X_train, X_test, y_train, y_test)
    elif option == 'LR':
        show_model('lr.joblib', ohe_X_train, ohe_X_test, y_train, y_test)
    elif option == 'RC':
        show_model('rc.joblib', ohe_X_train, ohe_X_test, y_train, y_test)
    elif option == 'RF':
        show_model('rf.joblib', ohe_X_train, ohe_X_test, y_train, y_test)
    elif option == 'AdaBoost':
        show_model('ab.joblib', ohe_X_train, ohe_X_test, y_train, y_test)
    elif option == 'LDA':
        show_model('lda.joblib', ohe_X_train, ohe_X_test, y_train, y_test)
    
if __name__ == '__main__':
    app()
    