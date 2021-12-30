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

modelos = ['Dummy', 'LR', 'AdaBoost', 'RC', 'RF']


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
    
if __name__ == '__main__':
    app()
    