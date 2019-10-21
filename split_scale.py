import pandas as pd
from sklearn.preprocessing import StandardScaler, PowerTransformer, MinMaxScaler, RobustScaler, QuantileTransformer
from sklearn.model_selection import train_test_split
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
import split_scale

def split_my_data(data, train_ratio=.80, seed=123):
    '''the function will take a dataframe and returns train and test dataframe split 
    where 80% is in train, and 20% in test. '''
    return train_test_split(data, train_size = train_ratio, random_state = seed)

## Types of scalers

def standard_scaler(train, test):
    scaler = StandardScaler(copy=True, with_mean=True, with_std=True).fit(train)
    train_scaled = pd.DataFrame(scaler.transform(train), columns=train.columns.values).set_index([train.index.values])
    test_scaled = pd.DataFrame(scaler.transform(test), columns=test.columns.values).set_index([test.index.values])
    return scaler, train_scaled, test_scaled

def my_inv_transform(scaler, train_scaled, test_scaled):
    train = pd.DataFrame(scaler.inverse_transform(train_scaled), columns=train_scaled.columns.values).set_index([train_scaled.index.values])
    test = pd.DataFrame(scaler.inverse_transform(test_scaled), columns=test_scaled.columns.values).set_index([test_scaled.index.values])
    return scaler, train, test

def uniform_scaler(train, test, seed=123):
    scaler = QuantileTransformer(n_quantiles=100, output_distribution='uniform', random_state=seed, copy=True).fit(train)
    train_scaled = pd.DataFrame(scaler.transform(train), columns=train.columns.values).set_index([train.index.values])
    test_scaled = pd.DataFrame(scaler.transform(test), columns=test.columns.values).set_index([test.index.values])
    return scaler, train_scaled, test_scaled

def gaussian_scaler(train, test, method='yeo-johnson'):
    scaler = PowerTransformer(method, standardize=False, copy=True).fit(train)
    train_scaled = pd.DataFrame(scaler.transform(train), columns=train.columns.values).set_index([train.index.values])
    test_scaled = pd.DataFrame(scaler.transform(test), columns=test.columns.values).set_index([test.index.values])
    return scaler, train_scaled, test_scaled

def my_minmax_scaler(train, test, minmax_range=(0,1)):
    scaler = MinMaxScaler(copy=True, feature_range=minmax_range).fit(train)
    train_scaled = pd.DataFrame(scaler.transform(train), columns=train.columns.values).set_index([train.index.values])
    test_scaled = pd.DataFrame(scaler.transform(test), columns=test.columns.values).set_index([test.index.values])
    return scaler, train_scaled, test_scaled

def iqr_robust_scaler(train, test):
    scaler = RobustScaler(quantile_range=(25.0,75.0), copy=True, with_centering=True, with_scaling=True).fit(train)
    train_scaled = pd.DataFrame(scaler.transform(train), columns=train.columns.values).set_index([train.index.values])
    test_scaled = pd.DataFrame(scaler.transform(test), columns=test.columns.values).set_index([test.index.values])
    return scaler, train_scaled, test_scaled


def split_scale_df(df):
    
    train, test = split_scale.split_my_data(df,train_ratio=.8,seed=123)

    scaler, train, test = split_scale.standard_scaler(train,test)

    X_train = train.drop(columns='tax_value')
    y_train = train[['tax_value']]
    X_test = test.drop(columns='tax_value')
    y_test = test[['tax_value']]
    ols_model = ols('y_train ~ X_train',data=train).fit()
    train['yhat'] = ols_model.predict(y_train)
    return train,test,X_train,y_train,X_test,y_test,ols_model

def split_unscale_df(df):
    
    train, test = split_scale.split_my_data(df,train_ratio=.8,seed=123)

    #scaler, train, test = split_scale.standard_scaler(train,test)

    X_train = train.drop(columns='tax_value')
    y_train = train[['tax_value']]
    X_test = test.drop(columns='tax_value')
    y_test = test[['tax_value']]
    ols_model = ols('y_train ~ X_train',data=train).fit()
    train['yhat'] = ols_model.predict(y_train)
    return train,test,X_train,y_train,X_test,y_test,ols_model
