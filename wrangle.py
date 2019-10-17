import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import split_scale
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import wrangle
import env

url = env.get_db_url('telco_churn')
def wrangle_churn():
    df = pd.read_sql("""
SELECT 
c.monthly_charges, c.tenure, c.total_charges, ct.contract_type_id as contract_type, c.internet_service_type_id AS internet_type
FROM customers c
JOIN contract_types ct USING(contract_type_id)
WHERE c.total_charges != ' '
""",url)

    df.total_charges = df.total_charges.astype(float)
    return df



#X and y train and test
def train_test(data_frame):
    train, test = split_scale.split_my_data(data_frame)
    return train,test
    # scaler, train,test = split_scale.standard_scaler(train,test)
    # #X and y train and test
    # X_train = train.drop(columns='total_charges')
    # y_train = train[['total_charges']]
    # X_test = test.drop(columns='total_charges')
    # y_test = test[['total_charges']]
    # return X_train,y_train,X_test,y_test