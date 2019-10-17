import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import split_scale
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import wrangle
import env

url = env.get_db_url('zillow')
def wrangle_zillow():
    df = pd.read_sql("""

SELECT 
p17.transactiondate,p.id,p.bathroomcnt as bathrooms,p.bedroomcnt as bedrooms, p.calculatedfinishedsquarefeet as sqft, p.taxvaluedollarcnt as tax_value
FROM propertylandusetype pl
JOIN
properties_2017 p ON p.propertylandusetypeid = pl.propertylandusetypeid
JOIN
predictions_2017 p17 ON p17.id = p.id
WHERE 
p.propertylandusetypeid in (279,261) 
AND 
(p17.transactiondate LIKE '%%2017-05%%' or p17.transactiondate LIKE '%%2017-06%%')
AND
p.calculatedfinishedsquarefeet IS NOT NULL
and
p.bedroomcnt > 0
and 
p.bathroomcnt > 0
and
p.taxvaluedollarcnt > 0
"""
,url)
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