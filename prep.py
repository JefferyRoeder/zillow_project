import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def tax_data_clean(df):
    df['county']=df['fips']
    df['county'] = np.where(df['fips']== 6037,'Los Angles',(np.where(df['fips']== 6059,'Orange',(np.where(df['fips']==6111,'Ventura',"")))))
    return df


def df_clean(df):
    df.drop('id',axis=1,inplace=True)
    df.drop('transactiondate',axis=1,inplace=True)
    return df

def distribution_plot(df):
    sns.distplot(df['bedrooms'],kde=False,rug=True)
    plt.show()
    sns.distplot(df['bathrooms'],kde=False,rug=True)
    plt.show()
    sns.distplot(df['sqft'],kde=False,rug=True)
    plt.show()