def tax_data_clean(df):
    df['county']=df['fips']
    df['county'] = np.where(df['fips']== 6037,'Los Angles',(np.where(df['fips']== 6059,'Orange',(np.where(df['fips']==6111,'Ventura',"")))))
