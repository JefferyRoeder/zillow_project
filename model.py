import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
from sklearn.preprocessing import StandardScaler, PowerTransformer, MinMaxScaler, RobustScaler, QuantileTransformer
from sklearn.model_selection import train_test_split

def residual(train):
    #creating residual of tax value vs expected tax value
    train['residual'] = train['yhat'] - train['tax_value']
    train['residual2'] = train.residual **2
    return train

from sklearn.linear_model import LinearRegression
#pick model type, Linear Regression for the MVP baseline
def linear_model(y_train,X_train):
    lm1 = LinearRegression()
    lm1.fit(X_train,y_train)
    lm1_y_intercept = lm1.intercept_
    lm1_coefficients = lm1.coef_

    print('{} = b + m1 * {} + m2 * {}'.format(y_train.columns[0], X_train.columns[0],X_train.columns[1]))
    print('    y-intercept  (b): %.2f' % lm1_y_intercept)
    print('    coefficient (m1): %.2f' % lm1_coefficients[0][0])
    print('    coefficient (m2): %.2f' % lm1_coefficients[0][1])

import statsmodels.api as sm
from statsmodels.formula.api import ols
import math
def errors(train,y_train,X_train):
    t_test = stats.ttest_ind(y_train,X_train.sqft)
    ols_model = sm.OLS(y_train,X_train)
    fit = ols_model.fit()
    train['residual'] = train['yhat'] - train['tax_value']
    train['residual2'] = train.residual **2
    sse = sum(train.residual2)
    mse = sse/len(train)
    rmse = math.sqrt(mse)
    #r2 = ols_model.rsquared
    print('SSE: ', sse,'\n MSE: ',mse,'\n RMSE: ',rmse)


def model_plot(y_train,X_train,train):

    train['tax_mean'] = train.tax_value.mean()

    y_train = y_train.tax_value
    df = pd.DataFrame({'actual': y_train,
                'lm1': train.yhat,
                'lm_baseline': train.tax_mean})\
    .melt(id_vars=['actual'], var_name='model', value_name='prediction')\
    .pipe((sns.relplot, 'data'), x='actual', y='prediction', hue='model')


    min, max = 0, 5000000
    plt.plot([min, max], [min, max], c='black', ls=':')
    plt.ylim(min, max)
    plt.xlim(min, max)
    plt.title('Predicted vs Actual Tax Value')
    plt.show()
    return df
