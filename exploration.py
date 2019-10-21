import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score

def corr_plot(train):
    plt.figure(figsize=(8,6))
    cor = train[['bathrooms','bedrooms','sqft','tax_value']].corr()
    sns.heatmap(cor,annot=True,cmap=plt.cm.Blues)
    plt.show()


import statsmodels.api as sm
from statsmodels.formula.api import ols
#OLS object to analyze features

def stats_test(y_train,X_train):
    t_test = stats.ttest_ind(y_train,X_train.sqft)
    ols_model = sm.OLS(y_train,X_train)
    fit = ols_model.fit()
    return t_test, fit.summary()

def coef_lasso(y_train,X_train):
    reg = LassoCV()
    reg.fit(X_train, y_train)
    coef = pd.Series(reg.coef_, index = X_train.columns)
    
    return coef.sort_values(ascending=False)


def r2d2(y_train,train_yhat):
    r2_lm1 = r2_score(y_train,train_yhat)
    return r2_lm1