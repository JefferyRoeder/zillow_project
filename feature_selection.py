import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score


def coef_lasso(y_train,X_train):
    reg = LassoCV()
    reg.fit(X_train, y_train)
    coef = pd.Series(reg.coef_, index = X_train.columns)
    
    return coef.sort_values(ascending=False)


def r2d2(y_train,train_yhat):
    r2_lm1 = r2_score(y_train,train_yhat)
    return r2_lm1